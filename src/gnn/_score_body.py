"""
_score_body.py — CORPS des sections 11-16 : extraction des embeddings, score
d'importance émergent (5 composantes), K-means, baselines (MLP/DeepWalk/stat, gated),
annotations BDD sénescence, visualisations, assemblage + export du ranking, edge_stats,
vgae_metrics.json, group_expression.

Les BDD de validation (§14/§16) proviennent du **registre déclaratif**
`data/gene_sets/registry.yaml` via `src/data/loaders/gene_sets.py` (sniff +
health-check + mode DB-free ; plus aucun téléchargement au runtime). Colonnes de
sortie généralisées `in_<name>` + `n_gene_sets` (alias `n_databases`). Cf.
technical/gene_sets.md, design_log §35 (2026-07-29).

⚠️ Comme les autres *_body.py : COMPILÉ puis exécuté par `_score.score_and_write()`
dans un dict-namespace pré-rempli (model + embeddings issus du train + bundle graphe +
config + chemins + imports). Sémantique module-level exacte du monolithe. Ne pas
importer/exécuter directement.

Extrait verbatim du monolithe gnn_vgae.py (split Tier 2.5). Écrit gene_ranking_vgae.csv,
gene_embeddings_vgae.csv, vgae_weights.pt, vgae_metrics.json, group_expression.tsv.
"""
# flake8: noqa  (noms résolus à l'exécution via le namespace injecté)
# =============================================================================
# 11. EXTRACTION DES EMBEDDINGS ET SCORE D'IMPORTANCE ÉMERGENT
# =============================================================================
# C'est la section la plus importante du pipeline : on extrait les embeddings
# du VGAE entraîné et on calcule un SCORE D'IMPORTANCE COMPOSITE à 5
# composantes. Ce score est "émergent" car il n'est PAS défini a priori
# (contrairement au log2FC qui est un test statistique). Il ÉMERGE de
# l'espace latent appris par le VGAE à partir de la topologie du graphe.
#
# Les 5 composantes capturent 5 aspects différents de l'"importance" :
#   1. emb_norm (norme de μ) : remarquabilité topologique
#   2. density (k-NN cosinus) : centralité dans l'espace latent
#   3. recon_fidelity (1 - erreur) : fiabilité de la reconstruction
#   4. certainty (1 - σ) : stabilité/confiance du modèle
#   5. specificity (entropie de Shannon) : spécificité d'expression
#
# Score final = moyenne des 5 composantes (pas de poids arbitraires).
print("\n" + "=" * 70)
print("11. Score d'importance émergent")
print("=" * 70)

# Extraire les embeddings du meilleur modèle (mode évaluation → pas de bruit).
# On utilise μ (la moyenne du posterior) comme embedding final, pas z
# (échantillon bruité). μ est déterministe et plus stable pour le ranking.
model.eval()
with torch.no_grad():
    z_final, mu_final, logvar_final = model.encode(x_dict, edge_index_dict, edge_attr_dict)
    gene_emb = mu_final.numpy()  # (n_genes, latent_dim=64)

# ── Score 1 : Excentricité dans l'espace latent ────────────────────────────────
# La norme L2 de μ mesure à quel point un gène est "excentrique" par rapport
# au centre de l'espace latent. Avec le décodeur cosinus, la norme est libre
# (non contrainte par la reconstruction) — une grande norme signifie que la
# KL n'a pas réussi à ramener ce gène vers la prior, ce qui indique un
# profil topologique distinctif que le modèle "refuse" de simplifier.
emb_norm = np.linalg.norm(gene_emb, axis=1).astype(np.float32)
emb_norm_score = emb_norm / (emb_norm.max() + 1e-8)

# ── Score 2 : Reconstruction error par gène ──────────────────────────────────
# Pour chaque gène, on mesure à quel point le VGAE reconstruit correctement
# ses arêtes. Un gène mal reconstruit a un comportement atypique dans le réseau.
with torch.no_grad():
    z_eval = mu_final
    recon_error = np.zeros(n_genes, dtype=np.float32)
    edge_count = np.zeros(n_genes, dtype=np.float32)

    # Pour chaque arête positive, calculer l'erreur via le décodeur cosinus
    # (cohérent avec le décodeur utilisé pendant l'entraînement)
    edge_list = list(all_gene_edges)
    edge_src = torch.tensor([pair[0] for pair in edge_list], dtype=torch.long)
    edge_dst = torch.tensor([pair[1] for pair in edge_list], dtype=torch.long)
    edge_idx = torch.stack([edge_src, edge_dst])
    scores = torch.sigmoid(model.decode(z_eval, edge_idx)).numpy()
    for i, (s, d) in enumerate(edge_list):
        error = 1.0 - scores[i]
        recon_error[s] += error
        recon_error[d] += error
        edge_count[s] += 1
        edge_count[d] += 1

    # Moyenne par gène (éviter division par zéro pour les gènes sans arête)
    safe_count = np.where(edge_count > 0, edge_count, 1.0)
    recon_error_mean = np.where(edge_count > 0, recon_error / safe_count, 0.0)
    recon_error_score = recon_error_mean / (recon_error_mean.max() + 1e-8)

# ── Score 3 : Incertitude (σ) ────────────────────────────────────────────────
# Un gène avec une grande σ est un gène dont le VGAE est "incertain" —
# le réseau autour de ce gène ne contraint pas bien son embedding.
sigma = torch.exp(0.5 * logvar_final).numpy()
sigma_mean = sigma.mean(axis=1).astype(np.float32)
uncertainty_score = sigma_mean / (sigma_mean.max() + 1e-8)

# ── Score 4 : Densité locale (k-NN dans l'espace latent) ────────────────────
# Un gène entouré de beaucoup de voisins proches est un hub fonctionnel.
#
# CORRECTION : avec le décodeur cosinus, on utilise maintenant la similarité
# cosinus pour la k-NN (cohérent avec l'entraînement). De plus, on normalise
# le score par RANG (pas par valeur) pour éliminer l'artefact des "gènes
# fantômes" qui formaient un paquet isolé avec densité=1.0 dégénérée.
# Le rang est invariant aux échelles : un paquet isolé n'aura plus
# automatiquement le score max.
from sklearn.neighbors import NearestNeighbors
k_neighbors = min(20, n_genes - 1)

# Normaliser les embeddings pour utiliser la similarité cosinus
# (cohérent avec le décodeur)
gene_emb_normed = gene_emb / (np.linalg.norm(gene_emb, axis=1, keepdims=True) + 1e-8)
knn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="cosine")
knn.fit(gene_emb_normed)
distances, neighbors = knn.kneighbors(gene_emb_normed)
# Exclure le gène lui-même (premier voisin = lui-même avec distance 0)
distances = distances[:, 1:]
neighbors = neighbors[:, 1:]

# Densité brute = inverse de la distance moyenne (cosine ∈ [0, 2])
mean_dist = distances.mean(axis=1)
local_density_raw = 1.0 / (mean_dist + 1e-8)

# Normalisation par RANG plutôt que par valeur — élimine les dégénérescences
# où plusieurs gènes ont la même distance dégénérée (paquets isolés)
from scipy.stats import rankdata
density_ranks = rankdata(local_density_raw, method="average")
density_score = (density_ranks / density_ranks.max()).astype(np.float32)

# Pénaliser les "paquets isolés" : si tous les voisins d'un gène sont entre eux
# à des distances quasi identiques (signe d'un cluster dégénéré), on réduit
# son score. Mesure : variance des distances aux k voisins.
dist_variance = distances.var(axis=1)
# Plus la variance est petite (= cluster homogène isolé), plus on pénalise
# Normaliser par rang également
variance_ranks = rankdata(dist_variance, method="average") / len(dist_variance)
# Score final : densité × diversité du voisinage
density_score = (density_score * variance_ranks).astype(np.float32)
density_score = density_score / (density_score.max() + 1e-8)

# ── Score 5 : Spécificité d'expression (entropie de Shannon) ────────────────
# Mesure à quel point l'expression d'un gène est SPÉCIFIQUE à un sous-ensemble
# de groupes plutôt qu'ubiquitaire. Un gène marqueur d'un cluster P16 a une
# faible entropie (signal concentré), un gène housekeeping a une haute entropie.
# Pour la sénescence, les gènes spécifiques aux clusters P16 sont les plus
# intéressants. La spécificité = 1 - entropie_normalisée.
#
# IMPORTANT : on filtre aussi les gènes silencieux (mean_expression total < seuil)
# car ils créent des artefacts (gènes "fantômes" dans des paquets isolés).
print("  Calcul de la spécificité d'expression (entropie de Shannon)...")

# mean_expr_per_group a déjà été calculé en section 5 sur les gènes filtrés.
# Shape : (n_groups, n_genes) → on transpose pour avoir (n_genes, n_groups)
expr_matrix = mean_expr_per_group.T  # (n_genes, n_groups)

# Total d'expression par gène (somme sur les groupes)
total_expr = expr_matrix.sum(axis=1)
# Probabilités p_g = expression dans groupe g / total
# Avec epsilon pour éviter log(0)
eps = 1e-8
prob = expr_matrix / (total_expr[:, None] + eps)
# Entropie de Shannon : H = -Σ p_g log(p_g)
entropy = -np.sum(prob * np.log(prob + eps), axis=1)
# Normaliser par log(n_groups) pour avoir entropie ∈ [0, 1]
max_entropy = np.log(len(CELL_GROUPS))
entropy_norm = entropy / max_entropy
# Spécificité = 1 - entropie : haute pour gènes spécifiques d'un groupe
specificity = 1.0 - entropy_norm

# Filtre anti-fantômes : un gène silencieux (total_expr trop bas) a une
# spécificité non significative (artefact numérique). On la met à 0.
# Seuil : 5e percentile de l'expression totale
silent_threshold = np.percentile(total_expr, 5)
specificity[total_expr < silent_threshold] = 0.0
specificity_score = specificity.astype(np.float32)
n_silenced = int((total_expr < silent_threshold).sum())
print(f"    Gènes silencieux filtrés : {n_silenced} (expression totale < {silent_threshold:.4f})")

# ── Score composite (combinaison non pondérée, 5 composantes) ───────────────
# POURQUOI PAS DE POIDS ? Chaque composante capture un aspect ORTHOGONAL
# de l'importance. Pondérer reviendrait à décider a priori que la norme est
# "plus importante" que la densité, ce qui serait arbitraire. La moyenne
# simple laisse le score émerger de l'espace latent sans biais humain.
#
# Ce que chaque composante capture :
#   emb_norm_score        : remarquabilité topologique (le gène est "loin du centre")
#   density_score         : centralité latente (hub dans l'espace des embeddings)
#   1 - recon_error_score : fiabilité (le VGAE reconstruit bien ses arêtes)
#   1 - uncertainty_score : certitude (σ faible = le modèle est confiant)
#   specificity_score     : spécificité d'expression (concentré sur certains groupes)
#
# Les composantes "inversées" (1 - x) transforment des scores "plus haut = pire"
# en scores "plus haut = mieux" :
#   - recon_error haute = mauvaise reconstruction → on veut l'inverse
#   - uncertainty haute = grande σ = modèle incertain → on veut l'inverse
importance_score = (
    emb_norm_score
    + density_score
    + (1 - recon_error_score)    # Fiabilité = 1 - erreur
    + (1 - uncertainty_score)    # Certitude = 1 - incertitude
    + specificity_score
) / 5.0
# Normalisation finale dans [0, 1]
importance_score = importance_score / (importance_score.max() + 1e-8)

print(f"  Composantes du score :")
print(f"    Norme μ (remarquabilité) : mean={emb_norm_score.mean():.3f}")
print(f"    Densité locale (corrigée): mean={density_score.mean():.3f}")
print(f"    1-Recon error (fiabilité): mean={(1-recon_error_score).mean():.3f}")
print(f"    1-Incertitude (certitude): mean={(1-uncertainty_score).mean():.3f}")
print(f"    Spécificité (entropie)   : mean={specificity_score.mean():.3f}")
print(f"    Score composite          : mean={importance_score.mean():.3f}")

top_idx = np.argsort(importance_score)[::-1][:15]
print(f"\n  Top 15 gènes par score d'importance émergent :")
print(f"    {'Gène':15s} {'Score':>7s} {'Norme':>7s} {'Densit':>7s} {'Recon':>7s} {'Certit':>7s} {'Spéc':>7s}")
for idx in top_idx:
    print(f"    {gene_symbols[idx]:15s} {importance_score[idx]:.4f} "
          f"{emb_norm_score[idx]:.4f} {density_score[idx]:.4f} "
          f"{1-recon_error_score[idx]:.4f} {1-uncertainty_score[idx]:.4f} "
          f"{specificity_score[idx]:.4f}")

# ── Clustering des embeddings ────────────────────────────────────────────────
# K-means sur les embeddings μ pour identifier des groupes fonctionnels
# de gènes. Les gènes d'un même cluster ont des embeddings proches →
# ils occupent un rôle similaire dans le graphe biologique.
# Le silhouette score mesure la qualité du clustering :
#   > 0.3 = structure claire, < 0.1 = pas de structure nette.
# n_init=10 : K-means est lancé 10 fois avec des initialisations différentes
# et le meilleur résultat (inertie minimale) est retenu.
print(f"\n  Clustering K-means (k={N_CLUSTERS})...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
gene_clusters = kmeans.fit_predict(gene_emb)
sil_score = silhouette_score(gene_emb, gene_clusters)
print(f"    Silhouette score : {sil_score:.4f}")

for c in range(N_CLUSTERS):
    n_in = (gene_clusters == c).sum()
    mean_imp = importance_score[gene_clusters == c].mean()
    print(f"    Cluster {c} : {n_in:5d} gènes, importance moyenne = {mean_imp:.4f}")

# ── Étapes optionnelles : toggles --no-baselines / --no-validation ───────────
RUN_BASELINES = not getattr(CLI_ARGS, "no_baselines", False)
RUN_VALIDATION = not getattr(CLI_ARGS, "no_validation", False)
# Sentinelles : si une section est sautée, ses variables aval restent définies.
mlp_auc = float("nan"); mlp_ap = float("nan")
mlp_gene_score = None; node2vec_score = None
databases = []
if not RUN_BASELINES:
    print("[skip] baselines entraînées (MLP §12 + DeepWalk §13bis) — --no-baselines")
if not RUN_VALIDATION:
    print("[skip] validation post-hoc BDD aging (§14) — --no-validation")

# ── Registre déclaratif des ensembles de gènes (validation post-hoc) ─────────
# Chargé UNE fois, ici, de façon INCONDITIONNELLE (les colonnes d'annotation
# §16 sont produites même sous --no-validation). Offline-first (aucun DL au
# runtime, cf. scripts/fetch_gene_sets.py). Registre absent ⇒ GENE_SETS = []
# ⇒ toutes les features BDD passent en OFF (aucun crash). Le health-check
# aligne chaque set sur l'univers du graphe (gene_symbols) et déclasse en
# AUTO_OFF les sets sans recouvrement (mauvais espace d'ID / espèce).
GENE_SETS = []
try:
    import sys as _sys
    # BASE_DIR = racine projet (fourni par le namespace appelant ; `__file__`
    # est retiré du ns par _score.exec — ne pas s'en servir ici).
    _loaders = os.path.join(BASE_DIR, "src", "data", "loaders")
    if _loaders not in _sys.path:
        _sys.path.insert(0, _loaders)
    import gene_sets as _gs  # noqa: E402

    _alias_map = {}
    try:  # rattrape les renommages HGNC (MARCH1→MARCHF1…) avant l'intersection
        from hgnc_alias import build_alias_map as _build_alias  # noqa: E402
        _alias_map = _build_alias(cache_dir=os.path.join(DATA_DIR, "omnipath"),
                                  download_if_missing=False)
    except Exception:  # noqa: BLE001
        _alias_map = {}

    GENE_SETS = _gs.load_registry(root=BASE_DIR)
    _gs.health_check(GENE_SETS, set(gene_symbols), alias_map=_alias_map or None)
    _gs.log_health(GENE_SETS)
    try:
        _gs.health_table(GENE_SETS).to_csv(
            os.path.join(OUT_DIR, "db_health.tsv"), sep="\t", index=False)
    except Exception as _e:  # noqa: BLE001
        print(f"  [warn] écriture db_health.tsv KO ({_e})")
    # `databases` (name, gènes) alimente §14 (Mann-Whitney), la PCA §15b et la
    # violin §15c. Exclut role='anchor' (peut toucher l'axe → hors validation).
    databases = _gs.validation_pairs(GENE_SETS)
except Exception as _e:  # noqa: BLE001 — jamais fatal : mode DB-free
    print(f"  [warn] registre gene-sets indisponible ({type(_e).__name__}: {_e}) "
          f"— annotations/validation BDD OFF.")
    GENE_SETS, databases = [], []

if RUN_BASELINES:
    # =============================================================================
    # 12. BASELINE MLP (même features, pas de graphe)
    # =============================================================================
    # OBJECTIF DE CETTE BASELINE : mesurer l'apport de la TOPOLOGIE du graphe.
    # Le MLP utilise les MÊMES features que le VGAE (is_tf, variance, degree, etc.)
    # mais NE fait PAS de message passing. Il prédit directement si une arête
    # existe entre deux gènes en concaténant leurs features.
    #
    # Si VGAE AUC >> MLP AUC → la structure du graphe (qui est voisin de qui)
    # apporte de l'information au-delà des features brutes.
    # Si VGAE AUC ≈ MLP AUC → les features seules suffisent, le graphe n'aide pas.
    #
    # C'est un test d'ABLATION : on retire le graphe et on regarde si le modèle
    # perd en performance. Si oui, le message passing est utile.
    print("\n" + "=" * 70)
    print("12. Baseline MLP (sans graphe)")
    print("=" * 70)


    class MLPBaseline(nn.Module):
        """
        MLP baseline pour la prédiction de liens (sans graphe).

        Pour prédire si une arête (i, j) existe :
          1. On concatène les features de i et j : h = [x_i ; x_j]  (dim = 2×gene_in)
          2. On passe h dans un MLP à 3 couches : 2×gene_in → hidden → hidden/2 → 1
          3. La sortie est un logit (pas de sigmoid, appliquée dans la loss)

        Le MLP n'a AUCUNE notion de voisinage ou de graphe : il ne voit que
        les features des deux gènes. Pas de message passing.
        """

        def __init__(self, in_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim * 2, hidden_dim),   # Concaténation → hidden
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),  # hidden → hidden/2
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),       # hidden/2 → 1 logit
            )

        def forward(self, x, edge_index):
            src, dst = edge_index
            # Concaténation des features des deux gènes de chaque paire
            h = torch.cat([x[src], x[dst]], dim=1)  # (n_pairs, 2×in_dim)
            return self.net(h).squeeze(-1)  # (n_pairs,) logits


    mlp = MLPBaseline(gene_features.shape[1], MLP_HIDDEN)
    mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=MLP_LR)

    # Entraînement du MLP — même loss et même split que le VGAE
    mlp.train()
    for epoch in range(MLP_EPOCHS):
        mlp_optimizer.zero_grad()

        # Mêmes arêtes positives et négatifs que le VGAE
        pos_scores = mlp(gene_features, pos_train_bidir)
        neg_edges = negative_sampling(pos_train_bidir, num_nodes=n_genes,
                                      num_neg_samples=pos_train_bidir.shape[1])
        neg_scores = mlp(gene_features, neg_edges)

        loss = (F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
                + F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores)))
        loss.backward()
        mlp_optimizer.step()

    # Évaluation MLP sur les mêmes arêtes test que le VGAE
    mlp.eval()
    with torch.no_grad():
        pos_s = torch.sigmoid(mlp(gene_features, pos_test_bidir))
        neg_test_mlp = negative_sampling(pos_test_bidir, num_nodes=n_genes,
                                         num_neg_samples=pos_test_bidir.shape[1])
        neg_s = torch.sigmoid(mlp(gene_features, neg_test_mlp))

        mlp_labels = torch.cat([torch.ones_like(pos_s), torch.zeros_like(neg_s)])
        mlp_scores = torch.cat([pos_s, neg_s])
        mlp_auc = roc_auc_score(mlp_labels.numpy(), mlp_scores.numpy())
        mlp_ap = average_precision_score(mlp_labels.numpy(), mlp_scores.numpy())

    print(f"  MLP — Test AUC: {mlp_auc:.4f}, AP: {mlp_ap:.4f}")
    print(f"  VGAE — Test AUC: {best_test_auc:.4f}")
    print(f"  Δ AUC (VGAE - MLP) = {best_test_auc - mlp_auc:+.4f}")
    if best_test_auc > mlp_auc:
        print(f"  → La topologie du graphe apporte +{(best_test_auc - mlp_auc)*100:.1f}% d'AUC")
    else:
        print(f"  → Le MLP fait aussi bien/mieux — la topologie n'aide pas ici")

    # --- Score MLP par gène (analogue de vgae_recon_fidelity) -------------------
    # Pour chaque gène, on prend la PROBABILITÉ MOYENNE que ses vraies arêtes
    # soient correctement prédites par le MLP. C'est la version sans-graphe
    # de `recon_fidelity` : un gène est "important MLP" si ses interactions
    # sont prédictibles uniquement à partir des features concaténées des deux
    # gènes, sans message passing. Utile comme 2e baseline pour voir ce que
    # le message passing apporte vraiment au scoring (pas juste à l'AUC).
    mlp.eval()
    with torch.no_grad():
        all_pos = torch.cat([pos_train_bidir, pos_test_bidir], dim=1)
        pos_scores_all = torch.sigmoid(mlp(gene_features, all_pos)).numpy()
    src_np = all_pos[0].numpy()
    dst_np = all_pos[1].numpy()
    mlp_score_sum = np.zeros(n_genes, dtype=np.float32)
    mlp_edge_cnt = np.zeros(n_genes, dtype=np.float32)
    np.add.at(mlp_score_sum, src_np, pos_scores_all)
    np.add.at(mlp_score_sum, dst_np, pos_scores_all)
    np.add.at(mlp_edge_cnt, src_np, 1.0)
    np.add.at(mlp_edge_cnt, dst_np, 1.0)
    mlp_gene_score = np.where(mlp_edge_cnt > 0,
                              mlp_score_sum / np.maximum(mlp_edge_cnt, 1.0),
                              0.0).astype(np.float32)
    # Normalisation [0, 1] pour comparer aux autres scores.
    mlp_gene_score = mlp_gene_score / (mlp_gene_score.max() + 1e-8)
    print(f"  Score MLP par gène : mean={mlp_gene_score.mean():.4f}, "
          f"gènes couverts={int((mlp_edge_cnt > 0).sum())}/{n_genes}")
    top_mlp = np.argsort(mlp_gene_score)[::-1][:10]
    print(f"  Top 10 gènes (baseline MLP) :")
    for idx in top_mlp:
        print(f"    {gene_symbols[idx]:15s} score={mlp_gene_score[idx]:.4f}")

# =============================================================================
# 13. BASELINE STATISTIQUE (ranking par expression différentielle)
# =============================================================================
# OBJECTIF : comparer le VGAE avec la méthode la plus simple possible —
# ranger les gènes par leur différence d'expression absolue entre P4 et P16.
# C'est un proxy de |log2FC| (sans test statistique formel car n=1 par condition).
#
# Si le VGAE identifie des gènes à haut score qui ont un BAS score statistique
# → ce sont des "candidats de découverte" : des gènes qui ne changeraient pas
# d'expression de manière dramatique mais qui ont un rôle important dans le
# réseau biologique (hubs, régulateurs, etc.). C'est la plus-value du VGAE.
print("\n" + "=" * 70)
print("13. Baseline statistique (ranking expression)")
print("=" * 70)

# Pseudo log2FC : |mean(avancé) - mean(baseline)| sur les données LogNormalize.
# Baseline = CELL_GROUPS[0] (HUVEC : P4 ; bulk : pro). "Avancé global" = moyenne
# des autres groupes (HUVEC : 4 clusters P16 ; bulk : sen). Générique au nommage
# et au nombre de groupes (byte-identique HUVEC).
p4_mean = group_stats[CELL_GROUPS[0]]["mean_expression"]
p16_means = np.array([
    group_stats[g]["mean_expression"] for g in CELL_GROUPS[1:]
])
p16_global_mean = p16_means.mean(axis=0)  # Moyenne sur les groupes "avancés"

# Valeur absolue car on s'intéresse aux gènes dérégulés (up OU down)
pseudo_lfc = np.abs(p16_global_mean - p4_mean).astype(np.float32)
stat_score = pseudo_lfc / (pseudo_lfc.max() + 1e-8)  # Normalisation [0, 1]

print(f"  Score statistique (|ΔExpr P16-P4|) : mean={stat_score.mean():.4f}")
top_stat = np.argsort(stat_score)[::-1][:10]
print(f"  Top 10 gènes (baseline stat) :")
for idx in top_stat:
    print(f"    {gene_symbols[idx]:15s} score={stat_score[idx]:.4f}")

if RUN_BASELINES:
    # =============================================================================
    # 13bis. BASELINE GRAPHE SANS VAE — DeepWalk (random walks + skip-gram)
    # =============================================================================
    # OBJECTIF : isoler la contribution du VAE par rapport à la pure topologie
    # du graphe. On apprend des embeddings de gènes via des random walks
    # uniformes + skip-gram (word2vec) — on utilise la STRUCTURE du graphe mais :
    #   - pas de features de noeud (ignore gene_features)
    #   - pas de reconstruction probabiliste (pas de μ, pas de σ)
    #   - pas d'hétérogénéité de types d'arêtes (on aplatit tout)
    #
    # Si le ranking VGAE ≈ ranking DeepWalk → le VGAE n'apporte que la topologie,
    # pas le VAE (et pas les features). Si VGAE ≠ DeepWalk → le VAE + features
    # apportent quelque chose qu'on peut défendre en soutenance.
    #
    # NOTE : implémentation self-contained en pur PyTorch (pas de dépendance
    # torch-cluster, qui fournit normalement le C++ des random walks dans
    # PyG.Node2Vec). Walks uniformes = DeepWalk = Node2Vec(p=1, q=1).
    print("\n" + "=" * 70)
    print("13bis. Baseline DeepWalk (graphe sans VAE, sans features)")
    print("=" * 70)

    # Concaténer toutes les arêtes gène↔gène en un graphe homogène non orienté.
    _gene_edges_list = []
    for et in data.edge_types:
        if et[0] == "gene" and et[2] == "gene":
            ei = data[et].edge_index
            if ei.numel() > 0:
                _gene_edges_list.append(ei)
    _all_gene_ei = (torch.cat(_gene_edges_list, dim=1) if _gene_edges_list
                    else torch.zeros((2, 0), dtype=torch.long))
    # Symétrisation (random walks sur graphe non orienté)
    _all_gene_ei = torch.cat([_all_gene_ei, _all_gene_ei.flip(0)], dim=1)
    # Dédoublonnage des arêtes
    _u = torch.unique(_all_gene_ei.t(), dim=0).t()

    # --- Adjacence en format CSR pour échantillonner des voisins en vectoriel ---
    # adj_flat   : (E,) IDs des voisins, concaténés par noeud source
    # adj_offset : (n+1,) décalages → les voisins du noeud i sont
    #              adj_flat[adj_offset[i] : adj_offset[i+1]]
    # degrees    : (n,) nombre de voisins par noeud
    _src = _u[0].cpu().numpy()
    _dst = _u[1].cpu().numpy()
    # Tri par source pour construire CSR
    _sort_idx = np.argsort(_src, kind="stable")
    _src = _src[_sort_idx]
    _dst = _dst[_sort_idx]
    degrees = np.bincount(_src, minlength=n_genes).astype(np.int64)
    adj_offset = np.concatenate([[0], np.cumsum(degrees)]).astype(np.int64)
    adj_flat = torch.as_tensor(_dst, dtype=torch.long)
    adj_offset_t = torch.as_tensor(adj_offset, dtype=torch.long)
    degrees_t = torch.as_tensor(degrees, dtype=torch.long)
    n_isolated = int((degrees_t == 0).sum())
    print(f"  Graphe gène↔gène homogène : {n_genes} noeuds, "
          f"{_u.shape[1]} arêtes (dédupliquées), {n_isolated} isolés")


    def _random_walk_step(current, adj_flat, adj_offset_t, degrees_t):
        """Un pas de marche aléatoire uniforme, vectorisé.

        Pour chaque noeud courant :
          - si deg > 0 : tirer un voisin uniformément parmi adj
          - si deg = 0 : rester sur place (pas de voisin disponible)
        """
        deg = degrees_t[current]                           # (B,)
        rand = torch.rand(current.shape[0])                # (B,) ∈ [0,1)
        # Index d'un voisin dans adj_flat = offset[node] + ⌊rand × deg⌋
        safe_deg = deg.clamp(min=1)                         # évite div par 0
        idx = (rand * safe_deg.float()).long().clamp(max=safe_deg - 1)
        neighbor = adj_flat[adj_offset_t[current] + idx]   # (B,)
        return torch.where(deg > 0, neighbor, current)


    def _random_walks(start_nodes, walk_length):
        walks = torch.zeros(start_nodes.shape[0], walk_length, dtype=torch.long)
        walks[:, 0] = start_nodes
        for step in range(1, walk_length):
            walks[:, step] = _random_walk_step(
                walks[:, step - 1], adj_flat, adj_offset_t, degrees_t)
        return walks


    # --- Hyperparamètres du skip-gram ---
    WALKS_PER_NODE = 10
    WALK_LENGTH = 20
    WINDOW = 5           # context_size=10 → fenêtre 5 de chaque côté
    NUM_NEG = 5          # négatifs tirés par paire positive
    N2V_EPOCHS = 5
    N2V_BATCH = 128
    N2V_LR = 0.01

    # --- Table des embeddings (une seule matrice, partagée center/context) ---
    n2v_emb_param = nn.Embedding(n_genes, LATENT_DIM)
    nn.init.uniform_(n2v_emb_param.weight,
                     a=-0.5 / LATENT_DIM, b=0.5 / LATENT_DIM)
    n2v_optim = torch.optim.Adam(n2v_emb_param.parameters(), lr=N2V_LR)

    # Pré-calcul des offsets pour extraire les paires (center, context) d'un walk.
    # Pour chaque décalage j ∈ [-WINDOW, WINDOW] \ {0}, on récupère
    # (walks[:, i], walks[:, i+j]) pour tout i valide.
    _pair_offsets = [j for j in range(-WINDOW, WINDOW + 1) if j != 0]

    n2v_emb_param.train()
    for epoch in range(N2V_EPOCHS):
        # Tous les noeuds de départ (répétés WALKS_PER_NODE fois), mélangés.
        all_starts = torch.arange(n_genes).repeat_interleave(WALKS_PER_NODE)
        all_starts = all_starts[torch.randperm(all_starts.shape[0])]

        loss_sum = 0.0
        n_batches = 0
        for b in range(0, all_starts.shape[0], N2V_BATCH):
            batch_starts = all_starts[b:b + N2V_BATCH]
            walks = _random_walks(batch_starts, WALK_LENGTH)     # (B, L)

            centers, contexts = [], []
            for j in _pair_offsets:
                i_start = max(0, -j)
                i_end = WALK_LENGTH - max(0, j)
                centers.append(walks[:, i_start:i_end].flatten())
                contexts.append(walks[:, i_start + j:i_end + j].flatten())
            centers = torch.cat(centers)                          # (P,)
            contexts = torch.cat(contexts)
            n_pairs = centers.shape[0]
            if n_pairs == 0:
                continue

            # Négatifs : tirage uniforme sur {0,…,n_genes-1}.
            neg = torch.randint(0, n_genes, (n_pairs, NUM_NEG))

            u = n2v_emb_param(centers)                            # (P, D)
            v_pos = n2v_emb_param(contexts)                       # (P, D)
            v_neg = n2v_emb_param(neg)                            # (P, K, D)

            pos_score = (u * v_pos).sum(dim=1)                    # (P,)
            neg_score = torch.bmm(v_neg, u.unsqueeze(2)).squeeze(2)  # (P, K)
            pos_loss = -F.logsigmoid(pos_score).mean()
            neg_loss = -F.logsigmoid(-neg_score).mean()
            loss = pos_loss + neg_loss

            n2v_optim.zero_grad()
            loss.backward()
            n2v_optim.step()
            loss_sum += float(loss)
            n_batches += 1

        avg = loss_sum / max(1, n_batches)
        print(f"  DeepWalk epoch {epoch+1}/{N2V_EPOCHS} — "
              f"loss_mean={avg:.4f} ({n_batches} batches)")

    n2v_emb_param.eval()
    n2v_emb = n2v_emb_param.weight.detach().cpu().numpy()         # (n_genes, LATENT_DIM)

    # Score par gène : norme + densité k-NN, même structure que vgae_importance
    # mais sans recon_fidelity / certainty / specificity (ne s'appliquent pas à
    # un modèle non probabiliste sans features).
    n2v_norm_raw = np.linalg.norm(n2v_emb, axis=1)
    n2v_norm_score = (n2v_norm_raw / (n2v_norm_raw.max() + 1e-8)).astype(np.float32)

    n2v_emb_normed = n2v_emb / (np.linalg.norm(n2v_emb, axis=1, keepdims=True) + 1e-8)
    _k = min(20, n_genes - 1)
    _knn_n2v = NearestNeighbors(n_neighbors=_k + 1, metric="cosine")
    _knn_n2v.fit(n2v_emb_normed)
    _d_n2v, _ = _knn_n2v.kneighbors(n2v_emb_normed)
    _d_n2v = _d_n2v[:, 1:]
    n2v_density_raw = 1.0 / (_d_n2v.mean(axis=1) + 1e-8)
    n2v_density_score = (n2v_density_raw / (n2v_density_raw.max() + 1e-8)).astype(np.float32)

    node2vec_score = ((n2v_norm_score + n2v_density_score) / 2.0).astype(np.float32)
    node2vec_score = node2vec_score / (node2vec_score.max() + 1e-8)

    print(f"  Score Node2Vec : mean={node2vec_score.mean():.4f}")
    top_n2v = np.argsort(node2vec_score)[::-1][:10]
    print(f"  Top 10 gènes (baseline Node2Vec) :")
    for idx in top_n2v:
        print(f"    {gene_symbols[idx]:15s} score={node2vec_score[idx]:.4f}")

if RUN_VALIDATION:
    # =============================================================================
    # 14. VALIDATION POST-HOC (BDD externes)
    # =============================================================================
    # OBJECTIF : vérifier que les gènes à haut score VGAE sont enrichis dans des
    # bases de données INDÉPENDANTES de vieillissement/sénescence.
    #
    # POINT CRUCIAL : ces bases de données ne sont JAMAIS utilisées dans
    # l'entraînement du VGAE. Elles ne servent qu'à ÉVALUER après coup (post-hoc).
    # Si les gènes à haut score VGAE sont surreprésentés dans GenAge/CellAge/etc.,
    # cela valide que le score capture quelque chose de biologiquement pertinent
    # pour la sénescence — sans circularité.
    #
    # 5 bases de données de validation :
    #   1. GenAge : gènes humains associés au vieillissement (organisme entier)
    #   2. CellAge : gènes associés à la sénescence cellulaire spécifiquement
    #   3. MSigDB aging : gene sets Hallmark liés au vieillissement (MSigDB)
    #   4. AgeAnno : DEGs du vieillissement en scRNA-seq (multi-tissus)
    #   5. Aging local : base locale de gènes liés à l'âge (custom)
    #
    # Test statistique : Mann-Whitney U unilatéral.
    #   H0 : les gènes de la BDD n'ont pas un score plus élevé que les autres
    #   H1 : les gènes de la BDD ont un score significativement plus élevé
    #   p < 0.05 → enrichissement significatif → le score capture le signal
    print("\n" + "=" * 70)
    print("14. Validation post-hoc (ensembles de gènes du registre)")
    print("   → PAS utilisées dans l'entraînement, uniquement pour évaluer")
    print("=" * 70)

    # Les sets proviennent du registre déclaratif chargé plus haut (offline,
    # sniff + health-check). `databases` = [(name, gènes)] des sets actifs de
    # rôle validation/annotation. Registre absent / tous AUTO_OFF ⇒ liste vide
    # ⇒ la validation est simplement sautée (mode DB-free, aucun crash).
    if not databases:
        print("  [skip] aucun ensemble de gènes actif — validation post-hoc OFF.")

    # ── Évaluation : les gènes à haut score sont-ils enrichis dans les BDD ? ────
    # Pour CHAQUE approche (VGAE, MLP-based, stat), on mesure l'enrichissement.
    # On utilise le Mann-Whitney U test : les gènes dans la BDD ont-ils un
    # score significativement plus élevé que les autres ?

    def evaluate_ranking(scores, score_name, gene_syms, databases):
        """
        Évalue un ranking par enrichissement dans les BDD externes.

        Pour chaque base de données, on compare les scores des gènes IN (dans la BDD)
        vs OUT (pas dans la BDD) avec un test de Mann-Whitney U unilatéral.
        Si p < 0.05, les gènes de la BDD ont un score significativement plus élevé
        → le score capture un signal pertinent pour cette BDD.
        """
        print(f"\n  [{score_name}]")
        print(f"    {'Base':20s} {'In_graph':>9s} {'MeanScore_in':>13s} "
              f"{'MeanScore_out':>14s} {'U_pvalue':>10s} {'Enrichi':>8s}")
        print("    " + "-" * 75)

        results = {}
        for db_name, db_genes in databases:
            in_graph = np.array([g in db_genes for g in gene_syms])
            n_in = in_graph.sum()
            if n_in < 5:
                print(f"    {db_name:20s} {n_in:9d}   (trop peu de gènes)")
                continue

            scores_in = scores[in_graph]
            scores_out = scores[~in_graph]
            mean_in = scores_in.mean()
            mean_out = scores_out.mean()

            # Test unilatéral : les gènes de la BDD ont-ils un score plus élevé ?
            _, p_val = mannwhitneyu(scores_in, scores_out, alternative="greater")
            enriched = "OUI" if p_val < 0.05 else "non"

            print(f"    {db_name:20s} {n_in:9d} {mean_in:13.4f} {mean_out:14.4f} "
                  f"{p_val:10.2e} {enriched:>8s}")
            results[db_name] = {"n": n_in, "mean_in": mean_in, "mean_out": mean_out,
                                "p_value": p_val, "enriched": p_val < 0.05}
        return results

    # `databases` est déjà construit depuis le registre (sets actifs). Vide en
    # mode DB-free → evaluate_ranking n'itère sur rien (aucune sortie).
    vgae_results = evaluate_ranking(importance_score, "VGAE (non supervisé)",
                                     gene_symbols, databases)
    stat_results = evaluate_ranking(stat_score, "Baseline statistique (|ΔExpr|)",
                                     gene_symbols, databases)

# =============================================================================
# 15. VISUALISATIONS
# =============================================================================
# 3 figures principales :
#   15a. Courbes d'entraînement (loss + AUC au fil des epochs)
#        → vérifie la convergence et l'absence de collapse
#   15b. PCA des embeddings (3 vues : score, clusters, BDD sénescence)
#        → visualise la structure de l'espace latent en 2D
#   15c. VGAE vs baseline statistique (scatter + violin)
#        → identifie les candidats de découverte (haut VGAE, bas stat)
print("\n" + "=" * 70)
print("15. Visualisations")
print("=" * 70)

# ── 15a. Loss et AUC d'entraînement ─────────────────────────────────────────
# Panneau gauche : loss totale (reconstruction + KL) au fil des epochs.
#   La loss devrait diminuer puis se stabiliser.
# Panneau droit : AUC test au fil des epochs.
#   AUC > 0.9 = bonne reconstruction, ~0.5 = collapse.
#   La ligne verte (MLP baseline) montre le niveau atteignable sans graphe.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, color="#3498DB", alpha=0.8)
axes[0].axvline(best_epoch, ls="--", color="grey", lw=0.8,
                label=f"Best (epoch {best_epoch+1})")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("VGAE — Loss (reconstruction + KL)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(eval_epochs, test_aucs, color="#E74C3C", alpha=0.8, marker="o", ms=3)
if RUN_BASELINES:
    axes[1].axhline(mlp_auc, color="#2ECC71", ls="--", lw=1.5, label=f"MLP baseline ({mlp_auc:.3f})")
axes[1].axhline(0.5, color="grey", ls=":", lw=1)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("AUC")
axes[1].set_title("VGAE vs MLP — Reconstruction d'arêtes (test)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "vgae_training.png"))
plt.close(fig)
print("  → vgae_training.png")

# ── 15b. PCA des embeddings ─────────────────────────────────────────────────
# PCA 2D sur les embeddings 64D pour visualisation.
# Les embeddings sont dans un espace de haute dimension (latent_dim=64) —
# la PCA projette sur les 2 axes de plus grande variance.
# 3 sous-figures :
#   Gauche : coloré par score d'importance → les gènes rouges sont les plus importants
#   Centre : coloré par cluster K-means → structure fonctionnelle
#   Droite : gènes des BDD de sénescence en rouge → vérification visuelle
pca = PCA_sk(n_components=2)
gene_pca = pca.fit_transform(gene_emb)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Coloré par score d'importance
sc0 = axes[0].scatter(gene_pca[:, 0], gene_pca[:, 1], c=importance_score,
                       cmap="YlOrRd", s=8, alpha=0.7, rasterized=True)
plt.colorbar(sc0, ax=axes[0], label="Score d'importance")
axes[0].set_title("Embeddings VGAE — Score d'importance émergent")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

# Coloré par cluster K-means
cluster_colors = plt.cm.tab10(np.linspace(0, 1, N_CLUSTERS))
for c in range(N_CLUSTERS):
    mask_c = gene_clusters == c
    axes[1].scatter(gene_pca[mask_c, 0], gene_pca[mask_c, 1],
                    c=[cluster_colors[c]], s=8, alpha=0.7, label=f"C{c}",
                    rasterized=True)
axes[1].legend(fontsize=7, ncol=2)
axes[1].set_title(f"Embeddings VGAE — K-means (k={N_CLUSTERS}, sil={sil_score:.3f})")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

# Coloré par présence dans les BDD de sénescence
any_db = np.array([
    any(g in db for _, db in databases) for g in gene_symbols
])
colors_db = np.where(any_db, "#E74C3C", "#CCCCCC")
axes[2].scatter(gene_pca[~any_db, 0], gene_pca[~any_db, 1],
                c="#CCCCCC", s=5, alpha=0.3, label="Autres", rasterized=True)
axes[2].scatter(gene_pca[any_db, 0], gene_pca[any_db, 1],
                c="#E74C3C", s=15, alpha=0.8, label="BDD sénescence", rasterized=True)
axes[2].legend()
axes[2].set_title("Embeddings VGAE — Gènes des BDD de sénescence")
axes[2].set_xlabel("PC1")
axes[2].set_ylabel("PC2")

fig.suptitle("Espace latent du VGAE (non supervisé)", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "vgae_embeddings.png"))
plt.close(fig)
print("  → vgae_embeddings.png")

# ── 15c. Comparaison des scores VGAE vs stat ────────────────────────────────
# Figure clé pour identifier les "candidats de découverte".
# Panneau gauche : scatter VGAE (y) vs stat (x).
#   - Diagonale : gènes identifiés par les deux approches (concordance)
#   - QUADRANT HAUT-GAUCHE : haut VGAE + bas stat → candidats de découverte
#     Ce sont des gènes que l'approche statistique simple (|ΔExpr|) ne
#     trouverait pas, mais que le VGAE identifie grâce à leur rôle dans
#     le réseau biologique. C'est la plus-value principale du VGAE.
#   - Quadrant bas-droite : haut stat + bas VGAE → gènes très dérégulés
#     mais pas centraux dans le réseau (housekeeping dérégulés, etc.)
# Panneau droit : violin plot du score VGAE par BDD de validation.
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter VGAE vs stat (tous les gènes en bleu)
axes[0].scatter(stat_score, importance_score, s=5, alpha=0.3,
                c="#3498DB", rasterized=True)
# Surligner les gènes des BDD de sénescence en rouge
for g_idx in range(n_genes):
    if any(gene_symbols[g_idx] in db for _, db in databases):
        axes[0].scatter(stat_score[g_idx], importance_score[g_idx],
                        s=20, c="#E74C3C", alpha=0.7, zorder=5)
axes[0].set_xlabel("Score statistique (|ΔExpr|)")
axes[0].set_ylabel("Score VGAE (émergent)")
axes[0].set_title("VGAE vs Baseline statistique")
axes[0].grid(True, alpha=0.3)

# Définition des candidats de découverte :
#   - haut VGAE = top 10% (percentile 90)
#   - bas stat = bottom 50% (médiane)
# Les gènes dans cette zone sont détectés par le VGAE MAIS PAS par la
# méthode statistique simple → découvertes potentielles.
high_vgae = importance_score > np.percentile(importance_score, 90)
low_stat = stat_score < np.percentile(stat_score, 50)
discovery_candidates = high_vgae & low_stat
n_disc = discovery_candidates.sum()
# Lignes pointillées délimitant le quadrant de découverte
axes[0].axhline(np.percentile(importance_score, 90), color="red", ls=":", alpha=0.5)
axes[0].axvline(np.percentile(stat_score, 50), color="red", ls=":", alpha=0.5)
axes[0].annotate(f"Candidats de\ndécouverte\n(n={n_disc})",
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, ha='left', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Distribution des scores par BDD
all_score_data = []
for db_name, db_genes in databases:
    in_mask = np.array([g in db_genes for g in gene_symbols])
    if in_mask.sum() > 0:
        all_score_data.extend([
            {"gene": g, "score": importance_score[i], "source": f"{db_name} (in)",
             "method": "VGAE"}
            for i, g in enumerate(gene_symbols) if in_mask[i]
        ])

if databases:
    axes[1].violinplot(
        [importance_score[np.array([g in db for g in gene_symbols])]
         for _, db in databases if np.array([g in db for g in gene_symbols]).sum() > 5],
        showmeans=True, showmedians=True,
    )
axes[1].set_xticks(range(1, len(databases) + 1))
axes[1].set_xticklabels([name for name, _ in databases], rotation=45, ha="right", fontsize=9)
axes[1].set_ylabel("Score VGAE")
axes[1].set_title("Score VGAE par base de données")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "vgae_vs_baseline.png"))
plt.close(fig)
print("  → vgae_vs_baseline.png")

# =============================================================================
# 16. EXPORT DES RÉSULTATS
# =============================================================================
# 3 fichiers de sortie principaux :
#   1. gene_ranking_vgae.csv : tableau complet (1 ligne par gène) avec tous
#      les scores, rangs, clusters, et annotations BDD. C'est le fichier
#      principal utilisé par perturb_top_genes.py pour sélectionner les
#      gènes à perturber in silico.
#   2. gene_embeddings_vgae.csv : embeddings bruts μ (n_genes × latent_dim).
#      Utilisés par gnn_perturbation.py pour charger le modèle et calculer
#      les effets des perturbations.
#   3. vgae_weights.pt : poids du modèle PyTorch (encoder + décodeur τ).
#      Utilisés par gnn_perturbation.py pour reconstruire le modèle et
#      simuler des perturbations (knockdown, knockout, overexpress).
print("\n" + "=" * 70)
print("16. Export des résultats")
print("=" * 70)

results = pd.DataFrame({"gene": gene_symbols})

# --- Scores (les 5 composantes + le composite + la baseline statistique) ---
results["vgae_importance"] = importance_score       # Score composite final
results["vgae_emb_norm"] = emb_norm_score           # Composante 1 : norme μ
results["vgae_density"] = density_score             # Composante 2 : densité k-NN
results["vgae_recon_fidelity"] = 1 - recon_error_score  # Composante 3 : 1-erreur
results["vgae_certainty"] = 1 - uncertainty_score   # Composante 4 : 1-σ
results["vgae_specificity"] = specificity_score     # Composante 5 : spécificité
results["stat_score"] = stat_score                  # Baseline 1 : |ΔExpr P16-P4| (toujours)
if RUN_BASELINES:
    results["mlp_score"] = mlp_gene_score           # Baseline 2 : MLP sans graphe (features only)
    results["node2vec_score"] = node2vec_score      # Baseline 3 : Node2Vec (graphe sans VAE, sans features)
results["cluster"] = gene_clusters                  # Cluster K-means (0 à N_CLUSTERS-1)

# --- Rangs (pour faciliter les comparaisons entre runs) ---
results["rank_vgae"] = results["vgae_importance"].rank(ascending=False).astype(int)
results["rank_stat"] = results["stat_score"].rank(ascending=False).astype(int)
if RUN_BASELINES:
    results["rank_mlp"] = results["mlp_score"].rank(ascending=False).astype(int)
    results["rank_node2vec"] = results["node2vec_score"].rank(ascending=False).astype(int)

# --- Candidats de découverte (haut VGAE, bas statistique) ---
# Flag binaire : 1 si le gène est dans le quadrant haut-gauche du scatter
results["discovery_candidate"] = discovery_candidates.astype(int)

# --- Annotations des ensembles de gènes (binaire, 0 ou 1) ---
# Une colonne `in_<name>` par set ACTIF du registre (OK/WARN) — nommage
# généralisé (plus de liste figée). Registre absent / tous AUTO_OFF ⇒ aucune
# colonne (mode DB-free propre). `n_gene_sets` = nombre de sets contenant le
# gène ; `n_databases` conservé comme alias rétro-compatible pour les scripts
# d'analyse V5.4.1. Un gène dans 0 set mais à haut score VGAE = découverte.
try:
    _ann = _gs.annotate(GENE_SETS, gene_symbols).reset_index(drop=True)
    for _c in _ann.columns:
        results[_c] = _ann[_c].values
    results["n_databases"] = results.get("n_gene_sets", 0)
except Exception as _e:  # noqa: BLE001 — jamais fatal
    print(f"  [warn] annotation gene-sets KO ({_e}) — colonnes in_* omises.")
    results["n_gene_sets"] = 0
    results["n_databases"] = 0

# Tri par score d'importance décroissant (les gènes les plus importants en premier)
results = results.sort_values("vgae_importance", ascending=False)
results.to_csv(os.path.join(OUT_DIR, "gene_ranking_vgae.csv"), index=False)
print(f"  → gene_ranking_vgae.csv ({len(results)} gènes)")

# Embeddings bruts (matrice n_genes × latent_dim) — index = symboles de gènes
emb_df = pd.DataFrame(gene_emb, index=gene_symbols)
emb_df.to_csv(os.path.join(OUT_DIR, "gene_embeddings_vgae.csv"))
print(f"  → gene_embeddings_vgae.csv ({gene_emb.shape})")

# Poids du modèle PyTorch (pour réutilisation dans les perturbations)
torch.save(model.state_dict(), os.path.join(OUT_DIR, "vgae_weights.pt"))
print("  → vgae_weights.pt")

# Sidecar JSON : historique d'entraînement + métriques finales — permet
# aux scripts d'analyse cross-seed de reconstruire courbes loss/AUC/AP
# sans devoir reparser log.txt. Format human-readable, indépendant de torch.
import json as _json

# --- Stats d'arêtes (V4.1) : breakdown par type + signe + couverture vs PPI ---
# Pour chaque edge_type construit en section 6/7, on rapporte :
#   - n_edges (taille du edge_index)
#   - n_pos / n_neg / n_unsigned (si edge_attr[:,1] = sign)
# Et aux niveaux agrégés :
#   - n_signed_total = somme des arêtes des types signés (signaling, tf_curated)
#   - frac_signed   = n_signed / total (mesure le poids causal du graphe)
#   - signed_non_ppi : arêtes signées dont la paire (a,b) n'a PAS d'arête PPI
#       → quantifie ce qu'OmniPath apporte au-delà de STRING (apport causal pur).
def _edge_stats(ei, attr=None):
    """Compte arêtes d'un type, et breakdown +/−/0 si attr[:,1]=sign."""
    n = int(ei.shape[1]) if ei.numel() > 0 else 0
    out = {"n_edges": n}
    if (attr is not None and attr.numel() > 0
            and attr.dim() == 2 and attr.shape[1] >= 2):
        # convention V4 : colonne 1 = sign ∈ {−1, 0, +1}
        sign_col = attr[:, 1]
        out["n_pos"]      = int((sign_col > 0).sum())
        out["n_neg"]      = int((sign_col < 0).sum())
        out["n_unsigned"] = int((sign_col == 0).sum())
    return out


# Catalogue { edge_type_name : (edge_index, edge_attr_ou_None) }. None pour
# les types sans edge_attr (same_pathway) ou non instanciés.
_edge_catalog = {
    "ppi":                   (edge_index_ppi,         edge_attr_ppi),
    "same_pathway":          (edge_index_pathway,     None),
    "regulates":             (edge_index_regulates,   edge_attr_regulates),
    "coexpression":          (edge_index_coexpr,      coexpr_w_tensor),
    "metabolic_cocatalysis": (edge_index_cocat,       edge_attr_cocat),
    "signaling":             (edge_index_signaling,   edge_attr_signaling),
    "tf_curated":            (edge_index_tf_curated,  edge_attr_tf_curated),
    "reactome_fi":           (edge_index_reactome_fi, edge_attr_reactome_fi),
    "expresses":             (edge_index_expresses,   edge_attr_expresses),
}
_per_type = {name: _edge_stats(ei, attr)
             for name, (ei, attr) in _edge_catalog.items()}

# Agrégats : types qui portent un signe biologique (causal/régulatoire).
# reactome_fi (V4.2) est signé (edge_attr=[score, sign]) → inclus.
_signed_types = ("signaling", "tf_curated", "regulates", "reactome_fi")
_signed_total_edges = sum(_per_type[t]["n_edges"] for t in _signed_types)
_total_gene_gene = sum(_per_type[t]["n_edges"] for t in
                       ("ppi", "same_pathway", "regulates", "coexpression",
                        "metabolic_cocatalysis", "signaling", "tf_curated",
                        "reactome_fi"))

# Couverture vs PPI : combien d'arêtes signées (signaling / tf_curated) NE
# sont PAS doublées par une arête PPI (bidirectionnelle) ? Mesure l'apport
# causal pur d'OmniPath au-delà de STRING.
_ppi_pairs: set[tuple[int, int]] = set()
if edge_index_ppi.numel() > 0:
    _ppi_pairs = {(int(s), int(d)) for s, d in zip(
        edge_index_ppi[0].tolist(), edge_index_ppi[1].tolist())}


def _count_non_ppi(ei) -> int:
    """Nb d'arêtes signées dont aucune des deux orientations n'est dans PPI."""
    if ei.numel() == 0 or not _ppi_pairs:
        return int(ei.shape[1]) if ei.numel() > 0 else 0
    cnt = 0
    for s, d in zip(ei[0].tolist(), ei[1].tolist()):
        if (int(s), int(d)) not in _ppi_pairs and (int(d), int(s)) not in _ppi_pairs:
            cnt += 1
    return cnt


_signaling_non_ppi  = _count_non_ppi(edge_index_signaling)
_tf_curated_non_ppi = _count_non_ppi(edge_index_tf_curated)
_reactome_fi_non_ppi = _count_non_ppi(edge_index_reactome_fi)

# Logs console — utile pour diag interactif sans ouvrir le JSON
print("\n  Stats arêtes (V4.1) :")
for name, st in _per_type.items():
    extra = ""
    if "n_pos" in st:
        extra = (f"  [+:{st['n_pos']} −:{st['n_neg']} "
                 f"0:{st['n_unsigned']}]")
    print(f"    {name:<22} : {st['n_edges']:>8}{extra}")
if (_per_type["signaling"]["n_edges"] > 0
        or _per_type["tf_curated"]["n_edges"] > 0
        or _per_type["reactome_fi"]["n_edges"] > 0):
    print(f"  Couverture causale hors PPI :")
    print(f"    signaling non-PPI   : {_signaling_non_ppi} "
          f"/ {_per_type['signaling']['n_edges']}")
    print(f"    tf_curated non-PPI  : {_tf_curated_non_ppi} "
          f"/ {_per_type['tf_curated']['n_edges']}")
    if _per_type["reactome_fi"]["n_edges"] > 0:
        print(f"    reactome_fi non-PPI : {_reactome_fi_non_ppi} "
              f"/ {_per_type['reactome_fi']['n_edges']}")
if _total_gene_gene > 0:
    print(f"  frac signed / total gene-gene : "
          f"{_signed_total_edges / _total_gene_gene:.3f}")

_edge_stats_block = {
    "per_type": _per_type,
    "aggregates": {
        "n_signed_total":   int(_signed_total_edges),
        "n_gene_gene_total": int(_total_gene_gene),
        "frac_signed_of_gene_gene": (
            float(_signed_total_edges / _total_gene_gene)
            if _total_gene_gene > 0 else 0.0),
    },
    "omnipath_vs_ppi": {
        "signaling_total":      _per_type["signaling"]["n_edges"],
        "signaling_non_ppi":    int(_signaling_non_ppi),
        "tf_curated_total":     _per_type["tf_curated"]["n_edges"],
        "tf_curated_non_ppi":   int(_tf_curated_non_ppi),
        "reactome_fi_total":    _per_type["reactome_fi"]["n_edges"],
        "reactome_fi_non_ppi":  int(_reactome_fi_non_ppi),
    },
    "include_omnipath_genes":   bool(MODULES["include_omnipath_genes"]),
    "n_omnipath_endpoints_in_graph": int(
        len(set(omnipath_endpoints) & set(gene_symbols))
        if omnipath_endpoints else 0),
}


# --- Drift-proof provenance helpers (mirror of gnn_vgae.py) -----------------
def _json_safe_metrics(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple, set)):
        return [_json_safe_metrics(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe_metrics(x) for k, x in v.items()}
    return str(v)


def _all_cli_args_metrics() -> dict:
    return {k: _json_safe_metrics(v) for k, v in sorted(vars(CLI_ARGS).items())}


def _runtime_env_metrics() -> dict:
    import os as _os
    return {
        "deterministic": _os.environ.get("GNN_DETERMINISTIC", "0") == "1",
        "pythonhashseed": _os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": _os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


_metrics = {
    "version": 1,
    "seed": int(getattr(CLI_ARGS, "seed", 42)),
    "run_tag": str(globals().get("RUN_TAG", "")),
    "best_epoch": int(best_epoch + 1),  # 1-indexé pour cohérence avec les logs
    "best_auc": float(best_test_auc),
    "best_ap": float(best_test_ap),
    "mlp_auc": float(mlp_auc) if RUN_BASELINES else None,
    "mlp_ap": float(mlp_ap) if RUN_BASELINES else None,
    "delta_auc_vgae_minus_mlp": float(best_test_auc - mlp_auc) if RUN_BASELINES else None,
    "n_epochs_planned": int(N_EPOCHS),
    "n_epochs_run": int(len(train_losses)),
    "early_stopped": bool(len(train_losses) < N_EPOCHS),
    "patience": int(PATIENCE),
    "n_genes": int(n_genes),
    "n_train_edges": int(len(train_edges)),
    "n_test_edges": int(len(test_edges)),
    "history": {
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": [float(x) for x in train_losses],
        "recon_loss": [float(x) for x in recon_losses],
        "kl_loss": [float(x) for x in kl_losses],
        "kl_beta": [float(x) for x in kl_betas],
    },
    "eval_history": {
        "epoch": [int(e + 1) for e in eval_epochs],
        "test_auc": [float(x) for x in test_aucs],
        "test_ap": [float(x) for x in test_aps],
    },
    "hyperparams": {
        "lr": float(LR),
        "kl_beta_max": float(KL_BETA_MAX),
        "kl_warmup_epochs": int(KL_WARMUP_EPOCHS),
        "free_bits": float(FREE_BITS),
        "latent_dim": int(LATENT_DIM),
        "edge_sample_ratio": float(EDGE_SAMPLE_RATIO),
        "grad_clip_norm": float(GRAD_CLIP_NORM),
        # V5 flags — requis par gnn_perturbation.load_run pour reconstruire
        # le bon encoder (SignedGATConv) et accepter le state_dict
        # (clés bilinear_decoder.*). Sans ces flags persistés, le load
        # silencieusement ignore les signes ou refuse le state_dict.
        "signed_message": bool(getattr(CLI_ARGS, "signed_message", False)),
        "signed_decoder": bool(getattr(CLI_ARGS, "signed_decoder", False)),
        "signed_loss_weight": float(getattr(CLI_ARGS, "signed_loss_weight", 0.0)),
        # V5.4 (decoder-split) — requis pour identifier les runs V5.4 et
        # reproduire le régime (cosinus non-signé + bilinéaire existence).
        "decoder_split": bool(getattr(CLI_ARGS, "decoder_split", False)),
        # V4.3 — choix méthode×prune amont (consommé par cross-method report).
        "coexpr_method": CLI_ARGS.coexpr_method,
        "coexpr_prune": CLI_ARGS.coexpr_prune,
        # Drift-proof provenance (2026-08-04). The curated keys above are an
        # ALLOWLIST and silently dropped every new flag (`--dedup-ppi-mirror`,
        # `--reactome-fi-directed`, determinism...), which made two runs on
        # DIFFERENT graphs indistinguishable from their metrics file. These two
        # blocks record everything; keep the named keys for existing consumers
        # (gnn_perturbation.load_run reads them by name) but never assume they
        # are complete. Same blocks are written to run_config.json.
        "cli_args": _all_cli_args_metrics(),
        "runtime_env": _runtime_env_metrics(),
    },
    "edge_stats": _edge_stats_block,
}
with open(os.path.join(OUT_DIR, "vgae_metrics.json"), "w") as _f:
    _json.dump(_metrics, _f, indent=2)
print("  → vgae_metrics.json")

# Expression par groupe cellulaire — requis par gnn_perturbation.py pour le
# shift pondéré au niveau gène. On exporte mean_expression et pct_expressing
# sous leur forme BRUTE (avant Z-scoring), car gnn_vgae.py applique un Z-score
# in-place à edge_attr_expresses ce qui détruit les valeurs originales dans le
# HeteroData sauvé.
_group_expr_rows = {"gene": gene_symbols}
for grp in CELL_GROUPS:
    _group_expr_rows[f"mean_{grp}"] = group_stats[grp]["mean_expression"]
    _group_expr_rows[f"pct_{grp}"] = group_stats[grp]["pct_expressing"]
pd.DataFrame(_group_expr_rows).to_csv(
    os.path.join(OUT_DIR, "group_expression.tsv"), sep="\t", index=False)
print(f"  → group_expression.tsv (mean + pct × {len(CELL_GROUPS)} groupes)")

# =============================================================================
# RÉSUMÉ
# =============================================================================
print("\n" + "=" * 70)
print("RÉSUMÉ — VGAE NON SUPERVISÉ + BASELINES")
print("=" * 70)

# Compter les candidats de découverte dans les BDD
disc_genes = gene_symbols[discovery_candidates]
disc_in_db = sum(1 for g in disc_genes
                 if any(g in db for _, db in databases))

_mlp_auc_str = f"{mlp_auc:.4f}" if RUN_BASELINES else "sautée (--no-baselines)"
_delta_str = ((f"{best_test_auc - mlp_auc:+.4f} "
               f"({'topologie utile' if best_test_auc > mlp_auc else 'topologie non utile'})")
              if RUN_BASELINES else "—")
print(f"""
Graphe hétérogène :
  Noeuds gene       : {n_genes} (features: is_tf, variance, ppi_degree, reg_degree)
  Noeuds cell_group : {len(CELL_GROUPS)}
  Arêtes expresses  : {edge_index_expresses.shape[1]} (7 features)
  Arêtes PPI        : {edge_index_ppi.shape[1]}
  Arêtes pathway    : {edge_index_pathway.shape[1]}
  Arêtes regulates  : {edge_index_regulates.shape[1]}
  Arêtes coexpr     : {edge_index_coexpr.shape[1] if edge_index_coexpr.numel() > 0 else 0}
  TOTAL gene↔gene   : {n_edges} arêtes uniques

VGAE :
  Architecture : {N_LAYERS} couches GATConv × {N_HEADS} têtes, latent={LATENT_DIM}
  Paramètres   : {total_params:,}
  Best epoch   : {best_epoch + 1}

Reconstruction d'arêtes (test) :
  VGAE  AUC : {best_test_auc:.4f}
  MLP   AUC : {_mlp_auc_str}
  Δ AUC     : {_delta_str}

Candidats de découverte :
  (haut score VGAE + bas score statistique)
  {n_disc} gènes, dont {disc_in_db} dans les BDD de sénescence

Fichiers : {OUT_DIR}/
Figures  : {FIG_DIR}/
""")
