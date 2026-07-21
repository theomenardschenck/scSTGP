"""
_train_body.py — CORPS des sections 8-10 : modèle VGAE, préparation des arêtes,
boucle d'entraînement (+ tête supervisée jointe optionnelle + finalize_supervised).

⚠️ Comme _graph_build_body.py : COMPILÉ puis exécuté par `_train.train_vgae()` dans
un dict-namespace pré-rempli (bundle graphe §1-7 + config + chemins + modèle + helpers
+ imports). Sémantique IDENTIQUE au niveau module du monolithe (assign/read sur le
même dict). Ne pas exécuter/importer directement.

Extrait verbatim du monolithe gnn_vgae.py (split Tier 2.5). Produit dans le namespace :
model (entraîné, best_vgae.pt rechargé), les tenseurs d'embedding, la tête supervisée
éventuelle, et les métriques — récupérés par gnn_vgae pour la section scoring (§11+).
"""
# flake8: noqa  (noms résolus à l'exécution via le namespace injecté)
# =============================================================================
# 8. MODÈLE VGAE
# =============================================================================
# Le VGAE (Variational Graph AutoEncoder) est un modèle génératif qui apprend
# des représentations (embeddings) de gènes en reconstruisant la structure
# du graphe. Il a deux composants :
#
#   1. ENCODER (HeteroGNN → μ, log(σ²)) :
#      Empile N couches de GATConv (Graph Attention Network) sur le graphe
#      hétérogène. Chaque couche agrège les messages des voisins avec un
#      mécanisme d'attention appris. Après N couches, chaque gène a un
#      vecteur h ∈ R^hidden qui encode son contexte local dans le graphe.
#      Deux têtes linéaires projettent h vers μ ∈ R^latent et log(σ²) ∈ R^latent.
#
#   2. DECODER (cosine similarity → probabilité d'arête) :
#      Prédit P(arête i↔j) = σ(τ · cos(z_i, z_j)), où z est échantillonné
#      du posterior q(z|x) = N(μ, diag(σ²)). Le décodeur est NON paramétrique
#      (pas de poids appris, seulement le scalaire τ).
#
#   LOSS = reconstruction_loss + β · KL_divergence
#      - Reconstruction : BCE entre les logits du décodeur et les labels
#        (1 pour les arêtes réelles, 0 pour les paires non connectées)
#      - KL : D_KL(q(z|x) || N(0,I)) régularise l'espace latent
#      - β : coefficient d'annealing (0 → KL_BETA_MAX)
# =============================================================================
print("\n" + "=" * 70)
print("8. Modèle VGAE")
print("=" * 70)


# Classes du modèle + SIGNED_EDGE_TYPES : SOURCE DE VÉRITÉ = _vgae_model.py
# (import-safe, sans effet de bord). Elles étaient auparavant DUPLIQUÉES ici et
# synchronisées à la main ; dédupliquées au split (Tier 2.5). Le code était
# strictement équivalent (seuls docstrings/prints diagnostiques différaient).
# Cf. docs/technical/vgae_model.md + design_log #split-gnn-vgae.
from _vgae_model import (  # noqa: E402  (import mid-module : après le catalogue de sections)
    SIGNED_EDGE_TYPES,
    SignedGATConv,
    BilinearSignedDecoder,
    _ScaledConv,
    HeteroEncoder,
    VGAE,
)


# =============================================================================
# V-sup : labels DEG pour la tête de classification (--supervised)
# =============================================================================
# Les FEATURES DE (--de-features) sont construites AU BUILD DU GRAPHE
# (_graph_build_body) — plus rien à concaténer ici. Ce bloc ne sert qu'à garantir
# la présence des LABELS pour la tête. SUP_LABELS n'est PAS propagé par le build
# (hors _CACHE_VARS : il casserait le dépicklage du cache --reuse-graph, cf. le
# commentaire dans _graph_build_body) → on le reconstruit ici quand --supervised.
# Chemin identique en build frais et en cache réutilisé.
SUP_LABELS = globals().get("SUP_LABELS")
if getattr(CLI_ARGS, "supervised", False) and SUP_LABELS is None:
    import sys as _sys_sup
    # Robuste aux 2 layouts : local (src/gnn/, src/data/preprocess/) ET cluster
    # à plat (tous les .py sous src/). On ajoute les candidats au sys.path.
    _HERE_DIR = os.path.dirname(os.path.abspath(__file__))
    for _cand in (os.path.join(_HERE_DIR, "..", "data", "preprocess"), _HERE_DIR):
        _cand = os.path.abspath(_cand)
        if os.path.isdir(_cand) and _cand not in _sys_sup.path:
            _sys_sup.path.insert(0, _cand)
    from build_supervised_labels import build_supervised_labels as _build_sup_labels
    SUP_LABELS = _build_sup_labels(
        gene_symbols, GNN_DATA_DIR,
        recompute=getattr(CLI_ARGS, "supervised_recompute_labels", False))
if getattr(CLI_ARGS, "de_features", False) or getattr(CLI_ARGS, "supervised", False):
    print("\n" + "=" * 70)
    print("MODE V-sup — VGAE reconstruit "
          f"{'+ features DE CIRCULAIRES ' if CLI_ARGS.de_features else ''}"
          f"{'+ tête classif jointe' if CLI_ARGS.supervised else ''}")
    print("=" * 70)

# --- Instanciation du modèle ---
# L'encoder prend les features brutes (8 dim gene, 3 dim cell_group) et
# produit μ, log(σ²) en dimension latent=64 pour chaque gène.
# Le VGAE enveloppe l'encoder + le décodeur cosinus avec τ.
encoder = HeteroEncoder(
    gene_in=gene_features.shape[1],       # n features actives (modulaire)
    cell_in=cell_group_features.shape[1],  # 3 features de groupe
    hidden=HIDDEN_DIM,                     # 128 (dimension cachée)
    latent=LATENT_DIM,                     # 64 (dimension de l'espace latent)
    n_layers=N_LAYERS,                     # 3 couches de message passing
    n_heads=N_HEADS,                       # 4 têtes d'attention
    dropout=DROPOUT,                       # 0.2
    available_edge_types=list(data.edge_types),  # filtre selon ablations
    # V4.2 : edge_dim overrides — coexpr 1→6 (differential) ;
    # ppi 1→2 si --dedup-ppi-signed annotate (colonne flag ajoutée).
    edge_dim_overrides=(lambda _o: _o or None)({
        **({("gene", "coexpression", "gene"): _COEXPR_DIM}
           if COEXPR_DIFFERENTIAL else {}),
        **({("gene", "ppi", "gene"): int(edge_attr_ppi.shape[1])}
           if getattr(CLI_ARGS, "dedup_ppi_signed", "off") == "annotate"
           and edge_attr_ppi.numel() > 0 else {}),
    }),
    # V4.2 : pondération γ_t par edge_type (message-level), toggleable.
    edge_type_weights=EDGE_TYPE_WEIGHTS or None,
    # V5 (TIER 1c.2) : SignedGATConv pour les edge_types signés si flag actif.
    signed_message=CLI_ARGS.signed_message,
    signed_edge_types=SIGNED_EDGE_TYPES,
)
# V5 (TIER 1c.3) : décodeur bilinéaire signé optionnel, instancié uniquement
# si --signed-decoder. Sinon, VGAE utilise uniquement le décodeur cosinus
# historique (backward-compat V4.x).
# V5.3 (TIER 1c.7) : tête sub-espace `signed_proj` paramétrée par
# --signed-decoder-dim (défaut LATENT_DIM ⇒ équivalent V5.2 numérique au
# load checkpoint).
SIGNED_DECODER_DIM = (CLI_ARGS.signed_decoder_dim
                      if CLI_ARGS.signed_decoder_dim is not None
                      else LATENT_DIM)
_bilinear_decoder = (BilinearSignedDecoder(LATENT_DIM, signed_dim=SIGNED_DECODER_DIM)
                     if CLI_ARGS.signed_decoder else None)
model = VGAE(encoder, bilinear_decoder=_bilinear_decoder)
if CLI_ARGS.signed_decoder:
    print(f"  VGAE V5 BilinearSignedDecoder actif (λ_signed="
          f"{CLI_ARGS.signed_loss_weight}, signed_dim={SIGNED_DECODER_DIM} "
          f"{'(= latent_dim, équiv V5.2 init)' if SIGNED_DECODER_DIM == LATENT_DIM else '(compression vs latent)'}).")

total_params = sum(p.numel() for p in model.parameters())
print(f"  VGAE : {N_LAYERS} couches GATConv, hidden={HIDDEN_DIM}, latent={LATENT_DIM}")
print(f"  Paramètres : {total_params:,}")

# =============================================================================
# 9. PRÉPARATION DES ARÊTES POUR L'ENTRAÎNEMENT
# =============================================================================
# Le VGAE apprend à reconstruire le graphe : étant donné des embeddings z,
# le décodeur prédit quelles paires (i, j) sont connectées. Pour évaluer
# la qualité de cette reconstruction, on a besoin d'un split train/test :
#   - TRAIN : arêtes utilisées pour calculer la loss (le modèle voit ces arêtes)
#   - TEST : arêtes masquées pendant l'entraînement (le modèle doit les prédire)
# L'AUC sur le test mesure la capacité de généralisation du modèle.
#
# IMPORTANT : on combine TOUS les types d'arêtes gene↔gene (PPI, pathway,
# régulation, coexpression, cocatalyse) en un seul pool. Le décodeur ne
# distingue pas les types d'arêtes — il prédit simplement "arête ou pas".
# Les types d'arêtes restent SÉPARÉS dans le graphe d'entrée de l'encoder
# (chaque type a son GATConv), mais la supervision est sur le pool combiné.
print("\n" + "=" * 70)
print("9. Préparation des arêtes d'entraînement")
print("=" * 70)

# Collecter toutes les arêtes gene↔gene (non dirigées, dédupliquées).
# On stocke chaque arête comme (min(i,j), max(i,j)) pour dédupliquer
# les arêtes bidirectionnelles (i→j et j→i comptent comme une seule paire).
#
# V5.2 (2026-05-29) : les edge_types SIGNÉS (signaling, tf_curated,
# reactome_fi) sont MAINTENANT inclus dans le pool de reconstruction.
# Justification (cf. §14bis.6vicies du rapport) :
#   - Avant V5.2 : signed edges étaient injectées dans le graphe pour
#     l'encoder mais ABSENTES de all_gene_edges → décodeur cosinus
#     aveugle au sous-graphe signed + fuite par negative sampling
#     (paires signed tirées comme "négatifs" alors qu'elles existent).
#   - À partir de V5.2 : signed edges contribuent à la loss recon
#     comme arêtes positives (sans distinction de signe — la sémantique
#     signed reste l'affaire du BilinearSignedDecoder via la
#     signed_aux_loss). Plus de fuite ; cosinus apprend l'existence
#     de toutes les arêtes du graphe.
#   - tf_curated_by est la copie inverse de tf_curated (mêmes paires
#     (TF, target) avec src/dst swappés) → dédup automatique via
#     (min, max), pas de double comptage.
#   - n_edges passe typiquement de ~100k à ~150k → AUC recon **non
#     comparable** aux versions V3/V4/V4.1/V5.1.
# V5.4 (decoder-split, §14bis.6duovicies) : si --decoder-split (+ --signed-decoder),
# les arêtes SIGNÉES quittent le pool cosinus (all_gene_edges) → elles sont
# reconstruites par le bilinéaire (existence, voir signed_exist_pairs plus bas).
# Le cosinus retrouve alors le pool V5.1 (non-signé) → AUC comparable V5.1 (~0.97).
# Sinon (V5.2/V5.3), les signées restent dans le pool cosinus (AUC ~0.94).
DECODER_SPLIT = (bool(getattr(CLI_ARGS, "decoder_split", False))
                 and CLI_ARGS.signed_decoder)
_unsigned_sources = [
    (ppi_src, ppi_dst),
    (react_src, react_dst),
    (reg_src, reg_dst), (reg_dst, reg_src),  # Regulates + regulated_by
    (coexpr_src, coexpr_dst),
    # V6.2 : reactome_fi non-orienté (sign=0, module) → pool cosinus non-signé.
    (globals().get("reactome_fi_und_src", []),
     globals().get("reactome_fi_und_dst", [])),
]
_signed_sources = [
    (op_sig_src, op_sig_dst),       # signaling (OmniPath kinase + SIGNOR)
    (op_tf_src, op_tf_dst),         # tf_curated (CollecTRI) — tf_curated_by inverse
    (reactome_fi_src, reactome_fi_dst),  # Reactome FI signé
]
# V6 : arêtes OmniPath supplémentaires (--omnipath-edges) → même pool signé
# (V5.2 : décodeur cosinus voit toutes les arêtes ; pas de fuite negsamp).
for _et_name, (_ex_src, _ex_dst) in globals().get(
        "omnipath_extra_edges", {}).items():
    _signed_sources.append((_ex_src, _ex_dst))
all_gene_edges = set()
for src_list, dst_list in (_unsigned_sources if DECODER_SPLIT
                           else _unsigned_sources + _signed_sources):
    for s, d in zip(src_list, dst_list):
        pair = (min(s, d), max(s, d))
        all_gene_edges.add(pair)

# V5.4 : pool d'EXISTENCE signé (décodé par le bilinéaire via
# predict_signed_existence). Paires signées dédupliquées, PRIVÉES de celles
# déjà présentes dans le pool cosinus (évite double label). Vide si pas de split.
signed_exist_pairs: set = set()
if DECODER_SPLIT:
    for src_list, dst_list in _signed_sources:
        for s, d in zip(src_list, dst_list):
            signed_exist_pairs.add((min(s, d), max(s, d)))
    signed_exist_pairs -= all_gene_edges
    print(f"  V5.4 decoder-split ON : pool cosinus {len(all_gene_edges)} "
          f"(non-signé) + pool existence bilinéaire {len(signed_exist_pairs)} (signé)")

all_edges = np.array(list(all_gene_edges))
n_edges = len(all_edges)
# Décomposition pour audit (V5.2) : combien d'arêtes ont été ajoutées par
# l'inclusion des signed ? Aide à comparer V5.1 (sans) vs V5.2 (avec).
_signed_pair_count = 0
for src_list, dst_list in [
    (op_sig_src, op_sig_dst),
    (op_tf_src, op_tf_dst),
    (reactome_fi_src, reactome_fi_dst),
]:
    for s, d in zip(src_list, dst_list):
        _signed_pair_count += 1
print(f"  Arêtes gene↔gene uniques : {n_edges} "
      f"(dont ~{_signed_pair_count} arêtes signées brutes contribuant au pool, "
      f"après dédup avec PPI/REACTOME)")

# Split train/test aléatoire — utilise CLI_ARGS.seed (défaut 42) pour
# reproductibilité ET multi-seed inter-runs (V3.6+).
# EDGE_SAMPLE_RATIO = 10% des arêtes sont réservées pour le test.
# Ce split est fait au niveau des paires non dirigées (pas des arêtes bidirectionnelles).
np.random.seed(CLI_ARGS.seed)
perm = np.random.permutation(n_edges)
n_test = max(1, int(n_edges * EDGE_SAMPLE_RATIO))
test_edge_idx = perm[:n_test]      # Premiers 10% → test
train_edge_idx = perm[n_test:]     # 90% restants → train

test_edges = all_edges[test_edge_idx]
train_edges = all_edges[train_edge_idx]

# Conversion en format PyG : tenseur [2, n_edges] avec src en ligne 0, dst en ligne 1.
pos_train = torch.tensor(train_edges.T, dtype=torch.long)
# Rendre bidirectionnel : pour chaque paire (i, j) du train, on ajoute (j, i).
# .flip(0) inverse les lignes src/dst → transforme (i→j) en (j→i).
# Concaténation → le tenseur contient (i→j) ET (j→i) pour chaque paire.
pos_train_bidir = torch.cat([pos_train, pos_train.flip(0)], dim=1)

pos_test = torch.tensor(test_edges.T, dtype=torch.long)
pos_test_bidir = torch.cat([pos_test, pos_test.flip(0)], dim=1)

print(f"  Train : {len(train_edges)} arêtes positives")
print(f"  Test  : {len(test_edges)} arêtes positives")

# V5.4 (decoder-split) : split 90/10 du pool d'existence signé, bidirectionnel,
# avec le MÊME seed que le pool cosinus (cohérence train/test). Ces tenseurs
# alimentent recon_loss_signed (existence bilinéaire) + l'AUC combinée.
pos_train_signed_bidir = None
pos_test_signed_bidir = None
if DECODER_SPLIT and signed_exist_pairs:
    _se = np.array(list(signed_exist_pairs))
    _perm_s = np.random.permutation(len(_se))
    _n_test_s = max(1, int(len(_se) * EDGE_SAMPLE_RATIO))
    _se_test = _se[_perm_s[:_n_test_s]]
    _se_train = _se[_perm_s[_n_test_s:]]
    _pt_s = torch.tensor(_se_train.T, dtype=torch.long)
    pos_train_signed_bidir = torch.cat([_pt_s, _pt_s.flip(0)], dim=1)
    _pte_s = torch.tensor(_se_test.T, dtype=torch.long)
    pos_test_signed_bidir = torch.cat([_pte_s, _pte_s.flip(0)], dim=1)
    print(f"  V5.4 signed-existence : {len(_se_train)} train / "
          f"{len(_se_test)} test (bilinéaire)")

# =============================================================================
# 10. ENTRAÎNEMENT DU VGAE
# =============================================================================
# Boucle d'entraînement standard PyTorch avec quelques spécificités VGAE :
#   1. Negative sampling : à chaque epoch, on échantillonne des paires NON
#      connectées (négatifs) pour équilibrer la loss. Sans ça, le modèle
#      apprendrait juste à prédire "arête partout" (AUC = 0.5).
#   2. KL annealing : β augmente linéairement de 0 à KL_BETA_MAX. Au début
#      de l'entraînement (β ≈ 0), le modèle apprend UNIQUEMENT à reconstruire.
#      La KL entre progressivement pour régulariser l'espace latent.
#   3. Free bits : minimum de KL par dimension (empêche le collapse).
#   4. Early stopping : on arrête si l'AUC test ne s'améliore plus.
#   5. Gradient clipping : empêche les explosions de gradient.
print("\n" + "=" * 70)
print("10. Entraînement du VGAE")
print("=" * 70)

# =============================================================================
# V-sup : TÊTE de classification JOINTE (multi-tâche) — cf. _supervised.py
# =============================================================================
# Si --supervised : on N'INTERROMPT PAS la reconstruction. On attache une tête
# de classification sur μ, entraînée CONJOINTEMENT dans la boucle VGAE
# (loss = recon + β·KL + λ·classif). L'encodeur reste entraîné par
# reconstruction → le run reste PERTURBATION-READY (Δμ standard). Défini AVANT
# l'optimizer (ses params y sont ajoutés).
_SUP_HEAD = None
_SUP_LABELS_T = _SUP_CONF_T = _SUP_TRAIN_MASK = _SUP_TEST_MASK = None
if getattr(CLI_ARGS, "supervised", False):
    import sys as _sys_sup2
    _HERE_DIR2 = os.path.dirname(os.path.abspath(__file__))
    if _HERE_DIR2 not in _sys_sup2.path:
        _sys_sup2.path.insert(0, _HERE_DIR2)
    from _supervised import SupervisedHead, node_split as _sup_node_split, weighted_bce
    _SUP_HEAD = SupervisedHead(LATENT_DIM, n_labels=len(SUP_LABELS.label_names))
    _SUP_LABELS_T = torch.tensor(SUP_LABELS.labels)
    _SUP_CONF_T = torch.tensor(SUP_LABELS.confidence)
    _SUP_TRAIN_MASK, _SUP_TEST_MASK = _sup_node_split(
        len(gene_symbols), 0.2, CLI_ARGS.seed)
    print(f"  [V-sup] tête classif JOINTE : latent={LATENT_DIM} → "
          f"{len(SUP_LABELS.label_names)} labels, λ={CLI_ARGS.supervised_loss_weight} "
          f"(recon + KL conservés → perturbation-ready)")

# Adam avec weight_decay = 1e-4 (L2 regularization sur les poids).
# Le weight_decay empêche les poids de devenir trop grands, ce qui
# complémente la KL (qui régularise l'espace latent, pas les poids).
_opt_params = list(model.parameters())
if _SUP_HEAD is not None:                 # V-sup : la tête est co-entraînée
    _opt_params += list(_SUP_HEAD.parameters())
optimizer = torch.optim.Adam(_opt_params, lr=LR, weight_decay=1e-4)

# Préparer les dictionnaires de features et d'arêtes pour l'encoder.
# x_dict : features des noeuds par type
# edge_index_dict : arêtes par type (pour le message passing de l'encoder)
# NOTE : l'encoder voit TOUTES les arêtes du graphe (y compris les arêtes test).
# Seul le DÉCODEUR est évalué sur les arêtes test. C'est la convention standard
# des VGAE : l'encoder a accès au graphe complet pour le message passing,
# mais la supervision (loss) est calculée uniquement sur les arêtes train/test.
x_dict = {"gene": data["gene"].x, "cell_group": data["cell_group"].x}
edge_index_dict = {}
edge_attr_dict = {}   # features d'arête passées aux GATConv via edge_dim
# Boucle dynamique sur tous les edge_types présents dans le graphe (modularité
# V3.5+) — auparavant hard-codée, donc cassait sur les ablations partielles.
for et_key in data.edge_types:
    store = data[et_key]
    edge_index_dict[et_key] = store.edge_index
    if "edge_attr" in store and store.edge_attr is not None:
        edge_attr_dict[et_key] = store.edge_attr

# V5 (TIER 1c.4) : pool des arêtes signées POSITIVES pour la loss auxiliaire.
# {edge_type: (edge_index_dirigé, sign∈{-1,0,+1})}. Construit une seule fois,
# indépendamment du split train/test du pool de reconstruction (smoke test :
# on entraîne sur l'ensemble du pool signé).
signed_pos_pool: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
if CLI_ARGS.signed_decoder:
    for et_key in data.edge_types:
        if tuple(et_key) not in SIGNED_EDGE_TYPES:
            continue
        ea = edge_attr_dict.get(et_key)
        ei = edge_index_dict.get(et_key)
        if ea is None or ea.ndim < 2 or ea.shape[1] < 2 or ei is None \
                or ei.numel() == 0:
            continue
        # Convention V4 : edge_attr=[score, sign]. Seules les arêtes signées
        # (sign != 0) contribuent à la loss auxiliaire — sign=0 = info inconnue.
        sign = ea[:, 1].float()
        mask = sign != 0
        if not mask.any():
            continue
        signed_pos_pool[tuple(et_key)] = (
            ei[:, mask].clone(),
            sign[mask].clone(),
        )
    if signed_pos_pool:
        _summary = {et[1]: int(ei.shape[1])
                    for et, (ei, _) in signed_pos_pool.items()}
        print(f"  V5 signed_pos_pool (auxiliaire) : {_summary}")
    else:
        print("  [warn] --signed-decoder ON mais signed_pos_pool vide "
              "(pas d'edge_type signé avec sign≠0 dans le graphe).")

# V5 phase 2 (TIER 1c.5 strict) : split TF hold-out pour gate rigoureux.
# On retire de la signed_aux_loss les arêtes dont src OU dst est un
# régulateur hold-out. L'encoder voit toujours ces arêtes via le
# message-passing → seul le SIGNE est masqué de la loss aux.
# Liste hold-out persistée dans run_config.json + signed_holdout_edges.tsv
# pour `test_signed_auc.py --mode holdout`.
signed_holdout_pool: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
HOLDOUT_TF_SET: list[str] = []
HOLDOUT_TF_SEED = (CLI_ARGS.holdout_signed_tf_seed
                   if CLI_ARGS.holdout_signed_tf_seed is not None
                   else CLI_ARGS.seed)
if (CLI_ARGS.signed_decoder and signed_pos_pool
        and CLI_ARGS.holdout_signed_tf_fraction > 0.0):
    # 1. Set des régulateurs = union des sym source de TOUTES les arêtes signées
    #    présentes (ie. les nœuds qui apparaissent en src d'au moins un
    #    edge_type signé). Pour tf_curated_by les sym src sont les targets ;
    #    mais comme on filtre ensuite par (src OU dst) ∈ hold-out, tout
    #    régulateur de tf_curated sera attrapé en tf_curated_by aussi.
    _regulator_idx = set()
    for et_key, (ei, _) in signed_pos_pool.items():
        _regulator_idx.update(ei[0].tolist())
    _regulator_syms = sorted(str(gene_symbols[i]) for i in _regulator_idx
                             if i < len(gene_symbols))
    if len(_regulator_syms) < 5:
        print(f"  [warn] --holdout-signed-tf-fraction ON mais seulement "
              f"{len(_regulator_syms)} régulateurs trouvés — hold-out désactivé.")
    else:
        _rng = np.random.default_rng(HOLDOUT_TF_SEED)
        _n_holdout = max(1, int(len(_regulator_syms)
                                * CLI_ARGS.holdout_signed_tf_fraction))
        HOLDOUT_TF_SET = sorted(_rng.choice(_regulator_syms,
                                            size=_n_holdout,
                                            replace=False).tolist())
        _holdout_idx_set = {gene_to_idx[s] for s in HOLDOUT_TF_SET
                            if s in gene_to_idx}

        # 2. Split par edge_type : edge en hold-out si src OR dst ∈ set.
        for et_key in list(signed_pos_pool.keys()):
            ei, sign = signed_pos_pool[et_key]
            _src = ei[0].numpy()
            _dst = ei[1].numpy()
            # vectorisé : True si src ou dst dans set
            _src_in = np.isin(_src, list(_holdout_idx_set))
            _dst_in = np.isin(_dst, list(_holdout_idx_set))
            _is_holdout = _src_in | _dst_in
            _is_train = ~_is_holdout

            if _is_holdout.sum() > 0:
                signed_holdout_pool[et_key] = (
                    ei[:, _is_holdout].clone(),
                    sign[_is_holdout].clone(),
                )
            if _is_train.sum() > 0:
                signed_pos_pool[et_key] = (
                    ei[:, _is_train].clone(),
                    sign[_is_train].clone(),
                )
            else:
                # Tout est en hold-out (rare avec X≤0.5) → on retire l'edge_type
                signed_pos_pool.pop(et_key, None)

        # 3. Log + persistance TSV des arêtes hold-out (pour test_signed_auc.py).
        _ho_summary = {et[1]: int(ei.shape[1])
                       for et, (ei, _) in signed_holdout_pool.items()}
        _tr_summary = {et[1]: int(ei.shape[1])
                       for et, (ei, _) in signed_pos_pool.items()}
        print(f"  V5 phase 2 hold-out signed TF (seed={HOLDOUT_TF_SEED}, "
              f"frac={CLI_ARGS.holdout_signed_tf_fraction:.0%}) :")
        print(f"    régulateurs : {len(HOLDOUT_TF_SET)}/{len(_regulator_syms)} "
              f"({100*len(HOLDOUT_TF_SET)/len(_regulator_syms):.1f}%)")
        print(f"    arêtes train (loss aux) : {_tr_summary}")
        print(f"    arêtes hold-out (eval pure) : {_ho_summary}")

        # TSV : 1 ligne par (edge_type, src_sym, dst_sym, sign).
        _ho_rows = []
        for et_key, (ei, sign) in signed_holdout_pool.items():
            for j in range(ei.shape[1]):
                _ho_rows.append({
                    "edge_type": et_key[1],
                    "src_sym": str(gene_symbols[int(ei[0, j])]),
                    "dst_sym": str(gene_symbols[int(ei[1, j])]),
                    "src_idx": int(ei[0, j]),
                    "dst_idx": int(ei[1, j]),
                    "sign": float(sign[j]),
                })
        if _ho_rows:
            pd.DataFrame(_ho_rows).to_csv(
                os.path.join(OUT_DIR, "signed_holdout_edges.tsv"),
                sep="\t", index=False,
            )
            print(f"    persisté : signed_holdout_edges.tsv "
                  f"({len(_ho_rows)} arêtes)")

        # 4. Réécriture du manifest avec le set hold-out résolu (= reproductible).
        try:
            with open(_MANIFEST_PATH) as _fh:
                _manifest = json.load(_fh)
            _manifest.update({
                "holdout_signed_tf_set": HOLDOUT_TF_SET,
                "holdout_signed_tf_seed_used": HOLDOUT_TF_SEED,
                "holdout_signed_edges_per_edge_type": _ho_summary,
                "signed_train_edges_per_edge_type": _tr_summary,
            })
            with open(_MANIFEST_PATH, "w") as _fh:
                json.dump(_manifest, _fh, indent=2)
            print(f"    manifest enrichi : {_MANIFEST_PATH}")
        except Exception as _e:
            print(f"    [warn] échec enrichissement manifest : {_e}")
elif CLI_ARGS.holdout_signed_tf_fraction > 0.0 and not CLI_ARGS.signed_decoder:
    print(f"  [warn] --holdout-signed-tf-fraction={CLI_ARGS.holdout_signed_tf_fraction} "
          f"ignoré (nécessite --signed-decoder).")

# Historique pour les plots
train_losses = []      # Loss totale à chaque epoch
recon_losses = []      # Loss de reconstruction à chaque epoch
kl_losses = []         # KL loss (non pondérée) à chaque epoch
kl_betas = []          # β du KL annealing à chaque epoch
test_aucs = []         # AUC test (évaluée toutes les 10 epochs)
test_aps = []          # AP test (évaluée toutes les 10 epochs)
eval_epochs = []       # Numéros d'epoch où on a évalué
best_test_auc = 0.0    # Meilleure AUC test observée
best_test_ap = 0.0     # AP au meilleur epoch (par AUC)
best_epoch = 0         # Epoch correspondante
# PATIENCE : si l'AUC val ne s'améliore pas pendant PATIENCE epochs, on arrête.
# Surchargeable via --patience (défaut 100).
PATIENCE = CLI_ARGS.patience
# GRAD_CLIP_NORM = 1.0 : on clamp la norme du gradient total à 1.0.
# Empêche les mises à jour explosives (ex : quand τ change brusquement).
GRAD_CLIP_NORM = 1.0
patience_counter = 0

model.train()  # Mode entraînement (active dropout + BatchNorm en mode train)
if _SUP_HEAD is not None:
    _SUP_HEAD.train()
for epoch in range(N_EPOCHS):
    optimizer.zero_grad()  # Remet les gradients à zéro

    # --- Forward pass de l'encoder ---
    # z = échantillon du posterior q(z|x) via reparametrization trick
    # mu, logvar = paramètres du posterior (pour la KL loss)
    z, mu, logvar = model.encode(x_dict, edge_index_dict, edge_attr_dict)

    # --- Negative sampling ---
    # Pour chaque arête positive (vraie connexion), on tire une arête
    # négative (paire non connectée). Le ratio 1:1 donne des classes
    # équilibrées. Les négatifs sont RE-ÉCHANTILLONNÉS à chaque epoch
    # pour que le modèle voie des négatifs variés (pas toujours les mêmes).
    neg_train = negative_sampling(
        pos_train_bidir, num_nodes=n_genes,
        num_neg_samples=pos_train_bidir.shape[1],
    )

    # --- Loss de reconstruction (BCE) ---
    # Le décodeur produit des logits (τ · cos) pour chaque paire.
    # binary_cross_entropy_with_logits applique sigmoid + BCE en une seule
    # opération (numériquement plus stable que sigmoid séparé).
    pos_scores = model.decode(z, pos_train_bidir)   # Logits pour les vrais voisins
    neg_scores = model.decode(z, neg_train)          # Logits pour les non-voisins

    # pos_loss : on veut que les logits des vrais voisins soient HAUTS (label=1)
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_scores, torch.ones_like(pos_scores)
    )
    # neg_loss : on veut que les logits des non-voisins soient BAS (label=0)
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores, torch.zeros_like(neg_scores)
    )
    recon_loss = pos_loss + neg_loss

    # --- V5.4 (decoder-split) : recon_loss_signed (EXISTENCE bilinéaire) ---
    # Le cosinus ne voit plus les arêtes signées ; le bilinéaire reconstruit
    # leur existence via predict_signed_existence = logsumexp(W+,W-,W0).
    # BCE(existence(pos signées), 1) + BCE(existence(neg), 0). Le SIGNE reste
    # porté par signed_aux_loss (predict_sign_score). Cf. §14bis.6duovicies.
    if DECODER_SPLIT and pos_train_signed_bidir is not None:
        _ps = pos_train_signed_bidir.to(z.device)
        _neg_s = negative_sampling(
            _ps, num_nodes=n_genes, num_neg_samples=_ps.shape[1],
        )
        _exist_pos = model.bilinear_decoder.predict_signed_existence(z, _ps)
        _exist_neg = model.bilinear_decoder.predict_signed_existence(z, _neg_s)
        recon_loss_signed = (
            F.binary_cross_entropy_with_logits(_exist_pos, torch.ones_like(_exist_pos))
            + F.binary_cross_entropy_with_logits(_exist_neg, torch.zeros_like(_exist_neg))
        )
        recon_loss = recon_loss + recon_loss_signed

    # --- V5.1 (TIER 1c.4) : loss auxiliaire signée CONTRASTIVE -----------
    # Cible binaire : sign>0 → 1 (activation), sign<0 → 0 (inhibition).
    # BCE sur le score SIGN-AGNOSTIQUE `predict_sign_score = logit_pos −
    # logit_neg`. Optimise directement la métrique du gate 1c.5.
    #
    # ⚠ Correction V5.1 (2026-05-29) — la formulation V5.0 utilisait
    # `forward_signed(z, ei, sign)` qui SÉLECTIONNAIT le canal selon le
    # sign cible. Conséquence : `logit_pos` était entraîné haut pour les
    # activations, `logit_neg` bas pour les inhibitions, mais les deux
    # canaux n'étaient PAS contraints l'un par rapport à l'autre. Le
    # score différentiel `logit_pos − logit_neg` était donc indépendant
    # du sign cible → AUC ≈ 0.5 sur le gate 1c.5 (mesuré 2026-05-29 sur
    # v5-full.s{1,2,3}, cf. §14bis.6septies du rapport).
    #
    # V5.1 fix : on entraîne directement le score différentiel, ce qui
    # force `logit_pos > logit_neg` pour activations et inverse pour
    # inhibitions. Cf. SGAT-bilinear Liu 2024 *NAR* §3.
    signed_aux_loss = torch.tensor(0.0, device=z.device)
    if CLI_ARGS.signed_decoder and signed_pos_pool:
        _aux_terms = []
        for et_key, (ei, sign) in signed_pos_pool.items():
            ei = ei.to(z.device)
            sign = sign.to(z.device)
            # Score différentiel : (z_src·W_+·z_dst) − (z_src·W_-·z_dst)
            score = model.bilinear_decoder.predict_sign_score(z, ei)
            target = (sign > 0).float()  # +1 → 1 ; -1 → 0
            _aux_terms.append(F.binary_cross_entropy_with_logits(
                score, target
            ))
        if _aux_terms:
            signed_aux_loss = torch.stack(_aux_terms).mean()

    # --- KL divergence avec annealing + free bits ---
    # KL ANNEALING : β augmente linéairement de 0 à KL_BETA_MAX pendant
    # les KL_WARMUP_EPOCHS premières epochs. Calendrier :
    #   epoch 0   → β = 0        (pas de KL, reconstruction pure)
    #   epoch 25  → β = KL_BETA_MAX/2  (KL progressive)
    #   epoch 50+ → β = KL_BETA_MAX    (régularisation complète)
    # Cela permet au modèle de d'abord apprendre à bien reconstruire les
    # arêtes (encoder des embeddings informatifs), PUIS d'introduire la
    # régularisation KL qui va lisser et structurer l'espace latent.
    kl_beta = min(KL_BETA_MAX, KL_BETA_MAX * epoch / max(1, KL_WARMUP_EPOCHS))
    kl_loss = model.kl_loss(mu, logvar, free_bits=FREE_BITS)

    # Loss totale = reconstruction + β × KL + λ_signed × signed_aux (V5)
    loss = recon_loss + kl_beta * kl_loss \
        + CLI_ARGS.signed_loss_weight * signed_aux_loss
    # V-sup : + λ × classification JOINTE (multi-tâche) sur μ. L'encodeur reste
    # entraîné par reconstruction ; la tête ajoute le signal DEG par cluster.
    if _SUP_HEAD is not None:
        _clf_loss = weighted_bce(_SUP_HEAD(mu), _SUP_LABELS_T, _SUP_CONF_T,
                                 _SUP_TRAIN_MASK)
        loss = loss + CLI_ARGS.supervised_loss_weight * _clf_loss
    loss.backward()
    # Gradient clipping : empêche les mises à jour explosives
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    train_losses.append(loss.item())
    recon_losses.append(recon_loss.item())
    kl_losses.append(kl_loss.item())
    kl_betas.append(kl_beta)

    # --- Évaluation sur les arêtes test (toutes les 10 epochs) ---
    # On évalue sur des arêtes que le modèle n'a PAS vues dans la loss.
    # L'AUC mesure la capacité du modèle à distinguer les vraies arêtes
    # (positifs) des paires non connectées (négatifs).
    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()  # Mode évaluation (désactive dropout, BatchNorm en mode eval)
        with torch.no_grad():  # Pas de calcul de gradient (économie mémoire + vitesse)
            z_eval, mu_eval, _ = model.encode(x_dict, edge_index_dict, edge_attr_dict)

            # Scores pour les arêtes test positives (vraies)
            pos_test_scores = torch.sigmoid(model.decode(z_eval, pos_test_bidir))
            # Négatifs test : nouvelles paires non connectées
            neg_test = negative_sampling(
                pos_test_bidir, num_nodes=n_genes,
                num_neg_samples=pos_test_bidir.shape[1],
            )
            neg_test_scores = torch.sigmoid(model.decode(z_eval, neg_test))

            # Labels : 1 pour positifs, 0 pour négatifs
            test_labels = torch.cat([
                torch.ones(pos_test_scores.shape[0]),
                torch.zeros(neg_test_scores.shape[0]),
            ])
            test_scores = torch.cat([pos_test_scores, neg_test_scores])

            # V5.4 (decoder-split) : AUC COMBINÉE sur tout le graphe — on ajoute
            # les scores d'EXISTENCE bilinéaire sur le test pool signé (que le
            # cosinus ne voit plus). La métrique rapportée couvre cosinus
            # (non-signé) + bilinéaire (signé). Cf. §14bis.6duovicies propriété 3.
            if DECODER_SPLIT and pos_test_signed_bidir is not None:
                _pts = pos_test_signed_bidir.to(z_eval.device)
                _neg_s = negative_sampling(
                    _pts, num_nodes=n_genes, num_neg_samples=_pts.shape[1],
                )
                _ex_pos = torch.sigmoid(
                    model.bilinear_decoder.predict_signed_existence(z_eval, _pts))
                _ex_neg = torch.sigmoid(
                    model.bilinear_decoder.predict_signed_existence(z_eval, _neg_s))
                test_labels = torch.cat([
                    test_labels,
                    torch.ones(_ex_pos.shape[0]), torch.zeros(_ex_neg.shape[0]),
                ])
                test_scores = torch.cat([test_scores, _ex_pos, _ex_neg])

            # AUC-ROC et Average Precision (AP)
            # AUC = 0.5 → modèle aléatoire (collapse)
            # AUC > 0.9 → le modèle reconstruit bien le graphe
            try:
                auc = roc_auc_score(test_labels.numpy(), test_scores.numpy())
                ap = average_precision_score(test_labels.numpy(), test_scores.numpy())
            except ValueError:
                auc, ap = 0.5, 0.5  # Fallback si toutes les prédictions sont identiques

            test_aucs.append(auc)
            test_aps.append(ap)
            eval_epochs.append(epoch)

            # Sauvegarder le meilleur modèle (par AUC test)
            if auc > best_test_auc:
                best_test_auc = auc
                best_test_ap = ap
                best_epoch = epoch
                patience_counter = 0  # Reset du compteur de patience
                torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_vgae.pt"))
                if _SUP_HEAD is not None:   # V-sup : tête au même best epoch
                    torch.save(_SUP_HEAD.state_dict(),
                               os.path.join(OUT_DIR, "best_sup_head.pt"))
            else:
                patience_counter += 10  # +10 car on évalue toutes les 10 epochs

        model.train()  # Retour en mode entraînement

        # Log détaillé toutes les 50 epochs
        if (epoch + 1) % 50 == 0:
            tau_val = np.exp(model.log_tau.item())  # τ actuel
            _signed_log = (f" sgn={signed_aux_loss.item():.4f}"
                           if CLI_ARGS.signed_decoder else "")
            print(f"    Epoch {epoch+1:3d}/{N_EPOCHS} — "
                  f"Loss: {loss.item():.4f} (recon={recon_loss.item():.4f} "
                  f"kl={kl_loss.item():.4f} β={kl_beta:.5f} τ={tau_val:.2f}"
                  f"{_signed_log}) — "
                  f"Test AUC: {auc:.4f}, AP: {ap:.4f}")

    # Early stopping — si l'AUC ne s'améliore plus, on arrête avant
    # de gaspiller des epochs et de risquer le collapse
    if patience_counter >= PATIENCE:
        print(f"\n  Early stopping à epoch {epoch+1} (pas d'amélioration "
              f"depuis {PATIENCE} epochs)")
        break

print(f"\n  Meilleur modèle : epoch {best_epoch+1}, AUC = {best_test_auc:.4f}")

# Charger le meilleur modèle (celui avec la meilleure AUC test).
# weights_only=True : sécurité PyTorch — ne charge que les tenseurs, pas
# d'objets Python arbitraires (évite les attaques de sérialisation).
model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_vgae.pt"), weights_only=True))

# =============================================================================
# V-sup : finalisation de la tête de classification après reconstruction
# =============================================================================
# Plus de re-save de hetero_graph_vgae.pt ici : les features DE font désormais
# partie du build, donc le graphe écrit par gnn_vgae.py (build ET --reuse-graph)
# porte déjà le gene.x exact que la perturbation doit recharger.
if _SUP_HEAD is not None:
    from _supervised import finalize_supervised
    _hp_best = os.path.join(OUT_DIR, "best_sup_head.pt")
    if os.path.exists(_hp_best):
        _SUP_HEAD.load_state_dict(torch.load(_hp_best, weights_only=True))
    finalize_supervised(
        model, _SUP_HEAD, x_dict, edge_index_dict, edge_attr_dict,
        SUP_LABELS, gene_symbols, OUT_DIR,
        hyperparams={"hidden": HIDDEN_DIM, "latent": LATENT_DIM,
                     "n_layers": N_LAYERS, "n_heads": N_HEADS,
                     "signed_message": CLI_ARGS.signed_message,
                     "signed_decoder": CLI_ARGS.signed_decoder,
                     # Required by load_supervised_run to rebuild the exact
                     # encoder/decoder shapes (V5 bilinear + γ_t per edge type).
                     "signed_decoder_dim": SIGNED_DECODER_DIM,
                     "edge_type_weights": EDGE_TYPE_WEIGHTS or None,
                     "de_features": bool(getattr(CLI_ARGS, "de_features", False))},
        train_mask=_SUP_TRAIN_MASK, test_mask=_SUP_TEST_MASK, data=data)
