"""
_config.py — parsing CLI du pipeline VGAE (import-safe, sans effet de bord).

Extrait du monolithe gnn_vgae.py (split Tier 2.5). Parseur argparse complet
(modularité graphe/features/hyperparamètres). Réutilisable indépendamment :

    from _config import parse_cli_args
    args = parse_cli_args([...])   # ou parse_cli_args() = sys.argv

Les DÉRIVATIONS (MODULES, GENE_FEATURE_FLAGS, RUN_TAG, ...) restent pour l'instant
dans gnn_vgae.py (extraction ultérieure).
"""
import argparse


def parse_cli_args(argv=None):
    p = argparse.ArgumentParser(add_help=True,
                                description="VGAE HUVEC — modulaire (sources & features).")
    # --- Sources de données (edge types) ---
    p.add_argument("--use-ppi", dest="use_ppi", action="store_true", default=True)
    p.add_argument("--no-ppi", dest="use_ppi", action="store_false")
    p.add_argument("--use-reactome", dest="use_reactome", action="store_true", default=True)
    p.add_argument("--no-reactome", dest="use_reactome", action="store_false")
    p.add_argument("--use-coexpr", dest="use_coexpr", action="store_true", default=True)
    p.add_argument("--no-coexpr", dest="use_coexpr", action="store_false")
    p.add_argument("--use-scenic-regulons", dest="use_scenic_regulons",
                   action="store_true", default=True)
    p.add_argument("--no-scenic-regulons", dest="use_scenic_regulons",
                   action="store_false")
    p.add_argument("--use-humess-edges", dest="use_humess_edges",
                   action="store_true", default=True,
                   help="arêtes metabolic_cocatalysis (GPR rules)")
    p.add_argument("--no-humess-edges", dest="use_humess_edges", action="store_false")
    p.add_argument("--use-humess-features", dest="use_humess_features",
                   action="store_true", default=True,
                   help="features de noeud imp_P4_z, imp_P16_z, imp_delta, has_humess")
    p.add_argument("--no-humess-features", dest="use_humess_features",
                   action="store_false")
    p.add_argument("--use-cell-group-edges", dest="use_cell_group_edges",
                   action="store_true", default=True,
                   help="arêtes bipartites cell_group ↔ gene (expresses/expressed_in)")
    p.add_argument("--no-cell-group-edges", dest="use_cell_group_edges",
                   action="store_false")
    # Convenience flag : --no-humess désactive arêtes ET features HuMess.
    p.add_argument("--no-humess", action="store_true", default=False,
                   help="raccourci : désactive --use-humess-edges ET --use-humess-features")

    # --- OmniPath (V4 — opt-in, OFF par défaut) ---
    # Sources pré-téléchargées via scripts/cache_omnipath.py — les compute
    # nodes Nautilus n'ayant pas Internet, on lit toujours depuis le cache TSV.
    p.add_argument("--use-omnipath-signaling", dest="use_omnipath_signaling",
                   action="store_true", default=False,
                   help="V4 : ajoute arêtes 'signaling' dirigées signées "
                        "(kinase-substrat OmniPath + SIGNOR causal). "
                        "edge_attr=[score, sign∈{−1,0,+1}].")
    p.add_argument("--no-omnipath-signaling", dest="use_omnipath_signaling",
                   action="store_false")
    p.add_argument("--use-omnipath-tf-curated", dest="use_omnipath_tf_curated",
                   action="store_true", default=False,
                   help="V4 : ajoute edge_type 'tf_curated' (CollecTRI, "
                        "fallback DoRothEA) à côté de pySCENIC 'regulates'. "
                        "~1186 TFs vs ~50 SCENIC, signe biologique préservé.")
    p.add_argument("--no-omnipath-tf-curated", dest="use_omnipath_tf_curated",
                   action="store_false")
    p.add_argument("--omnipath-cache-dir", default=None,
                   help="dossier des TSV OmniPath pré-téléchargés "
                        "(défaut : <DATA_DIR>/omnipath). Cf. "
                        "scripts/cache_omnipath.py")
    p.add_argument("--omnipath-download-if-missing", action="store_true",
                   default=False,
                   help="autorise le téléchargement à la volée si le cache "
                        "est absent (à n'utiliser que sur le frontal qui a "
                        "Internet ; OFF sur les compute nodes)")
    # V4.1 : include OmniPath endpoints in the selected gene set.
    p.add_argument("--include-omnipath-genes", dest="include_omnipath_genes",
                   action="store_true", default=False,
                   help="V4.1 : étend la sélection des gènes en section 3 avec "
                        "les endpoints d'OmniPath (CollecTRI + signaling) qui "
                        "sont aussi présents dans le scRNA-seq. Sans ce flag, "
                        "OmniPath ne fait qu'ajouter des arêtes entre gènes "
                        "déjà sélectionnés via PPI/SCENIC/coexpr/REACTOME ; "
                        "résultat : ~700 TFs CollecTRI sont perdus. "
                        "Requiert --use-omnipath-signaling et/ou "
                        "--use-omnipath-tf-curated.")
    # V6 : HGNC alias normalization when projecting OmniPath edges.
    p.add_argument("--omnipath-hgnc-alias", dest="omnipath_hgnc_alias",
                   action="store_true", default=True,
                   help="V6 (défaut ON) : canonicalise les symboles OmniPath "
                        "ET les clés gene_to_idx en symbole HGNC approuvé avant "
                        "la projection (hgnc_alias.py). Récupère les gènes "
                        "dérivés de nomenclature (H2AFZ↔H2AZ1 : 0→316 arêtes). "
                        "Cache data/omnipath/hgnc_alias_map.tsv.gz.")
    p.add_argument("--no-omnipath-hgnc-alias", dest="omnipath_hgnc_alias",
                   action="store_false",
                   help="Désactive la normalisation HGNC → match brut en "
                        "symbole exact (comportement legacy pré-V6).")
    # V6 Module 1 : OmniPath-derived NODE features (localisation intercell +
    # druggabilité + classe moléculaire) sur le type 'gene' (PAS de nouveau
    # type de nœud). Opt-in (défaut OFF, réversible). Exclusion fine possible
    # via --exclude-features op_secreted,op_druggable,is_lncrna,...
    p.add_argument("--use-omnipath-node-features",
                   dest="use_omnipath_node_features",
                   action="store_true", default=False,
                   help="V6 : ajoute les features de nœud OmniPath "
                        "(op_transmitter/receiver/secreted/plasma_membrane, "
                        "op_druggable/op_drug_records, is_protein_coding/"
                        "is_lncrna/is_mirna). Requiert graph/nodes.tsv.gz "
                        "(build_omnipath_graph) + hgnc_biotype_map. Offline-"
                        "safe : sources absentes → features à 0.")
    p.add_argument("--no-omnipath-node-features",
                   dest="use_omnipath_node_features", action="store_false")
    # V6 Module 1 (extension) : arêtes OmniPath supplémentaires projetées sur
    # les nœuds gène (protein↔protein signé [score,sign]). Liste modulaire —
    # ajouter/retirer un type à la fois. Source = graphe autonome
    # (data/omnipath/graph/edges.tsv.gz, build_omnipath_graph). signaling &
    # collectri_tf EXCLUS (déjà via --use-omnipath-signaling/-tf-curated).
    p.add_argument("--omnipath-edges", dest="omnipath_edges", default="",
                   help="V6 : liste séparée par virgules d'edge_types OmniPath "
                        "à ajouter : {transcriptional, tf_target, "
                        "kinase_substrate, ligand_receptor, pathway, "
                        "enzyme_substrate} ou 'all'. Défaut vide (OFF). "
                        "Offline-safe : graphe absent → aucune arête ajoutée.")

    # --- V4.2 : coexpression différentielle P4∪P16 (option A) ---
    p.add_argument("--coexpr-mode", choices=["p16_only", "differential"],
                   default="p16_only",
                   help="V4.2 : 'p16_only' = adjacencies.csv GRNBoost2 P16 "
                        "(comportement V4.1, edge_dim=1) ; 'differential' = "
                        "coexpr_diff.tsv (P4∪P16, option A, edge_dim=6 : "
                        "imp_p4, imp_p16, delta, cat_shared/p4/p16). "
                        "Cf. build_diff_coexpr.py + §V4.2 du rapport.")
    p.add_argument("--diff-coexpr-file",
                   default="data/pyscenic/diff_coexpr/coexpr_diff.tsv",
                   help="V4.2 : chemin du TSV différentiel produit par "
                        "build_diff_coexpr.py merge-adjacencies. Si laissé "
                        "à la valeur par défaut ET que --coexpr-method/"
                        "--coexpr-prune sont != défaut, le chemin sera "
                        "auto-résolu vers coexpr_diff.<method>.<prune>.tsv.")
    # --- V4.3 : grille comparaison méthodes GRN × élagage ---
    # Auto-résolution du chemin coexpr_diff selon (method, prune) si
    # --diff-coexpr-file n'a pas été surchargé. method='sklearn' +
    # prune='topk' = comportement V4.2 (fichier historique coexpr_diff.tsv).
    p.add_argument("--coexpr-method",
                   choices=["sklearn", "arboreto", "corr", "mi"],
                   default="sklearn",
                   help="V4.3 : méthode d'inférence GRN amont. sklearn = "
                        "grnboost2-local (défaut, comportement V4.2) ; "
                        "arboreto = grnboost2-diff canonique ; corr = "
                        "Pearson/Spearman ; mi = mutual_info_regression. "
                        "Persisté dans vgae_metrics.json.")
    p.add_argument("--coexpr-prune",
                   choices=["topk", "quantile", "mr", "zscore"],
                   default="topk",
                   help="V4.3 : méthode d'élagage des arêtes dans le "
                        "réseau coexpr. topk = per-target-topk (défaut) ; "
                        "quantile = global-quantile (baseline -) ; mr = "
                        "mutual-rank Obayashi 2018 ; zscore = z-score "
                        "per-target. Persisté dans vgae_metrics.json.")

    # --- V4.2 : Reactome FI (arêtes signées additionnelles) ---
    p.add_argument("--use-reactome-fi", dest="use_reactome_fi",
                   action="store_true", default=False,
                   help="V4.2 : ajoute edge_type 'reactome_fi' (Reactome "
                        "Functional Interactions, signed). ~45k arêtes "
                        "signées NOUVELLES (75% absentes de PPI/SIGNOR/"
                        "CollecTRI). edge_dim=2 [score, sign].")
    p.add_argument("--no-reactome-fi", dest="use_reactome_fi",
                   action="store_false")
    p.add_argument("--reactome-fi-file",
                   default="data/reactome_fi/FIsInGene_with_annotations.txt",
                   help="V4.2 : chemin du fichier Reactome FI décompressé "
                        "(Gene1, Gene2, Annotation, Direction, Score).")
    # ── V6.2 (2026-07-16) : exploitation de la colonne Direction ─────────────
    # Défaut = LEGACY V5/V6 (tout symétrisé, signe seul). Opt-in réversible.
    p.add_argument("--reactome-fi-directed", dest="reactome_fi_directed",
                   action="store_true", default=False,
                   help="V6.2 : ORIENTE les arêtes reactome_fi via la colonne "
                        "Direction (->/<-/<->/-|/|-) au lieu de tout symétriser. "
                        "Les arêtes à orientation inconnue ('-', 66%%) vont dans "
                        "l'edge_type séparé 'reactome_fi_undirected'. Défaut OFF "
                        "= comportement V5/V6 strict.")
    p.add_argument("--no-reactome-fi-undirected",
                   dest="no_reactome_fi_undirected",
                   action="store_true", default=False,
                   help="V6.2 (avec --reactome-fi-directed) : JETTE les arêtes "
                        "reactome_fi à orientation inconnue ('-'), ne garde que "
                        "le causal orienté.")
    p.add_argument("--reactome-fi-predicted", dest="reactome_fi_predicted",
                   action="store_true", default=False,
                   help="V6.2 : INCLUT les FI 'predicted' (computationnelles non "
                        "curées, ~29%%). Défaut OFF = exclues comme en V5/V6.")

    # --- V4.2 : pondération γ_t par edge_type (niveau message) ---
    # NB : la loss VGAE est poolée (PPI∪REACTOME∪reg∪coexpr dédupliqués),
    # donc on ne peut pas pondérer la *loss* par type proprement. Le
    # déséquilibre mesuré (‖h_PPI‖ ≈ 13× ‖h_signaling‖, §14bis.6bis) est
    # au niveau du MESSAGE-PASSING. γ_t scale donc la sortie de chaque
    # GATConv AVANT l'agrégation HeteroConv-sum. Toggleable pour A/B V4.2.
    p.add_argument("--edge-type-weights", default="",
                   help="V4.2 : pondération γ_t par edge_type, format "
                        "'ppi=0.1,coexpression=0.5,signaling=1.0'. Vide = "
                        "tous les γ_t=1.0 (comportement V4.1). Les types "
                        "non listés gardent γ=1.0. Appliqué au niveau du "
                        "message (pas de la loss). Cf. §14bis.6bis.")
    p.add_argument("--dedup-ppi-signed",
                   choices=["off", "remove", "annotate"], default="off",
                   help="V4.2 : si une paire (a,b) a une arête SIGNÉE "
                        "orientée (signaling/tf_curated/reactome_fi), "
                        "l'arête PPI non-signée est redondante. off "
                        "(DÉFAUT) = inchangé, diagnostic seul (compte "
                        "arêtes/gènes touchés). remove = supprime le "
                        "doublon PPI. annotate = garde l'arête PPI + "
                        "colonne flag has_signed_counterpart (edge_dim "
                        "1→2). Cf. §14bis.6quaterdecies. Bénéfice plein "
                        "avec V5 (message/decodeur signés).")

    # --- V5 : message-passing signé + décodeur bilinéaire signé (TIER 1c) ---
    # Tous opt-in, défaut OFF → backward-compat V4.x. Cf. §14bis.6septies
    # du rapport, prototypes dans src/gnn/_vgae_model.py:171-277.
    p.add_argument("--signed-message",
                   action="store_true", default=False,
                   help="V5 (TIER 1c.2) : remplace GATConv par SignedGATConv "
                        "pour les edge_types signés (signaling, tf_curated, "
                        "tf_curated_by, reactome_fi). Chaque message est "
                        "multiplié par son `sign` (colonne 1 de edge_attr) "
                        "⇒ une arête `sign=-1` propage `-W·h_j`. Ref : Derr "
                        "2018 ICDM SGCN §3.2. Défaut OFF (legacy).")
    p.add_argument("--signed-decoder",
                   action="store_true", default=False,
                   help="V5 (TIER 1c.3) : ajoute un BilinearSignedDecoder "
                        "(3 canaux W_+ / W_- / W_0) en parallèle du "
                        "décodeur cosinus. Active une loss auxiliaire "
                        "BCE(logit_bilin, sign∈{0,1}) sur les arêtes "
                        "positives des edge_types signés. La loss "
                        "principale (cosinus + KL) reste inchangée. Ref : "
                        "Liu 2024 NAR SGAT-bilinear, Yang 2015 ICLR "
                        "DistMult. Défaut OFF (legacy).")
    p.add_argument("--signed-loss-weight", type=float, default=1.0,
                   help="V5 (TIER 1c.4) : λ_signed multiplicateur de la "
                        "loss auxiliaire signée (ignoré si "
                        "--signed-decoder OFF). 1.0 = équivalent à la "
                        "recon_loss principale ; 0.5 = demi-poids. Défaut "
                        "1.0.")
    p.add_argument("--signed-decoder-dim", type=int, default=None,
                   help="V5.3 (TIER 1c.7) : dimension du sous-espace signed "
                        "(tête `signed_proj : R^latent → R^signed_dim` avant "
                        "le décodeur bilinéaire). Défaut None = LATENT_DIM "
                        "(pas de compression, signed_proj init=identité ⇒ "
                        "équivalent V5.2 au load checkpoint). Valeurs <"
                        "LATENT_DIM forcent la compression du sous-espace "
                        "signed (spécialisation). Réduit la concurrence "
                        "avec le décodeur cosinus (V5.2 : −0.022 AUC recon). "
                        "Cf. §14bis.6unvicesies.")

    # --- V5 phase 2 : VRAI hold-out signed pour gate 1c.5 rigoureux ---
    # Sans ces flags, la signed_aux_loss voit TOUTES les arêtes signées
    # à l'entraînement → gate 1c.5 reste in-sample. Avec --holdout-signed-tf-
    # fraction X > 0, on tire X% des « régulateurs » (= union des sym source
    # des edge_types signés) et on MASQUE leurs arêtes signées de la loss.
    # L'encodeur continue de voir ces arêtes via le message-passing
    # (SignedGATConv propage), mais leur signe n'est PAS appris.
    # test_signed_auc.py lit la liste persistée dans run_config.json et
    # évalue uniquement sur les arêtes hold-out → vrai test de
    # généralisation à des TFs jamais utilisés pour la loss.
    p.add_argument("--holdout-signed-tf-fraction", type=float, default=0.0,
                   help="V5 phase 2 : fraction des régulateurs (= union des "
                        "sym source des edge_types signés) à réserver au test "
                        "1c.5 rigoureux. Leurs arêtes signées sont retirées "
                        "de la signed_aux_loss (mais conservées dans le "
                        "graphe → encoder les voit toujours). 0.0 = pas de "
                        "hold-out (défaut, comportement V5.1). Valeur typique "
                        "défense : 0.2 (Liu 2024 NAR SGAT-bilinear §3).")
    p.add_argument("--holdout-signed-tf-seed", type=int, default=None,
                   help="V5 phase 2 : seed RNG pour le split TF hold-out. "
                        "Défaut : prend --seed (reproductibilité couplée au "
                        "training). Passer une valeur fixée (ex. 42) pour "
                        "comparer des configs A/B sur le MÊME set hold-out.")

    # --- Exclusion fine de features de noeud gene ---
    # Liste possible : is_tf, variance, ppi_degree, reg_degree,
    #                  imp_P4, imp_P16, imp_delta, has_humess
    p.add_argument("--exclude-features", default="",
                   help="liste séparée par des virgules de gene_features à exclure "
                        "(parmi : is_tf, variance, ppi_degree, reg_degree, "
                        "imp_P4, imp_P16, imp_delta, has_humess)")

    # --- Hyperparamètres de filtrage du graphe ---
    p.add_argument("--ppi-score-thresh", type=int, default=900,
                   help="seuil STRING combined_score (0-1000), defaut=900")
    p.add_argument("--coexpr-top-quantile", type=float, default=0.98,
                   help="quantile minimal pour garder une coexpression GRNBoost2 "
                        "(0.98 = top 2%%)")
    p.add_argument("--reactome-max-pathway", type=int, default=20,
                   help="taille max d'un pathway REACTOME pour rester informatif")

    # --- Étiquetage du run ---
    p.add_argument("--run-tag", default="auto",
                   help="suffixe ajouté à OUT_DIR pour ce run. 'auto' = construit "
                        "à partir des modules désactivés ; 'full' = baseline ; "
                        "ou nom libre.")
    p.add_argument("--seed", type=int, default=42,
                   help="graine aléatoire — appliquée à numpy + torch + cuda. "
                        "Inclus dans RUN_TAG (sauf si --run-tag <libre>) pour "
                        "ne pas écraser un run d'un autre seed.")

    p.add_argument("--n-epochs", dest="n_epochs", type=int, default=1200,
                   help="Nombre maximal d'epochs (early stopping arrête souvent "
                        "avant). Défaut 1000.")
    p.add_argument("--patience", type=int, default=150,
                   help="Patience de l'early stopping (epochs sans amélioration "
                        "AUC val avant arrêt). Défaut 100.")
    # --- Cache du graphe (itération rapide : saute la reconstruction §1-7) ---
    p.add_argument("--reuse-graph", dest="reuse_graph", action="store_true",
                   help="Réutilise le cache de build (§1-7) s'il est VALIDE "
                        "(même signature de config/sources/gènes). Sinon rebuild.")
    p.add_argument("--graph-cache", dest="graph_cache", default=None,
                   help="Chemin du cache de build (pickle). Défaut : "
                        "<OUT_DIR_BASE>/_graph_cache.pkl.")
    p.add_argument("--build-only", dest="build_only", action="store_true",
                   help="Construit (ou recharge) le graphe §1-7, écrit le cache, "
                        "puis S'ARRÊTE avant l'entraînement. Sert la règle Snakemake "
                        "'build_graph' (graphe bâti 1× puis réutilisé par tous les "
                        "seeds via --reuse-graph) et le debug rapide du build.")
    # --- Étapes optionnelles (généralisation : accélère les sweeps / portabilité) ---
    p.add_argument("--no-baselines", dest="no_baselines", action="store_true",
                   help="Saute les baselines ENTRAÎNÉES : MLP (§12) et DeepWalk/"
                        "Node2Vec (§13bis, lent). La baseline statistique |ΔExpr| "
                        "(§13, triviale) reste calculée. Sorties baseline absentes "
                        "du ranking (mlp_score/node2vec_score).")
    p.add_argument("--no-validation", dest="no_validation", action="store_true",
                   help="Saute la validation post-hoc sur BDD aging externes "
                        "(§14 : GenAge/CellAge/MSigDB/AgeAnno). Utile sans les "
                        "fichiers BDD ou sur un autre phénotype que la sénescence "
                        "(colonnes in_* / n_databases mises à 0).")
    # --- V5.4 (decoder-split, §14bis.6duovicies) ---
    p.add_argument("--decoder-split", dest="decoder_split", action="store_true",
                   help="V5.4 : route les arêtes SIGNÉES vers le décodeur "
                        "bilinéaire (existence via logsumexp) et libère le "
                        "cosinus des arêtes dirigées (recon cosinus = pool V5.1). "
                        "Nécessite --signed-decoder. Défaut OFF (backward-compat V5.3).")
    # --- V4.3-tune : hyperparamètres surchargeables (étaient hardcodés) ---
    p.add_argument("--kl-beta-max", dest="kl_beta_max", type=float, default=0.0005,
                   help="V4.3-tune : β final du KL annealing (0→kl_beta_max). "
                        "Défaut 0.0005. kl1 = 0.0001.")
    p.add_argument("--latent-dim", dest="latent_dim", type=int, default=64,
                   help="V4.3-tune : dimension de l'espace latent. Défaut 64.")
    # --- V-sup : VGAE circulaire (reconstruit) + features DE (+ tête option) ---
    # ARCHI : le VGAE reconstruit NORMALEMENT (recon + KL + décodeur → prêt pour
    # la perturbation Δμ standard) ; --de-features AJOUTE les stats DE aux nœuds
    # (circulaire) ; --supervised attache EN PLUS une tête de classification
    # entraînée CONJOINTEMENT (multi-tâche). Les deux activables indépendamment.
    # --- V6.3 : RAW expression as node features (distinct from --de-features) ---
    # A LEVEL per cell group (mean expression in P4, c0..c3), not a P4-vs-P16
    # CONTRAST: it does not hand the encoder the readout axis the way log2FC
    # does. Motivation: today NO node feature carries expression at all — it
    # lives only on the `expresses` edges — so perturbing a gene scales is_tf /
    # ppi_degree and never its expression. Pair with `--perturb-features expr`
    # (gnn_perturbation) to intervene on expression ONLY.
    p.add_argument("--expr-features", dest="expr_features", action="store_true",
                   default=False,
                   help="V6.3 : ajoute l'expression moyenne BRUTE par cell_group "
                        "comme features de nœud (expr_<group>). Niveau, pas "
                        "contraste → moins circulaire que --de-features, mais "
                        "l'axe reste partiellement lisible depuis les features "
                        "(à contrôler par la nulle N4). Défaut OFF.")
    p.add_argument("--no-expr-features", dest="expr_features",
                   action="store_false",
                   help="Désactive explicitement les features d'expression brute.")
    p.add_argument("--de-features", dest="de_features", action="store_true",
                   default=False,
                   help="V-sup : injecte les features DE de nœud (log2FC global/par "
                        "cluster, -log10 padj, Delta-pct) — CIRCULAIRES. Le VGAE "
                        "reconstruit normalement AVEC ces features. Défaut OFF "
                        "(anti-circularité). Standalone = 'VGAE circulaire'.")
    p.add_argument("--no-de-features", dest="de_features", action="store_false",
                   help="Désactive explicitement les features DE (ablation).")
    p.add_argument("--supervised", dest="supervised", action="store_true",
                   help="V-sup : attache une TÊTE de classification multi-label "
                        "(P4_vs_P16 + cluster_0..3) entraînée CONJOINTEMENT à la "
                        "reconstruction (loss = recon + β·KL + λ·classif). Importance "
                        "par cluster (saliency). Typiquement avec --de-features.")
    p.add_argument("--supervised-loss-weight", dest="supervised_loss_weight",
                   type=float, default=1.0,
                   help="V-sup : λ du terme de classification dans la loss jointe. "
                        "Défaut 1.0.")
    p.add_argument("--supervised-recompute-labels",
                   dest="supervised_recompute_labels", action="store_true",
                   help="V-sup : force le recalcul de la DE par cluster (scanpy "
                        "Wilcoxon) au lieu de lire le cache DEGs_P16_cluster_*.csv.")

    args, _unknown = p.parse_known_args(argv)

    # Application du raccourci --no-humess
    if args.no_humess:
        args.use_humess_edges = False
        args.use_humess_features = False

    # V4.3 : auto-résolution du chemin coexpr_diff si --diff-coexpr-file
    # n'a pas été surchargé (= reste sur la valeur par défaut historique).
    # method='sklearn' + prune='topk' → fichier V4.2 historique
    # `coexpr_diff.tsv` conservé (pas de cassure ascendante).
    _default_diff = "data/pyscenic/diff_coexpr/coexpr_diff.tsv"
    if args.diff_coexpr_file == _default_diff and (
            args.coexpr_method != "sklearn" or args.coexpr_prune != "topk"):
        args.diff_coexpr_file = (
            f"data/pyscenic/diff_coexpr/coexpr_diff."
            f"{args.coexpr_method}.{args.coexpr_prune}.tsv"
        )

    return args


# --- Dérivations de config (modules/features/run_tag) depuis les args parsés -----
import os as _os
from types import SimpleNamespace as _SNS

_DERIVE_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_config_derive.py")
with open(_DERIVE_PATH, encoding="utf-8") as _dfh:
    _DERIVE_CODE = compile(_dfh.read(), _DERIVE_PATH, "exec")


def derive_config(args):
    """Dérive modules actifs / feature flags / run_tag depuis les args parsés.
    Retourne un namespace (attributs = ex-globals du monolithe)."""
    ns = {"CLI_ARGS": args}
    exec(_DERIVE_CODE, ns)
    return _SNS(
        CLI_ARGS=args,
        EXCLUDED_FEATURES=ns["_EXCLUDED_FEATURES"],
        MODULES=ns["MODULES"],
        COEXPR_MODE=ns["COEXPR_MODE"],
        COEXPR_DIFFERENTIAL=ns["COEXPR_DIFFERENTIAL"],
        EDGE_TYPE_WEIGHTS=ns["EDGE_TYPE_WEIGHTS"],
        GENE_FEATURE_FLAGS=ns["GENE_FEATURE_FLAGS"],
        # V6.3 : raw-expression node features. Their names depend on CELL_GROUPS
        # (dataset-dependent), so the build registers the flags itself.
        USE_EXPR_NODE_FEATURES=ns["USE_EXPR_NODE_FEATURES"],
        EXPR_NODE_FEATURE_PREFIX=ns["EXPR_NODE_FEATURE_PREFIX"],
        OMNIPATH_EXTRA_EDGES=ns["OMNIPATH_EXTRA_EDGES"],
        RUN_TAG=ns["RUN_TAG"],
    )
