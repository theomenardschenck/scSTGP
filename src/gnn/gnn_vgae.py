"""
GNN VGAE — Priorisation non supervisée de gènes dans la sénescence (HUVEC)
==========================================================================
Approche NON SUPERVISÉE par Variational Graph AutoEncoder (VGAE) sur un
graphe hétérogène. Le VGAE apprend des embeddings de gènes en reconstruisant
la topologie du graphe — sans aucun label DEG.

PROBLÈME RÉSOLU : le pipeline supervisé précédent (gnn.py) souffrait de
circularité (features = log2FC/padj → labels = DEG basés sur ces mêmes stats).
Ici, le score d'importance ÉMERGE de l'espace latent, pas d'une formule.

MODE V-sup (--supervised) : plafond CIRCULAIRE assumé. Réutilise le MÊME graphe
et le MÊME backbone HeteroEncoder, mais entraîne l'encodeur end-to-end sur les
labels DEG multi-label (+ features DE en nœud) et calcule l'importance par
cluster. Opposé au VGAE non supervisé (mesure ce que la topologie seule capte).
Cf. _supervised.py + build_supervised_labels.py + docs/technical/gnn_supervised.md.

DONNÉES : GSE102090 (HUVEC), n=1 par condition (P4, P16).

ARCHITECTURE :
  1. Graphe hétérogène :
     - Noeuds "gene"       : features topologiques (is_tf, variance, ppi_degree)
                              PAS de log2FC/padj (supprime la circularité)
     - Noeuds "cell_group" : 5 groupes (P4, P16_cluster_0..3)
     - Arêtes "expresses"  : mean_expr, pct, std, cv, q25, q75, tf_activity
     - Arêtes "ppi"        : STRING (combined_score, unsigned)
     - Arêtes "regulates"  : pySCENIC TF→cible (weight)
     - Arêtes "same_pathway" : REACTOME
     - Arêtes "coexpression" : GRNBoost2 P16 (V4.1) OU différentiel
                               P4∪P16 (V4.2, --coexpr-mode differential,
                               edge_dim=6 option A)
     - Arêtes "signaling"/"tf_curated" : OmniPath/SIGNOR/CollecTRI signé (V4)
     - Arêtes "reactome_fi"  : Reactome Functional Interactions signé
                               (V4.2, --use-reactome-fi)

  2. VGAE : encoder HeteroGNN (GATConv) → μ, σ → z ~ N(μ,σ²) → decoder (inner product)
     Loss = reconstruction des arêtes + KL divergence
     V4.2 : pondération γ_t par edge_type au niveau message
            (--edge-type-weights, _ScaledConv ; cf. §14bis.6octies rapport)

  CHANGELOG en-tête :
   - V4   : OmniPath signed (signaling + tf_curated), edge_dim=2
   - V4.1 : --include-omnipath-genes (endpoints OmniPath dans gene set)
   - V4.1.1 : is_tf = pySCENIC ∪ CollecTRI (section 5)
   - V4.2 : coexpr différentielle P4∪P16 (option A, edge_dim 1→6),
            γ_t par edge_type (_ScaledConv message-level),
            Reactome FI signé (edge_type 'reactome_fi')
   - V6   : généralisation bulk/scRNA via env (GNN_EXPR_MATRIX/GNN_GROUP_META/
            GNN_CELL_GROUPS/GNN_HUMESS_CONDITIONS, cf. docs/technical/
            gnn_vgae_paths.md) + cache du build §1-7 (--reuse-graph) :
            recharge le graphe si la signature de config est identique
            (sources/matrice+mtime/conditions/features/flags → invalide si
            le nb de gènes change), sinon rebuild. logFC jamais en feature.
            metadata gatée GROUP_META (bulk = samplesheet d'échantillons ;
            scRNA = merged_P4_P16_metadata.csv). Étapes optionnelles :
            --no-baselines (saute MLP §12 + DeepWalk §13bis ; garde Stat §13)
            et --no-validation (saute BDD aging §14) — hors-signature cache.

  3. Score d'importance émergent :
     - Centralité dans l'espace latent (norme de μ)
     - Reconstruction error par gène
     - Distance aux clusters fonctionnels

  4. Comparaison avec :
     - Baseline MLP (mêmes features, pas de graphe)
     - Baseline statistique (ranking par log2FC)

  5. Validation post-hoc : GenAge, CellAge, MSigDB, AgeAnno (PAS d'entraînement)
"""

import os
import sys
import re
import json
import pickle
import argparse
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric.utils import negative_sampling
from sklearn.decomposition import PCA as PCA_sk
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score, silhouette_score
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# OmniPath integration (V4) — chargé lazily : si les flags --use-omnipath-*
# sont OFF (défaut), on n'importe rien. Sinon on importe le module local.
# Cela évite de planter un run baseline si `omnipath` n'est pas installé.
_OPI = None  # peuplé dans la section 6g si MODULES["use_omnipath_*"]

# =============================================================================
# ARGUMENTS CLI — MODULARITÉ DU GRAPHE & DES FEATURES
# =============================================================================
# Permet d'activer/désactiver chaque source de données et chaque feature pour
# les études d'ablation (cf. §11 du rapport : « qu'apporte vraiment HuMess ?
# la coexpression ? »). Les drapeaux par défaut reproduisent la config V3.3
# (toutes sources actives). Un manifest run_config.json est exporté dans le
# run_dir pour rendre chaque ablation auditable.
#
# Exemples :
#   python gnn_vgae.py                          # baseline V3.3 complet
#   python gnn_vgae.py --no-humess --run-tag no_humess
#   python gnn_vgae.py --no-coexpr --no-humess  # ablation double
#   python gnn_vgae.py --exclude-features ppi_degree,reg_degree
#   python gnn_vgae.py --ppi-score-thresh 700 --run-tag ppi700
# Si --run-tag vaut "auto" (défaut), il est dérivé des modules désactivés.

# Parseur CLI : déplacé dans _config.py (import-safe, réutilisable).
from _config import parse_cli_args  # noqa: E402


CLI_ARGS = parse_cli_args()

# Set des features à exclure (normalisées en minuscules, dépouillées d'espaces)
# Dérivations config (modules/features/run_tag) : déléguées à _config.py.
from _config import derive_config  # noqa: E402
_CFG = derive_config(CLI_ARGS)
_EXCLUDED_FEATURES  = _CFG.EXCLUDED_FEATURES
MODULES             = _CFG.MODULES
COEXPR_MODE         = _CFG.COEXPR_MODE
COEXPR_DIFFERENTIAL = _CFG.COEXPR_DIFFERENTIAL
EDGE_TYPE_WEIGHTS   = _CFG.EDGE_TYPE_WEIGHTS
GENE_FEATURE_FLAGS  = _CFG.GENE_FEATURE_FLAGS
RUN_TAG             = _CFG.RUN_TAG

# Application immédiate du seed à tous les RNG. Important de le faire AVANT
# tout import-side-effect ou allocation pour assurer la reproductibilité.
import random as _random
_random.seed(CLI_ARGS.seed)
np.random.seed(CLI_ARGS.seed)
torch.manual_seed(CLI_ARGS.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CLI_ARGS.seed)

# Mode DÉTERMINISTE opt-in (env GNN_DETERMINISTIC=1) — OFF par défaut, donc
# comportement de PRODUCTION inchangé (multi-thread, non bit-reproductible ; les
# réductions scatter-add CPU de PyG ne sont pas associatives → variance run-to-run
# même à seed fixe). Sert UNIQUEMENT le golden test du refactor : force threads=1
# + algorithmes déterministes pour comparer iso-comportement avant/après split.
# NB : PYTHONHASHSEED doit être posé AVANT le lancement (cf. tests/golden/run_golden.sh).
if os.environ.get("GNN_DETERMINISTIC", "0") == "1":
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as _e:
        print(f"[determinism] use_deterministic_algorithms indisponible : {_e}")
    print("[determinism] mode reproductible ON (threads=1, algos déterministes, seed complet)")

print("=" * 70)
print("Configuration modulaire :")
print("=" * 70)
for k, v in MODULES.items():
    flag = "ON " if v else "OFF"
    print(f"  [{flag}] {k}")
print(f"  Features actives  : "
      f"{[k for k, v in GENE_FEATURE_FLAGS.items() if v]}")
if _EXCLUDED_FEATURES:
    print(f"  Features exclues  : {sorted(_EXCLUDED_FEATURES)}")
print(f"  PPI threshold     : {CLI_ARGS.ppi_score_thresh}")
print(f"  Coexpr quantile   : {CLI_ARGS.coexpr_top_quantile}")
print(f"  REACTOME max size : {CLI_ARGS.reactome_max_pathway}")
print(f"  RUN_TAG           : {RUN_TAG}")
print()


# =============================================================================
# CONFIGURATION
# =============================================================================
# Cette section définit tous les chemins d'accès aux données et les
# hyperparamètres du pipeline. Les chemins pointent vers le cluster Nautilus
# (GLiCID) ; des chemins locaux commentés sont disponibles pour le debug.

# --- Chemins principaux sur le cluster Nautilus (GLiCID) ---
# LAB_DIR : racine de l'espace utilisateur sur le stockage partagé GLiCID.
# BASE_DIR : sous-dossier contenant le projet GNN (code + données).
# DATA_DIR : données d'entrée (scRNA-seq, PPI, pySCENIC, bases de données).
# SCENIC_DIR : sorties de pySCENIC (regulons, adjacencies, TF activity).
# OUT_DIR : dossier de sortie sur /scratch (écriture rapide, non sauvegardé).
# Chemins surchargeables par variable d'environnement (clone portable /
# orchestration Snakemake). Les défauts reproduisent le comportement
# historique sur GLiCID — exporter GNN_LAB_DIR / GNN_DATA_DIR /
# GNN_OUT_DIR_BASE (cf. workflow/Snakefile) pour pointer ailleurs.
# _REPO_ROOT : racine du dépôt (src/gnn/gnn_vgae.py → 3 niveaux au-dessus).
# Sert de défaut PORTABLE (local ET clone cluster), tout reste surchargeable env.
# --- Chemins : résolution déléguée à _paths.py (import-safe, réutilisable) ---
# Fix layout : find_repo_root remonte jusqu'au dossier `data/` → robuste au
# déploiement à plat du cluster (src/gnn_vgae.py) qui cassait dirname³.
# Les défauts (GLiCID) et les surcharges env (GNN_DATA_DIR/OUT_DIR_BASE/...) sont
# strictement identiques au monolithe. On dépaquette vers les globals historiques
# pour laisser le reste du script inchangé.
from _paths import resolve_paths, ensure_dirs  # noqa: E402
_PATHS = resolve_paths(CLI_ARGS, RUN_TAG)
LAB_DIR             = _PATHS.LAB_DIR
BASE_DIR            = _PATHS.BASE_DIR
DATA_DIR            = _PATHS.DATA_DIR
SCENIC_DIR          = _PATHS.SCENIC_DIR
OUT_DIR_BASE        = _PATHS.OUT_DIR_BASE
OUT_DIR             = _PATHS.OUT_DIR
GNN_DATA_DIR        = _PATHS.GNN_DATA_DIR
PPI_DIR             = _PATHS.PPI_DIR
DB_DIR              = _PATHS.DB_DIR
FIG_DIR             = _PATHS.FIG_DIR
OMNIPATH_CACHE_DIR  = _PATHS.OMNIPATH_CACHE_DIR
HUMESS_DIR          = _PATHS.HUMESS_DIR
HUMESS_CONDITIONS   = _PATHS.HUMESS_CONDITIONS
EXPR_MATRIX         = _PATHS.EXPR_MATRIX
GROUP_META          = _PATHS.GROUP_META
ensure_dirs(_PATHS)

# Manifest des modules activés/désactivés pour ce run — sert d'audit pour les
# études d'ablation (cf. perturb_report.py / cross_seed_report).
_MANIFEST_PATH = os.path.join(OUT_DIR, "run_config.json")
with open(_MANIFEST_PATH, "w") as _fh:
    json.dump({
        "run_tag": RUN_TAG,
        "seed": CLI_ARGS.seed,
        "modules": MODULES,
        "gene_feature_flags": GENE_FEATURE_FLAGS,
        "excluded_features": sorted(_EXCLUDED_FEATURES),
        "ppi_score_thresh": CLI_ARGS.ppi_score_thresh,
        "coexpr_top_quantile": CLI_ARGS.coexpr_top_quantile,
        "reactome_max_pathway": CLI_ARGS.reactome_max_pathway,
        "omnipath_cache_dir": OMNIPATH_CACHE_DIR,
        "omnipath_download_if_missing": CLI_ARGS.omnipath_download_if_missing,
        # V4.2
        "coexpr_mode": COEXPR_MODE,
        "diff_coexpr_file": CLI_ARGS.diff_coexpr_file,
        "use_reactome_fi": MODULES["use_reactome_fi"],
        "reactome_fi_file": CLI_ARGS.reactome_fi_file,
        "edge_type_weights": EDGE_TYPE_WEIGHTS,
        "dedup_ppi_signed": getattr(CLI_ARGS, "dedup_ppi_signed", "off"),
        # V4.3
        "coexpr_method": CLI_ARGS.coexpr_method,
        "coexpr_prune": CLI_ARGS.coexpr_prune,
        # V5 (TIER 1c) — wiring signed message + bilinear decoder
        "signed_message": CLI_ARGS.signed_message,
        "signed_decoder": CLI_ARGS.signed_decoder,
        "signed_loss_weight": CLI_ARGS.signed_loss_weight,
        # V5.4 (decoder-split) + V4.3-tune (kl_beta_max, latent_dim)
        "decoder_split": bool(getattr(CLI_ARGS, "decoder_split", False)),
        "kl_beta_max": CLI_ARGS.kl_beta_max,
        "latent_dim": CLI_ARGS.latent_dim,
        # V5.3 (TIER 1c.7) — tête sub-espace signed_proj
        "signed_decoder_dim": CLI_ARGS.signed_decoder_dim,
        # V5 phase 2 (TIER 1c.5 strict) — hold-out signed TF pour gate rigoureux
        # Les clés `holdout_signed_tf_set` et `holdout_signed_tf_seed_used`
        # sont enrichies plus tard (cf. section 10) une fois le pool construit.
        "holdout_signed_tf_fraction": CLI_ARGS.holdout_signed_tf_fraction,
        "holdout_signed_tf_seed": CLI_ARGS.holdout_signed_tf_seed,
    }, _fh, indent=2)
print(f"  Manifest écrit    : {_MANIFEST_PATH}")

# ── Paramètres du graphe ─────────────────────────────────────────────────────
# PPI_SCORE_THRESH : seuil de confiance STRING (0-1000). 900 = "highest
#   confidence". On ne garde que les interactions protéine-protéine très
#   fiables pour éviter le bruit dans le graphe.
PPI_SCORE_THRESH = CLI_ARGS.ppi_score_thresh
# REACTOME_MAX_PATHWAY : taille max d'un pathway REACTOME (en gènes).
#   Les très grands pathways (ex : "Metabolism") sont non informatifs —
#   ils connectent trop de gènes entre eux et diluent le signal.
#   Les très petits (< 2 gènes) ne créent pas d'arêtes utiles.
REACTOME_MAX_PATHWAY = CLI_ARGS.reactome_max_pathway
# COEXPR_TOP_QUANTILE : seules les co-expressions GRNBoost2 au-dessus de
#   ce quantile sont conservées. 0.98 = top 2% des poids → réseau épars
#   de haute confiance.
COEXPR_TOP_QUANTILE = CLI_ARGS.coexpr_top_quantile

# ── Paramètres du VGAE ──────────────────────────────────────────────────────
# HIDDEN_DIM : dimension des couches cachées du GNN (après projection des
#   features d'entrée). Chaque couche GATConv produit un vecteur de cette
#   taille pour chaque noeud. Avec N_HEADS=4, chaque tête travaille en
#   dimension HIDDEN_DIM/N_HEADS = 32, puis les résultats sont concaténés.
HIDDEN_DIM = 128
# LATENT_DIM : dimension de l'espace latent (μ et log(σ²) ont cette taille).
#   C'est l'espace dans lequel on calcule le score d'importance. 64 offre
#   un bon compromis entre expressivité et risque de KL collapse.
#   Surchargeable via --latent-dim (défaut 64).
LATENT_DIM = CLI_ARGS.latent_dim
# N_LAYERS : nombre de couches de message passing. Chaque couche permet
#   à un gène de "voir" un voisin de plus. Avec 3 couches, chaque gène
#   agrège l'information de ses voisins jusqu'à distance 3 dans le graphe.
N_LAYERS = 3
# N_HEADS : nombre de têtes d'attention dans GATConv. Le multi-head
#   attention permet au modèle d'apprendre différents types de relations
#   (ex : une tête pour la co-expression, une autre pour la régulation).
N_HEADS = 4
# DROPOUT : régularisation par extinction aléatoire de neurones pendant
#   l'entraînement. 0.2 = 20% des neurones sont désactivés à chaque
#   forward pass, ce qui réduit le surapprentissage.
DROPOUT = 0.2
# N_EPOCHS : nombre maximal d'epochs (itérations complètes sur les données).
#   L'early stopping arrêtera souvent avant (typiquement epoch 30-80).
#   Surchargeable via --n-epochs (défaut 1000).
N_EPOCHS = CLI_ARGS.n_epochs
# LR : learning rate de l'optimiseur Adam. 0.005 est relativement élevé
#   (typique des GNN qui convergent vite) mais compensé par le gradient
#   clipping et le weight decay.
LR = 0.005
# EDGE_SAMPLE_RATIO : fraction d'arêtes réservées pour le test. 10% des
#   arêtes sont masquées et le VGAE doit les prédire. Les 90% restantes
#   servent à l'entraînement.
EDGE_SAMPLE_RATIO = 0.1
# NEG_SAMPLE_RATIO : pour chaque arête positive (vraie connexion), on
#   échantillonne ce ratio d'arêtes négatives (paires non connectées).
#   1.0 = autant de négatifs que de positifs → classes équilibrées.
NEG_SAMPLE_RATIO = 1.0
# N_CLUSTERS : nombre de clusters K-means sur les embeddings finaux.
#   Sert à identifier des groupes fonctionnels de gènes dans l'espace latent.
N_CLUSTERS = 8

# ── Paramètres anti-KL-collapse ──────────────────────────────────────────────
# Le posterior collapse est un problème classique des VGAE : le modèle apprend
# à ignorer l'espace latent en produisant q(z|x) ≈ p(z) = N(0,I) pour tout x,
# ce qui donne des embeddings identiques et une AUC de 0.5.
# Solutions :
#   1. KL annealing : β commence à 0 et monte linéairement jusqu'à KL_BETA_MAX
#      pendant KL_WARMUP_EPOCHS. Le modèle apprend d'abord à bien reconstruire,
#      puis la régularisation KL entre progressivement.
#   2. Free bits : on impose un minimum de KL par dimension latente (FREE_BITS).
#      Si une dimension a KL < FREE_BITS, on ne la pénalise pas. Cela force le
#      modèle à utiliser au moins FREE_BITS nats d'information par dimension.
KL_BETA_MAX = CLI_ARGS.kl_beta_max   # β final ; --kl-beta-max (défaut 0.0005 ; kl1=0.0001)
KL_WARMUP_EPOCHS = 50     # Warmup court puis β stable — le cosinus n'a pas besoin de long warmup
FREE_BITS = 0.5           # Minimum KL par dimension latente (en nats)

# ── Paramètres du baseline MLP ───────────────────────────────────────────────
# Le MLP est une baseline "sans graphe" : il utilise les mêmes features que
# le VGAE mais ne fait PAS de message passing. Si le VGAE bat le MLP en AUC
# de reconstruction, cela prouve que la topologie du graphe apporte de
# l'information au-delà des features brutes.
MLP_HIDDEN = 64    # Dimension cachée du MLP (plus petit que le VGAE)
MLP_EPOCHS = 250   # Moins d'epochs car le MLP converge vite (pas de graphe)
MLP_LR = 0.001     # Learning rate plus faible que le VGAE (MLP est plus stable)

# ── Style des figures ────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 11})


# =============================================================================
# UTILITAIRES
# =============================================================================
def download_if_absent(url, local_path, label=""):
    """
    Ensure `local_path` exists, downloading from `url` on first use.

    Fetches external databases (STRING PPI, MSigDB REACTOME/Hallmarks, GenAge,
    CellAge, AgeAnno) and caches them locally. A custom User-Agent avoids
    rejections from some servers (e.g. Broad Institute).

    Offline-aware: on a node without internet (HPC compute nodes behind a proxy
    return 403) or when GNN_ALLOW_DOWNLOADS=0, a missing file raises a clear,
    actionable error instead of a cryptic urllib traceback — pre-stage the file
    on a node with internet (frontal) or rsync your local data/ cache.
    """
    if os.path.exists(local_path):
        print(f"    [cache] {label or os.path.basename(local_path)}")
        return
    _allow = os.environ.get("GNN_ALLOW_DOWNLOADS", "1").lower() not in ("0", "false", "no")
    if not _allow:
        raise FileNotFoundError(
            f"{local_path} absent et téléchargements désactivés "
            f"(GNN_ALLOW_DOWNLOADS=0). Pré-stagez ce fichier "
            f"({label or os.path.basename(local_path)}) : exécutez sur un nœud avec "
            f"internet (frontal) ou rsync depuis votre cache local data/.")
    print(f"    Téléchargement {label or os.path.basename(local_path)}...")
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (research)")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except Exception as _e:
        raise FileNotFoundError(
            f"Échec téléchargement {label or os.path.basename(local_path)} ({_e}). "
            f"Nœud probablement hors-ligne (HPC derrière proxy → 403). Pré-stagez "
            f"{local_path} : nœud avec internet (frontal) ou rsync du cache local "
            f"data/. URL : {url}") from _e
    with open(local_path, "wb") as f:
        f.write(raw)
    print(f"      OK ({len(raw) / 1e6:.1f} MB)")


# =============================================================================
# CACHE DU GRAPHE (§1-7) — réutilisation via --reuse-graph
# =============================================================================
# Le build du graphe (§1-7, ~40 min) peut être mis en cache (pickle). Avec
# --reuse-graph, on recharge le cache UNIQUEMENT si sa signature de config
# correspond : sources actives (MODULES), matrice/fichiers + mtime/taille,
# conditions HuMess, features exclues, flags CLI graphe. La matrice étant dans
# la signature (mtime+taille), tout changement de jeu de données — donc de
# NB DE GÈNES — invalide le cache. n_genes est stocké et ré-affiché au
# chargement pour vérification → jamais de graphe obsolète réutilisé.
_CACHE_VARS = ["CELL_GROUPS", "_COEXPR_DIM", "_dst", "_f", "_g", "_src", "b", "cell_group_features", "coexpr_dst", "coexpr_src", "coexpr_w_tensor", "col", "data", "edge_attr_cocat", "edge_attr_expresses", "edge_attr_ppi", "edge_attr_reactome_fi", "edge_attr_regulates", "edge_attr_signaling", "edge_attr_tf_curated", "edge_index_cocat", "edge_index_coexpr", "edge_index_expresses", "edge_index_pathway", "edge_index_ppi", "edge_index_reactome_fi", "edge_index_regulates", "edge_index_signaling", "edge_index_tf_curated", "f", "g", "gene_features", "gene_symbols", "gene_to_idx", "group_stats", "grp", "i", "idx", "j", "line", "mask", "mean_expr_per_group", "mu", "n_genes", "omnipath_endpoints", "op_sig_dst", "op_sig_src", "op_tf_dst", "op_tf_src", "pair", "parts", "ppi_dst", "ppi_src", "react_dst", "react_src", "reactome_fi_dst", "reactome_fi_src", "reg_dst", "reg_src", "score", "sign", "std", "target"]
import hashlib as _hashlib
def _mtime_sig(_p):
    try:
        _s = os.stat(_p); return (round(_s.st_mtime, 2), _s.st_size)
    except OSError:
        return None
def _resolve_expr_path():
    return EXPR_MATRIX if os.path.isabs(EXPR_MATRIX) else os.path.join(DATA_DIR, EXPR_MATRIX)
_TRAIN_ONLY = {"seed", "run_tag", "n_epochs", "patience", "reuse_graph",
               "graph_cache", "build_only", "device", "lr", "kl_beta_max",
               "no_baselines", "no_validation",
               # V-sup : n'affectent que l'étape post-build (features DE ajoutées
               # APRÈS restauration du cache + tête jointe) → cache réutilisable.
               "supervised", "de_features", "supervised_loss_weight",
               "supervised_recompute_labels"}  # post-build only
_sig_obj = {
    "env": {_k: os.environ.get(_k, "") for _k in
            ("GNN_EXPR_MATRIX", "GNN_GROUP_META", "GNN_CELL_GROUPS",
             "GNN_HUMESS_CONDITIONS", "GNN_HUMESS_DIR", "GNN_DATA_DIR")},
    "modules": sorted((_k, bool(_v)) for _k, _v in MODULES.items()),
    "excluded": sorted(_EXCLUDED_FEATURES),
    "humess_conditions": list(HUMESS_CONDITIONS),
    "cli": {_k: str(_v) for _k, _v in sorted(vars(CLI_ARGS).items())
            if _k not in _TRAIN_ONLY},
    "mtimes": {_p: _mtime_sig(_p) for _p in
               (_resolve_expr_path(), GROUP_META or "", HUMESS_DIR,
                getattr(CLI_ARGS, "diff_coexpr_file", "") or "")},
}
_SIG = _hashlib.md5(repr(_sig_obj).encode()).hexdigest()
_GRAPH_CACHE = CLI_ARGS.graph_cache or os.path.join(OUT_DIR_BASE, "_graph_cache.pkl")
_REUSE_OK = False
if getattr(CLI_ARGS, "reuse_graph", False) and os.path.exists(_GRAPH_CACHE):
    try:
        with open(_GRAPH_CACHE, "rb") as _fh:
            _cache = pickle.load(_fh)
    except Exception as _e:
        _cache = None
        print(f"[reuse-graph] cache illisible ({_e}) -> rebuild complet")
    if _cache is not None and _cache.get("_sig") == _SIG:
        globals().update({_k: _v for _k, _v in _cache.items() if _k != "_sig"})
        _REUSE_OK = True
        print(f"[reuse-graph] OK cache VALIDE (signature identique) -> sections 1-7 sautees "
              f"(n_genes={_cache.get('n_genes')}, {len(_cache.get('gene_symbols', []))} symboles)")
    elif _cache is not None:
        print("[reuse-graph] cache OBSOLETE (config/sources/fichiers/nb-genes "
              "differents) -> rebuild complet du graphe")
elif getattr(CLI_ARGS, "reuse_graph", False):
    print(f"[reuse-graph] aucun cache a {_GRAPH_CACHE} -> build puis mise en cache")

if not _REUSE_OK:
    # §1-7 : construction du graphe déléguée à _graph_build.py (import-safe,
    # réutilisable). Contrat = _CACHE_VARS (identique au cache --reuse-graph).
    from _graph_build import build_graph  # noqa: E402
    globals().update(build_graph(dict(globals())))
    # Sauvegarde du graphe complet pour réutilisation (perturbations, etc.)
    torch.save(data, os.path.join(OUT_DIR, "hetero_graph_vgae.pt"))

    # ----- mise en cache du build (sections 1-7) pour --reuse-graph -----
    # Robustesse : on saute les variables non-picklables (ex. handle de fichier
    # temporaire `f`/`fh` réutilisé en aval — non requis au reload) en testant
    # chaque valeur ; et on écrit dans un .tmp puis rename atomique pour ne
    # JAMAIS laisser un cache partiel/corrompu en cas d'échec.
    try:
        os.makedirs(os.path.dirname(_GRAPH_CACHE) or ".", exist_ok=True)
        _bundle = {"_sig": _SIG}
        _skipped = []
        for _k in _CACHE_VARS:
            if _k not in globals():
                continue
            _v = globals()[_k]
            try:
                pickle.dumps(_v, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                _skipped.append(_k)  # non-picklable & non requis au reload
                continue
            _bundle[_k] = _v
        _tmp = _GRAPH_CACHE + ".tmp"
        with open(_tmp, "wb") as _fh:
            pickle.dump(_bundle, _fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(_tmp, _GRAPH_CACHE)  # rename atomique
        print(f"[reuse-graph] cache ecrit -> {_GRAPH_CACHE} "
              f"(n_genes={globals().get('n_genes')}, {len(_bundle)-1} vars, sig={_SIG[:8]})"
              + (f" [skip non-picklables: {_skipped}]" if _skipped else ""))
    except Exception as _e:
        print(f"[reuse-graph] echec ecriture cache ({_e}) -- non bloquant")
        try:
            os.path.exists(_GRAPH_CACHE + ".tmp") and os.remove(_GRAPH_CACHE + ".tmp")
        except OSError:
            pass

# --- Arrêt build-only : graphe prêt (construit + mis en cache OU rechargé) →
# on s'arrête AVANT l'entraînement. Sert la règle Snakemake `build_graph`
# (graphe bâti 1× puis réutilisé par tous les seeds via --reuse-graph) et le
# debug rapide du build. build_only ∈ _TRAIN_ONLY → n'affecte pas la signature.
if getattr(CLI_ARGS, "build_only", False):
    print(f"\n[build-only] Graphe prêt (n_genes={globals().get('n_genes')}) + cache "
          f"→ arrêt avant l'entraînement (--build-only).")
    sys.exit(0)

# =============================================================================
# 8-10. MODÈLE + ENTRAÎNEMENT : délégués à _train.py (import-safe, réutilisable).
# Corps exécuté dans un namespace (sémantique module-level du monolithe). Produit
# model/embeddings/tête-sup/métriques, fusionnés dans les globals pour le scoring.
# =============================================================================
from _train import train_vgae  # noqa: E402
globals().update(train_vgae(dict(globals())))
# =============================================================================
# 11-16. SCORING + EXPORT : délégués à _score.py (import-safe, réutilisable).
# Corps exécuté dans un namespace (sémantique module-level). Écrit gene_ranking,
# gene_embeddings, vgae_weights, vgae_metrics, group_expression.
# =============================================================================
from _score import score_and_write  # noqa: E402
score_and_write(dict(globals()))
