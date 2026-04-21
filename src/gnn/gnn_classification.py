"""
GNN Hétérogène — Classification multi-label de l'implication dans la sénescence
================================================================================
Graphe hétérogène (PyTorch Geometric) avec pour tâche la CLASSIFICATION
MULTI-LABEL de chaque gène selon son rôle dans la sénescence.

Données : GSE102090 (HUVEC) — n=1 par condition (un échantillon P4, un P16).
Le pseudo-bulk gold standard (agrégation par réplicat + DESeq2/edgeR) est
impossible. Pour compenser, le pipeline intègre trois mesures de robustesse :

  1. CONSENSUS MULTI-MÉTHODES (Wilcoxon + MAST) : un gène n'est considéré DE
     que s'il est significatif dans les deux tests. MAST modélise le dropout
     et inclut la profondeur de séquençage comme covariable technique.

  2. BOOTSTRAP DE STABILITÉ : sous-échantillonnage de 80% des cellules × 100
     itérations. Le score de stabilité (fraction des bootstraps où le gène
     reste DE) mesure la robustesse au bruit de composition cellulaire.

  3. FEATURES DISTRIBUTIONNELLES : std, CV, quantiles par groupe cellulaire
     sur les arêtes d'expression — le meilleur substitut à la variabilité
     inter-réplicats quand on n'a qu'un seul échantillon par condition.

  4. LOSS PONDÉRÉE PAR CONFIANCE : la contribution de chaque gène à la loss
     est pondérée par son score de confiance (bootstrap × consensus), de sorte
     que les labels fiables guident davantage l'apprentissage.

Labels (basés sur le clustering scRNA-seq P4/P16) :
  - de_p4_vs_p16  : gène DE dans la transition globale P4 → P16
  - de_cluster_0  : gène DE spécifique au cluster 0 de P16
  - de_cluster_1  : gène DE spécifique au cluster 1 de P16
  - de_cluster_2  : gène DE spécifique au cluster 2 de P16
  - de_cluster_3  : gène DE spécifique au cluster 3 de P16

Noeuds :
  - "gene"       : features intrinsèques (is_tf, variance, log2FC, bootstrap_stability,
                    consensus_score, ppi_degree...)
  - "cell_group" : 5 groupes (P4, P16_cluster_0..3), features = stats du groupe

Arêtes :
  - ("cell_group", "expresses", "gene")   : mean_expr, pct, tf_activity, std, cv, q25, q75
  - ("gene", "expressed_in", "cell_group") : reverse
  - ("gene", "ppi", "gene")               : STRING PPI (combined_score)
  - ("gene", "same_pathway", "gene")       : REACTOME (même voie)
  - ("gene", "regulates", "gene")          : pySCENIC TF→cible (weight)
  - ("gene", "regulated_by", "gene")       : reverse régulation
  - ("gene", "coexpression", "gene")       : GRNBoost2 adjacencies (filtrées)

Validation post-hoc : GenAge, CellAge, MSigDB, AgeAnno (comparaison, pas labels).
"""

import os
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.decomposition import PCA as PCA_sk
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
GNN_DATA_DIR = os.path.join(DATA_DIR, "gnn_data")
RNASEQ_DIR = os.path.join(DATA_DIR, "RNAseq")
PPI_DIR = os.path.join(DATA_DIR, "PPI")
DB_DIR = os.path.join(DATA_DIR, "databases")
SCENIC_DIR = os.path.join(BASE_DIR, "..", "output", "pyscenic")
OUT_DIR = os.path.join(BASE_DIR, "..", "output", "gnn")
FIG_DIR = os.path.join(OUT_DIR, "figure")

for d in [PPI_DIR, DB_DIR, OUT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Paramètres du graphe ─────────────────────────────────────────────────────
PPI_SCORE_THRESH = 900       # Score STRING min — haute confiance (réduit ~500k → ~50-80k arêtes)
REACTOME_MAX_PATHWAY = 20    # Taille max pathway (évite les cliques géantes qui noient le signal)
COEXPR_TOP_QUANTILE = 0.98   # Top 2% des adjacencies GRNBoost2 (plus sélectif)
PADJ_THRESH = 0.05           # Seuil de significativité pour les DEGs

# ── Paramètres du modèle ─────────────────────────────────────────────────────
HIDDEN_DIM = 128
N_LAYERS = 3
N_EPOCHS = 600
LR = 0.001                   # LR plus faible pour convergence plus stable
MASK_RATIO = 0.2
DROPOUT = 0.3                # Dropout augmenté contre l'overfit
N_LABELS = 5  # de_p4_vs_p16, de_cluster_0, _1, _2, _3
EARLY_STOPPING_PATIENCE = 80 # Arrêt si pas d'amélioration pendant 80 epochs
LAMBDA_SCORE = 0.3           # Poids de la loss de scoring (vs classification)
N_HEADS = 4                  # Nombre de têtes d'attention GATConv

LABEL_NAMES = ["de_p4_vs_p16", "de_cluster_0", "de_cluster_1",
               "de_cluster_2", "de_cluster_3"]

# ── Style des figures ────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 11})


# =============================================================================
# UTILITAIRES
# =============================================================================
def download_if_absent(url, local_path, label=""):
    if os.path.exists(local_path):
        print(f"    [cache] {label or os.path.basename(local_path)}")
        return
    print(f"    Téléchargement {label or os.path.basename(local_path)}...")
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (research)")
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    with open(local_path, "wb") as f:
        f.write(raw)
    print(f"      OK ({len(raw) / 1e6:.1f} MB)")


# =============================================================================
# 1. CHARGEMENT DES DONNÉES scRNA-seq (P4 / P16)
# =============================================================================
print("=" * 70)
print("1. Chargement des données scRNA-seq (P4 / P16)")
print("=" * 70)

# ── Metadata : contient passage, cluster_P16, cell_state ────────────────────
metadata = pd.read_csv(os.path.join(GNN_DATA_DIR, "merged_P4_P16_metadata.csv"))
print(f"  Metadata : {len(metadata)} cellules")
print(f"    P4  : {(metadata['passage'] == 'P4').sum()} cellules")
print(f"    P16 : {(metadata['passage'] == 'P16').sum()} cellules")

# Clusters P16
p16_meta = metadata[metadata["passage"] == "P16"].copy()
p16_clusters = sorted(p16_meta["cluster_P16"].dropna().unique())
print(f"    Clusters P16 : {p16_clusters}")
for c in p16_clusters:
    n = (p16_meta["cluster_P16"] == c).sum()
    print(f"      Cluster {c} : {n} cellules")

# ── DEGs : labels multi-label pour chaque gène ──────────────────────────────
print("\n  Chargement des DEGs (labels)...")

degs_p4_p16 = pd.read_csv(os.path.join(GNN_DATA_DIR, "DEGs_P4_vs_P16.csv"))
degs_p4_p16_sig = set(
    degs_p4_p16[degs_p4_p16["p_val_adj"] < PADJ_THRESH]["gene"].dropna()
)
print(f"    DEGs P4 vs P16 : {len(degs_p4_p16_sig)} gènes significatifs")

degs_clusters = {}
for c in [0, 1, 2, 3]:
    path = os.path.join(GNN_DATA_DIR, f"DEGs_P16_cluster_{c}.csv")
    df = pd.read_csv(path)
    degs_clusters[c] = set(df[df["p_val_adj"] < PADJ_THRESH]["gene"].dropna())
    print(f"    DEGs cluster {c} : {len(degs_clusters[c])} gènes significatifs")

# ── Sélection des gènes : union de tous les DEGs ────────────────────────────
all_deg_genes = set(degs_p4_p16_sig)
for c in degs_clusters:
    all_deg_genes |= degs_clusters[c]

# On charge les noms de colonnes du CSV normalisé pour avoir la liste complète
# des gènes disponibles (sans charger toute la matrice)
with open(os.path.join(GNN_DATA_DIR, "merged_P4_P16_normalized.csv")) as f:
    header = f.readline().strip().split(",")
# Les 4 premières colonnes sont barcode, passage, cluster_P16, cell_state
all_available_genes = [g.strip('"') for g in header[4:]]
available_gene_set = set(all_available_genes)

# Gènes sélectionnés = tous les DEGs qui sont dans la matrice d'expression
gene_symbols = np.array(sorted(all_deg_genes & available_gene_set))
gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}
n_genes = len(gene_symbols)

print(f"\n  Gènes sélectionnés : {n_genes}")
print(f"    dont DE P4→P16      : {len(degs_p4_p16_sig & set(gene_symbols))}")
for c in [0, 1, 2, 3]:
    print(f"    dont DE cluster {c}  : {len(degs_clusters[c] & set(gene_symbols))}")

# ── Construction des labels multi-label ──────────────────────────────────────
labels = np.zeros((n_genes, N_LABELS), dtype=np.float32)
for i, g in enumerate(gene_symbols):
    if g in degs_p4_p16_sig:
        labels[i, 0] = 1.0
    for c in range(4):
        if g in degs_clusters[c]:
            labels[i, 1 + c] = 1.0

print(f"\n  Labels multi-label ({N_LABELS} classes) :")
for j, name in enumerate(LABEL_NAMES):
    pos = labels[:, j].sum()
    print(f"    {name:20s} : {int(pos):5d} positifs ({100*pos/n_genes:.1f}%)")

# ── Chargement des scores de confiance (bootstrap + consensus) ──────────────
# JUSTIFICATION : avec n=1 par condition (GSE102090), les labels DEG issus du
# test Wilcoxon cell-level contiennent inévitablement des faux positifs. Le
# script R exporte deux mesures de robustesse :
#   - bootstrap_stability : fraction des sous-échantillonnages (80% cellules)
#     où le gène reste significatif. Un gène stable (>70%) ne dépend pas de
#     quelques cellules outliers.
#   - consensus_score : 1.0 si DE à la fois en Wilcoxon ET MAST, 0.5 si un
#     seul test, 0.0 si aucun. Le consensus multi-méthodes réduit les faux
#     positifs propres à chaque test.
# Ces scores seront utilisés comme :
#   1. Features de noeuds (le GNN apprend que certains labels sont plus fiables)
#   2. Pondération de la loss (les gènes avec haute confiance pèsent plus)
print("\n  Chargement des scores de confiance (bootstrap + consensus)...")

# Bootstrap stabilité (P4 vs P16)
bootstrap_path = os.path.join(GNN_DATA_DIR, "bootstrap_stability_P4_vs_P16.csv")
if os.path.exists(bootstrap_path):
    boot_df = pd.read_csv(bootstrap_path)
    boot_map = dict(zip(boot_df["gene"], boot_df["bootstrap_stability"]))
    bootstrap_stability = np.array(
        [boot_map.get(g, 0.0) for g in gene_symbols], dtype=np.float32
    )
    print(f"    Bootstrap stabilité chargée : mean={bootstrap_stability.mean():.3f}, "
          f"stable(≥0.7)={int((bootstrap_stability >= 0.7).sum())}/{n_genes}")
else:
    print(f"    [WARN] {bootstrap_path} non trouvé — stabilité mise à 0.5 (neutre)")
    bootstrap_stability = np.full(n_genes, 0.5, dtype=np.float32)

# Consensus multi-méthodes (P4 vs P16)
consensus_path = os.path.join(GNN_DATA_DIR, "consensus_P4_vs_P16.csv")
if os.path.exists(consensus_path):
    cons_df = pd.read_csv(consensus_path)
    cons_map = dict(zip(cons_df["gene"], cons_df["consensus_score"]))
    consensus_score_p4p16 = np.array(
        [cons_map.get(g, 0.0) for g in gene_symbols], dtype=np.float32
    )
    print(f"    Consensus P4vP16 chargé : mean={consensus_score_p4p16.mean():.3f}, "
          f"consensus(=1.0)={int((consensus_score_p4p16 == 1.0).sum())}/{n_genes}")
else:
    print(f"    [WARN] {consensus_path} non trouvé — consensus mis à 0.5 (neutre)")
    consensus_score_p4p16 = np.full(n_genes, 0.5, dtype=np.float32)

# Consensus par cluster P16
consensus_cl_path = os.path.join(GNN_DATA_DIR, "consensus_clusters_P16.csv")
consensus_score_clusters = {}
if os.path.exists(consensus_cl_path):
    cons_cl_df = pd.read_csv(consensus_cl_path)
    for c in [0, 1, 2, 3]:
        cl_sub = cons_cl_df[cons_cl_df["cluster"] == c]
        cl_map = dict(zip(cl_sub["gene"], cl_sub["consensus_score"]))
        consensus_score_clusters[c] = np.array(
            [cl_map.get(g, 0.0) for g in gene_symbols], dtype=np.float32
        )
    print(f"    Consensus clusters chargé : 4 clusters")
else:
    print(f"    [WARN] {consensus_cl_path} non trouvé — consensus clusters mis à 0.5")
    for c in [0, 1, 2, 3]:
        consensus_score_clusters[c] = np.full(n_genes, 0.5, dtype=np.float32)

# Score de confiance agrégé par gène : combinaison bootstrap + consensus
# sur tous les labels (P4vP16 + 4 clusters)
# C'est le score qui sera utilisé pour pondérer la loss
confidence_per_label = np.column_stack([
    consensus_score_p4p16 * bootstrap_stability,    # label 0: P4 vs P16
    consensus_score_clusters[0],                     # label 1: cluster 0
    consensus_score_clusters[1],                     # label 2: cluster 1
    consensus_score_clusters[2],                     # label 3: cluster 2
    consensus_score_clusters[3],                     # label 4: cluster 3
])  # shape: (n_genes, 5)
# Score global = moyenne sur les labels actifs (où le gène est DE)
confidence_global = np.where(
    labels.sum(axis=1) > 0,
    (confidence_per_label * labels).sum(axis=1) / (labels.sum(axis=1) + 1e-8),
    0.5  # pour les non-DE, confiance neutre
).astype(np.float32)

print(f"    Confiance globale : mean={confidence_global.mean():.3f}, "
      f"min={confidence_global.min():.3f}, max={confidence_global.max():.3f}")

# =============================================================================
# 2. CALCUL DE L'EXPRESSION PAR GROUPE CELLULAIRE
# =============================================================================
print("\n" + "=" * 70)
print("2. Calcul de l'expression par groupe cellulaire")
print("=" * 70)

# Définition des 5 groupes cellulaires
CELL_GROUPS = ["P4", "P16_cluster_0", "P16_cluster_1",
               "P16_cluster_2", "P16_cluster_3"]

# On doit lire la matrice normalisée pour calculer mean_expr et pct_expressing
# par groupe. Pour optimiser la mémoire, on lit uniquement les colonnes utiles.
print("  Chargement de la matrice normalisée (colonnes sélectionnées)...")

# Colonnes à lire : les 4 premières (metadata) + les gènes sélectionnés
cols_to_read = ["barcode", "passage", "cluster_P16", "cell_state"] + gene_symbols.tolist()
normalized = pd.read_csv(
    os.path.join(GNN_DATA_DIR, "merged_P4_P16_normalized.csv"),
    usecols=cols_to_read,
)
print(f"  Matrice chargée : {normalized.shape}")

# Assigner chaque cellule à son groupe
def assign_group(row):
    if row["passage"] == "P4":
        return "P4"
    else:
        c = row["cluster_P16"]
        if pd.notna(c):
            return f"P16_cluster_{int(c)}"
    return None

normalized["group"] = normalized.apply(assign_group, axis=1)
normalized = normalized.dropna(subset=["group"])

# Calculer les statistiques par groupe × gène
# JUSTIFICATION : avec n=1 par condition (GSE102090), on ne peut pas estimer la
# variabilité biologique inter-réplicats. En revanche, la variabilité INTRA-GROUPE
# (entre cellules d'un même groupe) est le meilleur substitut disponible :
#   - std_expression : dispersion de l'expression au sein du groupe
#   - cv_expression  : coefficient de variation (std/mean) — mesure relative de bruit
#   - q25, q75       : quantiles — capturent la forme de la distribution
# Ces features permettent au GNN de distinguer un gène avec une forte moyenne
# mais très variable (moins fiable) d'un gène avec la même moyenne mais stable.
print("  Calcul des statistiques par groupe (mean, pct, std, CV, quantiles)...")
group_stats = {}
for grp in CELL_GROUPS:
    mask = normalized["group"] == grp
    sub = normalized.loc[mask, gene_symbols]
    n_cells = mask.sum()

    mean_expr = sub.mean(axis=0).values.astype(np.float32)
    pct_expr = (sub > 0).mean(axis=0).values.astype(np.float32)
    std_expr = sub.std(axis=0).values.astype(np.float32)
    cv_expr = std_expr / (mean_expr + 1e-8)
    q25 = sub.quantile(0.25, axis=0).values.astype(np.float32)
    q75 = sub.quantile(0.75, axis=0).values.astype(np.float32)

    group_stats[grp] = {
        "mean_expression": mean_expr,
        "pct_expressing": pct_expr,
        "std_expression": std_expr,
        "cv_expression": cv_expr,
        "q25": q25,
        "q75": q75,
        "n_cells": n_cells,
    }
    print(f"    {grp:20s} : {n_cells} cellules, "
          f"mean={mean_expr.mean():.3f}, pct={pct_expr.mean():.3f}, "
          f"std={std_expr.mean():.3f}, cv={cv_expr.mean():.3f}")

# Libérer la mémoire
del normalized

# =============================================================================
# 3. CHARGEMENT DES DONNÉES pySCENIC
# =============================================================================
print("\n" + "=" * 70)
print("3. Chargement des données pySCENIC")
print("=" * 70)

# ── 3a. Regulon edges : TF → gene (filtrés par motif) ──────────────────────
regulon_edges = pd.read_csv(os.path.join(SCENIC_DIR, "regulon_edges_TF_to_gene.csv"))
# Nettoyer les noms de TF : "SOX18(+)" → "SOX18"
regulon_edges["TF_clean"] = regulon_edges["TF"].str.replace(r"\(\+\)$", "", regex=True)
print(f"  Regulon edges : {len(regulon_edges)} interactions TF→cible")
print(f"    TFs uniques : {regulon_edges['TF_clean'].nunique()}")

# ── 3b. TF activity par cluster ─────────────────────────────────────────────
tf_activity = pd.read_csv(os.path.join(SCENIC_DIR, "mean_TF_activity_per_cluster.csv"),
                           index_col=0)
# Les colonnes sont "SOX18(+)", "SOX4(+)", etc.
tf_activity.columns = [c.replace("(+)", "") for c in tf_activity.columns]
# L'index est le numéro de cluster (0, 1, 2, 3)
print(f"  TF activity : {tf_activity.shape[0]} clusters × {tf_activity.shape[1]} TFs")

# ── 3c. Adjacencies GRNBoost2 (coexpression brute, à filtrer) ───────────────
print("  Chargement adjacencies GRNBoost2...")
adjacencies = pd.read_csv(os.path.join(SCENIC_DIR, "adjacencies.csv"))
importance_thresh = adjacencies["importance"].quantile(COEXPR_TOP_QUANTILE)
adjacencies_filtered = adjacencies[adjacencies["importance"] >= importance_thresh].copy()
print(f"  Adjacencies : {len(adjacencies)} totales → {len(adjacencies_filtered)} "
      f"retenues (top {100*(1-COEXPR_TOP_QUANTILE):.0f}%, seuil={importance_thresh:.2f})")

# =============================================================================
# 4. FEATURES DES NOEUDS
# =============================================================================
print("\n" + "=" * 70)
print("4. Construction des features de noeuds")
print("=" * 70)

# ── 4a. Gene features (propriétés intrinsèques uniquement) ──────────────────
# L'expression est sur les arêtes, ici on ne garde que ce qui est propre au gène.

# is_tf : le gène est-il un facteur de transcription (dans les regulons SCENIC) ?
scenic_tfs = set(regulon_edges["TF_clean"].unique())
is_tf = np.array([1.0 if g in scenic_tfs else 0.0 for g in gene_symbols],
                 dtype=np.float32)
print(f"  is_tf : {int(is_tf.sum())} TFs parmi les gènes sélectionnés")

# variance_across_groups : mesure de variabilité de l'expression entre groupes
mean_expr_per_group = np.array([
    group_stats[grp]["mean_expression"] for grp in CELL_GROUPS
])  # shape: (5, n_genes)
variance_across = mean_expr_per_group.var(axis=0).astype(np.float32)
variance_norm = variance_across / (variance_across.max() + 1e-8)

# log2FC P4 vs P16 global
degs_p4_p16_full = pd.read_csv(os.path.join(GNN_DATA_DIR, "DEGs_P4_vs_P16.csv"))
lfc_map = dict(zip(degs_p4_p16_full["gene"], degs_p4_p16_full["avg_log2FC"]))
log2fc = np.array([lfc_map.get(g, 0.0) for g in gene_symbols], dtype=np.float32)
log2fc_norm = log2fc / (np.abs(log2fc).max() + 1e-8)

# -log10(padj) P4 vs P16 : mesure de significativité (plus c'est élevé, plus c'est sûr)
padj_map = dict(zip(degs_p4_p16_full["gene"], degs_p4_p16_full["p_val_adj"]))
neg_log_padj = np.array([
    -np.log10(max(padj_map.get(g, 1.0), 1e-300)) for g in gene_symbols
], dtype=np.float32)
neg_log_padj_norm = neg_log_padj / (neg_log_padj.max() + 1e-8)

# delta_pct : différence de fraction de cellules exprimant le gène (P16 - P4)
pct1_map = dict(zip(degs_p4_p16_full["gene"], degs_p4_p16_full["pct.1"]))
pct2_map = dict(zip(degs_p4_p16_full["gene"], degs_p4_p16_full["pct.2"]))
delta_pct = np.array([
    pct1_map.get(g, 0.0) - pct2_map.get(g, 0.0) for g in gene_symbols
], dtype=np.float32)

# log2FC par cluster (4 features supplémentaires — spécificité cluster)
cluster_lfc = {}
for c in [0, 1, 2, 3]:
    path = os.path.join(GNN_DATA_DIR, f"DEGs_P16_cluster_{c}.csv")
    df = pd.read_csv(path)
    lfc_c = dict(zip(df["gene"], df["avg_log2FC"]))
    cluster_lfc[c] = np.array([lfc_c.get(g, 0.0) for g in gene_symbols], dtype=np.float32)

# Normaliser chaque log2FC cluster
for c in range(4):
    max_val = np.abs(cluster_lfc[c]).max() + 1e-8
    cluster_lfc[c] = cluster_lfc[c] / max_val

# NOTE : le degré PPI sera ajouté après la construction des arêtes PPI (section 5b)
# On stocke les features pré-PPI et on complétera ensuite.
# AJOUT : bootstrap_stability et consensus_score comme features de noeuds.
# Ces features permettent au GNN d'apprendre que certains gènes ont des labels
# plus fiables que d'autres, ce qui est crucial quand les labels sont bruités
# (inévitable avec n=1 par condition).
gene_features_pre_ppi = np.column_stack([
    is_tf,                    # [0] TF oui/non
    variance_norm,            # [1] variabilité inter-groupes
    log2fc_norm,              # [2] log2FC global P4→P16
    neg_log_padj_norm,        # [3] significativité (-log10 padj)
    delta_pct,                # [4] Δ% cellules exprimant
    cluster_lfc[0],           # [5] log2FC cluster 0
    cluster_lfc[1],           # [6] log2FC cluster 1
    cluster_lfc[2],           # [7] log2FC cluster 2
    cluster_lfc[3],           # [8] log2FC cluster 3
    bootstrap_stability,      # [9] stabilité bootstrap (robustesse au sous-échantillonnage)
    consensus_score_p4p16,    # [10] consensus Wilcoxon+MAST (0, 0.5, ou 1)
])
print(f"  Gene features (pré-PPI) : {gene_features_pre_ppi.shape[1]} features")
print(f"    [is_tf, variance, log2FC_global, -log10padj, delta_pct, "
      f"lfc_c0..c3, bootstrap_stability, consensus_score]")
print(f"    (degré PPI ajouté après section 5b)")

# ── 4b. Cell group features ─────────────────────────────────────────────────
# Propriétés intrinsèques de chaque groupe cellulaire
cell_group_features_list = []
for grp in CELL_GROUPS:
    is_senescent = 0.0 if grp == "P4" else 1.0
    n_cells_norm = group_stats[grp]["n_cells"] / metadata.shape[0]
    # Cluster index normalisé (P4=0, cluster 0..3 = 0.25..1.0)
    if grp == "P4":
        cluster_idx = 0.0
    else:
        c = int(grp.split("_")[-1])
        cluster_idx = (c + 1) / 4.0
    cell_group_features_list.append([is_senescent, n_cells_norm, cluster_idx])

cell_group_features = torch.tensor(cell_group_features_list, dtype=torch.float)
print(f"  Cell group features : {cell_group_features.shape}")
print(f"    [is_senescent, n_cells_frac, cluster_idx]")
for i, grp in enumerate(CELL_GROUPS):
    print(f"    {grp:20s} : {cell_group_features[i].tolist()}")

# =============================================================================
# 5. CONSTRUCTION DES ARÊTES
# =============================================================================
print("\n" + "=" * 70)
print("5. Construction des arêtes")
print("=" * 70)

# ── 5a. Arêtes cell_group → gene (expression) ──────────────────────────────
# Features d'arête enrichies avec les statistiques distributionnelles :
#   [mean_expr, pct_expressing, tf_activity, std_expr, cv_expr, q25, q75]
# JUSTIFICATION : mean et pct seuls résument très grossièrement la distribution
# d'expression d'un gène dans un groupe. Avec n=1 par condition, on ne peut pas
# estimer la variabilité inter-réplicats, mais la variabilité intra-groupe
# (std, CV, quantiles) est le meilleur proxy disponible. Un gène dont l'expression
# est très dispersée au sein d'un groupe est moins informatif qu'un gène stable.
expr_src, expr_dst = [], []
expr_attrs = []

for grp_idx, grp in enumerate(CELL_GROUPS):
    stats = group_stats[grp]
    mean_expr = stats["mean_expression"]
    pct_expr = stats["pct_expressing"]
    std_expr = stats["std_expression"]
    cv_expr = stats["cv_expression"]
    q25 = stats["q25"]
    q75 = stats["q75"]

    # TF activity pour ce cluster (seulement pour P16 clusters)
    if grp == "P4":
        cluster_id = None
    else:
        cluster_id = int(grp.split("_")[-1])

    for gene_idx in range(n_genes):
        gene_name = gene_symbols[gene_idx]
        me = float(mean_expr[gene_idx])
        pe = float(pct_expr[gene_idx])
        se = float(std_expr[gene_idx])
        ce = float(cv_expr[gene_idx])
        q2 = float(q25[gene_idx])
        q7 = float(q75[gene_idx])

        # TF activity : si le gène est un TF et qu'on a l'activité pour ce cluster
        tf_act = 0.0
        if cluster_id is not None and gene_name in tf_activity.columns:
            if cluster_id in tf_activity.index:
                tf_act = float(tf_activity.loc[cluster_id, gene_name])

        expr_src.append(grp_idx)
        expr_dst.append(gene_idx)
        expr_attrs.append([me, pe, tf_act, se, ce, q2, q7])

edge_index_expresses = torch.tensor([expr_src, expr_dst], dtype=torch.long)
edge_attr_expresses = torch.tensor(expr_attrs, dtype=torch.float)

# Normaliser chaque feature d'arête indépendamment (z-score)
for col in range(edge_attr_expresses.shape[1]):
    col_data = edge_attr_expresses[:, col]
    mu, std = col_data.mean(), col_data.std() + 1e-8
    edge_attr_expresses[:, col] = (col_data - mu) / std

EXPR_FEAT_NAMES = "mean_expr, pct, tf_activity, std_expr, cv_expr, q25, q75"
print(f"  expresses (cell_group→gene) : {edge_index_expresses.shape[1]} arêtes, "
      f"{edge_attr_expresses.shape[1]} features [{EXPR_FEAT_NAMES}] (z-scored)")

# ── 5b. Arêtes gene ↔ gene : PPI STRING ─────────────────────────────────────
PPI_FILE = os.path.join(PPI_DIR, "9606.protein.links.v12.0.txt.gz")
PPI_ALIAS_FILE = os.path.join(PPI_DIR, "9606.protein.aliases.v12.0.txt.gz")

download_if_absent(
    "https://stringdb-static.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
    PPI_FILE, "STRING links"
)
download_if_absent(
    "https://stringdb-static.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz",
    PPI_ALIAS_FILE, "STRING aliases"
)

print("  Chargement STRING...")
aliases = pd.read_csv(PPI_ALIAS_FILE, sep="\t", compression="gzip")
aliases_filt = aliases[
    (aliases["alias"].isin(gene_symbols)) &
    (aliases["source"].str.contains("Ensembl_HGNC", na=False))
]
sym2string = dict(zip(aliases_filt["alias"], aliases_filt["#string_protein_id"]))
string2sym = {v: k for k, v in sym2string.items()}
string_ids = set(sym2string.values())
print(f"    Gènes mappés STRING : {len(sym2string)} / {n_genes}")

ppi = pd.read_csv(PPI_FILE, sep=" ", compression="gzip")
ppi_filt = ppi[
    (ppi["protein1"].isin(string_ids)) &
    (ppi["protein2"].isin(string_ids)) &
    (ppi["combined_score"] >= PPI_SCORE_THRESH)
]

ppi_src, ppi_dst, ppi_w = [], [], []
for _, row in ppi_filt.iterrows():
    s1, s2 = string2sym.get(row["protein1"]), string2sym.get(row["protein2"])
    if s1 and s2 and s1 in gene_to_idx and s2 in gene_to_idx:
        i, j = gene_to_idx[s1], gene_to_idx[s2]
        ppi_src.extend([i, j])
        ppi_dst.extend([j, i])
        ppi_w.extend([row["combined_score"] / 1000.0] * 2)

edge_index_ppi = torch.tensor([ppi_src, ppi_dst], dtype=torch.long)
edge_attr_ppi = torch.tensor(ppi_w, dtype=torch.float).unsqueeze(1)
print(f"  ppi (gene↔gene) : {len(ppi_src) // 2} interactions ({len(ppi_src)} arêtes)")

# ── Finaliser gene_features avec le degré PPI ──────────────────────────────
ppi_degree = np.zeros(n_genes, dtype=np.float32)
for idx in ppi_src:
    ppi_degree[idx] += 1
ppi_degree_norm = ppi_degree / (ppi_degree.max() + 1e-8)

gene_features = torch.tensor(
    np.column_stack([gene_features_pre_ppi, ppi_degree_norm]),
    dtype=torch.float,
)
print(f"  Gene features finales : {gene_features.shape}")
print(f"    [is_tf, variance, log2FC_global, -log10padj, delta_pct, "
      f"lfc_c0..c3, bootstrap_stability, consensus_score, ppi_degree]")

# ── 5c. Arêtes gene ↔ gene : REACTOME (même voie) ──────────────────────────
MSIGDB_REACTOME = os.path.join(DB_DIR, "c2.cp.reactome.symbols.gmt")
download_if_absent(
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/c2.cp.reactome.v2024.1.Hs.symbols.gmt",
    MSIGDB_REACTOME, "MSigDB REACTOME pathways"
)

reactome_pathways = {}
with open(MSIGDB_REACTOME) as f:
    for line in f:
        parts = line.strip().split("\t")
        genes_in_pw = set(parts[2:]) & set(gene_symbols)
        if 2 <= len(genes_in_pw) <= REACTOME_MAX_PATHWAY:
            reactome_pathways[parts[0]] = genes_in_pw

print(f"    REACTOME : {len(reactome_pathways)} voies (2-{REACTOME_MAX_PATHWAY} gènes)")

react_src, react_dst = [], []
react_pairs = set()
for pw_genes in reactome_pathways.values():
    gene_list = sorted(pw_genes)
    for i_idx in range(len(gene_list)):
        for j_idx in range(i_idx + 1, len(gene_list)):
            g1, g2 = gene_list[i_idx], gene_list[j_idx]
            if g1 in gene_to_idx and g2 in gene_to_idx:
                pair = (min(gene_to_idx[g1], gene_to_idx[g2]),
                        max(gene_to_idx[g1], gene_to_idx[g2]))
                if pair not in react_pairs:
                    react_pairs.add(pair)
                    react_src.extend([pair[0], pair[1]])
                    react_dst.extend([pair[1], pair[0]])

edge_index_pathway = torch.tensor([react_src, react_dst], dtype=torch.long)
print(f"  same_pathway (gene↔gene) : {len(react_pairs)} paires ({len(react_src)} arêtes)")

# ── 5d. Arêtes gene → gene : TF regulon (pySCENIC) ─────────────────────────
reg_src, reg_dst, reg_w = [], [], []
reg_pairs = set()
for _, row in regulon_edges.iterrows():
    tf = row["TF_clean"]
    target = row["target_gene"]
    weight = row["weight"]
    if tf in gene_to_idx and target in gene_to_idx:
        tf_idx, target_idx = gene_to_idx[tf], gene_to_idx[target]
        pair = (tf_idx, target_idx)
        if pair not in reg_pairs:
            reg_pairs.add(pair)
            reg_src.append(tf_idx)
            reg_dst.append(target_idx)
            reg_w.append(float(weight))

edge_index_regulates = torch.tensor([reg_src, reg_dst], dtype=torch.long)
edge_attr_regulates = torch.tensor(reg_w, dtype=torch.float).unsqueeze(1)
# Normaliser les poids de régulation
if edge_attr_regulates.numel() > 0:
    edge_attr_regulates = edge_attr_regulates / (edge_attr_regulates.max() + 1e-8)

edge_index_regulated_by = torch.tensor([reg_dst, reg_src], dtype=torch.long)
print(f"  regulates (TF→gene, pySCENIC) : {len(reg_pairs)} liens")

# ── 5e. Arêtes gene ↔ gene : coexpression (GRNBoost2, filtrée) ─────────────
coexpr_src, coexpr_dst, coexpr_w = [], [], []
coexpr_pairs = set()
for _, row in adjacencies_filtered.iterrows():
    g1, g2 = str(row["TF"]), str(row["target"])
    if g1 in gene_to_idx and g2 in gene_to_idx:
        i, j = gene_to_idx[g1], gene_to_idx[g2]
        pair = (min(i, j), max(i, j))
        if pair not in coexpr_pairs:
            coexpr_pairs.add(pair)
            coexpr_src.extend([i, j])
            coexpr_dst.extend([j, i])
            imp = float(row["importance"])
            coexpr_w.extend([imp, imp])

edge_index_coexpr = torch.tensor(
    [coexpr_src, coexpr_dst] if coexpr_src else [[],[]],
    dtype=torch.long,
)
coexpr_w_tensor = torch.tensor(coexpr_w, dtype=torch.float).unsqueeze(1) if coexpr_w else torch.zeros((0,1))
# Normaliser
if coexpr_w_tensor.numel() > 0:
    coexpr_w_tensor = coexpr_w_tensor / (coexpr_w_tensor.max() + 1e-8)

print(f"  coexpression (gene↔gene, GRNBoost2) : {len(coexpr_pairs)} paires ({len(coexpr_src)} arêtes)")

# =============================================================================
# 5f. CALCUL DU TARGET IMPACT_SCORE (pour la tête de ranking)
# =============================================================================
# impact_score = n_contextes_DE × mean(|log2FC|) × centralité_graphe
# C'est un score continu qui reflète l'importance globale d'un gène dans
# la progression de la sénescence.
print("\n  Calcul du target impact_score...")

# Composante 1 : nombre de contextes DE (0-5), normalisé
n_contextes_de = labels.sum(axis=1)  # shape: (n_genes,)
n_contextes_norm = n_contextes_de / N_LABELS  # [0, 1]

# Composante 2 : mean(|log2FC|) à travers tous les contextes
# Utiliser le log2FC global + les log2FC par cluster
all_lfc = np.column_stack([
    np.abs(log2fc),
    np.abs(cluster_lfc[0] * (np.abs(cluster_lfc[0]).max() + 1e-8)),  # dé-normaliser
    np.abs(cluster_lfc[1] * (np.abs(cluster_lfc[1]).max() + 1e-8)),
    np.abs(cluster_lfc[2] * (np.abs(cluster_lfc[2]).max() + 1e-8)),
    np.abs(cluster_lfc[3] * (np.abs(cluster_lfc[3]).max() + 1e-8)),
])
mean_abs_lfc = all_lfc.mean(axis=1).astype(np.float32)
mean_abs_lfc_norm = mean_abs_lfc / (mean_abs_lfc.max() + 1e-8)

# Composante 3 : centralité dans le graphe (degré total normalisé)
# Combiner PPI + pathway + régulation + coexpression
total_degree = np.zeros(n_genes, dtype=np.float32)
for idx in ppi_src:
    total_degree[idx] += 1
for idx in react_src:
    total_degree[idx] += 1
for idx in reg_src + reg_dst:
    total_degree[idx] += 1
for idx in coexpr_src:
    total_degree[idx] += 1
centrality_norm = total_degree / (total_degree.max() + 1e-8)

# Score composite
impact_scores = (
    0.4 * n_contextes_norm
    + 0.35 * mean_abs_lfc_norm
    + 0.25 * centrality_norm
).astype(np.float32)

# Normaliser entre 0 et 1
impact_scores = impact_scores / (impact_scores.max() + 1e-8)

print(f"  Impact score : min={impact_scores.min():.3f}, max={impact_scores.max():.3f}, "
      f"mean={impact_scores.mean():.3f}")
top_impact = np.argsort(impact_scores)[::-1][:10]
print(f"  Top 10 gènes par impact score :")
for idx in top_impact:
    print(f"    {gene_symbols[idx]:15s}  score={impact_scores[idx]:.3f}  "
          f"(n_DE={int(n_contextes_de[idx])}, |lfc|={mean_abs_lfc[idx]:.2f}, "
          f"centrality={centrality_norm[idx]:.3f})")

# =============================================================================
# 6. ASSEMBLAGE DU GRAPHE HÉTÉROGÈNE
# =============================================================================
print("\n" + "=" * 70)
print("6. Assemblage du graphe hétérogène")
print("=" * 70)

data = HeteroData()

# ── Noeuds ───────────────────────────────────────────────────────────────────
data["gene"].x = gene_features
data["gene"].num_nodes = n_genes
data["gene"].y = torch.tensor(labels, dtype=torch.float)  # multi-label (n_genes, 5)
data["gene"].impact_score = torch.tensor(impact_scores, dtype=torch.float)  # target ranking
# Confiance par label et globale — utilisées pour pondérer la loss (section 8)
data["gene"].confidence_per_label = torch.tensor(confidence_per_label, dtype=torch.float)  # (n_genes, 5)
data["gene"].confidence_global = torch.tensor(confidence_global, dtype=torch.float)  # (n_genes,)

data["cell_group"].x = cell_group_features
data["cell_group"].num_nodes = len(CELL_GROUPS)

# ── Arêtes ───────────────────────────────────────────────────────────────────
data["cell_group", "expresses", "gene"].edge_index = edge_index_expresses
data["cell_group", "expresses", "gene"].edge_attr = edge_attr_expresses

data["gene", "expressed_in", "cell_group"].edge_index = torch.stack([
    edge_index_expresses[1], edge_index_expresses[0]
])
data["gene", "expressed_in", "cell_group"].edge_attr = edge_attr_expresses

data["gene", "ppi", "gene"].edge_index = edge_index_ppi
data["gene", "ppi", "gene"].edge_attr = edge_attr_ppi

data["gene", "same_pathway", "gene"].edge_index = edge_index_pathway

data["gene", "regulates", "gene"].edge_index = edge_index_regulates
data["gene", "regulates", "gene"].edge_attr = edge_attr_regulates

data["gene", "regulated_by", "gene"].edge_index = edge_index_regulated_by
data["gene", "regulated_by", "gene"].edge_attr = edge_attr_regulates

if edge_index_coexpr.numel() > 0:
    data["gene", "coexpression", "gene"].edge_index = edge_index_coexpr
    data["gene", "coexpression", "gene"].edge_attr = coexpr_w_tensor

# ── Masques train / test ─────────────────────────────────────────────────────
np.random.seed(42)
perm = np.random.permutation(n_genes)
n_test = max(1, int(n_genes * MASK_RATIO))
test_idx = perm[:n_test]
train_idx = perm[n_test:]

train_mask = torch.zeros(n_genes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask = torch.zeros(n_genes, dtype=torch.bool)
test_mask[test_idx] = True

data["gene"].train_mask = train_mask
data["gene"].test_mask = test_mask

print(f"\n  Graphe final :")
print(f"    Noeuds gene       : {n_genes} (features={gene_features.shape[1]})")
print(f"    Noeuds cell_group : {len(CELL_GROUPS)} (features={cell_group_features.shape[1]})")
print(f"    Arêtes expresses       : {edge_index_expresses.shape[1]}")
print(f"    Arêtes expressed_in    : {edge_index_expresses.shape[1]}")
print(f"    Arêtes ppi             : {edge_index_ppi.shape[1]}")
print(f"    Arêtes same_pathway    : {edge_index_pathway.shape[1]}")
print(f"    Arêtes regulates       : {edge_index_regulates.shape[1]}")
print(f"    Arêtes regulated_by    : {edge_index_regulated_by.shape[1]}")
if edge_index_coexpr.numel() > 0:
    print(f"    Arêtes coexpression    : {edge_index_coexpr.shape[1]}")
print(f"    Train / Test           : {train_mask.sum().item()} / {test_mask.sum().item()} gènes")

torch.save(data, os.path.join(OUT_DIR, "hetero_graph.pt"))
print(f"  → Sauvé : hetero_graph.pt")

# =============================================================================
# 7. MODÈLE GNN HÉTÉROGÈNE (CLASSIFICATION MULTI-LABEL + RANKING)
# =============================================================================
print("\n" + "=" * 70)
print("7. Définition du modèle")
print("=" * 70)


class HeteroGNN(nn.Module):
    """
    GNN hétérogène multi-tâche avec attention (GATConv).

    Deux têtes de sortie :
      1. Classifier : 5 logits multi-label (DE dans chaque contexte)
      2. Scorer     : 1 score continu d'importance [0, 1]

    GATConv fournit des poids d'attention par arête, permettant de mesurer
    l'importance structurelle de chaque gène (combien d'attention il reçoit).
    """

    def __init__(self, gene_in, cell_in, hidden, n_layers, n_labels,
                 n_heads=4, dropout=0.2):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads

        # hidden doit être divisible par n_heads pour GATConv
        assert hidden % n_heads == 0, f"hidden ({hidden}) doit être divisible par n_heads ({n_heads})"
        self.head_dim = hidden // n_heads

        self.gene_proj = nn.Linear(gene_in, hidden)
        self.cell_proj = nn.Linear(cell_in, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        edge_types = [
            ("gene", "ppi", "gene"),
            ("gene", "same_pathway", "gene"),
            ("gene", "regulates", "gene"),
            ("gene", "regulated_by", "gene"),
            ("cell_group", "expresses", "gene"),
            ("gene", "expressed_in", "cell_group"),
            ("gene", "coexpression", "gene"),
        ]

        for _ in range(n_layers):
            conv_dict = {}
            for et in edge_types:
                conv_dict[et] = GATConv(
                    hidden, self.head_dim, heads=n_heads,
                    concat=True,  # output = head_dim × n_heads = hidden
                    dropout=dropout, add_self_loops=False,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(nn.ModuleDict({
                "gene": nn.BatchNorm1d(hidden),
                "cell_group": nn.BatchNorm1d(hidden),
            }))

        self.dropout = nn.Dropout(dropout)

        # Tête 1 : Classification multi-label
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_labels),
        )

        # Tête 2 : Score d'importance (ranking)
        self.scorer = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            "gene": F.relu(self.gene_proj(x_dict["gene"])),
            "cell_group": F.relu(self.cell_proj(x_dict["cell_group"])),
        }

        for i in range(self.n_layers):
            x_prev = {k: v.clone() for k, v in x_dict.items()}

            active_edges = {
                k: v for k, v in edge_index_dict.items()
                if v.numel() > 0
            }
            x_dict = self.convs[i](x_dict, active_edges)

            for key in x_dict:
                x_dict[key] = self.norms[i][key](x_dict[key])
                x_dict[key] = F.relu(x_dict[key])
                x_dict[key] = self.dropout(x_dict[key])
                x_dict[key] = x_dict[key] + x_prev[key]

        gene_emb = x_dict["gene"]

        # Tête 1 : logits multi-label
        logits = self.classifier(gene_emb)             # (n_genes, n_labels)
        # Tête 2 : score d'importance
        score = torch.sigmoid(self.scorer(gene_emb))   # (n_genes, 1) ∈ [0, 1]

        return logits, score.squeeze(-1), x_dict


model = HeteroGNN(
    gene_in=gene_features.shape[1],
    cell_in=cell_group_features.shape[1],
    hidden=HIDDEN_DIM,
    n_layers=N_LAYERS,
    n_labels=N_LABELS,
    n_heads=N_HEADS,
    dropout=DROPOUT,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Modèle : {N_LAYERS} couches GATConv, hidden={HIDDEN_DIM}, "
      f"heads={N_HEADS}, labels={N_LABELS}")
print(f"  Paramètres : {total_params:,}")
print(f"\n{model}")

# =============================================================================
# 8. ENTRAÎNEMENT
# =============================================================================
print("\n" + "=" * 70)
print("8. Entraînement")
print("=" * 70)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=30, min_lr=1e-5
)

# Pondération des classes pour gérer le déséquilibre pos/neg par label
pos_counts = labels[train_idx].sum(axis=0)
neg_counts = len(train_idx) - pos_counts
pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-8), dtype=torch.float)
print(f"  Pos weights (déséquilibre) : {pos_weight.tolist()}")

# PONDÉRATION PAR CONFIANCE (confidence-weighted loss)
# JUSTIFICATION : avec n=1 par condition, les labels DEG contiennent des faux
# positifs. Plutôt que de traiter tous les labels comme équivalents, on pondère
# la contribution de chaque gène à la loss par son score de confiance :
#   - confiance élevée (consensus Wilcox+MAST, stable au bootstrap) → poids fort
#   - confiance faible (un seul test, instable) → poids réduit
# On utilise reduction='none' pour appliquer la pondération par gène, puis on
# moyenne manuellement. Le plancher à 0.3 évite d'ignorer complètement les
# gènes à faible confiance (ils contribuent encore, mais moins).
CONFIDENCE_FLOOR = 0.3  # Poids minimum pour tout gène (évite de "muter" les labels)
criterion_cls_raw = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
criterion_score = nn.MSELoss()

# Poids de confiance par gène (plancher + normalisation)
confidence_weights = data["gene"].confidence_global.clone()
confidence_weights = torch.clamp(confidence_weights, min=CONFIDENCE_FLOOR)
# Normaliser pour que la moyenne = 1 (pas de changement d'échelle global de la loss)
confidence_weights = confidence_weights / confidence_weights.mean()
print(f"  Confidence weights : min={confidence_weights.min():.3f}, "
      f"max={confidence_weights.max():.3f}, mean={confidence_weights.mean():.3f}")

x_dict = {"gene": data["gene"].x, "cell_group": data["cell_group"].x}

edge_index_dict = {}
for edge_type_key in [
    ("cell_group", "expresses", "gene"),
    ("gene", "expressed_in", "cell_group"),
    ("gene", "ppi", "gene"),
    ("gene", "same_pathway", "gene"),
    ("gene", "regulates", "gene"),
    ("gene", "regulated_by", "gene"),
]:
    edge_index_dict[edge_type_key] = data[edge_type_key].edge_index

if ("gene", "coexpression", "gene") in data.edge_types:
    edge_index_dict[("gene", "coexpression", "gene")] = data["gene", "coexpression", "gene"].edge_index

targets = data["gene"].y                    # (n_genes, 5)
target_scores = data["gene"].impact_score   # (n_genes,)

train_losses, test_losses = [], []
train_f1s, test_f1s = [], []
train_score_losses, test_score_losses = [], []
best_test_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

model.train()
for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    logits, pred_scores, _ = model(x_dict, edge_index_dict)

    # Loss multi-tâche : classification pondérée par confiance + ranking
    # La loss BCE brute a shape (n_train, 5) — on pondère par gène avant de moyenner
    raw_cls_loss = criterion_cls_raw(logits[train_mask], targets[train_mask])  # (n_train, 5)
    train_conf_w = confidence_weights[train_mask].unsqueeze(1)  # (n_train, 1)
    loss_cls = (raw_cls_loss * train_conf_w).mean()
    loss_score = criterion_score(pred_scores[train_mask], target_scores[train_mask])
    train_loss = loss_cls + LAMBDA_SCORE * loss_score

    train_loss.backward()
    optimizer.step()

    # Évaluation (même pondération par confiance pour la loss test)
    model.eval()
    with torch.no_grad():
        logits_eval, scores_eval, _ = model(x_dict, edge_index_dict)
        raw_test_cls = criterion_cls_raw(logits_eval[test_mask], targets[test_mask])
        test_conf_w = confidence_weights[test_mask].unsqueeze(1)
        test_loss_cls = (raw_test_cls * test_conf_w).mean().item()
        test_loss_score = criterion_score(scores_eval[test_mask], target_scores[test_mask]).item()
        test_loss = test_loss_cls + LAMBDA_SCORE * test_loss_score

        # F1 score
        pred_train = (torch.sigmoid(logits_eval[train_mask]) > 0.5).numpy()
        pred_test = (torch.sigmoid(logits_eval[test_mask]) > 0.5).numpy()
        true_train = targets[train_mask].numpy()
        true_test = targets[test_mask].numpy()

        f1_train = f1_score(true_train, pred_train, average="macro", zero_division=0)
        f1_test = f1_score(true_test, pred_test, average="macro", zero_division=0)
    model.train()

    train_losses.append(train_loss.item())
    test_losses.append(test_loss)
    train_f1s.append(f1_train)
    test_f1s.append(f1_test)
    train_score_losses.append(loss_score.item())
    test_score_losses.append(test_loss_score)

    # LR scheduler
    scheduler.step(test_loss)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_epoch = epoch
        epochs_without_improvement = 0
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pt"))
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 50 == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"    Epoch {epoch+1:3d}/{N_EPOCHS} — "
              f"Loss: {train_loss.item():.4f} (cls={loss_cls.item():.4f} "
              f"score={loss_score.item():.4f}) — "
              f"Test: {test_loss:.4f} — F1: {f1_test:.3f} — LR: {current_lr:.1e}")

    # Early stopping
    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"\n  Early stopping à l'epoch {epoch+1} "
              f"(pas d'amélioration depuis {EARLY_STOPPING_PATIENCE} epochs)")
        break

print(f"\n  Meilleur modèle : epoch {best_epoch+1}, test loss = {best_test_loss:.5f}")

# Charger le meilleur modèle
model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pt"), weights_only=True))

# =============================================================================
# 9. ÉVALUATION ET VISUALISATION
# =============================================================================
print("\n" + "=" * 70)
print("9. Évaluation et visualisation")
print("=" * 70)

model.eval()
with torch.no_grad():
    logits_final, pred_impact, embeddings = model(x_dict, edge_index_dict)
    probs = torch.sigmoid(logits_final).numpy()
    preds = (probs > 0.5).astype(int)
    pred_impact_np = pred_impact.numpy()
    gene_emb = embeddings["gene"].numpy()

# ── 9a. Métriques détaillées par label ──────────────────────────────────────
print("\n  Métriques par label (test set) :")
print(f"  {'Label':20s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'AUROC':>7s} {'Support':>8s}")
print("  " + "-" * 60)

test_true = labels[test_mask.numpy()]
test_pred = preds[test_mask.numpy()]
test_prob = probs[test_mask.numpy()]

for j, name in enumerate(LABEL_NAMES):
    support = int(test_true[:, j].sum())
    if support > 0:
        prec = precision_score(test_true[:, j], test_pred[:, j], zero_division=0)
        rec = recall_score(test_true[:, j], test_pred[:, j], zero_division=0)
        f1 = f1_score(test_true[:, j], test_pred[:, j], zero_division=0)
        try:
            auc = roc_auc_score(test_true[:, j], test_prob[:, j])
        except ValueError:
            auc = float("nan")
    else:
        prec = rec = f1 = auc = float("nan")
    print(f"  {name:20s} {prec:7.3f} {rec:7.3f} {f1:7.3f} {auc:7.3f} {support:8d}")

f1_macro = f1_score(test_true, test_pred, average="macro", zero_division=0)
f1_micro = f1_score(test_true, test_pred, average="micro", zero_division=0)
print(f"\n  F1 macro : {f1_macro:.4f}")
print(f"  F1 micro : {f1_micro:.4f}")

# Métriques du score d'importance (ranking)
from scipy.stats import spearmanr, kendalltau
test_impact_true = impact_scores[test_mask.numpy()]
test_impact_pred = pred_impact_np[test_mask.numpy()]
score_mse = np.mean((test_impact_true - test_impact_pred) ** 2)
score_corr = np.corrcoef(test_impact_true, test_impact_pred)[0, 1]
spearman_r, _ = spearmanr(test_impact_true, test_impact_pred)
kendall_t, _ = kendalltau(test_impact_true, test_impact_pred)

print(f"\n  Score d'importance (test set) :")
print(f"    MSE         : {score_mse:.5f}")
print(f"    Pearson r   : {score_corr:.4f}")
print(f"    Spearman ρ  : {spearman_r:.4f}")
print(f"    Kendall τ   : {kendall_t:.4f}")

# ── 9b. Courbes de loss, F1 et score ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

axes[0].plot(train_losses, label="Train", color="#3498DB", alpha=0.8)
axes[0].plot(test_losses, label="Test", color="#E74C3C", alpha=0.8)
axes[0].axvline(best_epoch, ls="--", color="grey", lw=0.8, label=f"Best (epoch {best_epoch+1})")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss totale")
axes[0].set_title("Loss multi-tâche (BCE + λ·MSE)")
axes[0].legend()
axes[0].set_yscale("log")
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_f1s, label="Train F1", color="#3498DB", alpha=0.8)
axes[1].plot(test_f1s, label="Test F1", color="#E74C3C", alpha=0.8)
axes[1].axvline(best_epoch, ls="--", color="grey", lw=0.8)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("F1 macro")
axes[1].set_title("F1 — Classification multi-label")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(train_score_losses, label="Train MSE", color="#3498DB", alpha=0.8)
axes[2].plot(test_score_losses, label="Test MSE", color="#E74C3C", alpha=0.8)
axes[2].axvline(best_epoch, ls="--", color="grey", lw=0.8)
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("MSE")
axes[2].set_title("Loss — Score d'importance (ranking)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "training_loss_f1.png"))
plt.close(fig)
print("  → training_loss_f1.png")

# ── 9c. Heatmap des probabilités pour les top gènes ────────────────────────
# Top gènes = ceux avec la plus forte probabilité maximale sur les 5 labels
top_n = 40
max_prob = probs.max(axis=1)
top_genes_idx = np.argsort(max_prob)[::-1][:top_n]

heatmap_data = pd.DataFrame(
    probs[top_genes_idx],
    index=[gene_symbols[i] for i in top_genes_idx],
    columns=LABEL_NAMES,
)

fig, ax = plt.subplots(figsize=(10, 12))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd",
            ax=ax, linewidths=0.5, vmin=0, vmax=1)
ax.set_title(f"Top {top_n} gènes — Probabilité par label de sénescence", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "probability_heatmap.png"))
plt.close(fig)
print("  → probability_heatmap.png")

# ── 9d. Prédiction vs Réel — échantillon de gènes du test set ──────────────
n_show = 30  # nombre de gènes à afficher
test_indices = np.where(test_mask.numpy())[0]
# Mélanger et prendre un échantillon représentatif (mix positifs / négatifs)
np.random.seed(0)
show_idx = np.random.choice(test_indices, size=min(n_show, len(test_indices)), replace=False)
# Trier par nombre de labels vrais décroissant pour une lecture plus claire
show_idx = show_idx[np.argsort(-labels[show_idx].sum(axis=1))]

show_genes = gene_symbols[show_idx]
show_true = labels[show_idx]
show_pred = preds[show_idx]
show_prob = probs[show_idx]

fig, axes = plt.subplots(1, 3, figsize=(22, max(6, n_show * 0.35)))

# Panel 1 : Labels réels (0/1)
sns.heatmap(pd.DataFrame(show_true, index=show_genes, columns=LABEL_NAMES),
            annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=1,
            linewidths=0.5, cbar=False, ax=axes[0])
axes[0].set_title("Réel (0/1)", fontsize=12, fontweight="bold")

# Panel 2 : Probabilités prédites (continues)
sns.heatmap(pd.DataFrame(show_prob, index=show_genes, columns=LABEL_NAMES),
            annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
            linewidths=0.5, cbar=True, ax=axes[1])
axes[1].set_title("Probabilité prédite", fontsize=12, fontweight="bold")
axes[1].set_yticklabels([])

# Panel 3 : Concordance (vert = correct, rouge = erreur)
concordance = (show_true == show_pred).astype(float)
cmap_conc = sns.color_palette(["#E74C3C", "#2ECC71"], as_cmap=True)
sns.heatmap(pd.DataFrame(concordance, index=show_genes, columns=LABEL_NAMES),
            annot=False, cmap=cmap_conc, vmin=0, vmax=1,
            linewidths=0.5, cbar=False, ax=axes[2])
# Annoter avec ✓ / ✗
for i in range(concordance.shape[0]):
    for j in range(concordance.shape[1]):
        symbol = "✓" if concordance[i, j] == 1 else "✗"
        color = "white" if concordance[i, j] == 0 else "black"
        axes[2].text(j + 0.5, i + 0.5, symbol, ha="center", va="center",
                     fontsize=9, color=color, fontweight="bold")
axes[2].set_title("Concordance", fontsize=12, fontweight="bold")
axes[2].set_yticklabels([])

# Accuracy globale sur les gènes affichés
acc = concordance.mean()
fig.suptitle(f"Prédiction vs Réel — {len(show_idx)} gènes du test set "
             f"(accuracy cellule = {acc:.1%})", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "test_prediction_vs_real.png"))
plt.close(fig)
print("  → test_prediction_vs_real.png")

# ── 9e. Ranking — Score d'importance prédit vs réel ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1 : scatter plot true vs predicted impact score
for mask_name, mask_arr, color, marker in [
    ("Train", train_mask.numpy(), "#3498DB", "o"),
    ("Test", test_mask.numpy(), "#E74C3C", "^"),
]:
    axes[0].scatter(impact_scores[mask_arr], pred_impact_np[mask_arr],
                    c=color, s=15, alpha=0.5, label=mask_name, marker=marker)
lims = [0, max(impact_scores.max(), pred_impact_np.max()) + 0.05]
axes[0].plot(lims, lims, "k--", lw=0.8, alpha=0.5)
axes[0].set_xlabel("Impact score réel")
axes[0].set_ylabel("Impact score prédit")
axes[0].set_title(f"Score d'importance — Prédit vs Réel\n"
                   f"(Spearman ρ={spearman_r:.3f}, Pearson r={score_corr:.3f})")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Panel 2 : Top 30 gènes par score d'importance prédit (barplot)
n_top_rank = 30
ranking = np.argsort(pred_impact_np)[::-1]
top_rank_idx = ranking[:n_top_rank]
top_rank_genes = gene_symbols[top_rank_idx]
top_rank_pred = pred_impact_np[top_rank_idx]
top_rank_true = impact_scores[top_rank_idx]

y_pos = np.arange(n_top_rank)
bar_width = 0.35
axes[1].barh(y_pos - bar_width/2, top_rank_pred, bar_width,
             label="Prédit", color="#E74C3C", alpha=0.8)
axes[1].barh(y_pos + bar_width/2, top_rank_true, bar_width,
             label="Réel", color="#3498DB", alpha=0.8)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(top_rank_genes, fontsize=8)
axes[1].invert_yaxis()
axes[1].set_xlabel("Score d'importance")
axes[1].set_title(f"Top {n_top_rank} gènes — Classement par importance")
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3, axis="x")

fig.suptitle("Ranking des gènes par importance dans la sénescence", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "gene_ranking.png"))
plt.close(fig)
print("  → gene_ranking.png")

# ── 9f. PCA des embeddings ──────────────────────────────────────────────────
pca = PCA_sk(n_components=2)
gene_pca = pca.fit_transform(gene_emb)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Coloré par label P4→P16
sc0 = axes[0].scatter(gene_pca[:, 0], gene_pca[:, 1], c=probs[:, 0],
                       cmap="YlOrRd", s=10, alpha=0.7, rasterized=True)
plt.colorbar(sc0, ax=axes[0], label="P(DE P4→P16)")
axes[0].set_title("Embeddings — P(DE P4→P16)")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

# Coloré par le cluster le plus probable (parmi les 4 clusters)
cluster_probs = probs[:, 1:5]
dominant_cluster = cluster_probs.argmax(axis=1)
max_cluster_prob = cluster_probs.max(axis=1)
# Ne colorier que si la prob est > 0.3
dominant_cluster_plot = np.where(max_cluster_prob > 0.3, dominant_cluster, -1)
colors = np.array(["#95A5A6"] * n_genes)  # gris par défaut
cluster_colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12"]
for c in range(4):
    mask_c = dominant_cluster_plot == c
    colors[mask_c] = cluster_colors[c]

axes[1].scatter(gene_pca[:, 0], gene_pca[:, 1], c=colors,
                s=10, alpha=0.7, rasterized=True)
for c in range(4):
    axes[1].scatter([], [], c=cluster_colors[c], label=f"Cluster {c}")
axes[1].scatter([], [], c="#95A5A6", label="Non spécifique")
axes[1].legend(fontsize=8)
axes[1].set_title("Embeddings — Cluster dominant (prob>0.3)")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

# Train vs Test
colors_mask = np.where(train_mask.numpy(), "#3498DB", "#E74C3C")
axes[2].scatter(gene_pca[:, 0], gene_pca[:, 1], c=colors_mask,
                s=10, alpha=0.7, rasterized=True)
axes[2].set_title("Embeddings — Train (bleu) vs Test (rouge)")
axes[2].set_xlabel("PC1")
axes[2].set_ylabel("PC2")

fig.suptitle("Espace latent du GNN", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "embeddings_pca.png"))
plt.close(fig)
print("  → embeddings_pca.png")

# ── 9e. Distribution des probabilités par label ────────────────────────────
fig, axes = plt.subplots(1, N_LABELS, figsize=(4 * N_LABELS, 4), sharey=True)
for j, name in enumerate(LABEL_NAMES):
    axes[j].hist(probs[:, j], bins=40, alpha=0.7, color="#3498DB", edgecolor="white")
    axes[j].axvline(0.5, color="red", ls="--", lw=1)
    axes[j].set_xlabel("Probabilité")
    axes[j].set_title(name, fontsize=10)
axes[0].set_ylabel("Nombre de gènes")
fig.suptitle("Distribution des probabilités prédites", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "probability_distributions.png"))
plt.close(fig)
print("  → probability_distributions.png")

# =============================================================================
# 10. VALIDATION PAR LES BASES DE DONNÉES EXTERNES
# =============================================================================
print("\n" + "=" * 70)
print("10. Validation par les bases de données externes")
print("=" * 70)
print("  (GenAge, CellAge, MSigDB, AgeAnno — utilisées comme vérification)")

# ── Téléchargement des bases ────────────────────────────────────────────────
# GenAge
GENAGE_ZIP = os.path.join(DB_DIR, "genage_human.zip")
GENAGE_FILE = os.path.join(DB_DIR, "genage_human.csv")
download_if_absent(
    "https://genomics.senescence.info/genes/human_genes.zip",
    GENAGE_ZIP, "GenAge"
)
if not os.path.exists(GENAGE_FILE):
    with zipfile.ZipFile(GENAGE_ZIP, "r") as z:
        csv_names = [n for n in z.namelist() if n.endswith(".csv")]
        if csv_names:
            with z.open(csv_names[0]) as f:
                content = f.read()
            with open(GENAGE_FILE, "wb") as out:
                out.write(content)

genage = pd.read_csv(GENAGE_FILE)
genage_symbols = set(genage["symbol"].dropna()) if "symbol" in genage.columns else set()

# MSigDB Hallmarks
MSIGDB_HALLMARK = os.path.join(DB_DIR, "h.all.symbols.gmt")
download_if_absent(
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt",
    MSIGDB_HALLMARK, "MSigDB Hallmarks"
)
msigdb_sets = {}
with open(MSIGDB_HALLMARK) as f:
    for line in f:
        parts = line.strip().split("\t")
        msigdb_sets[parts[0]] = set(parts[2:])

AGING_KEYWORDS = ["SENESCENCE", "P53", "APOPTOSIS", "INFLAMMATORY", "TNFA",
                  "IL6", "KRAS", "MTORC", "REACTIVE_OXYGEN", "DNA_REPAIR",
                  "OXIDATIVE", "AGING"]
msigdb_aging_genes = set()
for name, genes in msigdb_sets.items():
    if any(kw in name.upper() for kw in AGING_KEYWORDS):
        msigdb_aging_genes |= genes

# CellAge
CELLAGE_ZIP = os.path.join(DB_DIR, "cellAge.zip")
CELLAGE_FILE = os.path.join(DB_DIR, "cellage3.tsv")
download_if_absent(
    "https://genomics.senescence.info/cells/cellAge.zip",
    CELLAGE_ZIP, "CellAge"
)
if not os.path.exists(CELLAGE_FILE):
    with zipfile.ZipFile(CELLAGE_ZIP, "r") as z:
        tsv_names = [n for n in z.namelist() if n.lower().endswith(('.tsv', '.csv'))]
        if tsv_names:
            with z.open(tsv_names[0]) as f:
                content = f.read()
            with open(CELLAGE_FILE, "wb") as out:
                out.write(content)

cellage = pd.read_csv(CELLAGE_FILE, sep='\t', engine='python',
                       on_bad_lines='skip', quoting=3, dtype=str)
cellage_symbol_col = None
for col in cellage.columns:
    if "symbol" in col.lower() or "gene" in col.lower() or "name" in col.lower():
        cellage_symbol_col = col
        break
if cellage_symbol_col is None:
    cellage_symbol_col = cellage.columns[0]
cellage_symbols = set(cellage[cellage_symbol_col].dropna().str.strip())

# AgeAnno
AGEANNO_DIR = os.path.join(DB_DIR, "ageanno")
os.makedirs(AGEANNO_DIR, exist_ok=True)
AGEANNO_DEGS = os.path.join(AGEANNO_DIR, "aging_DEGs.txt")
download_if_absent(
    "https://raw.githubusercontent.com/vikkihuangkexin/AgeAnno/main/scRNA/Aging-related%20DEGs.txt",
    AGEANNO_DEGS, "AgeAnno DEGs"
)
ageanno_degs = pd.read_csv(AGEANNO_DEGS, sep=",", encoding="latin-1")
ageanno_genes = set(ageanno_degs["gene"].dropna().unique())

# Local aging DB
aging_local = pd.read_csv(os.path.join(DATA_DIR, "human_age_related_gene.csv"))
aging_local_symbols = set(aging_local["Symbol"].dropna())

# ── Enrichissement : les gènes prédits comme DE sont-ils enrichis dans les BDD ? ──
print("\n  Analyse d'enrichissement :")
print(f"  {'Base':20s} {'N_total':>8s} {'In_graph':>9s} {'Pred_DE':>8s} {'%_pred':>8s} {'Expected':>9s} {'Enrichment':>11s}")
print("  " + "-" * 75)

# Gènes prédits comme importants (prob > 0.5 sur au moins un label)
pred_any_positive = (probs > 0.5).any(axis=1)
pred_positive_genes = set(gene_symbols[i] for i in range(n_genes) if pred_any_positive[i])
n_pred_pos = len(pred_positive_genes)
frac_pred = n_pred_pos / n_genes if n_genes > 0 else 0

for db_name, db_genes in [
    ("GenAge", genage_symbols),
    ("CellAge", cellage_symbols),
    ("MSigDB aging", msigdb_aging_genes),
    ("AgeAnno", ageanno_genes),
    ("Aging local", aging_local_symbols),
]:
    in_graph = db_genes & set(gene_symbols)
    pred_and_db = pred_positive_genes & db_genes
    n_in = len(in_graph)
    n_pred = len(pred_and_db)
    pct = 100 * n_pred / n_in if n_in > 0 else 0
    expected = frac_pred * n_in
    enrichment = n_pred / expected if expected > 0 else float("nan")
    print(f"  {db_name:20s} {len(db_genes):8d} {n_in:9d} {n_pred:8d} "
          f"{pct:7.1f}% {expected:9.1f} {enrichment:10.2f}x")

# ── Heatmap : probabilités des gènes des BDD ────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax_idx, (db_name, db_genes) in enumerate([
    ("GenAge", genage_symbols),
    ("CellAge", cellage_symbols),
    ("MSigDB aging", msigdb_aging_genes),
    ("AgeAnno", ageanno_genes),
]):
    ax = axes[ax_idx // 2, ax_idx % 2]
    in_graph_mask = np.array([g in db_genes for g in gene_symbols])
    not_in_mask = ~in_graph_mask

    for j, name in enumerate(LABEL_NAMES):
        if in_graph_mask.sum() > 0:
            ax.hist(probs[in_graph_mask, j], bins=30, alpha=0.4,
                    label=f"{name} (in DB)" if j == 0 else None)

    # Box plot comparaison
    data_in = probs[in_graph_mask].mean(axis=1) if in_graph_mask.sum() > 0 else []
    data_out = probs[not_in_mask].mean(axis=1) if not_in_mask.sum() > 0 else []

    ax.hist(data_in, bins=30, alpha=0.6, color="#E74C3C", label=f"In {db_name}", edgecolor="white")
    ax.hist(data_out, bins=30, alpha=0.4, color="#3498DB", label=f"Not in {db_name}", edgecolor="white")
    ax.set_title(f"{db_name} ({in_graph_mask.sum()} gènes dans le graphe)")
    ax.legend(fontsize=8)
    ax.set_xlabel("Probabilité moyenne")

fig.suptitle("Distribution des probabilités : gènes des BDD vs reste", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "validation_databases.png"))
plt.close(fig)
print("  → validation_databases.png")

# =============================================================================
# 11. EXPORT DES RÉSULTATS
# =============================================================================
print("\n" + "=" * 70)
print("11. Export")
print("=" * 70)

# Table complète des résultats
results = pd.DataFrame({"gene": gene_symbols})
results["split"] = np.where(train_mask.numpy(), "train", "test")

# Score d'importance (ranking) — prédit et réel
results["impact_score_predicted"] = pred_impact_np
results["impact_score_real"] = impact_scores
results["rank"] = results["impact_score_predicted"].rank(ascending=False).astype(int)

# Probabilités et prédictions par label
for j, name in enumerate(LABEL_NAMES):
    results[f"prob_{name}"] = probs[:, j]
    results[f"pred_{name}"] = preds[:, j]
    results[f"true_{name}"] = labels[:, j].astype(int)

# Résumés multi-label
results["prob_max"] = probs.max(axis=1)
results["prob_mean"] = probs.mean(axis=1)
results["n_labels_predicted"] = preds.sum(axis=1)
results["n_labels_true"] = labels.sum(axis=1).astype(int)

# Scores de confiance (robustesse des labels)
results["bootstrap_stability"] = bootstrap_stability
results["consensus_score_p4p16"] = consensus_score_p4p16
results["confidence_global"] = confidence_global

# Présence dans les BDD externes (validation)
results["in_genage"] = [1 if g in genage_symbols else 0 for g in gene_symbols]
results["in_cellage"] = [1 if g in cellage_symbols else 0 for g in gene_symbols]
results["in_msigdb_aging"] = [1 if g in msigdb_aging_genes else 0 for g in gene_symbols]
results["in_ageanno"] = [1 if g in ageanno_genes else 0 for g in gene_symbols]
results["in_aging_local"] = [1 if g in aging_local_symbols else 0 for g in gene_symbols]
results["n_databases"] = (results[["in_genage", "in_cellage", "in_msigdb_aging",
                                    "in_ageanno", "in_aging_local"]].sum(axis=1))

# Trier par score d'importance prédit (ranking)
results = results.sort_values("impact_score_predicted", ascending=False)
results.to_csv(os.path.join(OUT_DIR, "gene_predictions.csv"), index=False)
print(f"  → gene_predictions.csv ({len(results)} gènes, trié par impact_score)")

# Embeddings
gene_emb_df = pd.DataFrame(gene_emb, index=gene_symbols)
gene_emb_df.to_csv(os.path.join(OUT_DIR, "gene_embeddings.csv"))
print(f"  → gene_embeddings.csv ({gene_emb.shape})")

# Modèle
torch.save(model.state_dict(), os.path.join(OUT_DIR, "model_weights.pt"))
print("  → model_weights.pt")

# =============================================================================
# RÉSUMÉ
# =============================================================================
print("\n" + "=" * 70)
print("RÉSUMÉ — GNN MULTI-TÂCHE (CLASSIFICATION + RANKING) SÉNESCENCE")
print("=" * 70)
print(f"""
Graphe hétérogène :
  Noeuds gene       : {n_genes} (features : {gene_features.shape[1]})
  Noeuds cell_group : {len(CELL_GROUPS)} ({', '.join(CELL_GROUPS)})
  Arêtes expresses       : {edge_index_expresses.shape[1]} (features: {EXPR_FEAT_NAMES})
  Arêtes expressed_in    : {edge_index_expresses.shape[1]}
  Arêtes PPI             : {edge_index_ppi.shape[1]}
  Arêtes pathway         : {edge_index_pathway.shape[1]}
  Arêtes regulates       : {edge_index_regulates.shape[1]} (pySCENIC, weighted)
  Arêtes regulated_by    : {edge_index_regulated_by.shape[1]}
  Arêtes coexpression    : {edge_index_coexpr.shape[1] if edge_index_coexpr.numel() > 0 else 0} (GRNBoost2 top {100*(1-COEXPR_TOP_QUANTILE):.0f}%)

Labels (basés sur le clustering scRNA-seq) :
  {LABEL_NAMES[0]:20s} : {int(labels[:, 0].sum()):5d} gènes positifs
  {LABEL_NAMES[1]:20s} : {int(labels[:, 1].sum()):5d} gènes positifs
  {LABEL_NAMES[2]:20s} : {int(labels[:, 2].sum()):5d} gènes positifs
  {LABEL_NAMES[3]:20s} : {int(labels[:, 3].sum()):5d} gènes positifs
  {LABEL_NAMES[4]:20s} : {int(labels[:, 4].sum()):5d} gènes positifs

Robustesse des labels (compensation n=1 par condition) :
  Bootstrap stabilité   : mean={bootstrap_stability.mean():.3f}, stable(≥0.7)={int((bootstrap_stability >= 0.7).sum())}
  Consensus (Wilcox+MAST) : score=1.0 pour {int((consensus_score_p4p16 == 1.0).sum())} gènes
  Confiance globale     : mean={confidence_global.mean():.3f}, min={confidence_global.min():.3f}
  Loss pondérée         : plancher={CONFIDENCE_FLOOR}, weights mean={confidence_weights.mean():.3f}

Modèle HeteroGNN (GATConv, multi-tâche) :
  Architecture : {N_LAYERS} couches GATConv × {N_HEADS} têtes d'attention
  Hidden dim   : {HIDDEN_DIM}
  Paramètres   : {total_params:,}
  Best epoch   : {best_epoch + 1}
  Lambda score : {LAMBDA_SCORE}

Performance classification (test set, {test_mask.sum().item()} gènes) :
  Loss totale : {best_test_loss:.5f}
  F1 macro    : {f1_macro:.4f}
  F1 micro    : {f1_micro:.4f}

Performance ranking (test set) :
  Score MSE   : {score_mse:.5f}
  Pearson r   : {score_corr:.4f}
  Spearman ρ  : {spearman_r:.4f}
  Kendall τ   : {kendall_t:.4f}

Top 5 gènes par score d'importance prédit :
  {"Rang":>4s}  {"Gène":15s}  {"Score":>7s}  {"Labels":>7s}
{chr(10).join(f"  {i+1:4d}  {gene_symbols[ranking[i]]:15s}  {pred_impact_np[ranking[i]]:.4f}  {int(labels[ranking[i]].sum()):7d}" for i in range(5))}

Validation externe :
  GenAge     : {len(genage_symbols)} gènes
  CellAge    : {len(cellage_symbols)} gènes
  MSigDB     : {len(msigdb_aging_genes)} gènes
  AgeAnno    : {len(ageanno_genes)} gènes

Fichiers : {OUT_DIR}/
Figures  : {FIG_DIR}/
""")
