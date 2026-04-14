"""
GNN VGAE — Priorisation non supervisée de gènes dans la sénescence (HUVEC)
==========================================================================
Approche NON SUPERVISÉE par Variational Graph AutoEncoder (VGAE) sur un
graphe hétérogène. Le VGAE apprend des embeddings de gènes en reconstruisant
la topologie du graphe — sans aucun label DEG.

PROBLÈME RÉSOLU : le pipeline supervisé précédent (gnn.py) souffrait de
circularité (features = log2FC/padj → labels = DEG basés sur ces mêmes stats).
Ici, le score d'importance ÉMERGE de l'espace latent, pas d'une formule.

DONNÉES : GSE102090 (HUVEC), n=1 par condition (P4, P16).

ARCHITECTURE :
  1. Graphe hétérogène :
     - Noeuds "gene"       : features topologiques (is_tf, variance, ppi_degree)
                              PAS de log2FC/padj (supprime la circularité)
     - Noeuds "cell_group" : 5 groupes (P4, P16_cluster_0..3)
     - Arêtes "expresses"  : mean_expr, pct, std, cv, q25, q75, tf_activity
     - Arêtes "ppi"        : STRING (combined_score)
     - Arêtes "regulates"  : pySCENIC TF→cible (weight)
     - Arêtes "same_pathway" : REACTOME
     - Arêtes "coexpression" : GRNBoost2 (top 2%)

  2. VGAE : encoder HeteroGNN (GATConv) → μ, σ → z ~ N(μ,σ²) → decoder (inner product)
     Loss = reconstruction des arêtes + KL divergence

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
import re
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

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory sur nautilus
LAB_DIR = "/LAB-DATA/GLiCID/users/USER@univ-nantes.fr/"
BASE_DIR = os.path.join(LAB_DIR, "gnn")
DATA_DIR = os.path.join(BASE_DIR, "data")
SCENIC_DIR = os.path.join(BASE_DIR, "output", "pyscenic")
OUT_DIR = "/scratch/nautilus/users/USER@univ-nantes.fr/gnn_vgae"

# directory local
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(BASE_DIR, "..", "data")
# SCENIC_DIR = os.path.join(BASE_DIR, "..", "output", "pyscenic")
# OUT_DIR = os.path.join(BASE_DIR, "..", "output", "gnn_vgae")

# Commun
GNN_DATA_DIR = os.path.join(DATA_DIR, "gnn_data")
PPI_DIR = os.path.join(DATA_DIR, "PPI")
DB_DIR = os.path.join(DATA_DIR, "databases")
FIG_DIR = os.path.join(OUT_DIR, "figure")

# HuMess (métabolisme — CarveMe + Corner Sampling)
HUMESS_DIR = os.path.join(BASE_DIR, "humess", "output_huvec")
# Local fallback si la structure diffère :
# HUMESS_DIR = "/home/USER/M2/S2/Stage/Projet_Colin/humess/output_huvec"
HUMESS_CONDITIONS = ["P4", "P16"]

for d in [PPI_DIR, DB_DIR, OUT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Paramètres du graphe ─────────────────────────────────────────────────────
PPI_SCORE_THRESH = 900
REACTOME_MAX_PATHWAY = 20
COEXPR_TOP_QUANTILE = 0.98

# ── Paramètres du VGAE ──────────────────────────────────────────────────────
HIDDEN_DIM = 128
LATENT_DIM = 64           # Dimension de l'espace latent (μ, σ)
N_LAYERS = 3
N_HEADS = 4
DROPOUT = 0.2
N_EPOCHS = 400
LR = 0.005
EDGE_SAMPLE_RATIO = 0.1   # Fraction d'arêtes pour train/test split
NEG_SAMPLE_RATIO = 1.0    # Ratio négatifs/positifs pour la reconstruction
N_CLUSTERS = 8            # Nombre de clusters K-means sur les embeddings

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
KL_BETA_MAX = 0.0005      # β final — calibré empiriquement : AUC > 0.90 à ce β, dégrade au-delà
KL_WARMUP_EPOCHS = 50     # Warmup court puis β stable — le cosinus n'a pas besoin de long warmup
FREE_BITS = 0.5           # Minimum KL par dimension latente (en nats)

# ── Paramètres du baseline MLP ───────────────────────────────────────────────
MLP_HIDDEN = 64
MLP_EPOCHS = 200
MLP_LR = 0.001

# ── Style ────────────────────────────────────────────────────────────────────
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
# 1. CHARGEMENT DES DONNÉES scRNA-seq
# =============================================================================
print("=" * 70)
print("1. Chargement des données scRNA-seq (P4 / P16)")
print("=" * 70)

metadata = pd.read_csv(os.path.join(GNN_DATA_DIR, "merged_P4_P16_metadata.csv"))
print(f"  Metadata : {len(metadata)} cellules")
print(f"    P4  : {(metadata['passage'] == 'P4').sum()} cellules")
print(f"    P16 : {(metadata['passage'] == 'P16').sum()} cellules")

p16_meta = metadata[metadata["passage"] == "P16"].copy()
p16_clusters = sorted(p16_meta["cluster_P16"].dropna().unique())
print(f"    Clusters P16 : {p16_clusters}")

# Liste de TOUS les gènes disponibles (pas de filtrage par DEG)
# On prend tous les gènes de la matrice normalisée — le VGAE n'a pas besoin
# de sélection par DEG puisqu'il n'utilise pas de labels.
with open(os.path.join(GNN_DATA_DIR, "merged_P4_P16_normalized.csv")) as f:
    header = f.readline().strip().split(",")
all_available_genes = [g.strip('"') for g in header[4:]]

# Mais on ne garde que les gènes qui ont au moins une interaction dans le graphe
# (PPI, SCENIC, ou pathway). Les gènes isolés n'apportent rien au VGAE.
# Cette filtration sera faite APRÈS le chargement des arêtes (section 3).
print(f"  Gènes dans la matrice : {len(all_available_genes)}")

# =============================================================================
# 2. CHARGEMENT DES DONNÉES pySCENIC
# =============================================================================
print("\n" + "=" * 70)
print("2. Chargement des données pySCENIC")
print("=" * 70)

regulon_edges = pd.read_csv(os.path.join(SCENIC_DIR, "regulon_edges_TF_to_gene.csv"))
regulon_edges["TF_clean"] = regulon_edges["TF"].str.replace(r"\(\+\)$", "", regex=True)
print(f"  Regulon edges : {len(regulon_edges)} interactions TF→cible")

tf_activity = pd.read_csv(
    os.path.join(SCENIC_DIR, "mean_TF_activity_per_cluster.csv"), index_col=0
)
tf_activity.columns = [c.replace("(+)", "") for c in tf_activity.columns]
print(f"  TF activity : {tf_activity.shape[0]} clusters × {tf_activity.shape[1]} TFs")

adjacencies = pd.read_csv(os.path.join(SCENIC_DIR, "adjacencies.csv"))
importance_thresh = adjacencies["importance"].quantile(COEXPR_TOP_QUANTILE)
adjacencies_filtered = adjacencies[adjacencies["importance"] >= importance_thresh].copy()
print(f"  Adjacencies : {len(adjacencies)} → {len(adjacencies_filtered)} "
      f"(top {100*(1-COEXPR_TOP_QUANTILE):.0f}%, seuil={importance_thresh:.2f})")

# =============================================================================
# 3. SÉLECTION DES GÈNES — BASÉE SUR LA CONNECTIVITÉ (pas les DEGs)
# =============================================================================
# JUSTIFICATION : dans le VGAE, un gène isolé (sans arête) ne reçoit aucune
# information par message passing et ne contribue pas à la reconstruction.
# On garde les gènes qui participent à au moins un type d'interaction.
print("\n" + "=" * 70)
print("3. Sélection des gènes par connectivité")
print("=" * 70)

available_set = set(all_available_genes)

# Gènes dans le réseau de régulation SCENIC
scenic_genes = set(regulon_edges["TF_clean"]) | set(regulon_edges["target_gene"])
scenic_genes &= available_set

# Gènes dans la co-expression GRNBoost2 (filtrée)
coexpr_genes = set(adjacencies_filtered["TF"].astype(str)) | set(adjacencies_filtered["target"].astype(str))
coexpr_genes &= available_set

# Gènes dans STRING PPI — chargement léger pour identifier les gènes connectés
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

aliases = pd.read_csv(PPI_ALIAS_FILE, sep="\t", compression="gzip")
aliases_filt = aliases[
    (aliases["alias"].isin(available_set)) &
    (aliases["source"].str.contains("Ensembl_HGNC", na=False))
]
sym2string = dict(zip(aliases_filt["alias"], aliases_filt["#string_protein_id"]))
string2sym = {v: k for k, v in sym2string.items()}
string_ids = set(sym2string.values())

ppi_raw = pd.read_csv(PPI_FILE, sep=" ", compression="gzip")
ppi_hc = ppi_raw[
    (ppi_raw["protein1"].isin(string_ids)) &
    (ppi_raw["protein2"].isin(string_ids)) &
    (ppi_raw["combined_score"] >= PPI_SCORE_THRESH)
]
ppi_genes = set()
for _, row in ppi_hc.iterrows():
    s1, s2 = string2sym.get(row["protein1"]), string2sym.get(row["protein2"])
    if s1 and s2:
        ppi_genes.update([s1, s2])

# REACTOME
MSIGDB_REACTOME = os.path.join(DB_DIR, "c2.cp.reactome.symbols.gmt")
download_if_absent(
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/c2.cp.reactome.v2024.1.Hs.symbols.gmt",
    MSIGDB_REACTOME, "MSigDB REACTOME"
)
reactome_pathways = {}
reactome_genes = set()
with open(MSIGDB_REACTOME) as f:
    for line in f:
        parts = line.strip().split("\t")
        genes_in_pw = set(parts[2:]) & available_set
        if 2 <= len(genes_in_pw) <= REACTOME_MAX_PATHWAY:
            reactome_pathways[parts[0]] = genes_in_pw
            reactome_genes |= genes_in_pw

# Union de tous les gènes connectés
connected_genes = scenic_genes | coexpr_genes | ppi_genes | reactome_genes
gene_symbols = np.array(sorted(connected_genes & available_set))
gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}
n_genes = len(gene_symbols)

print(f"  Gènes connectés : {n_genes}")
print(f"    SCENIC       : {len(scenic_genes & set(gene_symbols))}")
print(f"    Co-expression: {len(coexpr_genes & set(gene_symbols))}")
print(f"    PPI          : {len(ppi_genes & set(gene_symbols))}")
print(f"    REACTOME     : {len(reactome_genes & set(gene_symbols))}")

# =============================================================================
# 4. CALCUL DE L'EXPRESSION PAR GROUPE — FEATURES D'ARÊTES
# =============================================================================
# L'expression est placée sur les ARÊTES (cell_group → gene), pas sur les
# noeuds gene. C'est ce qui supprime la circularité : les noeuds gene n'ont
# que des features topologiques.
print("\n" + "=" * 70)
print("4. Expression par groupe cellulaire (features d'arêtes)")
print("=" * 70)

CELL_GROUPS = ["P4", "P16_cluster_0", "P16_cluster_1",
               "P16_cluster_2", "P16_cluster_3"]

print("  Chargement de la matrice normalisée...")
cols_to_read = ["barcode", "passage", "cluster_P16", "cell_state"] + gene_symbols.tolist()
normalized = pd.read_csv(
    os.path.join(GNN_DATA_DIR, "merged_P4_P16_normalized.csv"),
    usecols=cols_to_read,
).copy()  # .copy() défragmente le DataFrame après read_csv avec usecols

def assign_group(row):
    if row["passage"] == "P4":
        return "P4"
    c = row["cluster_P16"]
    if pd.notna(c):
        return f"P16_cluster_{int(c)}"
    return None

normalized["group"] = normalized.apply(assign_group, axis=1)
normalized = normalized.dropna(subset=["group"])
print(f"  Matrice : {normalized.shape}")

# Statistiques par groupe : mean, pct, std, cv, q25, q75
print("  Calcul des statistiques (mean, pct, std, cv, q25, q75)...")
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
        "mean_expression": mean_expr, "pct_expressing": pct_expr,
        "std_expression": std_expr, "cv_expression": cv_expr,
        "q25": q25, "q75": q75, "n_cells": n_cells,
    }
    print(f"    {grp:20s} : {n_cells} cellules, mean={mean_expr.mean():.3f}")

del normalized

# =============================================================================
# 5. FEATURES DES NOEUDS (topologiques uniquement — PAS de log2FC/padj)
# =============================================================================
# JUSTIFICATION : les features de noeuds ne doivent PAS contenir l'information
# qui servait de labels (log2FC, padj, delta_pct). On ne garde que :
#   - is_tf : le gène est-il un TF (propriété intrinsèque)
#   - variance_across_groups : variabilité de l'expression entre groupes
#     (pas le log2FC qui est un test stat, mais la variance brute)
# Le degré PPI et le degré de régulation seront ajoutés après section 6.
print("\n" + "=" * 70)
print("5. Features des noeuds (topologiques)")
print("=" * 70)

scenic_tfs = set(regulon_edges["TF_clean"].unique())
is_tf = np.array([1.0 if g in scenic_tfs else 0.0 for g in gene_symbols],
                 dtype=np.float32)
print(f"  is_tf : {int(is_tf.sum())} TFs")

mean_expr_per_group = np.array([
    group_stats[grp]["mean_expression"] for grp in CELL_GROUPS
])
variance_across = mean_expr_per_group.var(axis=0).astype(np.float32)
variance_norm = variance_across / (variance_across.max() + 1e-8)

# ── Cell group features ─────────────────────────────────────────────────────
cell_group_features_list = []
for grp in CELL_GROUPS:
    is_senescent = 0.0 if grp == "P4" else 1.0
    n_cells_norm = group_stats[grp]["n_cells"] / metadata.shape[0]
    if grp == "P4":
        cluster_idx = 0.0
    else:
        c = int(grp.split("_")[-1])
        cluster_idx = (c + 1) / 4.0
    cell_group_features_list.append([is_senescent, n_cells_norm, cluster_idx])
cell_group_features = torch.tensor(cell_group_features_list, dtype=torch.float)

# =============================================================================
# 6. CONSTRUCTION DES ARÊTES
# =============================================================================
print("\n" + "=" * 70)
print("6. Construction des arêtes")
print("=" * 70)

# ── 6a. cell_group → gene (expression) ──────────────────────────────────────
expr_src, expr_dst, expr_attrs = [], [], []
for grp_idx, grp in enumerate(CELL_GROUPS):
    stats = group_stats[grp]
    cluster_id = None if grp == "P4" else int(grp.split("_")[-1])

    for gene_idx in range(n_genes):
        gene_name = gene_symbols[gene_idx]
        tf_act = 0.0
        if cluster_id is not None and gene_name in tf_activity.columns:
            if cluster_id in tf_activity.index:
                tf_act = float(tf_activity.loc[cluster_id, gene_name])

        expr_src.append(grp_idx)
        expr_dst.append(gene_idx)
        expr_attrs.append([
            float(stats["mean_expression"][gene_idx]),
            float(stats["pct_expressing"][gene_idx]),
            tf_act,
            float(stats["std_expression"][gene_idx]),
            float(stats["cv_expression"][gene_idx]),
            float(stats["q25"][gene_idx]),
            float(stats["q75"][gene_idx]),
        ])

edge_index_expresses = torch.tensor([expr_src, expr_dst], dtype=torch.long)
edge_attr_expresses = torch.tensor(expr_attrs, dtype=torch.float)
for col in range(edge_attr_expresses.shape[1]):
    col_data = edge_attr_expresses[:, col]
    mu, std = col_data.mean(), col_data.std() + 1e-8
    edge_attr_expresses[:, col] = (col_data - mu) / std
print(f"  expresses : {edge_index_expresses.shape[1]} arêtes, "
      f"7 features [mean, pct, tf_act, std, cv, q25, q75]")

# ── 6b. PPI STRING ──────────────────────────────────────────────────────────
ppi_src, ppi_dst, ppi_w = [], [], []
for _, row in ppi_hc.iterrows():
    s1, s2 = string2sym.get(row["protein1"]), string2sym.get(row["protein2"])
    if s1 and s2 and s1 in gene_to_idx and s2 in gene_to_idx:
        i, j = gene_to_idx[s1], gene_to_idx[s2]
        ppi_src.extend([i, j])
        ppi_dst.extend([j, i])
        ppi_w.extend([row["combined_score"] / 1000.0] * 2)

edge_index_ppi = torch.tensor([ppi_src, ppi_dst], dtype=torch.long)
edge_attr_ppi = torch.tensor(ppi_w, dtype=torch.float).unsqueeze(1)
print(f"  ppi : {len(ppi_src)//2} interactions ({len(ppi_src)} arêtes)")

# ── 6c. REACTOME ────────────────────────────────────────────────────────────
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
print(f"  pathway : {len(react_pairs)} paires ({len(react_src)} arêtes)")

# ── 6d. Regulon (pySCENIC) ──────────────────────────────────────────────────
reg_src, reg_dst, reg_w = [], [], []
reg_pairs = set()
for _, row in regulon_edges.iterrows():
    tf, target = row["TF_clean"], row["target_gene"]
    if tf in gene_to_idx and target in gene_to_idx:
        pair = (gene_to_idx[tf], gene_to_idx[target])
        if pair not in reg_pairs:
            reg_pairs.add(pair)
            reg_src.append(pair[0])
            reg_dst.append(pair[1])
            reg_w.append(float(row["weight"]))

edge_index_regulates = torch.tensor([reg_src, reg_dst], dtype=torch.long)
edge_attr_regulates = torch.tensor(reg_w, dtype=torch.float).unsqueeze(1)
if edge_attr_regulates.numel() > 0:
    edge_attr_regulates = edge_attr_regulates / (edge_attr_regulates.max() + 1e-8)
edge_index_regulated_by = torch.tensor([reg_dst, reg_src], dtype=torch.long)
print(f"  regulates : {len(reg_pairs)} liens TF→cible")

# ── 6e. Co-expression (GRNBoost2) ───────────────────────────────────────────
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
    [coexpr_src, coexpr_dst] if coexpr_src else [[], []], dtype=torch.long
)
coexpr_w_tensor = (torch.tensor(coexpr_w, dtype=torch.float).unsqueeze(1)
                   if coexpr_w else torch.zeros((0, 1)))
if coexpr_w_tensor.numel() > 0:
    coexpr_w_tensor = coexpr_w_tensor / (coexpr_w_tensor.max() + 1e-8)
print(f"  coexpression : {len(coexpr_pairs)} paires ({len(coexpr_src)} arêtes)")

# ── 6f. HuMess : cocatalysis (A) + importance métabolique (B) ───────────────
# A : arête entre gènes catalysant la même réaction (GPR rules), union P4 ∪ P16.
#     edge_attr binaire [in_P4, in_P16] → permet au modèle d'apprendre la
#     condition-spécificité du lien métabolique.
# B : importance métabolique par gène (max sur les réactions catalysées),
#     log1p + z-score, par condition + delta. Feature mask `has_humess` pour
#     éviter que le modèle interprète les 0 imputés comme un signal.
print("\n  HuMess (cocatalysis + importance)...")

_gene_token_re = re.compile(r"[A-Za-z0-9_\-\.]+")
_gpr_skip = {"and", "or", "AND", "OR", "And", "Or"}


def _parse_gpr(gpr_str):
    """Extrait les symboles de gène d'une règle GPR (sépare and/or/parenthèses)."""
    cleaned = gpr_str.replace("(", " ").replace(")", " ")
    return {t for t in _gene_token_re.findall(cleaned) if t not in _gpr_skip}


# --- A : parse gr-rules par condition ---
reaction_to_genes = {}   # cond -> {reaction: set(genes)}
gene_to_reactions = {c: {} for c in HUMESS_CONDITIONS}  # cond -> {gene: set(reactions)}

for cond in HUMESS_CONDITIONS:
    path = os.path.join(HUMESS_DIR, "models", cond, "stats", "carveme.gr-rules.tsv")
    cond_map = {}
    if not os.path.exists(path):
        print(f"    [warn] introuvable : {path}")
        reaction_to_genes[cond] = cond_map
        continue
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rxn, gpr = parts[0], parts[1]
            genes = _parse_gpr(gpr)
            cond_map[rxn] = genes
            for g in genes:
                gene_to_reactions[cond].setdefault(g, set()).add(rxn)
    reaction_to_genes[cond] = cond_map
    print(f"    {cond} : {len(cond_map)} réactions, "
          f"{len(gene_to_reactions[cond])} gènes")

# Construction des paires de cocatalysis (indices graphe)
cocat_pair_flags = {}  # (i,j) -> [in_P4, in_P16]
for cond_idx, cond in enumerate(HUMESS_CONDITIONS):
    for rxn, genes in reaction_to_genes[cond].items():
        idx_list = sorted({gene_to_idx[g] for g in genes if g in gene_to_idx})
        for a in range(len(idx_list)):
            for b in range(a + 1, len(idx_list)):
                pair = (idx_list[a], idx_list[b])
                flags = cocat_pair_flags.setdefault(pair, [0.0, 0.0])
                flags[cond_idx] = 1.0

cocat_src, cocat_dst, cocat_attr = [], [], []
for (i, j), flags in cocat_pair_flags.items():
    cocat_src.extend([i, j])
    cocat_dst.extend([j, i])
    cocat_attr.append(flags)
    cocat_attr.append(flags)

edge_index_cocat = torch.tensor(
    [cocat_src, cocat_dst] if cocat_src else [[], []], dtype=torch.long
)
edge_attr_cocat = (torch.tensor(cocat_attr, dtype=torch.float)
                   if cocat_attr else torch.zeros((0, 2)))
print(f"    cocatalysis : {len(cocat_pair_flags)} paires "
      f"({edge_index_cocat.shape[1]} arêtes, attr=[in_P4, in_P16])")

# --- B : importance métabolique par gène ---
gene_importance = {c: np.zeros(n_genes, dtype=np.float32) for c in HUMESS_CONDITIONS}
gene_in_model = {c: np.zeros(n_genes, dtype=np.float32) for c in HUMESS_CONDITIONS}

for cond in HUMESS_CONDITIONS:
    path = os.path.join(HUMESS_DIR, "models", cond, "cs",
                        f"cs_gene_to_importance_{cond}.tsv")
    if not os.path.exists(path):
        print(f"    [warn] introuvable : {path}")
        continue
    df_imp = pd.read_csv(path, sep="\t")
    # format attendu : symbol, bigg, importance (une ligne par (gène, réaction))
    df_imp = df_imp.dropna(subset=["symbol", "importance"])
    # max importance par symbole (agrégation sur réactions catalysées)
    agg = df_imp.groupby("symbol")["importance"].max()
    for sym, val in agg.items():
        if sym in gene_to_idx:
            gene_importance[cond][gene_to_idx[sym]] = float(val)
            gene_in_model[cond][gene_to_idx[sym]] = 1.0

# log1p + z-score par condition (sur gènes présents uniquement, extrapolé à 0 ailleurs)
def _log1p_zscore(arr, mask):
    out = np.zeros_like(arr)
    if mask.sum() < 2:
        return out
    vals = np.log1p(arr[mask.astype(bool)])
    mu, sd = vals.mean(), vals.std() + 1e-8
    out[mask.astype(bool)] = (vals - mu) / sd
    return out

imp_P4_z = _log1p_zscore(gene_importance["P4"], gene_in_model["P4"])
imp_P16_z = _log1p_zscore(gene_importance["P16"], gene_in_model["P16"])
imp_delta = imp_P16_z - imp_P4_z  # positif = plus important en P16
has_humess = ((gene_in_model["P4"] + gene_in_model["P16"]) > 0).astype(np.float32)

print(f"    importance  : P4={int(gene_in_model['P4'].sum())} gènes, "
      f"P16={int(gene_in_model['P16'].sum())} gènes, "
      f"has_humess={int(has_humess.sum())}/{n_genes}")

# ── Finaliser les features de noeuds gene avec les degrés ────────────────────
ppi_degree = np.zeros(n_genes, dtype=np.float32)
for idx in ppi_src:
    ppi_degree[idx] += 1
ppi_degree_norm = ppi_degree / (ppi_degree.max() + 1e-8)

reg_degree = np.zeros(n_genes, dtype=np.float32)
for idx in reg_src + reg_dst:
    reg_degree[idx] += 1
reg_degree_norm = reg_degree / (reg_degree.max() + 1e-8)

gene_features = torch.tensor(
    np.column_stack([
        is_tf, variance_norm, ppi_degree_norm, reg_degree_norm,
        imp_P4_z, imp_P16_z, imp_delta, has_humess,
    ]),
    dtype=torch.float,
)
print(f"\n  Gene features : {gene_features.shape}")
print(f"    [is_tf, variance, ppi_degree, reg_degree,")
print(f"     met_imp_P4, met_imp_P16, met_imp_delta, has_humess]")
print(f"    PAS de log2FC, padj, delta_pct (circularité supprimée)")

# =============================================================================
# 7. ASSEMBLAGE DU GRAPHE HÉTÉROGÈNE
# =============================================================================
print("\n" + "=" * 70)
print("7. Assemblage du graphe hétérogène")
print("=" * 70)

data = HeteroData()

data["gene"].x = gene_features
data["gene"].num_nodes = n_genes
data["cell_group"].x = cell_group_features
data["cell_group"].num_nodes = len(CELL_GROUPS)

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

if edge_index_cocat.numel() > 0:
    data["gene", "metabolic_cocatalysis", "gene"].edge_index = edge_index_cocat
    data["gene", "metabolic_cocatalysis", "gene"].edge_attr = edge_attr_cocat

print(f"  Noeuds gene       : {n_genes} (features={gene_features.shape[1]})")
print(f"  Noeuds cell_group : {len(CELL_GROUPS)}")
print(f"  Arêtes ppi        : {edge_index_ppi.shape[1]}")
print(f"  Arêtes pathway    : {edge_index_pathway.shape[1]}")
print(f"  Arêtes regulates  : {edge_index_regulates.shape[1]}")
print(f"  Arêtes coexpr     : {edge_index_coexpr.shape[1] if edge_index_coexpr.numel() > 0 else 0}")
print(f"  Arêtes cocat      : {edge_index_cocat.shape[1] if edge_index_cocat.numel() > 0 else 0}")

torch.save(data, os.path.join(OUT_DIR, "hetero_graph_vgae.pt"))

# =============================================================================
# 8. MODÈLE VGAE
# =============================================================================
# Le VGAE a deux composants :
#   1. Encoder (HeteroGNN) : produit μ et log(σ²) pour chaque noeud gene
#      → z = μ + ε·σ (reparametrization trick)
#   2. Decoder (inner product) : P(arête i↔j) = σ(z_i · z_j)
#      → pas de paramètres, juste un produit scalaire
#
# Loss = reconstruction_loss (BCE sur les arêtes) + β·KL_divergence
#   - reconstruction_loss : le VGAE doit prédire quelles paires de gènes
#     sont connectées (positifs) vs non connectées (négatifs)
#   - KL_divergence : régularise l'espace latent vers N(0,I)
# =============================================================================
print("\n" + "=" * 70)
print("8. Modèle VGAE")
print("=" * 70)


class HeteroEncoder(nn.Module):
    """
    Encoder hétérogène pour le VGAE.

    Produit μ et log(σ²) pour chaque noeud gene via message passing
    sur le graphe hétérogène. Les noeuds cell_group participent au
    message passing mais ne sont pas encodés dans l'espace latent
    (seuls les gènes nous intéressent pour la priorisation).
    """

    def __init__(self, gene_in, cell_in, hidden, latent, n_layers,
                 n_heads=4, dropout=0.2):
        super().__init__()
        self.n_layers = n_layers
        head_dim = hidden // n_heads

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
            ("gene", "metabolic_cocatalysis", "gene"),
        ]

        for _ in range(n_layers):
            conv_dict = {}
            for et in edge_types:
                conv_dict[et] = GATConv(
                    hidden, head_dim, heads=n_heads,
                    concat=True, dropout=dropout, add_self_loops=False,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(nn.ModuleDict({
                "gene": nn.BatchNorm1d(hidden),
                "cell_group": nn.BatchNorm1d(hidden),
            }))

        self.dropout = nn.Dropout(dropout)

        # Deux têtes pour la partie variationnelle : μ et log(σ²)
        self.mu_head = nn.Linear(hidden, latent)
        self.logvar_head = nn.Linear(hidden, latent)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            "gene": F.relu(self.gene_proj(x_dict["gene"])),
            "cell_group": F.relu(self.cell_proj(x_dict["cell_group"])),
        }

        for i in range(self.n_layers):
            x_prev = {k: v.clone() for k, v in x_dict.items()}
            active_edges = {k: v for k, v in edge_index_dict.items() if v.numel() > 0}
            x_dict = self.convs[i](x_dict, active_edges)

            for key in x_dict:
                x_dict[key] = self.norms[i][key](x_dict[key])
                x_dict[key] = F.relu(x_dict[key])
                x_dict[key] = self.dropout(x_dict[key])
                x_dict[key] = x_dict[key] + x_prev[key]

        gene_h = x_dict["gene"]
        mu = self.mu_head(gene_h)
        logvar = self.logvar_head(gene_h)

        return mu, logvar


class VGAE(nn.Module):
    """
    Variational Graph AutoEncoder.

    Encoder : HeteroGNN → μ, log(σ²)
    Reparametrization : z = μ + ε·exp(0.5·log(σ²)), ε ~ N(0,I)
    Decoder : P(arête i↔j) = σ(τ · cos(z_i, z_j))  (cosine similarity)

    Le décodeur utilise la similarité cosinus au lieu du produit scalaire brut.
    Cela découple la DIRECTION des embeddings (utilisée pour la reconstruction)
    de leur NORME (contrôlée par la KL). Sans ça, la KL pousse μ→0, ce qui
    écrase les inner products vers 0 et cause le collapse (AUC→0.5).
    τ (température) est un paramètre appris qui contrôle la dynamique des logits.
    """

    def __init__(self, encoder, tau_init=2.0, tau_max=3.0):
        super().__init__()
        self.encoder = encoder
        # Température apprise pour le décodeur cosinus — initialisée à tau_init.
        # τ est clampé à tau_max pour empêcher les logits trop grands
        # qui causent des prédictions sur-confiantes → surapprentissage.
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(tau_init))))
        self.log_tau_max = np.log(tau_max)

    def reparametrize(self, mu, logvar):
        """Reparametrization trick : z = μ + ε·σ"""
        # Clamp logvar pour éviter l'explosion numérique (σ trop grand ou trop petit)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        # En évaluation, on utilise directement μ (pas d'aléa)
        return mu

    def encode(self, x_dict, edge_index_dict):
        mu, logvar = self.encoder(x_dict, edge_index_dict)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

    def decode(self, z, edge_index):
        """
        Decoder cosinus : τ · cos(z_i, z_j) → logits.

        cos(z_i, z_j) ∈ [-1, 1] ne dépend que de la direction des embeddings,
        pas de leur norme. La KL peut donc réduire ||μ|| sans affecter la
        reconstruction. τ (appris) contrôle la confiance des prédictions.
        """
        src, dst = edge_index
        z_src = F.normalize(z[src], dim=1)  # L2-normalisation → vecteurs unitaires
        z_dst = F.normalize(z[dst], dim=1)
        cos_sim = (z_src * z_dst).sum(dim=1)
        # Clamp τ pour éviter le surapprentissage par sur-confiance
        log_tau_clamped = torch.clamp(self.log_tau, max=self.log_tau_max)
        tau = torch.exp(log_tau_clamped)
        return tau * cos_sim

    def kl_loss(self, mu, logvar, free_bits=0.0):
        """
        KL divergence avec free bits : D_KL(q(z|x) || p(z)), p(z) = N(0,I).

        Free bits : on calcule la KL par dimension latente et on impose un
        minimum de `free_bits` nats. Les dimensions sous ce seuil ne sont pas
        pénalisées, ce qui force le modèle à utiliser l'espace latent plutôt
        que de le collapser vers la prior.
        """
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        # KL par dimension : shape (n_genes, latent_dim)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        # Moyenne sur les gènes, garde les dimensions : shape (latent_dim,)
        kl_per_dim_mean = kl_per_dim.mean(dim=0)
        # Free bits : clamp chaque dimension au minimum
        kl_clamped = torch.clamp(kl_per_dim_mean, min=free_bits)
        return kl_clamped.sum()


# Instanciation
encoder = HeteroEncoder(
    gene_in=gene_features.shape[1],
    cell_in=cell_group_features.shape[1],
    hidden=HIDDEN_DIM,
    latent=LATENT_DIM,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    dropout=DROPOUT,
)
model = VGAE(encoder)

total_params = sum(p.numel() for p in model.parameters())
print(f"  VGAE : {N_LAYERS} couches GATConv, hidden={HIDDEN_DIM}, latent={LATENT_DIM}")
print(f"  Paramètres : {total_params:,}")

# =============================================================================
# 9. PRÉPARATION DES ARÊTES POUR L'ENTRAÎNEMENT
# =============================================================================
# Le VGAE s'entraîne sur la reconstruction d'arêtes gene↔gene.
# On combine TOUTES les arêtes gene↔gene (PPI + pathway + régulation + coexpr)
# en un seul ensemble, puis on split train/test.
print("\n" + "=" * 70)
print("9. Préparation des arêtes d'entraînement")
print("=" * 70)

# Collecter toutes les arêtes gene↔gene (non dirigées, dédupliquées)
all_gene_edges = set()
for src_list, dst_list in [
    (ppi_src, ppi_dst),
    (react_src, react_dst),
    (reg_src, reg_dst), (reg_dst, reg_src),
    (coexpr_src, coexpr_dst),
]:
    for s, d in zip(src_list, dst_list):
        pair = (min(s, d), max(s, d))
        all_gene_edges.add(pair)

all_edges = np.array(list(all_gene_edges))
n_edges = len(all_edges)
print(f"  Arêtes gene↔gene uniques : {n_edges}")

# Split train/test (on masque des arêtes que le VGAE devra reconstruire)
np.random.seed(42)
perm = np.random.permutation(n_edges)
n_test = max(1, int(n_edges * EDGE_SAMPLE_RATIO))
test_edge_idx = perm[:n_test]
train_edge_idx = perm[n_test:]

test_edges = all_edges[test_edge_idx]
train_edges = all_edges[train_edge_idx]

# Arêtes positives (format PyG : [2, n_edges])
pos_train = torch.tensor(train_edges.T, dtype=torch.long)
# Rendre bidirectionnel
pos_train_bidir = torch.cat([pos_train, pos_train.flip(0)], dim=1)

pos_test = torch.tensor(test_edges.T, dtype=torch.long)
pos_test_bidir = torch.cat([pos_test, pos_test.flip(0)], dim=1)

print(f"  Train : {len(train_edges)} arêtes positives")
print(f"  Test  : {len(test_edges)} arêtes positives")

# =============================================================================
# 10. ENTRAÎNEMENT DU VGAE
# =============================================================================
print("\n" + "=" * 70)
print("10. Entraînement du VGAE")
print("=" * 70)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

x_dict = {"gene": data["gene"].x, "cell_group": data["cell_group"].x}
edge_index_dict = {}
for et_key in [
    ("cell_group", "expresses", "gene"),
    ("gene", "expressed_in", "cell_group"),
    ("gene", "ppi", "gene"),
    ("gene", "same_pathway", "gene"),
    ("gene", "regulates", "gene"),
    ("gene", "regulated_by", "gene"),
]:
    edge_index_dict[et_key] = data[et_key].edge_index
if ("gene", "coexpression", "gene") in data.edge_types:
    edge_index_dict[("gene", "coexpression", "gene")] = \
        data["gene", "coexpression", "gene"].edge_index
if ("gene", "metabolic_cocatalysis", "gene") in data.edge_types:
    edge_index_dict[("gene", "metabolic_cocatalysis", "gene")] = \
        data["gene", "metabolic_cocatalysis", "gene"].edge_index

train_losses = []
test_aucs = []
eval_epochs = []  # Epochs où on évalue (pour le plot)
best_test_auc = 0.0
best_epoch = 0
PATIENCE = 80             # Early stopping : patience modérée, le pic est typiquement autour de epoch 30-50
GRAD_CLIP_NORM = 1.0     # Gradient clipping pour la stabilité
patience_counter = 0

model.train()
for epoch in range(N_EPOCHS):
    optimizer.zero_grad()

    z, mu, logvar = model.encode(x_dict, edge_index_dict)

    # Arêtes négatives (paires de gènes NON connectés)
    neg_train = negative_sampling(
        pos_train_bidir, num_nodes=n_genes,
        num_neg_samples=pos_train_bidir.shape[1],
    )

    # Loss de reconstruction
    pos_scores = model.decode(z, pos_train_bidir)
    neg_scores = model.decode(z, neg_train)

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_scores, torch.ones_like(pos_scores)
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores, torch.zeros_like(neg_scores)
    )
    recon_loss = pos_loss + neg_loss

    # KL divergence avec annealing + free bits
    # β augmente linéairement de 0 à KL_BETA_MAX pendant les KL_WARMUP_EPOCHS
    # premières epochs. Cela permet au modèle d'apprendre d'abord à bien
    # reconstruire les arêtes avant que la régularisation KL ne s'applique.
    kl_beta = min(KL_BETA_MAX, KL_BETA_MAX * epoch / max(1, KL_WARMUP_EPOCHS))
    kl_loss = model.kl_loss(mu, logvar, free_bits=FREE_BITS)

    loss = recon_loss + kl_beta * kl_loss
    loss.backward()
    # Gradient clipping — empêche les mises à jour explosives qui
    # pourraient déstabiliser l'encodeur ou la température τ
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    train_losses.append(loss.item())

    # Évaluation sur les arêtes test
    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            z_eval, mu_eval, _ = model.encode(x_dict, edge_index_dict)

            pos_test_scores = torch.sigmoid(model.decode(z_eval, pos_test_bidir))
            neg_test = negative_sampling(
                pos_test_bidir, num_nodes=n_genes,
                num_neg_samples=pos_test_bidir.shape[1],
            )
            neg_test_scores = torch.sigmoid(model.decode(z_eval, neg_test))

            test_labels = torch.cat([
                torch.ones(pos_test_scores.shape[0]),
                torch.zeros(neg_test_scores.shape[0]),
            ])
            test_scores = torch.cat([pos_test_scores, neg_test_scores])

            try:
                auc = roc_auc_score(test_labels.numpy(), test_scores.numpy())
                ap = average_precision_score(test_labels.numpy(), test_scores.numpy())
            except ValueError:
                auc, ap = 0.5, 0.5

            test_aucs.append(auc)
            eval_epochs.append(epoch)

            if auc > best_test_auc:
                best_test_auc = auc
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_vgae.pt"))
            else:
                patience_counter += 10  # On évalue toutes les 10 epochs

        model.train()

        if (epoch + 1) % 50 == 0:
            tau_val = np.exp(model.log_tau.item())
            print(f"    Epoch {epoch+1:3d}/{N_EPOCHS} — "
                  f"Loss: {loss.item():.4f} (recon={recon_loss.item():.4f} "
                  f"kl={kl_loss.item():.4f} β={kl_beta:.5f} τ={tau_val:.2f}) — "
                  f"Test AUC: {auc:.4f}, AP: {ap:.4f}")

    # Early stopping — si l'AUC ne s'améliore plus, on arrête avant
    # de gaspiller des epochs et de risquer le collapse
    if patience_counter >= PATIENCE:
        print(f"\n  Early stopping à epoch {epoch+1} (pas d'amélioration "
              f"depuis {PATIENCE} epochs)")
        break

print(f"\n  Meilleur modèle : epoch {best_epoch+1}, AUC = {best_test_auc:.4f}")

# Charger le meilleur modèle
model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_vgae.pt"), weights_only=True))

# =============================================================================
# 11. EXTRACTION DES EMBEDDINGS ET SCORE D'IMPORTANCE ÉMERGENT
# =============================================================================
print("\n" + "=" * 70)
print("11. Score d'importance émergent")
print("=" * 70)

model.eval()
with torch.no_grad():
    z_final, mu_final, logvar_final = model.encode(x_dict, edge_index_dict)
    gene_emb = mu_final.numpy()  # On utilise μ (pas z bruité)

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
# Pas de poids arbitraires — on laisse le score émerger de la moyenne
# des différentes perspectives. Chaque composante capture un aspect différent :
#   norme       = remarquabilité topologique
#   densité     = centralité dans l'espace latent (corrigée anti-paquets)
#   recon       = fiabilité de la reconstruction
#   certitude   = stabilité (faible σ)
#   spécificité = signal d'expression concentré sur certains groupes
importance_score = (
    emb_norm_score
    + density_score
    + (1 - recon_error_score)
    + (1 - uncertainty_score)
    + specificity_score
) / 5.0
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
print(f"\n  Clustering K-means (k={N_CLUSTERS})...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
gene_clusters = kmeans.fit_predict(gene_emb)
sil_score = silhouette_score(gene_emb, gene_clusters)
print(f"    Silhouette score : {sil_score:.4f}")

for c in range(N_CLUSTERS):
    n_in = (gene_clusters == c).sum()
    mean_imp = importance_score[gene_clusters == c].mean()
    print(f"    Cluster {c} : {n_in:5d} gènes, importance moyenne = {mean_imp:.4f}")

# =============================================================================
# 12. BASELINE MLP (même features, pas de graphe)
# =============================================================================
# Le MLP utilise les mêmes features que le VGAE mais SANS message passing.
# Il sert à mesurer l'apport de la topologie du graphe.
# On l'entraîne sur la même tâche : reconstruire les arêtes gene↔gene
# à partir des features concaténées de chaque paire.
print("\n" + "=" * 70)
print("12. Baseline MLP (sans graphe)")
print("=" * 70)


class MLPBaseline(nn.Module):
    """MLP qui prédit l'existence d'une arête à partir des features concaténées."""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index):
        src, dst = edge_index
        h = torch.cat([x[src], x[dst]], dim=1)
        return self.net(h).squeeze(-1)


mlp = MLPBaseline(gene_features.shape[1], MLP_HIDDEN)
mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=MLP_LR)

mlp.train()
for epoch in range(MLP_EPOCHS):
    mlp_optimizer.zero_grad()

    # Positifs
    pos_scores = mlp(gene_features, pos_train_bidir)
    neg_edges = negative_sampling(pos_train_bidir, num_nodes=n_genes,
                                  num_neg_samples=pos_train_bidir.shape[1])
    neg_scores = mlp(gene_features, neg_edges)

    loss = (F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
            + F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores)))
    loss.backward()
    mlp_optimizer.step()

# Évaluation MLP
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

# =============================================================================
# 13. BASELINE STATISTIQUE (ranking par log2FC)
# =============================================================================
# Pour comparer avec une approche sans ML : on rank les gènes par leur
# |log2FC| moyen entre P4 et les clusters P16.
print("\n" + "=" * 70)
print("13. Baseline statistique (ranking expression)")
print("=" * 70)

# Calculer un proxy de log2FC à partir des données normalisées (sans DEGs)
p4_mean = group_stats["P4"]["mean_expression"]
p16_means = np.array([
    group_stats[f"P16_cluster_{c}"]["mean_expression"] for c in range(4)
])
p16_global_mean = p16_means.mean(axis=0)

# Pseudo log2FC (sur données LogNormalize, c'est une différence de logs)
pseudo_lfc = np.abs(p16_global_mean - p4_mean).astype(np.float32)
stat_score = pseudo_lfc / (pseudo_lfc.max() + 1e-8)

print(f"  Score statistique (|ΔExpr P16-P4|) : mean={stat_score.mean():.4f}")
top_stat = np.argsort(stat_score)[::-1][:10]
print(f"  Top 10 gènes (baseline stat) :")
for idx in top_stat:
    print(f"    {gene_symbols[idx]:15s} score={stat_score[idx]:.4f}")

# =============================================================================
# 14. VALIDATION POST-HOC (BDD externes)
# =============================================================================
print("\n" + "=" * 70)
print("14. Validation post-hoc (GenAge, CellAge, MSigDB, AgeAnno)")
print("   → PAS utilisées dans l'entraînement, uniquement pour évaluer")
print("=" * 70)

# ── Téléchargement des BDD ───────────────────────────────────────────────────
GENAGE_ZIP = os.path.join(DB_DIR, "genage_human.zip")
GENAGE_FILE = os.path.join(DB_DIR, "genage_human.csv")
download_if_absent(
    "https://genomics.senescence.info/genes/human_genes.zip",
    GENAGE_ZIP, "GenAge"
)
if not os.path.exists(GENAGE_FILE):
    with zipfile.ZipFile(GENAGE_ZIP, "r") as z_file:
        csv_names = [n for n in z_file.namelist() if n.endswith(".csv")]
        if csv_names:
            with z_file.open(csv_names[0]) as f:
                with open(GENAGE_FILE, "wb") as out:
                    out.write(f.read())

genage = pd.read_csv(GENAGE_FILE)
genage_symbols = set(genage["symbol"].dropna()) if "symbol" in genage.columns else set()

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
for name, genes_set in msigdb_sets.items():
    if any(kw in name.upper() for kw in AGING_KEYWORDS):
        msigdb_aging_genes |= genes_set

CELLAGE_ZIP = os.path.join(DB_DIR, "cellAge.zip")
CELLAGE_FILE = os.path.join(DB_DIR, "cellage3.tsv")
download_if_absent(
    "https://genomics.senescence.info/cells/cellAge.zip",
    CELLAGE_ZIP, "CellAge"
)
if not os.path.exists(CELLAGE_FILE):
    with zipfile.ZipFile(CELLAGE_ZIP, "r") as z_file:
        tsv_names = [n for n in z_file.namelist() if n.lower().endswith(('.tsv', '.csv'))]
        if tsv_names:
            with z_file.open(tsv_names[0]) as f:
                with open(CELLAGE_FILE, "wb") as out:
                    out.write(f.read())

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

AGEANNO_DIR = os.path.join(DB_DIR, "ageanno")
os.makedirs(AGEANNO_DIR, exist_ok=True)
AGEANNO_DEGS = os.path.join(AGEANNO_DIR, "aging_DEGs.txt")
download_if_absent(
    "https://raw.githubusercontent.com/vikkihuangkexin/AgeAnno/main/scRNA/Aging-related%20DEGs.txt",
    AGEANNO_DEGS, "AgeAnno DEGs"
)
ageanno_degs = pd.read_csv(AGEANNO_DEGS, sep=",", encoding="latin-1")
ageanno_genes = set(ageanno_degs["gene"].dropna().unique())

aging_local = pd.read_csv(os.path.join(DATA_DIR, "human_age_related_gene.csv"))
aging_local_symbols = set(aging_local["Symbol"].dropna())

# ── Évaluation : les gènes à haut score sont-ils enrichis dans les BDD ? ────
# Pour CHAQUE approche (VGAE, MLP-based, stat), on mesure l'enrichissement.
# On utilise le Mann-Whitney U test : les gènes dans la BDD ont-ils un
# score significativement plus élevé que les autres ?

def evaluate_ranking(scores, score_name, gene_syms, databases):
    """Évalue un ranking par enrichissement dans les BDD externes."""
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

databases = [
    ("GenAge", genage_symbols),
    ("CellAge", cellage_symbols),
    ("MSigDB aging", msigdb_aging_genes),
    ("AgeAnno", ageanno_genes),
    ("Aging local", aging_local_symbols),
]

vgae_results = evaluate_ranking(importance_score, "VGAE (non supervisé)",
                                 gene_symbols, databases)
stat_results = evaluate_ranking(stat_score, "Baseline statistique (|ΔExpr|)",
                                 gene_symbols, databases)

# =============================================================================
# 15. VISUALISATIONS
# =============================================================================
print("\n" + "=" * 70)
print("15. Visualisations")
print("=" * 70)

# ── 15a. Loss et AUC d'entraînement ─────────────────────────────────────────
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
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter VGAE vs stat
axes[0].scatter(stat_score, importance_score, s=5, alpha=0.3,
                c="#3498DB", rasterized=True)
# Marquer les gènes des BDD
for g_idx in range(n_genes):
    if any(gene_symbols[g_idx] in db for _, db in databases):
        axes[0].scatter(stat_score[g_idx], importance_score[g_idx],
                        s=20, c="#E74C3C", alpha=0.7, zorder=5)
axes[0].set_xlabel("Score statistique (|ΔExpr|)")
axes[0].set_ylabel("Score VGAE (émergent)")
axes[0].set_title("VGAE vs Baseline statistique")
axes[0].grid(True, alpha=0.3)

# Les gènes intéressants = haut VGAE mais bas statistique (quadrant haut-gauche)
high_vgae = importance_score > np.percentile(importance_score, 90)
low_stat = stat_score < np.percentile(stat_score, 50)
discovery_candidates = high_vgae & low_stat
n_disc = discovery_candidates.sum()
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
print("\n" + "=" * 70)
print("16. Export des résultats")
print("=" * 70)

results = pd.DataFrame({"gene": gene_symbols})

# Scores
results["vgae_importance"] = importance_score
results["vgae_emb_norm"] = emb_norm_score
results["vgae_density"] = density_score
results["vgae_recon_fidelity"] = 1 - recon_error_score
results["vgae_certainty"] = 1 - uncertainty_score
results["vgae_specificity"] = specificity_score
results["stat_score"] = stat_score
results["cluster"] = gene_clusters

# Rang
results["rank_vgae"] = results["vgae_importance"].rank(ascending=False).astype(int)
results["rank_stat"] = results["stat_score"].rank(ascending=False).astype(int)

# Candidats de découverte (haut VGAE, bas statistique)
results["discovery_candidate"] = discovery_candidates.astype(int)

# BDD
results["in_genage"] = [1 if g in genage_symbols else 0 for g in gene_symbols]
results["in_cellage"] = [1 if g in cellage_symbols else 0 for g in gene_symbols]
results["in_msigdb_aging"] = [1 if g in msigdb_aging_genes else 0 for g in gene_symbols]
results["in_ageanno"] = [1 if g in ageanno_genes else 0 for g in gene_symbols]
results["in_aging_local"] = [1 if g in aging_local_symbols else 0 for g in gene_symbols]
results["n_databases"] = results[["in_genage", "in_cellage", "in_msigdb_aging",
                                   "in_ageanno", "in_aging_local"]].sum(axis=1)

results = results.sort_values("vgae_importance", ascending=False)
results.to_csv(os.path.join(OUT_DIR, "gene_ranking_vgae.csv"), index=False)
print(f"  → gene_ranking_vgae.csv ({len(results)} gènes)")

# Embeddings
emb_df = pd.DataFrame(gene_emb, index=gene_symbols)
emb_df.to_csv(os.path.join(OUT_DIR, "gene_embeddings_vgae.csv"))
print(f"  → gene_embeddings_vgae.csv ({gene_emb.shape})")

# Modèle
torch.save(model.state_dict(), os.path.join(OUT_DIR, "vgae_weights.pt"))
print("  → vgae_weights.pt")

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
  MLP   AUC : {mlp_auc:.4f}
  Δ AUC     : {best_test_auc - mlp_auc:+.4f} ({'topologie utile' if best_test_auc > mlp_auc else 'topologie non utile'})

Candidats de découverte :
  (haut score VGAE + bas score statistique)
  {n_disc} gènes, dont {disc_in_db} dans les BDD de sénescence

Fichiers : {OUT_DIR}/
Figures  : {FIG_DIR}/
""")
