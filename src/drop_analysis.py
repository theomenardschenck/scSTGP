"""
Analyse DROP-seq : HUVEC prolifératives (D2P4) vs sénescentes (D2P16)
=====================================================================
GSE102090 — Matrices 10X (MTX + barcodes + genes)
- QC cellulaire (gènes/cellule, UMI, % mito)
- Filtrage et normalisation
- Gènes hautement variables (HVG)
- PCA + t-SNE
- Clustering (Leiden-like via agglomératif)
- Gènes marqueurs par cluster
- Comparaison prolifératif vs sénescent
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import mmread
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "DROPseq")
OUT_DIR = os.path.join(BASE_DIR, "..", "output", "dropseq")
FIG_DIR = os.path.join(OUT_DIR, "figure")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

COLORS_COND = {"D2P4 (Prolif.)": "#2ECC71", "D2P16 (Sénesc.)": "#E74C3C"}

# =============================================================================
# 1. Chargement des données
# =============================================================================
print("=" * 60)
print("1. Chargement des données")
print("=" * 60)

# Genes (communs aux deux échantillons)
genes = pd.read_csv(os.path.join(DATA_DIR, "GSE102090_genes.tsv"),
                     sep="\t", header=None, names=["ensembl_id", "symbol"])
print(f"  Gènes : {len(genes)}")

# Charger chaque échantillon
samples = {
    "D2P4 (Prolif.)": {
        "matrix": os.path.join(DATA_DIR, "GSM2723761_D2P4_matrix.mtx"),
        "barcodes": os.path.join(DATA_DIR, "GSM2723761_D2P4_barcodes.tsv"),
    },
    "D2P16 (Sénesc.)": {
        "matrix": os.path.join(DATA_DIR, "GSM2723762_D2P16_matrix.mtx"),
        "barcodes": os.path.join(DATA_DIR, "GSM2723762_D2P16_barcodes.tsv"),
    },
}

matrices = []
barcodes_all = []
conditions = []

for name, paths in samples.items():
    print(f"  Chargement {name}...")
    mat = mmread(paths["matrix"]).T.tocsr()  # cellules × gènes
    bc = pd.read_csv(paths["barcodes"], sep="\t", header=None, names=["barcode"])
    print(f"    {mat.shape[0]} cellules × {mat.shape[1]} gènes")
    matrices.append(mat)
    barcodes_all.append(bc["barcode"].values)
    conditions.extend([name] * mat.shape[0])

# Concaténer
X = hstack(matrices, format="csr") if matrices[0].shape[1] != matrices[1].shape[1] else csr_matrix(
    np.vstack([m.toarray() for m in matrices])
)
# Les deux matrices ont le même nombre de gènes, empilons verticalement
X = csr_matrix(np.vstack([m.toarray() for m in matrices]))
barcodes = np.concatenate(barcodes_all)
conditions = np.array(conditions)
gene_names = genes["symbol"].values
gene_ids = genes["ensembl_id"].values

print(f"\n  Matrice combinée : {X.shape[0]} cellules × {X.shape[1]} gènes")

# =============================================================================
# 2. QC — Métriques par cellule
# =============================================================================
print("\n" + "=" * 60)
print("2. Contrôle qualité (QC)")
print("=" * 60)

# Convertir en dense pour les calculs QC
X_dense = X.toarray().astype(np.float32)

n_genes_per_cell = (X_dense > 0).sum(axis=1)
total_counts = X_dense.sum(axis=1)

# Gènes mitochondriaux
mito_mask = np.array([g.startswith("MT-") for g in gene_names])
n_mito = mito_mask.sum()
print(f"  Gènes mitochondriaux détectés : {n_mito}")
if total_counts.sum() > 0:
    pct_mito = X_dense[:, mito_mask].sum(axis=1) / total_counts * 100
else:
    pct_mito = np.zeros(X_dense.shape[0])

# DataFrame QC
qc = pd.DataFrame({
    "n_genes": n_genes_per_cell,
    "total_counts": total_counts,
    "pct_mito": pct_mito,
    "condition": conditions,
})

print(f"\n  Avant filtrage : {len(qc)} cellules")
for cond in COLORS_COND:
    sub = qc[qc["condition"] == cond]
    print(f"    {cond}: {len(sub)} cellules, "
          f"gènes/cellule médian = {sub['n_genes'].median():.0f}, "
          f"UMI médian = {sub['total_counts'].median():.0f}, "
          f"% mito médian = {sub['pct_mito'].median():.1f}%")

# ── Figure QC ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for metric, ax, title, ylabel in [
    ("n_genes", axes[0], "Gènes détectés par cellule", "Nombre de gènes"),
    ("total_counts", axes[1], "UMI totaux par cellule", "Total UMI"),
    ("pct_mito", axes[2], "% reads mitochondriaux", "% mito"),
]:
    sns.violinplot(data=qc, x="condition", y=metric, hue="condition",
                   palette=COLORS_COND, ax=ax, inner="quartile", cut=0, legend=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

fig.suptitle("Contrôle qualité par cellule", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "qc_violin.png"), bbox_inches="tight")
plt.close(fig)
print("\n  → Sauvé : qc_violin.png")

# ── Figure QC scatter ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for i, cond in enumerate(COLORS_COND):
    mask = qc["condition"] == cond
    axes[i].scatter(qc.loc[mask, "total_counts"], qc.loc[mask, "n_genes"],
                    c=COLORS_COND[cond], s=3, alpha=0.3, rasterized=True)
    axes[i].set_xlabel("Total UMI")
    axes[i].set_ylabel("Nombre de gènes")
    axes[i].set_title(cond)

fig.suptitle("UMI vs Gènes détectés", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "qc_scatter.png"), bbox_inches="tight")
plt.close(fig)
print("  → Sauvé : qc_scatter.png")

# =============================================================================
# 3. Filtrage
# =============================================================================
print("\n" + "=" * 60)
print("3. Filtrage des cellules et gènes")
print("=" * 60)

MIN_GENES = 200
MAX_GENES = 6000
MAX_MITO = 20
MIN_CELLS_PER_GENE = 10

# Filtrage cellules
cell_mask = (n_genes_per_cell >= MIN_GENES) & (n_genes_per_cell <= MAX_GENES) & (pct_mito <= MAX_MITO)
X_filt = X_dense[cell_mask]
conditions_filt = conditions[cell_mask]
barcodes_filt = barcodes[cell_mask]

print(f"  Cellules retenues : {cell_mask.sum()} / {len(cell_mask)}")
print(f"    Filtrées (< {MIN_GENES} gènes) : {(n_genes_per_cell < MIN_GENES).sum()}")
print(f"    Filtrées (> {MAX_GENES} gènes) : {(n_genes_per_cell > MAX_GENES).sum()}")
print(f"    Filtrées (> {MAX_MITO}% mito) : {(pct_mito > MAX_MITO).sum()}")

# Filtrage gènes
gene_cell_counts = (X_filt > 0).sum(axis=0)
gene_mask = gene_cell_counts >= MIN_CELLS_PER_GENE
X_filt = X_filt[:, gene_mask]
gene_names_filt = gene_names[gene_mask]
gene_ids_filt = gene_ids[gene_mask]

print(f"  Gènes retenus : {gene_mask.sum()} / {len(gene_mask)} (exprimés dans ≥ {MIN_CELLS_PER_GENE} cellules)")
print(f"  Matrice filtrée : {X_filt.shape[0]} cellules × {X_filt.shape[1]} gènes")

# Libérer mémoire
del X_dense, X

# =============================================================================
# 4. Normalisation et log-transform
# =============================================================================
print("\n" + "=" * 60)
print("4. Normalisation")
print("=" * 60)

# Normalisation par taille de librairie (CPM-like, target = médiane)
cell_totals = X_filt.sum(axis=1, keepdims=True)
median_total = np.median(cell_totals)
X_norm = X_filt / cell_totals * median_total

# Log1p
X_log = np.log1p(X_norm)

print(f"  Facteur de normalisation (médiane) : {median_total:.0f}")
print(f"  Matrice normalisée : {X_log.shape}")

# =============================================================================
# 5. Gènes hautement variables (HVG)
# =============================================================================
print("\n" + "=" * 60)
print("5. Sélection des gènes hautement variables (HVG)")
print("=" * 60)

gene_means = X_log.mean(axis=0)
gene_vars = X_log.var(axis=0)
# Dispersion = var / mean (coefficient de variation au carré pour log data)
gene_cv2 = np.where(gene_means > 0, gene_vars / gene_means, 0)

# Sélectionner le top N par variance
N_HVG = 2000
hvg_indices = np.argsort(gene_vars)[::-1][:N_HVG]
hvg_mask = np.zeros(len(gene_names_filt), dtype=bool)
hvg_mask[hvg_indices] = True

print(f"  {N_HVG} HVG sélectionnés (top variance)")

# Figure HVG
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(gene_means[~hvg_mask], gene_vars[~hvg_mask], s=2, alpha=0.3, c="#BBBBBB",
           label=f"Autres ({(~hvg_mask).sum()})", rasterized=True)
ax.scatter(gene_means[hvg_mask], gene_vars[hvg_mask], s=4, alpha=0.5, c="#E74C3C",
           label=f"HVG ({hvg_mask.sum()})", rasterized=True)
ax.set_xlabel("Expression moyenne (log)")
ax.set_ylabel("Variance (log)")
ax.set_title("Sélection des gènes hautement variables")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "hvg_selection.png"))
plt.close(fig)
print("  → Sauvé : hvg_selection.png")

# =============================================================================
# 6. PCA
# =============================================================================
print("\n" + "=" * 60)
print("6. PCA")
print("=" * 60)

X_hvg = X_log[:, hvg_mask]
X_scaled = StandardScaler().fit_transform(X_hvg)

N_PCS = 30
pca = PCA(n_components=N_PCS, random_state=42)
X_pca = pca.fit_transform(X_scaled)

cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
print(f"  {N_PCS} PC calculées")
print(f"  Variance cumulée (10 PC) : {cumvar[9]:.1f}%")
print(f"  Variance cumulée (20 PC) : {cumvar[19]:.1f}%")
print(f"  Variance cumulée (30 PC) : {cumvar[29]:.1f}%")

# Elbow plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(range(1, N_PCS + 1), pca.explained_variance_ratio_ * 100, color="#3498DB", edgecolor="white")
axes[0].set_xlabel("Composante principale")
axes[0].set_ylabel("% variance expliquée")
axes[0].set_title("Scree plot")

axes[1].plot(range(1, N_PCS + 1), cumvar, "o-", color="#E74C3C")
axes[1].axhline(80, ls="--", color="grey", lw=0.8)
axes[1].set_xlabel("Nombre de PC")
axes[1].set_ylabel("% variance cumulée")
axes[1].set_title("Variance cumulée")

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "pca_elbow.png"))
plt.close(fig)
print("  → Sauvé : pca_elbow.png")

# PCA scatter coloré par condition
fig, ax = plt.subplots(figsize=(9, 7))
for cond, color in COLORS_COND.items():
    mask = conditions_filt == cond
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, s=5, alpha=0.4,
               label=cond, rasterized=True)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA — DROP-seq HUVEC")
ax.legend(markerscale=3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "pca_condition.png"))
plt.close(fig)
print("  → Sauvé : pca_condition.png")

# =============================================================================
# 7. t-SNE
# =============================================================================
print("\n" + "=" * 60)
print("7. t-SNE")
print("=" * 60)

N_PCS_TSNE = 15
print(f"  Calcul t-SNE sur {N_PCS_TSNE} PC (perplexity=30)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000, init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(X_pca[:, :N_PCS_TSNE])
print("  t-SNE terminé")

# t-SNE coloré par condition
fig, ax = plt.subplots(figsize=(9, 7))
for cond, color in COLORS_COND.items():
    mask = conditions_filt == cond
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, s=5, alpha=0.4,
               label=cond, rasterized=True)

ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title("t-SNE — DROP-seq HUVEC")
ax.legend(markerscale=3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "tsne_condition.png"))
plt.close(fig)
print("  → Sauvé : tsne_condition.png")

# =============================================================================
# 8. Clustering
# =============================================================================
print("\n" + "=" * 60)
print("8. Clustering (agglomératif sur graphe KNN)")
print("=" * 60)

N_CLUSTERS = 8
connectivity = kneighbors_graph(X_pca[:, :N_PCS_TSNE], n_neighbors=15, include_self=False)
clustering = AgglomerativeClustering(
    n_clusters=N_CLUSTERS, connectivity=connectivity, linkage="ward"
)
labels = clustering.fit_predict(X_pca[:, :N_PCS_TSNE])
print(f"  {N_CLUSTERS} clusters identifiés")

for c in range(N_CLUSTERS):
    n_c = (labels == c).sum()
    n_prolif = ((labels == c) & (conditions_filt == "D2P4 (Prolif.)")).sum()
    n_sen = ((labels == c) & (conditions_filt == "D2P16 (Sénesc.)")).sum()
    print(f"    Cluster {c}: {n_c} cellules ({n_prolif} prolif / {n_sen} sénesc)")

# t-SNE coloré par cluster
cluster_palette = sns.color_palette("tab10", N_CLUSTERS)
fig, axes = plt.subplots(1, 2, figsize=(17, 7))

# Par cluster
for c in range(N_CLUSTERS):
    mask = labels == c
    axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[cluster_palette[c]], s=5, alpha=0.4,
                    label=f"Cluster {c}", rasterized=True)
axes[0].set_xlabel("t-SNE 1")
axes[0].set_ylabel("t-SNE 2")
axes[0].set_title("t-SNE — Clusters")
axes[0].legend(markerscale=3, fontsize=8)

# Par condition
for cond, color in COLORS_COND.items():
    mask = conditions_filt == cond
    axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, s=5, alpha=0.4,
                    label=cond, rasterized=True)
axes[1].set_xlabel("t-SNE 1")
axes[1].set_ylabel("t-SNE 2")
axes[1].set_title("t-SNE — Condition")
axes[1].legend(markerscale=3)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "tsne_clusters.png"))
plt.close(fig)
print("  → Sauvé : tsne_clusters.png")

# Composition des clusters
fig, ax = plt.subplots(figsize=(10, 5))
cluster_comp = pd.DataFrame({"cluster": labels, "condition": conditions_filt})
comp_table = cluster_comp.groupby(["cluster", "condition"]).size().unstack(fill_value=0)
comp_table.plot(kind="bar", stacked=True, color=[COLORS_COND.get(c, "grey") for c in comp_table.columns],
                ax=ax, edgecolor="white", lw=0.5)
ax.set_xlabel("Cluster")
ax.set_ylabel("Nombre de cellules")
ax.set_title("Composition des clusters par condition")
ax.legend(title="Condition")
plt.xticks(rotation=0)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "cluster_composition.png"))
plt.close(fig)
print("  → Sauvé : cluster_composition.png")

# =============================================================================
# 9. Gènes marqueurs par cluster (Wilcoxon rank-sum)
# =============================================================================
print("\n" + "=" * 60)
print("9. Gènes marqueurs par cluster")
print("=" * 60)

N_TOP_MARKERS = 10
all_markers = []

for c in range(N_CLUSTERS):
    in_cluster = labels == c
    out_cluster = ~in_cluster

    # Calculer le test pour chaque gène
    pvals = []
    logfcs = []
    for g in range(X_log.shape[1]):
        vals_in = X_log[in_cluster, g]
        vals_out = X_log[out_cluster, g]
        mean_in = vals_in.mean()
        mean_out = vals_out.mean()
        lfc = mean_in - mean_out  # déjà en log
        if vals_in.std() == 0 and vals_out.std() == 0:
            pvals.append(1.0)
        else:
            _, pval = mannwhitneyu(vals_in, vals_out, alternative="two-sided")
            pvals.append(pval)
        logfcs.append(lfc)

    marker_df = pd.DataFrame({
        "gene": gene_names_filt,
        "ensembl_id": gene_ids_filt,
        "log2FC": logfcs,
        "pvalue": pvals,
        "cluster": c,
    })
    marker_df = marker_df.sort_values("pvalue")
    top_up = marker_df[marker_df["log2FC"] > 0].head(N_TOP_MARKERS)
    all_markers.append(top_up)
    print(f"  Cluster {c} — top marqueurs : {', '.join(top_up['gene'].head(5).tolist())}")

markers_df = pd.concat(all_markers, ignore_index=True)
markers_df.to_csv(os.path.join(OUT_DIR, "cluster_markers.csv"), index=False)
print(f"\n  → Sauvé : cluster_markers.csv ({len(markers_df)} marqueurs)")

# =============================================================================
# 10. Heatmap des marqueurs
# =============================================================================
print("\n" + "=" * 60)
print("10. Heatmap des top marqueurs")
print("=" * 60)

# Top 5 par cluster
top5_per_cluster = markers_df.groupby("cluster").head(5)
marker_genes = top5_per_cluster["gene"].unique()
marker_gene_idx = [np.where(gene_names_filt == g)[0][0] for g in marker_genes if g in gene_names_filt]

# Ordonner les cellules par cluster
cell_order = np.argsort(labels)
heatmap_data = X_log[cell_order][:, marker_gene_idx]

# Sous-échantillonner pour la visualisation (max 200 cellules par cluster)
sampled_idx = []
for c in range(N_CLUSTERS):
    cluster_cells = np.where(labels[cell_order] == c)[0]
    if len(cluster_cells) > 200:
        sampled_idx.extend(np.random.choice(cluster_cells, 200, replace=False).tolist())
    else:
        sampled_idx.extend(cluster_cells.tolist())
sampled_idx = sorted(sampled_idx)

heatmap_sub = heatmap_data[sampled_idx]
labels_sub = labels[cell_order][sampled_idx]

# Z-score par gène
heatmap_z = (heatmap_sub - heatmap_sub.mean(axis=0)) / (heatmap_sub.std(axis=0) + 1e-8)
heatmap_z = np.clip(heatmap_z, -3, 3)

fig, ax = plt.subplots(figsize=(16, 8))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("bwr", ["#3498DB", "white", "#E74C3C"])
im = ax.imshow(heatmap_z.T, aspect="auto", cmap=cmap, vmin=-3, vmax=3, interpolation="none")

# Gène labels
ax.set_yticks(range(len(marker_genes)))
gene_labels = [g for g in marker_genes if g in gene_names_filt]
ax.set_yticklabels(gene_labels, fontsize=7)

# Séparer les clusters visuellement
cluster_boundaries = []
for c in range(N_CLUSTERS):
    cluster_cells = np.where(labels_sub == c)[0]
    if len(cluster_cells) > 0:
        cluster_boundaries.append(cluster_cells[-1])
        mid = cluster_cells[len(cluster_cells) // 2]
        ax.text(mid, -1.5, str(c), ha="center", fontsize=9, fontweight="bold")
for b in cluster_boundaries[:-1]:
    ax.axvline(b + 0.5, color="black", lw=0.5)

ax.set_xlabel("Cellules (ordonnées par cluster)")
ax.set_ylabel("Gènes marqueurs")
ax.set_title("Top 5 gènes marqueurs par cluster (Z-score)")
plt.colorbar(im, ax=ax, label="Z-score", shrink=0.6)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "marker_heatmap.png"), bbox_inches="tight")
plt.close(fig)
print("  → Sauvé : marker_heatmap.png")

# =============================================================================
# 11. Expression de gènes d'intérêt (sénescence)
# =============================================================================
print("\n" + "=" * 60)
print("11. Gènes d'intérêt — marqueurs de sénescence")
print("=" * 60)

SENESCENCE_GENES = ["CDKN2A", "CDKN1A", "TP53", "RB1", "LMNB1", "IL6", "IL8",
                    "CXCL8", "MKI67", "PCNA", "CCNA2", "CCNB1", "CDK1", "CDK2",
                    "GLB1", "SERPINE1", "IGFBP7", "TGFB1"]

found_genes = [g for g in SENESCENCE_GENES if g in gene_names_filt]
not_found = [g for g in SENESCENCE_GENES if g not in gene_names_filt]
if not_found:
    print(f"  Gènes non trouvés : {', '.join(not_found)}")
print(f"  Gènes trouvés : {len(found_genes)}")

n_cols = 4
n_rows = (len(found_genes) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
axes = axes.flatten()

for i, gene in enumerate(found_genes):
    g_idx = np.where(gene_names_filt == gene)[0][0]
    expr = X_log[:, g_idx]
    sc = axes[i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=expr, s=3, alpha=0.5,
                         cmap="viridis", rasterized=True)
    axes[i].set_title(gene, fontsize=11, fontweight="bold")
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    plt.colorbar(sc, ax=axes[i], shrink=0.6)

# Masquer les axes vides
for j in range(len(found_genes), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Expression de marqueurs de sénescence / prolifération (t-SNE)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "senescence_markers_tsne.png"), bbox_inches="tight")
plt.close(fig)
print("  → Sauvé : senescence_markers_tsne.png")

# =============================================================================
# 12. Gènes différentiellement exprimés entre conditions
# =============================================================================
print("\n" + "=" * 60)
print("12. Expression différentielle Prolif. vs Sénesc.")
print("=" * 60)

prolif_mask = conditions_filt == "D2P4 (Prolif.)"
sen_mask = conditions_filt == "D2P16 (Sénesc.)"

de_results = []
for g in range(X_log.shape[1]):
    vals_p = X_log[prolif_mask, g]
    vals_s = X_log[sen_mask, g]
    lfc = vals_s.mean() - vals_p.mean()
    if vals_p.std() == 0 and vals_s.std() == 0:
        pval = 1.0
    else:
        _, pval = mannwhitneyu(vals_p, vals_s, alternative="two-sided")
    pct_p = (vals_p > 0).mean() * 100
    pct_s = (vals_s > 0).mean() * 100
    de_results.append({
        "gene": gene_names_filt[g],
        "ensembl_id": gene_ids_filt[g],
        "log2FC": lfc,
        "pvalue": pval,
        "pct_prolif": pct_p,
        "pct_sen": pct_s,
    })

de_df = pd.DataFrame(de_results)
# Correction BH
from scipy.stats import false_discovery_control
try:
    de_df["padj"] = false_discovery_control(de_df["pvalue"], method="bh")
except AttributeError:
    # Fallback manual BH
    n = len(de_df)
    sorted_idx = np.argsort(de_df["pvalue"].values)
    padj = np.ones(n)
    pvals_sorted = de_df["pvalue"].values[sorted_idx]
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            padj[sorted_idx[i]] = pvals_sorted[i]
        else:
            padj[sorted_idx[i]] = min(padj[sorted_idx[i + 1]], pvals_sorted[i] * n / (i + 1))
    padj = np.clip(padj, 0, 1)
    de_df["padj"] = padj

de_df["neg_log10_padj"] = -np.log10(de_df["padj"].clip(lower=1e-300))

# Classification
de_df["regulation"] = "NS"
de_df.loc[(de_df["padj"] < 0.05) & (de_df["log2FC"] > 0.5), "regulation"] = "Up (Sénesc.)"
de_df.loc[(de_df["padj"] < 0.05) & (de_df["log2FC"] < -0.5), "regulation"] = "Down (Sénesc.)"

n_up = (de_df["regulation"] == "Up (Sénesc.)").sum()
n_down = (de_df["regulation"] == "Down (Sénesc.)").sum()
print(f"  Seuils : padj < 0.05, |log2FC| > 0.5")
print(f"  Up (sénescence) : {n_up}")
print(f"  Down (sénescence) : {n_down}")

# Volcano plot
fig, ax = plt.subplots(figsize=(10, 8))
colors_de = {"NS": "#BBBBBB", "Up (Sénesc.)": "#E74C3C", "Down (Sénesc.)": "#3498DB"}

for reg, color in colors_de.items():
    sub = de_df[de_df["regulation"] == reg]
    ax.scatter(sub["log2FC"], sub["neg_log10_padj"], c=color, s=5, alpha=0.5,
               label=f"{reg} ({len(sub)})", rasterized=True)

# Annoter les top
top_de = de_df[de_df["regulation"] != "NS"].nlargest(15, "neg_log10_padj")
for _, row in top_de.iterrows():
    ax.annotate(row["gene"], (row["log2FC"], row["neg_log10_padj"]),
                fontsize=7, fontweight="bold", ha="center", va="bottom",
                xytext=(0, 4), textcoords="offset points")

ax.axhline(-np.log10(0.05), ls="--", color="grey", lw=0.8)
ax.axvline(0.5, ls="--", color="grey", lw=0.8)
ax.axvline(-0.5, ls="--", color="grey", lw=0.8)
ax.set_xlabel("log2 Fold Change (Sénescent / Prolifératif)")
ax.set_ylabel("-log10(padj)")
ax.set_title("Volcano Plot — DROP-seq Prolifératif vs Sénescent")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "volcano_dropseq.png"))
plt.close(fig)
print("  → Sauvé : volcano_dropseq.png")

# Export
de_sig = de_df[de_df["regulation"] != "NS"].sort_values("padj")
de_sig.to_csv(os.path.join(OUT_DIR, "DE_genes_prolif_vs_sen.csv"), index=False)
de_df.to_csv(os.path.join(OUT_DIR, "DE_all_genes.csv"), index=False)
print(f"  → Sauvé : DE_genes_prolif_vs_sen.csv ({len(de_sig)} gènes)")
print(f"  → Sauvé : DE_all_genes.csv ({len(de_df)} gènes)")

# =============================================================================
# 13. Résumé final
# =============================================================================
print("\n" + "=" * 60)
print("RÉSUMÉ DE L'ANALYSE DROP-seq")
print("=" * 60)
print(f"""
Dataset : GSE102090 — DROP-seq HUVEC
Échantillons :
  D2P4 (Prolifératif, passage 4)  : {(conditions_filt == 'D2P4 (Prolif.)').sum()} cellules
  D2P16 (Sénescent, passage 16)   : {(conditions_filt == 'D2P16 (Sénesc.)').sum()} cellules
  Total après filtrage             : {len(conditions_filt)} cellules
  Gènes retenus                   : {X_log.shape[1]}

Clustering : {N_CLUSTERS} clusters (agglomératif, Ward)
HVG : {N_HVG} gènes

Expression différentielle (padj < 0.05, |log2FC| > 0.5) :
  Up (sénescence)   : {n_up}
  Down (sénescence) : {n_down}

Figures sauvées dans : {FIG_DIR}/
Tables sauvées dans  : {OUT_DIR}/
""")
