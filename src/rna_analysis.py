"""
Analyse RNA-seq : HUVEC prolifératives vs sénescentes (GSE98440)
================================================================
- Counts normalisés + analyse différentielle (DESeq2 pré-calculée)
- Figures : Volcano, MA plot, Heatmap, PCA, barplots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "RNAseq")
OUT_DIR = os.path.join(BASE_DIR, "..", "output", "rnaseq")
FIG_DIR = os.path.join(OUT_DIR, "figure")
print(FIG_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

COUNTS_FILE = os.path.join(DATA_DIR, "GSE98440_norm_counts_HUVECpro_sen.csv")
DE_FILE = os.path.join(DATA_DIR, "GSE98440_diff_expr_analysis_afterNorm_HUVEC_2reps.txt")

# ── Paramètres ───────────────────────────────────────────────────────────────
PADJ_THRESH = 0.05
LFC_THRESH = 1.0  # |log2FC| > 1

# ── Style global ─────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

# =============================================================================
# 1. Chargement des données
# =============================================================================
print("=" * 60)
print("1. Chargement des données")
print("=" * 60)

# Counts normalisés
counts = pd.read_csv(COUNTS_FILE, sep="\t", index_col=0)
print(f"  Counts : {counts.shape[0]} gènes × {counts.shape[1]} échantillons")
print(f"  Échantillons : {list(counts.columns)}")

# Analyse différentielle
de = pd.read_csv(DE_FILE, sep="\t")
# Nettoyer : supprimer les lignes spéciales (__ambiguous, __no_feature, etc.)
de = de[de["ensembl_gene_id"].str.startswith("ENSG")].copy()
de = de.dropna(subset=["padj", "log2FoldChange"])
de["neg_log10_padj"] = -np.log10(de["padj"].clip(lower=1e-300))
print(f"  DE table : {de.shape[0]} gènes avec padj non-NA")

# Classification des gènes
de["regulation"] = "NS"  # Not Significant
de.loc[(de["padj"] < PADJ_THRESH) & (de["log2FoldChange"] > LFC_THRESH), "regulation"] = "Up"
de.loc[(de["padj"] < PADJ_THRESH) & (de["log2FoldChange"] < -LFC_THRESH), "regulation"] = "Down"

n_up = (de["regulation"] == "Up").sum()
n_down = (de["regulation"] == "Down").sum()
n_ns = (de["regulation"] == "NS").sum()
print(f"\n  Gènes significatifs (padj < {PADJ_THRESH}, |log2FC| > {LFC_THRESH}) :")
print(f"    Up-régulés (sénescence)   : {n_up}")
print(f"    Down-régulés (sénescence) : {n_down}")
print(f"    Non significatifs         : {n_ns}")

# =============================================================================
# 2. Volcano Plot
# =============================================================================
print("\n" + "=" * 60)
print("2. Volcano Plot")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 8))

colors = {"NS": "#BBBBBB", "Up": "#E74C3C", "Down": "#3498DB"}
for reg in ["NS", "Up", "Down"]:
    subset = de[de["regulation"] == reg]
    ax.scatter(
        subset["log2FoldChange"],
        subset["neg_log10_padj"],
        c=colors[reg],
        s=8,
        alpha=0.6,
        label=f"{reg} ({len(subset)})",
        rasterized=True,
    )

# Annoter le top 15 gènes les plus significatifs (up + down)
top_genes = de[de["regulation"].isin(["Up", "Down"])].nlargest(15, "neg_log10_padj")
for _, row in top_genes.iterrows():
    label = row["hgnc_symbol"] if pd.notna(row["hgnc_symbol"]) else row["ensembl_gene_id"]
    ax.annotate(
        label,
        (row["log2FoldChange"], row["neg_log10_padj"]),
        fontsize=7,
        fontweight="bold",
        ha="center",
        va="bottom",
        xytext=(0, 5),
        textcoords="offset points",
    )

ax.axhline(-np.log10(PADJ_THRESH), ls="--", lw=0.8, color="grey")
ax.axvline(LFC_THRESH, ls="--", lw=0.8, color="grey")
ax.axvline(-LFC_THRESH, ls="--", lw=0.8, color="grey")
ax.set_xlabel("log2 Fold Change (Sénescent / Prolifératif)")
ax.set_ylabel("-log10(padj)")
ax.set_title("Volcano Plot — HUVEC Prolifératives vs Sénescentes")
ax.legend(loc="upper right", framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "volcano_plot.png"))
plt.close(fig)
print("  → Sauvé : volcano_plot.png")

# =============================================================================
# 3. MA Plot
# =============================================================================
print("\n" + "=" * 60)
print("3. MA Plot")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 7))

for reg in ["NS", "Up", "Down"]:
    subset = de[de["regulation"] == reg]
    ax.scatter(
        np.log10(subset["baseMean"].clip(lower=1)),
        subset["log2FoldChange"],
        c=colors[reg],
        s=6,
        alpha=0.5,
        label=f"{reg} ({len(subset)})",
        rasterized=True,
    )

ax.axhline(0, color="black", lw=0.5)
ax.axhline(LFC_THRESH, ls="--", lw=0.8, color="grey")
ax.axhline(-LFC_THRESH, ls="--", lw=0.8, color="grey")
ax.set_xlabel("log10(baseMean)")
ax.set_ylabel("log2 Fold Change")
ax.set_title("MA Plot — HUVEC Prolifératives vs Sénescentes")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "ma_plot.png"))
plt.close(fig)
print("  → Sauvé : ma_plot.png")

# =============================================================================
# 4. PCA sur les counts normalisés
# =============================================================================
print("\n" + "=" * 60)
print("4. PCA")
print("=" * 60)

# Log-transform pour la PCA
log_counts = np.log2(counts + 1)

pca = PCA(n_components=2)
pcs = pca.fit_transform(log_counts.T)  # échantillons en lignes

conditions = ["Prolifératif" if "pro" in c else "Sénescent" for c in counts.columns]
pca_df = pd.DataFrame({
    "PC1": pcs[:, 0],
    "PC2": pcs[:, 1],
    "Condition": conditions,
    "Sample": counts.columns,
})

fig, ax = plt.subplots(figsize=(8, 7))
cond_colors = {"Prolifératif": "#2ECC71", "Sénescent": "#E74C3C"}
for cond, color in cond_colors.items():
    sub = pca_df[pca_df["Condition"] == cond]
    ax.scatter(sub["PC1"], sub["PC2"], c=color, s=120, label=cond, edgecolors="black", zorder=3)
    for _, row in sub.iterrows():
        ax.annotate(row["Sample"], (row["PC1"], row["PC2"]),
                     fontsize=8, ha="left", va="bottom", xytext=(5, 5),
                     textcoords="offset points")

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("PCA — Counts normalisés (log2)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "pca.png"))
plt.close(fig)
print(f"  PC1 : {pca.explained_variance_ratio_[0]*100:.1f}% variance")
print(f"  PC2 : {pca.explained_variance_ratio_[1]*100:.1f}% variance")
print("  → Sauvé : pca.png")

# =============================================================================
# 5. Heatmap des top gènes DE
# =============================================================================
print("\n" + "=" * 60)
print("5. Heatmap des top gènes DE")
print("=" * 60)

N_TOP = 50
sig_genes = de[de["regulation"].isin(["Up", "Down"])].copy()
top_de = sig_genes.reindex(sig_genes["neg_log10_padj"].abs().nlargest(N_TOP).index)

# Mapper ensembl → symbol
gene_map = dict(zip(de["ensembl_gene_id"], de["hgnc_symbol"]))

# Filtrer ceux présents dans la matrice de counts
top_ids = [g for g in top_de["ensembl_gene_id"] if g in counts.index]
heatmap_data = log_counts.loc[top_ids]

# Renommer avec symboles
heatmap_data.index = [
    gene_map.get(g, g) if pd.notna(gene_map.get(g)) else g
    for g in heatmap_data.index
]

# Z-score par gène
heatmap_z = heatmap_data.subtract(heatmap_data.mean(axis=1), axis=0).divide(
    heatmap_data.std(axis=1), axis=0
)

fig = plt.figure(figsize=(8, 14))
cmap = LinearSegmentedColormap.from_list("bwr", ["#3498DB", "white", "#E74C3C"])
g = sns.clustermap(
    heatmap_z,
    cmap=cmap,
    center=0,
    vmin=-2.5,
    vmax=2.5,
    row_cluster=True,
    col_cluster=True,
    yticklabels=True,
    xticklabels=True,
    figsize=(8, 14),
    dendrogram_ratio=(0.1, 0.05),
    cbar_kws={"label": "Z-score"},
    linewidths=0.3,
)
g.fig.suptitle(f"Top {N_TOP} gènes DE — Z-score (log2 counts)", y=1.01, fontsize=14)
g.savefig(os.path.join(FIG_DIR, "heatmap_top_DE.png"), bbox_inches="tight")
plt.close("all")
print(f"  → Sauvé : heatmap_top_DE.png ({len(top_ids)} gènes)")

# =============================================================================
# 6. Distribution des log2FC
# =============================================================================
print("\n" + "=" * 60)
print("6. Distribution des log2FC")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogramme global
axes[0].hist(de["log2FoldChange"], bins=100, color="#7F8C8D", edgecolor="white", lw=0.3)
axes[0].axvline(0, color="black", lw=0.8)
axes[0].axvline(LFC_THRESH, ls="--", color="#E74C3C", lw=0.8, label=f"LFC = ±{LFC_THRESH}")
axes[0].axvline(-LFC_THRESH, ls="--", color="#3498DB", lw=0.8)
axes[0].set_xlabel("log2 Fold Change")
axes[0].set_ylabel("Nombre de gènes")
axes[0].set_title("Distribution des log2FC (tous les gènes)")
axes[0].legend()

# Histogramme p-values
axes[1].hist(de["pvalue"], bins=50, color="#9B59B6", edgecolor="white", lw=0.3)
axes[1].axvline(0.05, ls="--", color="red", lw=1, label="p = 0.05")
axes[1].set_xlabel("p-value")
axes[1].set_ylabel("Nombre de gènes")
axes[1].set_title("Distribution des p-values")
axes[1].legend()

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "distributions.png"))
plt.close(fig)
print("  → Sauvé : distributions.png")

# =============================================================================
# 7. Barplot des top gènes Up et Down
# =============================================================================
print("\n" + "=" * 60)
print("7. Barplot top gènes Up/Down")
print("=" * 60)

N_BAR = 20

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

for i, (direction, color, title) in enumerate([
    ("Up", "#E74C3C", "Top Up-régulés (sénescence)"),
    ("Down", "#3498DB", "Top Down-régulés (sénescence)"),
]):
    sub = de[de["regulation"] == direction].copy()
    sub["abs_lfc"] = sub["log2FoldChange"].abs()
    top = sub.nlargest(N_BAR, "abs_lfc")
    top["label"] = top.apply(
        lambda r: r["hgnc_symbol"] if pd.notna(r["hgnc_symbol"]) else r["ensembl_gene_id"],
        axis=1,
    )
    top = top.sort_values("log2FoldChange", ascending=(direction == "Up"))

    axes[i].barh(top["label"], top["log2FoldChange"], color=color, edgecolor="white")
    axes[i].set_xlabel("log2 Fold Change")
    axes[i].set_title(title)
    axes[i].axvline(0, color="black", lw=0.5)

fig.suptitle(f"Top {N_BAR} gènes par direction (padj < {PADJ_THRESH}, |LFC| > {LFC_THRESH})",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "barplot_top_genes.png"), bbox_inches="tight")
plt.close(fig)
print("  → Sauvé : barplot_top_genes.png")

# =============================================================================
# 8. Corrélation entre échantillons
# =============================================================================
print("\n" + "=" * 60)
print("8. Corrélation inter-échantillons")
print("=" * 60)

corr = log_counts.corr(method="pearson")

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(
    corr,
    annot=True,
    fmt=".3f",
    cmap="YlOrRd",
    vmin=corr.values.min(),
    vmax=1,
    square=True,
    ax=ax,
    linewidths=0.5,
)
ax.set_title("Corrélation de Pearson (log2 counts)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"))
plt.close(fig)
print("  → Sauvé : correlation_heatmap.png")

# =============================================================================
# 9. Tables récapitulatives
# =============================================================================
print("\n" + "=" * 60)
print("9. Tables récapitulatives")
print("=" * 60)

# Top 30 gènes les plus significatifs
top30 = de.nlargest(30, "neg_log10_padj")[
    ["hgnc_symbol", "ensembl_gene_id", "log2FoldChange", "padj", "baseMean", "regulation"]
].copy()
top30["padj"] = top30["padj"].apply(lambda x: f"{x:.2e}")
top30.to_csv(os.path.join(OUT_DIR, "top30_DE_genes.csv"), index=False)
print("  → Sauvé : top30_DE_genes.csv")

# Tous les gènes significatifs
all_sig = de[de["regulation"].isin(["Up", "Down"])][
    ["hgnc_symbol", "ensembl_gene_id", "log2FoldChange", "padj", "baseMean", "regulation"]
].sort_values("padj")
all_sig.to_csv(os.path.join(OUT_DIR, "all_significant_DE_genes.csv"), index=False)
print(f"  → Sauvé : all_significant_DE_genes.csv ({len(all_sig)} gènes)")

# =============================================================================
# 10. Résumé final
# =============================================================================
print("\n" + "=" * 60)
print("RÉSUMÉ DE L'ANALYSE")
print("=" * 60)
print(f"""
Dataset : GSE98440 — HUVEC prolifératives vs sénescentes
Échantillons : 3 prolifératifs + 3 sénescents (6 total)
Gènes analysés : {de.shape[0]}

Seuils : padj < {PADJ_THRESH}, |log2FC| > {LFC_THRESH}
  Up-régulés (sénescence)   : {n_up}
  Down-régulés (sénescence) : {n_down}
  Non significatifs         : {n_ns}

Top 5 Up-régulés :
{de[de['regulation']=='Up'].nlargest(5, 'neg_log10_padj')[['hgnc_symbol','log2FoldChange','padj']].to_string(index=False)}

Top 5 Down-régulés :
{de[de['regulation']=='Down'].nlargest(5, 'neg_log10_padj')[['hgnc_symbol','log2FoldChange','padj']].to_string(index=False)}

Figures sauvées dans : {FIG_DIR}/
""")
