#!/usr/bin/env python3
"""
viz_explorer.py — Visualisations exploratoires "défense-ready" pour le
ranking VGAE post-perturbation.

Contrairement à `visualize_global.py` (UMAP, network, cross-version),
ce module se concentre sur des **visualisations scientifiques** des
résultats : heatmaps drivers×clusters, distributions par evidence_tier,
bubble plots aging DBs, radar plots cluster-spécifiques.

Pensé pour la présentation/publication : tous les outputs sont aussi
exportés en SVG (vectoriel, retouchable en post-prod) en plus du PNG.

Sous-commandes
--------------
    heatmap-clusters    Heatmap top-N drivers × cell_groups (cosine signé).
    tier-distributions  Distribution driver_score / |diff| / cosine / DE
                        par evidence_tier (A_confirmed … E_noise).
    aging-bubbles       Bubble plot top-N × aging DBs (taille = driver_score,
                        couleur = canon_cosine signé).
    radar               Radar/polar plot top-N drivers (5 axes : P4, c0..c3).
                        Révèle les drivers cluster-spécifiques.
    lollipop-clusters   Top-K drivers cluster-spécifiques par cluster
                        sénescent (c1, c2, c3).
    all                 Toutes les figures avec defaults.

Usage typique
-------------
    python src/validation/viz_explorer.py all \\
        --version-dir output/gnn_vgae/V4.1/full+no-humess/cross_seed_v4.1-full+no-humess_axisV4 \\
        --raw-runs    output/gnn_vgae/V4.1/full+no-humess/v4.1-full+no-humess.s* \\
        --out-dir     output/gnn_vgae/V4.1/full+no-humess/figures_explorer

Outputs PNG @ 200 dpi + SVG vectoriel dans <out-dir>/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Style global
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

# Palette evidence_tier — consistante à travers toutes les figures
TIER_COLORS = {
    "A_confirmed":  "#2ca02c",   # vert
    "B_discovery":  "#1f77b4",   # bleu
    "C_effector":   "#ff7f0e",   # orange
    "D_hub":        "#d62728",   # rouge
    "E_noise":      "#999999",   # gris
}
TIER_ORDER = ["A_confirmed", "B_discovery", "C_effector", "D_hub", "E_noise"]

# Ordre canonique des cell_groups
CELL_GROUP_ORDER = ["P4", "P16_cluster_0", "P16_cluster_1",
                    "P16_cluster_2", "P16_cluster_3"]
CELL_GROUP_LABELS = {
    "P4":             "P4\n(proliferative)",
    "P16_cluster_0":  "c0\n(prolif-pers P16)",
    "P16_cluster_1":  "c1\n(ECM-mild)",
    "P16_cluster_2":  "c2\n(OIS)",
    "P16_cluster_3":  "c3\n(SASP-inflam)",
}

# Aging DBs canoniques (alignées sur ora_consensus.py / cross_seed_gene_ranking)
AGING_DBS = ["in_senmayo", "in_cellage", "in_genage",
             "in_msigdb_aging", "in_ageanno", "in_aging_local"]
# Variantes nommage possibles (fallback)
AGING_DB_FALLBACKS = {
    "in_senmayo": "SenMayo",
    "in_cellage": "CellAge",
    "in_genage": "GenAge",
    "in_msigdb_aging": "MSigDB aging",
    "in_ageanno": "AgeAnno",
    "in_aging_local": "Fridman",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_ranking(version_dir: Path) -> pd.DataFrame:
    """Charge cross_seed_gene_ranking.tsv avec lookup robuste."""
    candidates = [
        version_dir / "cross_seed_gene_ranking.tsv",
        version_dir / "cross_seed_report" / "cross_seed_gene_ranking.tsv",
    ]
    for child in version_dir.iterdir() if version_dir.is_dir() else []:
        if child.is_dir() and child.name.startswith("cross_seed"):
            candidates.append(child / "cross_seed_gene_ranking.tsv")
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            key = "target" if "target" in df.columns else "gene_symbol"
            print(f"[load] {p} ({len(df)} gènes)")
            return df.set_index(key)
    raise FileNotFoundError(f"cross_seed_gene_ranking.tsv introuvable sous {version_dir}")


def load_per_cluster_cosine(raw_runs: list[Path]) -> pd.DataFrame:
    """
    Agrège les `proj_signed_cosine_global_<cellgroup>` à travers seeds × modes
    depuis les `perturbation_all_genes_axisV4_<mode>.tsv` bruts.

    Returns
    -------
    DataFrame (gene × cell_group) avec cosine moyenne sur seeds × modes.
    """
    modes = ["knockout", "knockdown", "overexpress"]
    pieces = []
    for run_dir in raw_runs:
        for mode in modes:
            # Essaye axisV4 d'abord, puis axisV3, puis sans suffixe
            for suffix in ["_axisV4", "_axisV3", ""]:
                p = run_dir / f"perturbation_all_genes{suffix}_{mode}.tsv"
                if p.exists():
                    break
            else:
                continue
            df = pd.read_csv(p, sep="\t")
            if "target" not in df.columns:
                continue
            cols = ["target"]
            for cg in CELL_GROUP_ORDER:
                col = f"proj_signed_cosine_global_{cg}"
                if col in df.columns:
                    cols.append(col)
            if len(cols) < 2:
                continue
            sub = df[cols].copy()
            sub["_seed"] = run_dir.name
            sub["_mode"] = mode
            pieces.append(sub)

    if not pieces:
        raise FileNotFoundError(
            f"Aucun perturbation_all_genes_*.tsv trouvé dans {raw_runs}"
        )
    big = pd.concat(pieces, ignore_index=True)

    # Cosine moyenne sur seeds × modes (chaque mode pondéré pareil)
    value_cols = [c for c in big.columns if c.startswith("proj_signed_cosine_global_")]
    agg = big.groupby("target")[value_cols].mean()
    # Renomme colonnes : proj_signed_cosine_global_P4 → P4, etc.
    agg.columns = [c.replace("proj_signed_cosine_global_", "") for c in agg.columns]
    print(f"[load] per-cluster cosine agrégé sur {len(raw_runs)} runs × {len(modes)} modes "
          f"→ {len(agg)} gènes × {len(agg.columns)} cell_groups")
    return agg


def load_group_expression(raw_runs: list[Path]) -> pd.DataFrame:
    """
    Charge `group_expression.tsv` depuis le premier run disponible.

    Le fichier est généralement **invariant** entre seeds (calculé en amont
    du clustering VGAE — cf. CLAUDE.md §13.10.6). On prend donc le premier
    run avec un `group_expression.tsv` existant.

    Returns
    -------
    DataFrame indexé par `gene`, colonnes `mean_<cell_group>` +
    `pct_<cell_group>`.
    """
    for run in raw_runs:
        p = run / "group_expression.tsv"
        if p.exists():
            df = pd.read_csv(p, sep="\t").set_index("gene")
            print(f"[load] group_expression depuis {p} ({len(df)} gènes)")
            return df
    raise FileNotFoundError(
        f"group_expression.tsv introuvable dans les runs : "
        f"{[r.name for r in raw_runs]}"
    )


def filter_tier(df: pd.DataFrame, allowed: list[str] | None = None) -> pd.DataFrame:
    """Restreint le DataFrame aux evidence_tier autorisés."""
    if allowed is None:
        allowed = ["A_confirmed", "B_discovery", "C_effector"]
    if "evidence_tier" not in df.columns:
        return df
    return df[df["evidence_tier"].isin(allowed)].copy()


def _ensure_outdir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


_SVG_ENABLED = False   # SVG opt-in (--svg) ; PNG seul par défaut (cleanup 2026-06-18)


def _savefig(fig, base_path: Path, also_svg: bool | None = None) -> Path:
    """Sauvegarde PNG (+ SVG si --svg / also_svg). PNG seul par défaut."""
    svg = _SVG_ENABLED if also_svg is None else also_svg
    fig.savefig(base_path.with_suffix(".png"))
    if svg:
        fig.savefig(base_path.with_suffix(".svg"))
    plt.close(fig)
    print(f"[save] {base_path}.png" + ("+svg" if svg else ""))
    return base_path


# ---------------------------------------------------------------------------
# Figure 1 — Heatmap drivers × cell_groups (cosine signé), épurée
# ---------------------------------------------------------------------------
def _select_top_drivers_split(
    ranking: pd.DataFrame,
    per_cluster: pd.DataFrame | None,
    top_n: int,
    tiers: list[str],
    sort_by: str,
    balanced: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Helper partagé entre les 2 heatmaps : sélectionne les drivers,
    aligne avec per_cluster, et split en pro/anti par sign(canon_cosine).

    Args:
      balanced : si True, prend top (top_n // 2) anti + top (top_n // 2)
                 pro indépendamment (= représentation équilibrée des 2
                 directions). Si False (défaut), prend top_n drivers
                 globalement puis split a posteriori (= reflète le ratio
                 réel anti/pro dans le top du ranking).

    Returns
    -------
    (sub, mat, n_anti) où :
      - sub : DataFrame ranking restreint et **réordonné** (anti d'abord,
              puis séparateur, puis pro), trié par driver_score décroissant
              dans chaque bloc.
      - mat : matrice cosine alignée (None si per_cluster absent → fallback
              cosines ranking-level).
      - n_anti : nb de drivers anti-sen (indice où placer le séparateur).
    """
    filtered = filter_tier(ranking, tiers)

    if balanced:
        # Top-(top_n//2) anti + top-(top_n//2) pro **indépendamment**
        half = top_n // 2
        anti = (filtered[filtered["canon_cosine"] < 0]
                .sort_values(sort_by, ascending=False).head(half))
        pro = (filtered[filtered["canon_cosine"] >= 0]
               .sort_values(sort_by, ascending=False).head(top_n - half))
        sub = pd.concat([anti, pro])
    else:
        sub = filtered.sort_values(sort_by, ascending=False).head(top_n)
        anti = sub[sub["canon_cosine"] < 0].sort_values(sort_by, ascending=False)
        pro = sub[sub["canon_cosine"] >= 0].sort_values(sort_by, ascending=False)
        sub = pd.concat([anti, pro])

    if per_cluster is not None:
        common = [g for g in sub.index if g in per_cluster.index]
        sub = sub.loc[common]
        cols = [c for c in CELL_GROUP_ORDER if c in per_cluster.columns]
        mat = per_cluster.loc[common, cols]
    else:
        cols_use = [c for c in ["cosine_quiescent_like", "cosine_senescent",
                                "canon_cosine", "KO_cos", "KD_cos", "OE_cos"]
                    if c in sub.columns]
        mat = sub[cols_use]

    n_anti = int((sub["canon_cosine"] < 0).sum())
    return sub, mat, n_anti


def fig_heatmap_clusters(
    ranking: pd.DataFrame,
    per_cluster: pd.DataFrame | None,
    out_dir: Path,
    top_n: int = 50,
    tiers: list[str] | None = None,
    sort_by: str = "driver_score",
    balanced: bool = False,
) -> Path:
    """
    Heatmap top-N drivers × cell_groups, color = cosine signé.

    Version épurée 2026-05-13 : retire tier colorbar + driver_score bar
    + tags + gridlines pour mettre en avant **les noms de gènes** et le
    pattern cosine pur.

    Sort en 2 blocs : anti-sen (canon_cosine < 0) puis pro-sen
    (canon_cosine ≥ 0), séparés par une ligne horizontale.

    Args:
      balanced : si True → top (top_n // 2) anti + top (top_n // 2) pro
                 (représentation équilibrée). Si False → top_n
                 globalement puis split a posteriori (reflète le ratio
                 réel ; les drivers anti dominent généralement V3/V4).
    """
    tiers = tiers or ["A_confirmed", "B_discovery", "C_effector"]
    sub, mat, n_anti = _select_top_drivers_split(
        ranking, per_cluster, top_n, tiers, sort_by, balanced=balanced)

    if per_cluster is not None:
        col_labels = [CELL_GROUP_LABELS.get(c, c) for c in mat.columns]
        x_label = "cosine signé post-perturbation par cell_group"
    else:
        col_labels = [c.replace("_cos", "").replace("cosine_", "")
                      for c in mat.columns]
        x_label = "cosines (ranking-level fallback)"

    n_genes = len(sub)
    # Plus large + plus haut pour lisibilité des labels gènes
    fig, ax = plt.subplots(figsize=(max(7, len(mat.columns) * 1.4 + 2),
                                     max(10, n_genes * 0.30)))

    vmax = max(abs(mat.values.min()), abs(mat.values.max())) or 1.0
    im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax,
                   extent=(0, len(mat.columns), n_genes, 0))

    # Séparateur visuel pro/anti
    if 0 < n_anti < n_genes:
        ax.axhline(n_anti, color="black", linewidth=1.5, linestyle="--", alpha=0.8)

    # Annotations latérales : blocs anti / pro
    if n_anti > 0:
        ax.text(-0.15, n_anti / 2, "↓ anti-sen\n(cos<0)",
               transform=ax.get_yaxis_transform(), rotation=90,
               ha="center", va="center", fontsize=10, fontweight="bold",
               color="#1f4d7d")
    if n_anti < n_genes:
        ax.text(-0.15, (n_anti + n_genes) / 2, "↑ pro-sen\n(cos>0)",
               transform=ax.get_yaxis_transform(), rotation=90,
               ha="center", va="center", fontsize=10, fontweight="bold",
               color="#7d1f1f")

    # Axes : noms gènes très lisibles, cell_groups en bas
    ax.set_xticks(np.arange(len(mat.columns)) + 0.5)
    ax.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(np.arange(n_genes) + 0.5)
    ax.set_yticklabels(sub.index, fontsize=9, fontfamily="monospace")
    ax.set_xlabel(x_label, fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)  # désactive la grille (contraste cmap suffit)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                        orientation="vertical")
    cbar.set_label("cosine signé\n(post-perturbation)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    mode_tag = " (balanced)" if balanced else ""
    ax.set_title(f"Top-{top_n} drivers VGAE — cosine × cell_groups{mode_tag}\n"
                f"(n_anti={n_anti}, n_pro={n_genes - n_anti}, sort: "
                f"{sort_by} desc par bloc)",
                fontsize=11)

    fig.tight_layout()
    suffix = "_balanced" if balanced else ""
    return _savefig(fig,
                    out_dir / f"heatmap_drivers_clusters_top{top_n}{suffix}")


# ---------------------------------------------------------------------------
# Figure 1bis — Heatmap expression du gène × cell_groups (comparaison)
# ---------------------------------------------------------------------------
def fig_heatmap_expression(
    ranking: pd.DataFrame,
    per_cluster: pd.DataFrame | None,
    group_expr: pd.DataFrame,
    out_dir: Path,
    top_n: int = 50,
    tiers: list[str] | None = None,
    sort_by: str = "driver_score",
    expression_mode: str = "zscore",
    balanced: bool = False,
) -> Path:
    """
    Heatmap d'**expression mean** du gène par cell_group, top-N drivers.

    **Même ordre de gènes** que `fig_heatmap_clusters` (anti d'abord, pro
    ensuite) → permet la comparaison côte-à-côte. Lecture biologique :
    un gène avec cosine fort négatif sur c3 (pro-sen) ET expression
    élevée sur c3 = corrélation effet↔expression (classique).
    Un gène avec cosine fort négatif sur c3 MAIS expression égale partout
    = driver "graph-only" (le shift VGAE ne reflète pas l'expression
    locale → effet de propagation pure).

    Args:
      expression_mode : 'zscore' (défaut, z-score par gène à travers
                       clusters → spécificité), 'raw' (mean LogNormalize),
                       'log_raw' (log1p de raw).
      balanced : si True → top (top_n // 2) anti + top (top_n // 2) pro.
    """
    tiers = tiers or ["A_confirmed", "B_discovery", "C_effector"]
    sub, _, n_anti = _select_top_drivers_split(
        ranking, per_cluster, top_n, tiers, sort_by, balanced=balanced)

    # Construit la matrice expression
    expr_cols = [f"mean_{c}" for c in CELL_GROUP_ORDER
                 if f"mean_{c}" in group_expr.columns]
    if not expr_cols:
        raise KeyError(f"Aucune colonne mean_<cell_group> dans group_expr "
                       f"(colonnes : {list(group_expr.columns)[:10]}...)")
    cell_groups_present = [c.replace("mean_", "") for c in expr_cols]

    common = [g for g in sub.index if g in group_expr.index]
    sub = sub.loc[common]
    n_genes = len(sub)
    # Recompute n_anti pour le sous-set commun
    n_anti = int((sub["canon_cosine"] < 0).sum())

    raw = group_expr.loc[common, expr_cols].copy()
    raw.columns = cell_groups_present

    if expression_mode == "zscore":
        # Z-score par gène à travers les clusters (spécificité)
        mat = raw.sub(raw.mean(axis=1), axis=0)
        std = raw.std(axis=1).replace(0, 1)
        mat = mat.div(std, axis=0)
        cmap = "RdBu_r"
        vmax = max(abs(mat.values.min()), abs(mat.values.max())) or 1.0
        vmin = -vmax
        cbar_label = "z-score expression\n(spécificité par gène)"
    elif expression_mode == "log_raw":
        mat = np.log1p(raw)
        cmap = "viridis"
        vmin, vmax = mat.values.min(), mat.values.max()
        cbar_label = "log1p(mean expression)"
    else:  # raw
        mat = raw
        cmap = "viridis"
        vmin, vmax = mat.values.min(), mat.values.max()
        cbar_label = "mean expression\n(LogNormalize)"

    col_labels = [CELL_GROUP_LABELS.get(c, c) for c in cell_groups_present]

    fig, ax = plt.subplots(figsize=(max(7, len(mat.columns) * 1.4 + 2),
                                     max(10, n_genes * 0.30)))

    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=(0, len(mat.columns), n_genes, 0))

    if 0 < n_anti < n_genes:
        ax.axhline(n_anti, color="black", linewidth=1.5, linestyle="--", alpha=0.8)

    if n_anti > 0:
        ax.text(-0.15, n_anti / 2, "↓ anti-sen\n(cos<0)",
               transform=ax.get_yaxis_transform(), rotation=90,
               ha="center", va="center", fontsize=10, fontweight="bold",
               color="#1f4d7d")
    if n_anti < n_genes:
        ax.text(-0.15, (n_anti + n_genes) / 2, "↑ pro-sen\n(cos>0)",
               transform=ax.get_yaxis_transform(), rotation=90,
               ha="center", va="center", fontsize=10, fontweight="bold",
               color="#7d1f1f")

    ax.set_xticks(np.arange(len(mat.columns)) + 0.5)
    ax.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(np.arange(n_genes) + 0.5)
    ax.set_yticklabels(sub.index, fontsize=9, fontfamily="monospace")
    ax.set_xlabel(f"expression mean par cell_group ({expression_mode})",
                 fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)  # désactive la grille (contraste cmap suffit)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                        orientation="vertical")
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    mode_tag = " (balanced)" if balanced else ""
    ax.set_title(f"Top-{top_n} drivers VGAE — expression × cell_groups{mode_tag}\n"
                f"(même ordre que heatmap cosine ; n_anti={n_anti}, "
                f"n_pro={n_genes - n_anti})",
                fontsize=11)

    fig.tight_layout()
    suffix_parts = []
    if expression_mode != "zscore":
        suffix_parts.append(expression_mode)
    if balanced:
        suffix_parts.append("balanced")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    return _savefig(fig,
                    out_dir / f"heatmap_drivers_expression_top{top_n}{suffix}")


# ---------------------------------------------------------------------------
# Figure 1ter — Heatmap marker genes par cluster (staircase de spécificité)
# ---------------------------------------------------------------------------
def fig_heatmap_cluster_markers(
    group_expr: pd.DataFrame,
    out_dir: Path,
    top_per_cluster: int = 15,
    ranking: pd.DataFrame | None = None,
    restrict_to_tiers: list[str] | None = None,
    min_total_expr: float = 0.05,
) -> Path:
    """
    Heatmap **staircase de marqueurs par cluster** — gènes les plus
    spécifiquement exprimés dans chaque cell_group, empilés par cluster.

    Pour chaque cluster `c` :
      1. Filtre les gènes avec mean_expression total ≥ `min_total_expr`
         (élimine le bruit transcriptomique).
      2. Calcule z-score par gène à travers les clusters.
      3. Prend les top-`top_per_cluster` par z-score positif sur `c`.
      4. Garde uniquement les gènes pas déjà sélectionnés par un
         cluster précédent (déduplication greedy).

    Produit un pattern visuel **diagonal/staircase** : chaque bloc de
    cluster montre une "boîte" de forte expression sur sa propre colonne,
    pâle ailleurs. Style classique des Seurat / scRNA-seq marker
    heatmaps (Stuart et al. 2019).

    Args:
      ranking : si fourni + `restrict_to_tiers`, filtre l'univers de
                gènes au sous-set evidence_tier ∈ tiers (e.g. A/B/C).
                Sinon, tous les gènes de `group_expr` sont considérés.
    """
    expr_cols = [f"mean_{c}" for c in CELL_GROUP_ORDER
                 if f"mean_{c}" in group_expr.columns]
    if not expr_cols:
        raise KeyError("Aucune colonne mean_<cell_group> dans group_expr")
    cell_groups_present = [c.replace("mean_", "") for c in expr_cols]

    raw = group_expr[expr_cols].copy()
    raw.columns = cell_groups_present
    # Filtre bruit
    total = raw.sum(axis=1)
    raw = raw[total >= min_total_expr * len(cell_groups_present)]

    # Restrict à un sous-set de tiers si demandé
    if ranking is not None and restrict_to_tiers:
        keep = ranking[ranking["evidence_tier"].isin(restrict_to_tiers)].index
        raw = raw.loc[raw.index.intersection(keep)]

    # Z-score par gène à travers clusters
    z = raw.sub(raw.mean(axis=1), axis=0)
    std = raw.std(axis=1).replace(0, 1)
    z = z.div(std, axis=0)

    # Selection greedy : top-K par cluster, sans réutiliser un gène déjà pris
    selected: list[tuple[str, str]] = []  # (cluster, gene)
    used: set[str] = set()
    for cg in cell_groups_present:
        # Tri par z-score décroissant sur cg, exclure gènes déjà pris
        candidates = z[cg].sort_values(ascending=False)
        n_picked = 0
        for gene, score in candidates.items():
            if gene in used:
                continue
            if score <= 0:
                break  # ne prend pas de gènes "non spécifiques"
            selected.append((cg, gene))
            used.add(gene)
            n_picked += 1
            if n_picked >= top_per_cluster:
                break

    if not selected:
        raise RuntimeError("Aucun gène marqueur sélectionné (vérifier min_total_expr)")

    # Construit la matrice ordonnée
    ordered_genes = [g for _, g in selected]
    mat = z.loc[ordered_genes]
    # Cluster index pour les séparateurs
    cluster_of_gene = pd.Series([c for c, _ in selected], index=ordered_genes)

    # Calcule les frontières de blocs
    block_sizes = cluster_of_gene.value_counts().reindex(cell_groups_present).fillna(0).astype(int)
    block_starts = block_sizes.cumsum().shift(fill_value=0).tolist()
    block_ends = block_sizes.cumsum().tolist()

    n_genes = len(ordered_genes)
    fig, ax = plt.subplots(figsize=(max(7, len(mat.columns) * 1.4 + 2),
                                     max(12, n_genes * 0.22)))

    vmax = max(abs(mat.values.min()), abs(mat.values.max())) or 1.0
    im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax,
                   extent=(0, len(mat.columns), n_genes, 0))

    # Séparateurs entre blocs cluster
    for end in block_ends[:-1]:
        if end > 0:
            ax.axhline(end, color="black", linewidth=1.2,
                      linestyle="-", alpha=0.7)

    # Annotations latérales : label de chaque bloc cluster
    for cg, start, size in zip(cell_groups_present, block_starts, block_sizes):
        if size <= 0:
            continue
        mid = start + size / 2
        ax.text(-0.20, mid, CELL_GROUP_LABELS.get(cg, cg).replace("\n", " "),
               transform=ax.get_yaxis_transform(), rotation=90,
               ha="center", va="center", fontsize=10, fontweight="bold",
               color="#333")

    # Axes
    col_labels = [CELL_GROUP_LABELS.get(c, c) for c in cell_groups_present]
    ax.set_xticks(np.arange(len(mat.columns)) + 0.5)
    ax.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(np.arange(n_genes) + 0.5)
    ax.set_yticklabels(ordered_genes, fontsize=8, fontfamily="monospace")
    ax.set_xlabel("z-score expression\n(spécificité par gène)", fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)  # contraste cmap suffit

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("z-score expression", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    restrict_tag = ""
    if restrict_to_tiers:
        restrict_tag = f" (restreint à {'/'.join(t.split('_')[0] for t in restrict_to_tiers)})"
    ax.set_title(f"Marqueurs cluster-spécifiques — top {top_per_cluster} "
                f"par cell_group{restrict_tag}\n"
                f"({n_genes} gènes — déduplication greedy)",
                fontsize=11)

    fig.tight_layout()
    suffix = f"_tiers-{'-'.join(t[0] for t in restrict_to_tiers)}" \
        if restrict_to_tiers else ""
    return _savefig(fig,
                    out_dir / f"heatmap_cluster_markers_top{top_per_cluster}{suffix}")


# ---------------------------------------------------------------------------
# Figure 1quinquies — Clustermap expression (hierarchical, sans split forcé)
# ---------------------------------------------------------------------------
def fig_clustermap_expression(
    ranking: pd.DataFrame,
    group_expr: pd.DataFrame,
    out_dir: Path,
    n_anti: int = 25,
    n_pro: int = 25,
    tiers: list[str] | None = None,
    sort_by: str = "driver_score",
    linkage_method: str = "average",
    distance_metric: str = "correlation",
) -> Path:
    """
    Clustermap hiérarchique top-K anti + top-K pro, **sans split forcé**.

    Principe :
      - On prend les top-K anti-sen et top-K pro-sen (par driver_score).
      - Les 50 gènes (K=25) sont clusterisés **par similarité d'expression**
        (z-score × cell_groups, dist = correlation, linkage = average).
      - Le dendrogramme groupe naturellement les gènes au pattern
        d'expression similaire → l'utilisateur lit visuellement les clades.
      - Bandes de couleur latérales (annotations) montrent direction
        (pro/anti) + evidence_tier sans imposer l'ordre.

    Lecture biologique attendue :
      - 1 clade "expr prolif-side" (P4/c0 forts) — mélange anti-sen
        activators of proliferation + pro-sen inducers latents.
      - 1 clade "expr sen-side" (c1/c2/c3 forts) — mélange anti-sen
        senolytic candidates + pro-sen SASP effectors.
      - Les drivers "outliers" (clade séparé) sont les plus
        intéressants — pattern d'expression atypique.

    Args:
      n_anti, n_pro : top-K par direction (défaut 25/25 = 50 gènes).
      linkage_method : average | ward | complete | single.
      distance_metric : correlation | euclidean | cosine.
        correlation = mesure de pattern (shape, ignore magnitude),
        adapté aux patterns d'expression (Eisen et al. 1998).
    """
    tiers = tiers or ["A_confirmed", "B_discovery", "C_effector"]
    filtered = filter_tier(ranking, tiers)

    anti = (filtered[filtered["canon_cosine"] < 0]
            .sort_values(sort_by, ascending=False).head(n_anti))
    pro = (filtered[filtered["canon_cosine"] >= 0]
           .sort_values(sort_by, ascending=False).head(n_pro))
    sub = pd.concat([anti, pro])

    # Détecte le mode (both / anti-only / pro-only) pour adapter
    # annotations, titre et filename.
    has_anti = n_anti > 0 and not anti.empty
    has_pro = n_pro > 0 and not pro.empty
    if not has_anti and not has_pro:
        raise ValueError("n_anti + n_pro = 0 ; rien à plotter.")
    mode = "both" if (has_anti and has_pro) else ("anti" if has_anti else "pro")

    # Z-score expression matrix (par gène à travers cell_groups)
    expr_cols = [f"mean_{c}" for c in CELL_GROUP_ORDER
                 if f"mean_{c}" in group_expr.columns]
    if not expr_cols:
        raise KeyError("Aucune colonne mean_<cell_group> dans group_expr")
    cell_groups_present = [c.replace("mean_", "") for c in expr_cols]

    common = [g for g in sub.index if g in group_expr.index]
    sub = sub.loc[common]
    raw = group_expr.loc[common, expr_cols].copy()
    raw.columns = cell_groups_present
    z = raw.sub(raw.mean(axis=1), axis=0).div(raw.std(axis=1).replace(0, 1), axis=0)

    # Annotations latérales — adaptées au mode
    if mode == "both":
        row_direction = pd.Series(
            np.where(sub["canon_cosine"] < 0, "#1f4d7d", "#7d1f1f"),
            index=sub.index, name="direction (anti=blue, pro=red)"
        )
        row_tier = sub["evidence_tier"].map(TIER_COLORS).rename("evidence_tier")
        row_colors = pd.concat([row_direction, row_tier], axis=1)
    else:
        # Single-direction : la bande "direction" devient inutile (tous
        # même couleur). On garde uniquement evidence_tier — l'info
        # spatiale d'expression est déjà portée par la heatmap elle-même.
        row_colors = (sub["evidence_tier"]
                      .map(TIER_COLORS)
                      .rename("evidence_tier")
                      .to_frame())

    # Symétrie vmax/vmin sur z (cmap divergent)
    vmax = max(abs(z.values.min()), abs(z.values.max())) or 1.0

    # Style local : pas de whitegrid (clustermap a son propre layout)
    with sns.axes_style("white"):
        g = sns.clustermap(
            z,
            row_cluster=True,
            col_cluster=False,   # garde ordre canonique P4, c0..c3
            method=linkage_method,
            metric=distance_metric,
            cmap="RdBu_r",
            center=0,
            vmin=-vmax, vmax=vmax,
            figsize=(max(8, len(z.columns) * 1.4 + 4),
                     max(11, len(z) * 0.30)),
            row_colors=row_colors,
            xticklabels=[CELL_GROUP_LABELS.get(c, c)
                         for c in cell_groups_present],
            yticklabels=True,
            cbar_pos=(0.02, 0.78, 0.04, 0.16),
            dendrogram_ratio=(0.18, 0.08),
            colors_ratio=0.025,
            linewidths=0,  # pas de gridlines
        )

    # Tweak label sizes
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(),
                                  rotation=20, ha="right", fontsize=10)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(),
                                  fontsize=9, fontfamily="monospace")
    g.ax_heatmap.set_xlabel("z-score expression par cell_group",
                            fontsize=10)
    g.ax_heatmap.set_ylabel("")

    # Colorbar label
    g.cax.set_title("z-score\nexpression", fontsize=9, pad=4)
    g.cax.tick_params(labelsize=8)

    # Légende personnalisée pour les bandes de couleur — adaptée au mode
    legend_handles = []
    if mode == "both":
        legend_handles += [
            mpatches.Patch(color="#1f4d7d", label="anti-sen (cos<0)"),
            mpatches.Patch(color="#7d1f1f", label="pro-sen (cos>0)"),
        ]
    # Single-direction : pas de bande supplémentaire → seulement tier ci-dessous.
    legend_handles += [
        mpatches.Patch(color=TIER_COLORS[t], label=t)
        for t in TIER_ORDER if t in sub["evidence_tier"].values
    ]
    g.ax_heatmap.legend(
        handles=legend_handles,
        loc="upper left", bbox_to_anchor=(1.18, 1.0),
        fontsize=8, frameon=True, title="annotations",
        title_fontsize=9,
    )

    # Titre + filename adaptés au mode
    if mode == "both":
        title_main = (f"Clustermap expression top-{n_anti} anti + "
                     f"top-{n_pro} pro ({linkage_method}/{distance_metric})")
        title_sub = ("Les gènes au pattern d'expression similaire se regroupent "
                    "automatiquement — la couleur 'direction' à gauche révèle "
                    "si pro/anti se mélangent dans un clade.")
        fname = f"clustermap_top{n_anti}anti+{n_pro}pro"
    elif mode == "anti":
        title_main = (f"Clustermap expression top-{n_anti} anti-sénescence "
                     f"({linkage_method}/{distance_metric})")
        title_sub = ("Clades attendus : (1) prolif-side = activators of "
                    "proliferation (H2AFZ/HMGB1) ; (2) sen-side = candidates "
                    "sénolytiques (FHL2/EDN1/JUN).")
        fname = f"clustermap_top{n_anti}anti_only"
    else:  # pro
        title_main = (f"Clustermap expression top-{n_pro} pro-sénescence "
                     f"({linkage_method}/{distance_metric})")
        title_sub = ("Clades attendus : (1) prolif-side = inducers latents "
                    "(CEBPB/ATF4/DDIT3) ; (2) sen-side = SASP effectors "
                    "(p16/IL6/CXCL1).")
        fname = f"clustermap_top{n_pro}pro_only"

    g.figure.suptitle(f"{title_main}\n{title_sub}", fontsize=11, y=1.005)

    out_path = out_dir / f"{fname}_{linkage_method}_{distance_metric}"
    g.figure.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    if _SVG_ENABLED:
        g.figure.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(g.figure)
    print(f"[save] {out_path}.png" + ("+svg" if _SVG_ENABLED else ""))
    return out_path


# ---------------------------------------------------------------------------
# Figure 1quater — Quadrant biocategory (anti/pro × prolif/sen expression)
# ---------------------------------------------------------------------------
BIOCATEGORY_LABELS = {
    ("anti", "prolif"): "Activator of proliferation",
    ("anti", "sen"):    "Senolytic target candidate",
    ("pro",  "prolif"): "Senescence inducer",
    ("pro",  "sen"):    "SASP effector / marker",
}
BIOCATEGORY_COLORS = {
    ("anti", "prolif"): "#1f77b4",   # bleu — pro-proliferation
    ("anti", "sen"):    "#9467bd",   # violet — senolytic (rare, intéressant)
    ("pro",  "prolif"): "#ff7f0e",   # orange — inducteur latent
    ("pro",  "sen"):    "#d62728",   # rouge — effecteur SASP
}


def _compute_biocategory(
    ranking: pd.DataFrame,
    group_expr: pd.DataFrame,
    prolif_groups: tuple[str, ...] = ("P4", "P16_cluster_0"),
    sen_groups: tuple[str, ...] = ("P16_cluster_1", "P16_cluster_2", "P16_cluster_3"),
) -> pd.DataFrame:
    """
    Catégorise chaque gène en 4 quadrants biologiques (anti/pro × prolif/sen).

    expression_score = mean(z[prolif_groups]) − mean(z[sen_groups])
                       > 0 → expression dominante côté prolif
                       < 0 → expression dominante côté sen

    senescence_effect = sign(canon_cosine)
                       < 0 → anti-sen (KO induit P4)
                       ≥ 0 → pro-sen (KO/OE induit sen)

    Returns
    -------
    DataFrame `ranking` augmenté de :
      - `expr_score` (float, − à +)
      - `expr_side` ∈ {prolif, sen}
      - `effect_side` ∈ {anti, pro}
      - `bio_category` (1 des 4 labels)
    """
    # Z-score par gène à travers tous les cell_groups
    cols = [f"mean_{c}" for c in CELL_GROUP_ORDER
            if f"mean_{c}" in group_expr.columns]
    raw = group_expr[cols].copy()
    raw.columns = [c.replace("mean_", "") for c in cols]
    z = raw.sub(raw.mean(axis=1), axis=0).div(raw.std(axis=1).replace(0, 1), axis=0)

    prolif_cols = [c for c in prolif_groups if c in z.columns]
    sen_cols = [c for c in sen_groups if c in z.columns]
    expr_score = z[prolif_cols].mean(axis=1) - z[sen_cols].mean(axis=1)

    df = ranking.copy()
    df["expr_score"] = expr_score.reindex(df.index)
    df["expr_side"] = np.where(df["expr_score"] > 0, "prolif", "sen")
    df["effect_side"] = np.where(df["canon_cosine"] < 0, "anti", "pro")
    df["bio_category"] = df.apply(
        lambda r: BIOCATEGORY_LABELS.get((r["effect_side"], r["expr_side"]),
                                         "unknown").split("\n")[0],
        axis=1,
    )
    return df


def fig_quadrant_biocategory(
    ranking: pd.DataFrame,
    group_expr: pd.DataFrame,
    out_dir: Path,
    top_n_label: int = 30,
    tiers: list[str] | None = None,
    label_top_per_quadrant: int = 8,
) -> Path:
    """
    Scatter quadrant 2×2 : expr_score (x) × canon_cosine (y).

    4 quadrants annotés avec interprétation biologique :
      Q1 (top-left) : anti-sen × expr-prolif → "Activator of proliferation"
                       (KO induit sen, exprimé en prolif).
      Q2 (top-right) : anti-sen × expr-sen → "Senolytic target candidate"
                       (KO induit P4 mais gène exprimé en sen ; cibles
                       BCL-2 / BCL-XL / pro-survie sénescente).
      Q3 (bottom-left) : pro-sen × expr-prolif → "Senescence inducer"
                         (OE induit sen, latent en prolif ; e.g. CEBPB).
      Q4 (bottom-right) : pro-sen × expr-sen → "SASP effector / marker"
                          (OE renforce sen, déjà exprimé en sen ; e.g. p16).

    Référencé sur le mapping cell_group V3.3 (§13.10 du rapport).
    Articles : van Deursen 2014 *Nature* (senolytic), Zhu 2015 *Aging
    Cell* (Navitoclax), Salama 2014 *Genes Dev* (CEBPB inducer), Coppé
    2008 *PLoS Biol* (SASP).
    """
    tiers = tiers or ["A_confirmed", "B_discovery", "C_effector"]
    df = _compute_biocategory(ranking, group_expr)
    df = filter_tier(df, tiers).dropna(subset=["expr_score", "canon_cosine"])

    fig, ax = plt.subplots(figsize=(11, 9))

    # Background : zones colorées par quadrant (très transparent)
    xmin, xmax = df["expr_score"].min() * 1.1, df["expr_score"].max() * 1.1
    ymin, ymax = df["canon_cosine"].min() * 1.1, df["canon_cosine"].max() * 1.1
    quadrants = [
        # (x_range, y_range, effect_side, expr_side)
        ((xmin, 0), (ymin, 0), "anti", "sen"),
        ((0, xmax), (ymin, 0), "anti", "prolif"),
        ((xmin, 0), (0, ymax), "pro", "sen"),
        ((0, xmax), (0, ymax), "pro", "prolif"),
    ]
    for (xr, yr, eff, exp) in quadrants:
        color = BIOCATEGORY_COLORS[(eff, exp)]
        ax.fill_betweenx([yr[0], yr[1]], xr[0], xr[1], color=color, alpha=0.06)

    # Lignes diviseurs centrales
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)

    # Scatter principal — taille = driver_score, couleur = bio_category
    for (eff, exp), label in BIOCATEGORY_LABELS.items():
        sub = df[(df["effect_side"] == eff) & (df["expr_side"] == exp)]
        sizes = 30 + 250 * sub["driver_score"]
        ax.scatter(sub["expr_score"], sub["canon_cosine"],
                   s=sizes, c=BIOCATEGORY_COLORS[(eff, exp)],
                   alpha=0.65, edgecolor="black", linewidths=0.4,
                   label=f"{label.split(chr(10))[0]} (n={len(sub)})")

    # Annotations textuelles : top-K par quadrant
    for (eff, exp), label in BIOCATEGORY_LABELS.items():
        sub = (df[(df["effect_side"] == eff) & (df["expr_side"] == exp)]
               .nlargest(label_top_per_quadrant, "driver_score"))
        for gene, row in sub.iterrows():
            ax.annotate(gene, (row["expr_score"], row["canon_cosine"]),
                       fontsize=7, xytext=(3, 3), textcoords="offset points",
                       color="#222", fontweight="bold")

    # Annotations des quadrants — placées à l'extrême haut/bas pour
    # bien séparer visuellement les 2 paires (pro-sen en haut, anti-sen
    # en bas), avec des limites étendues pour éviter le chevauchement
    # avec les points.
    y_range = ymax - ymin
    ymin_ext = ymin - 0.18 * y_range  # marge basse pour bas-labels
    ymax_ext = ymax + 0.12 * y_range  # marge haute pour haut-labels

    x_mid_left = xmin + 0.25 * (xmax - xmin)
    x_mid_right = xmin + 0.75 * (xmax - xmin)
    y_top = ymax + 0.06 * y_range
    y_bot = ymin - 0.10 * y_range

    quadrant_pos = {
        ("pro",  "sen"):    (x_mid_left,  y_top),  # haut-gauche
        ("pro",  "prolif"): (x_mid_right, y_top),  # haut-droite
        ("anti", "sen"):    (x_mid_left,  y_bot),  # bas-gauche
        ("anti", "prolif"): (x_mid_right, y_bot),  # bas-droite
    }
    for (eff, exp), (x, y) in quadrant_pos.items():
        ax.text(x, y, BIOCATEGORY_LABELS[(eff, exp)],
               fontsize=11, ha="center", va="center", fontweight="bold",
               color=BIOCATEGORY_COLORS[(eff, exp)],
               bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                         edgecolor=BIOCATEGORY_COLORS[(eff, exp)],
                         linewidth=1.5, alpha=0.95))

    ax.set_xlabel("expr_score = mean z(P4,c0) − mean z(c1,c2,c3)\n"
                 "→ x>0 : exprimé prolif-side  ;  x<0 : exprimé sen-side",
                 fontsize=10)
    ax.set_ylabel("canon_cosine\n"
                 "→ y>0 : pro-sen (OE induit sen)  ;  y<0 : anti-sen (KO induit sen)",
                 fontsize=10)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin_ext, ymax_ext)
    ax.grid(False)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
             frameon=True, title="bio_category", title_fontsize=9)

    ax.set_title("Drivers VGAE — quadrant biologique\n"
                "(senolytic = anti-sen × expr-sen ; SASP effector = pro-sen × expr-sen ;\n"
                " activator-prolif = anti-sen × expr-prolif ; inducer = pro-sen × expr-prolif)",
                fontsize=11)

    fig.tight_layout()
    out_path = _savefig(fig, out_dir / "quadrant_biocategory")

    # Export TSV de la catégorisation pour usage downstream
    tsv_path = out_dir / "biocategory_assignment.tsv"
    cols_export = ["expr_score", "expr_side", "canon_cosine", "effect_side",
                   "bio_category", "driver_score", "discovery_score",
                   "evidence_tier", "is_tf", "n_aging_dbs",
                   "is_de_significant"]
    cols_export = [c for c in cols_export if c in df.columns]
    df.sort_values("driver_score", ascending=False)[cols_export].to_csv(
        tsv_path, sep="\t", float_format="%.4f"
    )
    print(f"[save] {tsv_path}")

    return out_path


# ---------------------------------------------------------------------------
# Figure 2 — Distributions par evidence_tier
# ---------------------------------------------------------------------------
def fig_tier_distributions(ranking: pd.DataFrame, out_dir: Path) -> Path:
    """
    4 violin plots par evidence_tier (A, B, C, D, E) :
      A — driver_score
      B — |canon_diff| log-scale
      C — canon_cosine (signed)
      D — DE log2FC × −log10(padj) (effect)
    """
    df = ranking.copy()
    if "evidence_tier" not in df.columns:
        raise KeyError("evidence_tier column missing")

    df = df[df["evidence_tier"].isin(TIER_ORDER)].copy()
    df["evidence_tier"] = pd.Categorical(df["evidence_tier"],
                                          categories=TIER_ORDER,
                                          ordered=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Helper violin uniforme
    def _violin(ax, y_col, title, transform=None, ylabel=None, log=False):
        data = df[[y_col, "evidence_tier"]].dropna()
        if transform is not None:
            data = data.copy()
            data[y_col] = transform(data[y_col])
        sns.violinplot(data=data, x="evidence_tier", y=y_col, ax=ax,
                      hue="evidence_tier", legend=False,
                      palette=TIER_COLORS, inner="quartile",
                      cut=0, density_norm="width")
        if log:
            ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel or y_col)
        ax.set_title(title)
        # Annotation n par tier
        counts = df["evidence_tier"].value_counts()
        for i, t in enumerate(TIER_ORDER):
            n = counts.get(t, 0)
            ax.text(i, ax.get_ylim()[1] * 0.95, f"n={n}",
                   ha="center", fontsize=7, color="#444")

    _violin(axes[0, 0], "driver_score",
           "A — driver_score (graph-intrinsic + perturbation)",
           ylabel="driver_score ∈ [0, 1]")
    _violin(axes[0, 1], "canon_diff",
           "B — |canon_diff| (effet cumulé signé)",
           transform=lambda x: x.abs(), ylabel="|canon_diff| (log-symlog)",
           log=True)
    _violin(axes[1, 0], "canon_cosine",
           "C — canon_cosine (directionalité pure)",
           ylabel="canon_cosine ∈ [−1, +1]")

    # Effect DE signé : log2fc × −log10(padj), clipped
    if "de_log2fc_p4_vs_p16" in df.columns and "de_neglog10_padj" in df.columns:
        df_de = df.copy()
        df_de["de_effect"] = (df_de["de_log2fc_p4_vs_p16"]
                              * df_de["de_neglog10_padj"].clip(upper=50))
        sns.violinplot(data=df_de[df_de["de_effect"].abs() < 100],
                      x="evidence_tier", y="de_effect",
                      hue="evidence_tier", legend=False,
                      palette=TIER_COLORS, ax=axes[1, 1], inner="quartile",
                      cut=0, density_norm="width")
        axes[1, 1].set_xlabel("")
        axes[1, 1].set_ylabel("log2FC × −log10(padj)")
        axes[1, 1].set_title("D — Effet DE signé (MAST)")
        axes[1, 1].axhline(0, c="black", lw=0.5)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "DE magnitudes\nnon disponibles",
                       ha="center", va="center", transform=axes[1, 1].transAxes)

    fig.suptitle("Distribution des métriques de scoring par evidence_tier",
                fontsize=13, y=1.02)
    return _savefig(fig, out_dir / "tier_distributions")


# ---------------------------------------------------------------------------
# Figure 3 — Bubble plot drivers × aging DBs
# ---------------------------------------------------------------------------
def fig_aging_bubbles(ranking: pd.DataFrame, out_dir: Path,
                      top_n: int = 50, tiers: list[str] | None = None) -> Path:
    """
    Bubble plot : rows = top-N drivers, cols = aging DBs.
    Taille = driver_score, couleur = canon_cosine signé.

    Si pas de colonnes `in_<db>` explicites, fallback sur n_aging_dbs comme
    score agrégé (1 colonne unique).
    """
    tiers = tiers or ["A_confirmed", "B_discovery", "C_effector"]
    sub = filter_tier(ranking, tiers).sort_values("driver_score", ascending=False).head(top_n)

    # Identifier les colonnes aging DBs disponibles
    db_cols = [c for c in AGING_DBS if c in sub.columns]
    if not db_cols:
        # Fallback : single "n_aging_dbs" column
        if "n_aging_dbs" not in sub.columns:
            raise KeyError("Ni colonnes in_<db> ni n_aging_dbs dans le ranking")
        # Synthétise un seul "n_aging_dbs" colonne
        sub_mat = sub[["n_aging_dbs"]].copy()
        sub_mat["n_aging_dbs"] = sub_mat["n_aging_dbs"].astype(int)
        n_dbs_max = int(sub_mat["n_aging_dbs"].max())
        col_labels = [f"DB#{i+1}" for i in range(n_dbs_max)]
        # Construit une matrice (gene, db) binaire reconstituée — limite : on
        # ne sait pas dans QUELLE db chaque gène est, juste combien.
        # On crée un faux pattern visuel : remplit les n_aging_dbs premières
        # colonnes. Pas idéal mais informatif. À améliorer si les vraies
        # colonnes in_<db> sont disponibles.
        mat = pd.DataFrame(0, index=sub_mat.index, columns=col_labels)
        for g, n in sub_mat["n_aging_dbs"].items():
            mat.loc[g, col_labels[:n]] = 1
        warn = "(reconstruit depuis n_aging_dbs — colonnes in_<db> absentes)"
    else:
        mat = sub[db_cols].astype(float)
        col_labels = [AGING_DB_FALLBACKS.get(c, c.replace("in_", "")) for c in db_cols]
        warn = ""

    n_genes = len(sub)
    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 1.0),
                                     max(8, n_genes * 0.22)))

    # Plot : scatter avec taille = driver_score, couleur = canon_cosine
    for i, (gene, row) in enumerate(sub.iterrows()):
        for j, col in enumerate(mat.columns):
            present = mat.iloc[i, j] > 0
            if present:
                size = 50 + 700 * sub.loc[gene, "driver_score"]
                cosine = sub.loc[gene, "canon_cosine"]
                ax.scatter(j, i, s=size, c=[cosine], cmap="RdBu_r",
                          vmin=-1, vmax=1, edgecolor="black", linewidths=0.5,
                          alpha=0.85)

    # Hack pour avoir une colorbar sans dépendre des points individuels
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-1, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("canon_cosine signé", fontsize=9)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(n_genes))
    ax.set_yticklabels(sub.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, len(col_labels) - 0.5)
    ax.set_ylim(n_genes - 0.5, -0.5)
    ax.set_title(f"Top-{top_n} drivers × aging DBs\n"
                f"(taille = driver_score, couleur = canon_cosine) {warn}",
                fontsize=11)
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")

    return _savefig(fig, out_dir / f"aging_bubbles_top{top_n}")


# ---------------------------------------------------------------------------
# Figure 4 — Radar/polar plot top drivers
# ---------------------------------------------------------------------------
def fig_radar(ranking: pd.DataFrame, per_cluster: pd.DataFrame,
              out_dir: Path, top_n: int = 12,
              tiers: list[str] | None = None) -> Path:
    """
    Radar plot top-N drivers — chaque polygone à 5 sommets (P4, c0..c3).
    Révèle les drivers cluster-spécifiques (polygone très excentré sur 1 axe)
    vs pan-cluster (polygone régulier).

    Grille 3×4 (12 drivers par défaut).
    """
    tiers = tiers or ["A_confirmed", "B_discovery"]
    sub = filter_tier(ranking, tiers).sort_values("driver_score", ascending=False).head(top_n)
    genes = [g for g in sub.index if g in per_cluster.index]
    if not genes:
        raise RuntimeError("Aucun gene du top-N trouvé dans per_cluster")
    genes = genes[:top_n]

    cell_groups = [c for c in CELL_GROUP_ORDER if c in per_cluster.columns]
    n_axes = len(cell_groups)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    n_cols = 4
    n_rows = (len(genes) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                            subplot_kw=dict(projection="polar"),
                            constrained_layout=True)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    # Échelle commune pour comparabilité
    vmax = per_cluster.loc[genes, cell_groups].abs().max().max()
    vmax = max(vmax, 0.1) * 1.1

    for k, gene in enumerate(genes):
        ax = axes[k]
        vals = per_cluster.loc[gene, cell_groups].values
        vals_closed = list(vals) + [vals[0]]

        tier = sub.loc[gene, "evidence_tier"]
        color = TIER_COLORS.get(tier, "#777")

        # Cercle 0 pour reference
        ax.plot(np.linspace(0, 2*np.pi, 100), [0]*100, c="black", lw=0.5, alpha=0.5)

        # Aire signée : remplissage rouge si négatif, bleu si positif
        ax.fill(angles_closed,
               [v if v > 0 else 0 for v in vals_closed],
               color="#1f77b4", alpha=0.45, label="anti-sen")
        ax.fill(angles_closed,
               [abs(v) if v < 0 else 0 for v in vals_closed],
               color="#d62728", alpha=0.45, label="pro-sen")
        # Contour
        ax.plot(angles_closed, [abs(v) for v in vals_closed],
               c=color, lw=1.5)

        ax.set_xticks(angles)
        ax.set_xticklabels([CELL_GROUP_LABELS.get(c, c).split("\n")[0]
                            for c in cell_groups], fontsize=7)
        ax.set_ylim(0, vmax)
        ax.set_yticks([vmax / 2, vmax])
        ax.set_yticklabels([f"{vmax/2:.2f}", f"{vmax:.2f}"], fontsize=6)
        ax.set_rlabel_position(180 / n_axes)

        ds = sub.loc[gene, "driver_score"]
        ax.set_title(f"{gene}\n[{tier.split('_')[0]}] driver={ds:.2f}",
                    fontsize=9, pad=10, color=color, fontweight="bold")

    # Cacher subplots inutilisés
    for k in range(len(genes), len(axes)):
        axes[k].axis("off")

    # Légende globale
    fig.legend(handles=[
        mpatches.Patch(color="#1f77b4", alpha=0.5, label="anti-sen (cosine > 0)"),
        mpatches.Patch(color="#d62728", alpha=0.5, label="pro-sen (cosine < 0)"),
    ], loc="lower center", ncol=2, fontsize=9,
       bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Radar top-{top_n} drivers — |cosine| par cell_group",
                fontsize=13, y=1.01)
    return _savefig(fig, out_dir / f"radar_top{top_n}")


# ---------------------------------------------------------------------------
# Figure 5 — Lollipop cluster-specific drivers
# ---------------------------------------------------------------------------
def fig_lollipop_clusters(ranking: pd.DataFrame, per_cluster: pd.DataFrame,
                          out_dir: Path, top_k: int = 10,
                          target_clusters: list[str] | None = None) -> Path:
    """
    Pour chaque cluster cible (c1, c2, c3 par défaut — sénescents) :
    top-K drivers les plus *spécifiques* à ce cluster, mesurés par
    `specificity = |cosine[cluster]| - mean(|cosine[other_clusters]|)`.

    Lollipop horizontal : ligne = cosine spécifique, point = couleur par
    evidence_tier.
    """
    target_clusters = target_clusters or ["P16_cluster_1", "P16_cluster_2",
                                          "P16_cluster_3"]
    target_clusters = [c for c in target_clusters if c in per_cluster.columns]

    # Joint
    common = [g for g in ranking.index if g in per_cluster.index]
    pc = per_cluster.loc[common]
    rk = ranking.loc[common]

    # Calcule specificity par cluster cible
    abs_pc = pc.abs()
    fig, axes = plt.subplots(1, len(target_clusters),
                            figsize=(5 * len(target_clusters), 8),
                            constrained_layout=True)
    if len(target_clusters) == 1:
        axes = [axes]

    for ax, tgt in zip(axes, target_clusters):
        others = [c for c in pc.columns if c != tgt]
        spec = abs_pc[tgt] - abs_pc[others].mean(axis=1)
        # Garde uniquement les gènes où le sign cosine[tgt] est cohérent
        signed_spec = spec * np.sign(pc[tgt])
        ranking_with_spec = rk.assign(spec=spec, signed_spec=signed_spec)

        # Top-K par specificity absolue (anti ou pro)
        top = ranking_with_spec.reindex(spec.abs().nlargest(top_k * 2).index)
        # Filtre : evidence_tier sérieux (A/B/C)
        top = top[top["evidence_tier"].isin(["A_confirmed", "B_discovery", "C_effector"])]
        top = top.head(top_k).sort_values("signed_spec")

        if top.empty:
            ax.text(0.5, 0.5, f"Aucun driver spécifique\nde {tgt}",
                   ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        ys = np.arange(len(top))
        colors = [TIER_COLORS.get(t, "#777") for t in top["evidence_tier"]]
        ax.hlines(ys, 0, top["signed_spec"].values,
                 colors=colors, linewidth=2.2)
        ax.scatter(top["signed_spec"].values, ys, c=colors, s=80,
                  edgecolor="black", linewidth=0.5, zorder=3)
        ax.axvline(0, c="black", lw=0.5)
        ax.set_yticks(ys)
        ax.set_yticklabels(top.index, fontsize=8)
        # Tag DE / TF
        for i, (g, row) in enumerate(top.iterrows()):
            tags = []
            if row.get("is_tf", 0): tags.append("TF")
            if row.get("is_de_significant", False): tags.append("DE")
            if row.get("n_aging_dbs", 0) >= 2: tags.append("lit")
            if tags:
                x = top["signed_spec"].iloc[i]
                offset = 0.005 * np.sign(x) if x != 0 else 0.005
                ax.text(x + offset, i, " ".join(tags),
                       fontsize=6.5, va="center",
                       ha="left" if x >= 0 else "right", color="#333")

        ax.set_title(f"{CELL_GROUP_LABELS.get(tgt, tgt)}\n"
                    f"top-{top_k} drivers spécifiques",
                    fontsize=10)
        ax.set_xlabel("specificity signed = |cos[tgt]| − mean(|cos[other]|) "
                     "× sign(cos[tgt])", fontsize=8)
        ax.grid(True, axis="x", alpha=0.3, linestyle=":")

    # Légende tier
    legend_handles = [mpatches.Patch(color=TIER_COLORS[t], label=t)
                      for t in ["A_confirmed", "B_discovery", "C_effector"]]
    fig.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(0.99, 0.99), fontsize=8, frameon=True,
              title="evidence_tier")

    fig.suptitle("Drivers cluster-spécifiques (cosine cluster ≫ cosine moyenne autres)",
                fontsize=12, y=1.01)
    return _savefig(fig, out_dir / f"lollipop_cluster_specific_top{top_k}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--version-dir", type=Path, required=True,
                  help="Dossier contenant cross_seed_gene_ranking.tsv "
                       "(ou cross_seed_*/<.tsv>).")
    p.add_argument("--raw-runs", type=Path, nargs="+", default=None,
                  help="Runs bruts (...s1, ...s2) contenant les "
                       "perturbation_all_genes_*.tsv pour le per-cluster cosine.")
    p.add_argument("--svg", action="store_true",
                   help="exporte aussi en SVG (défaut : PNG seul).")
    p.add_argument("--out-dir", type=Path, default=None,
                  help="Dossier de sortie (défaut : <version-dir>/figures_explorer).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_h = sub.add_parser("heatmap-clusters", help="Heatmap drivers × cell_groups (cosine)")
    _add_common_args(p_h)
    p_h.add_argument("--top-n", type=int, default=50)
    p_h.add_argument("--tiers", nargs="+", default=["A_confirmed", "B_discovery", "C_effector"])
    p_h.add_argument("--sort-by", default="driver_score",
                    choices=["driver_score", "discovery_score", "validation_score"])
    p_h.add_argument("--balanced", action="store_true",
                    help="top (top_n//2) anti + top (top_n//2) pro "
                         "(défaut : top_n global puis split a posteriori).")

    p_e = sub.add_parser("heatmap-expression",
                        help="Heatmap expression du gène × cell_groups (même ordre que cosine)")
    _add_common_args(p_e)
    p_e.add_argument("--top-n", type=int, default=50)
    p_e.add_argument("--tiers", nargs="+", default=["A_confirmed", "B_discovery", "C_effector"])
    p_e.add_argument("--sort-by", default="driver_score",
                    choices=["driver_score", "discovery_score", "validation_score"])
    p_e.add_argument("--expression-mode", default="zscore",
                    choices=["zscore", "raw", "log_raw"])
    p_e.add_argument("--balanced", action="store_true",
                    help="Voir heatmap-clusters --balanced.")

    p_m = sub.add_parser("heatmap-cluster-markers",
                        help="Marqueurs cluster-spécifiques (staircase z-score)")
    _add_common_args(p_m)
    p_m.add_argument("--top-per-cluster", type=int, default=15,
                    help="Nb de gènes-marqueurs sélectionnés par cluster (défaut 15).")
    p_m.add_argument("--restrict-tiers", nargs="*", default=None,
                    help="Restreint l'univers aux tiers donnés (e.g. "
                         "A_confirmed B_discovery). Défaut : tous les gènes.")

    p_q = sub.add_parser("quadrant",
                        help="Quadrant 2×2 biocategory (anti/pro × prolif/sen)")
    _add_common_args(p_q)
    p_q.add_argument("--tiers", nargs="+",
                    default=["A_confirmed", "B_discovery", "C_effector"])
    p_q.add_argument("--label-top-per-quadrant", type=int, default=8,
                    help="Nb de gènes labellisés par quadrant (défaut 8).")

    p_cm = sub.add_parser("clustermap",
                         help="Clustermap hiérarchique top-K anti + top-K pro")
    _add_common_args(p_cm)
    p_cm.add_argument("--n-anti", type=int, default=25)
    p_cm.add_argument("--n-pro", type=int, default=25)
    p_cm.add_argument("--direction", choices=["both", "anti", "pro"],
                     default="both",
                     help="Mode : both (défaut, top-K de chaque), "
                          "anti (top-K anti seuls), pro (top-K pro seuls). "
                          "Override --n-anti/--n-pro selon le mode.")
    p_cm.add_argument("--tiers", nargs="+",
                     default=["A_confirmed", "B_discovery", "C_effector"])
    p_cm.add_argument("--linkage", default="average",
                     choices=["average", "ward", "complete", "single"],
                     help="Méthode de linkage hiérarchique (défaut: average).")
    p_cm.add_argument("--metric", default="correlation",
                     choices=["correlation", "euclidean", "cosine"],
                     help="Métrique de distance (défaut: correlation, "
                          "préférée pour patterns d'expression — "
                          "Eisen et al. 1998).")

    p_t = sub.add_parser("tier-distributions",
                        help="Distributions par evidence_tier")
    _add_common_args(p_t)

    p_b = sub.add_parser("aging-bubbles", help="Bubble plot drivers × aging DBs")
    _add_common_args(p_b)
    p_b.add_argument("--top-n", type=int, default=50)

    p_r = sub.add_parser("radar", help="Radar polar top-N drivers")
    _add_common_args(p_r)
    p_r.add_argument("--top-n", type=int, default=12)

    p_l = sub.add_parser("lollipop-clusters",
                        help="Lollipop drivers cluster-spécifiques")
    _add_common_args(p_l)
    p_l.add_argument("--top-k", type=int, default=10)

    p_a = sub.add_parser("all", help="Toutes les figures avec defaults")
    _add_common_args(p_a)

    args = ap.parse_args()
    global _SVG_ENABLED
    _SVG_ENABLED = getattr(args, "svg", False)
    out_dir = _ensure_outdir(args.out_dir or args.version_dir / "figures_explorer")
    ranking = load_ranking(args.version_dir)

    # Per-cluster only si raw-runs fourni
    per_cluster = None
    if args.raw_runs:
        try:
            per_cluster = load_per_cluster_cosine(args.raw_runs)
        except Exception as e:
            print(f"[warn] per-cluster cosine échec ({e}); fallback ranking-level.")

    if args.cmd == "heatmap-clusters":
        fig_heatmap_clusters(ranking, per_cluster, out_dir,
                            top_n=args.top_n, tiers=args.tiers,
                            sort_by=args.sort_by,
                            balanced=args.balanced)
    elif args.cmd == "heatmap-expression":
        if not args.raw_runs:
            raise SystemExit("--raw-runs requis pour heatmap-expression "
                             "(besoin de group_expression.tsv).")
        group_expr = load_group_expression(args.raw_runs)
        fig_heatmap_expression(ranking, per_cluster, group_expr, out_dir,
                               top_n=args.top_n, tiers=args.tiers,
                               sort_by=args.sort_by,
                               expression_mode=args.expression_mode,
                               balanced=args.balanced)
    elif args.cmd == "heatmap-cluster-markers":
        if not args.raw_runs:
            raise SystemExit("--raw-runs requis pour heatmap-cluster-markers "
                             "(besoin de group_expression.tsv).")
        group_expr = load_group_expression(args.raw_runs)
        fig_heatmap_cluster_markers(group_expr, out_dir,
                                    top_per_cluster=args.top_per_cluster,
                                    ranking=ranking,
                                    restrict_to_tiers=args.restrict_tiers)
    elif args.cmd == "quadrant":
        if not args.raw_runs:
            raise SystemExit("--raw-runs requis pour quadrant "
                             "(besoin de group_expression.tsv).")
        group_expr = load_group_expression(args.raw_runs)
        fig_quadrant_biocategory(ranking, group_expr, out_dir,
                                  tiers=args.tiers,
                                  label_top_per_quadrant=args.label_top_per_quadrant)
    elif args.cmd == "clustermap":
        if not args.raw_runs:
            raise SystemExit("--raw-runs requis pour clustermap "
                             "(besoin de group_expression.tsv).")
        group_expr = load_group_expression(args.raw_runs)
        # Override n_anti/n_pro selon --direction
        n_anti = args.n_anti if args.direction in ("both", "anti") else 0
        n_pro = args.n_pro if args.direction in ("both", "pro") else 0
        fig_clustermap_expression(ranking, group_expr, out_dir,
                                   n_anti=n_anti, n_pro=n_pro,
                                   tiers=args.tiers,
                                   linkage_method=args.linkage,
                                   distance_metric=args.metric)
    elif args.cmd == "tier-distributions":
        fig_tier_distributions(ranking, out_dir)
    elif args.cmd == "aging-bubbles":
        fig_aging_bubbles(ranking, out_dir, top_n=args.top_n)
    elif args.cmd == "radar":
        if per_cluster is None:
            raise SystemExit("--raw-runs requis pour radar (besoin per-cluster cosine).")
        fig_radar(ranking, per_cluster, out_dir, top_n=args.top_n)
    elif args.cmd == "lollipop-clusters":
        if per_cluster is None:
            raise SystemExit("--raw-runs requis pour lollipop (besoin per-cluster cosine).")
        fig_lollipop_clusters(ranking, per_cluster, out_dir, top_k=args.top_k)
    elif args.cmd == "all":
        # Heatmap cosine — 2 versions (global top-50 + balanced 25/25)
        fig_heatmap_clusters(ranking, per_cluster, out_dir, top_n=50,
                            balanced=False)
        fig_heatmap_clusters(ranking, per_cluster, out_dir, top_n=50,
                            balanced=True)
        # Heatmap expression (requiert raw-runs)
        if args.raw_runs:
            try:
                group_expr = load_group_expression(args.raw_runs)
                fig_heatmap_expression(ranking, per_cluster, group_expr, out_dir,
                                       top_n=50, expression_mode="zscore",
                                       balanced=False)
                fig_heatmap_expression(ranking, per_cluster, group_expr, out_dir,
                                       top_n=50, expression_mode="zscore",
                                       balanced=True)
                # Marqueurs cluster-spécifiques (staircase) — 2 versions :
                # tous gènes + restrict à A/B/C drivers
                fig_heatmap_cluster_markers(group_expr, out_dir,
                                            top_per_cluster=15,
                                            ranking=ranking)
                fig_heatmap_cluster_markers(group_expr, out_dir,
                                            top_per_cluster=15,
                                            ranking=ranking,
                                            restrict_to_tiers=["A_confirmed",
                                                               "B_discovery",
                                                               "C_effector"])
                # Quadrant biocategory (anti/pro × prolif/sen)
                fig_quadrant_biocategory(ranking, group_expr, out_dir)
                # Clustermap hiérarchique — 3 versions : mixte + anti-only + pro-only
                fig_clustermap_expression(ranking, group_expr, out_dir,
                                           n_anti=25, n_pro=25)
                fig_clustermap_expression(ranking, group_expr, out_dir,
                                           n_anti=30, n_pro=0)
                fig_clustermap_expression(ranking, group_expr, out_dir,
                                           n_anti=0, n_pro=30)
            except Exception as e:
                print(f"[warn] heatmaps expression/markers/quadrant/clustermap échouées ({e})")
        else:
            print("[warn] --raw-runs absent : heatmap-expression + markers sautés.")
        fig_tier_distributions(ranking, out_dir)
        fig_aging_bubbles(ranking, out_dir, top_n=50)
        if per_cluster is not None:
            fig_radar(ranking, per_cluster, out_dir, top_n=12)
            fig_lollipop_clusters(ranking, per_cluster, out_dir, top_k=10)
        else:
            print("[warn] --raw-runs absent : radar + lollipop sautés.")
        print(f"\n[all] toutes les figures écrites dans {out_dir}")


if __name__ == "__main__":
    main()
