#!/usr/bin/env python3
"""
plot_pathway_heatmap.py — Heatmap pathway × config × direction pour V5.4.1.

Génère :
  - heatmap_pathway_antisene.png/svg  : rang des voies anti-sénescence cross-config
  - heatmap_pathway_prosene.png/svg   : rang des voies pro-sénescence cross-config
  - dotplot_top30_pathways.png/svg    : dot-plot top-30 anti-sén (rang × config)
  - tier1_gene_ranks.png/svg          : rang des gènes Tier-1 cross-config
  - pathway_rank_table.tsv            : tableau brut exportable (anti+pro)

Note directions : la colonne `direction` du cross_seed_pathway_ranking inclut
des variantes `... (mixed)` ; le filtrage se fait par startswith (cf.
build_rank_matrix) pour ne pas perdre les voies à modes partiellement discordants.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Config par défaut
# ---------------------------------------------------------------------------
GOOD_CONFIGS = [
    ("baseline",                       "baseline (s1+s2)",          "#2196F3"),
    ("ex-degrees",                     "ex-degrees (s1+s2)",         "#4CAF50"),
    ("coexpr-legacy-arboreto",         "coexpr-legacy (s1+s2)",     "#9C27B0"),
    ("coexpr-arboreto.quantile-limited","coexpr-ql (s1+s2)",        "#FF9800"),
    ("no-humess",                      "no-humess (s1)",             "#795548"),
    ("no-rfi",                         "no-rfi (s1)",                "#607D8B"),
    ("no-scenic",                      "no-scenic (s1)",             "#F44336"),
    ("no-dedup",                       "no-dedup (s1)",              "#00BCD4"),
    ("no-reactome",                    "no-reactome (s1)",           "#FF5722"),
]

# Nettoyage du nom de pathway pour affichage
def _clean_pw(pw: str) -> str:
    s = pw.replace("pw_", "").replace("_", " ")
    # Abrège les noms très longs
    words = s.split()
    if len(words) > 7:
        s = " ".join(words[:7]) + "…"
    return s.title()


def load_rankings(reports_dir: Path, configs: list) -> dict[str, pd.DataFrame]:
    """Charge les pathway_rankings pour chaque config."""
    dfs = {}
    for cfg_key, cfg_label, _ in configs:
        fpath = reports_dir / cfg_key / "cross_seed_pathway_ranking.tsv"
        if not fpath.exists():
            print(f"  [skip] {cfg_key} : {fpath} absent")
            continue
        df = pd.read_csv(fpath, sep="\t")
        df = df.sort_values("driver_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        dfs[cfg_label] = df
    return dfs


def build_rank_matrix(dfs: dict, direction: str, top_n_select: int = 50) -> pd.DataFrame:
    """
    Construit la matrice pathway × config avec le rang.
    Garde les pathways présents dans le top-N d'au moins 1 config.

    `direction` = "anti-senescence" | "pro-senescence". On inclut les variantes
    `... (mixed)` (modes KO/KD/OE en désaccord partiel mais direction primaire
    identique) via startswith — sinon la heatmap pro-sénescence perd ~2/3 des
    voies (66 pures vs 124 mixed sur baseline).
    """
    def _match_dir(s: pd.Series) -> pd.Series:
        return s.str.startswith(direction)

    # Collect top-N pathways per config for given direction
    all_pw = set()
    for df in dfs.values():
        sub = df[_match_dir(df["direction"])].head(top_n_select)
        all_pw.update(sub["pathway"].tolist())

    rows = []
    for pw in all_pw:
        row = {"pathway": pw}
        for label, df in dfs.items():
            sub = df[_match_dir(df["direction"])]
            match = sub[sub["pathway"] == pw]
            if len(match) > 0:
                row[label] = int(match.iloc[0]["rank"])
            else:
                row[label] = np.nan
        rows.append(row)

    mat = pd.DataFrame(rows).set_index("pathway")
    # Sort by median rank (best = lowest)
    mat["_med"] = mat.apply(lambda r: np.nanmedian(r.dropna()), axis=1)
    mat["_n"] = mat.apply(lambda r: r.dropna().shape[0], axis=1)
    mat = mat.sort_values(["_n", "_med"], ascending=[False, True]).drop(columns=["_med", "_n"])
    return mat


def plot_heatmap(mat: pd.DataFrame, configs: list, direction: str,
                 out_dir: Path, max_rows: int = 40, vmax: int = 200):
    """Heatmap rang pathway × config."""
    mat_plot = mat.head(max_rows).copy()
    cols_ordered = [lbl for _, lbl, _ in configs if lbl in mat_plot.columns]
    mat_plot = mat_plot[cols_ordered]

    n_pw, n_cfg = mat_plot.shape
    fig_h = max(6, n_pw * 0.28)
    fig_w = max(8, n_cfg * 1.1 + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Normalise rang : 1=vert foncé, vmax+=gris clair, nan=blanc
    data = mat_plot.values.astype(float)
    data_clipped = np.where(np.isnan(data), np.nan, np.clip(data, 1, vmax))

    # Custom colormap : vert→jaune→rouge (bas rang = bon)
    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color="#EEEEEE")

    im = ax.imshow(data_clipped, aspect="auto", cmap=cmap,
                   vmin=1, vmax=vmax, interpolation="nearest")

    # Annotations : rang numérique
    for i in range(n_pw):
        for j in range(n_cfg):
            v = data[i, j]
            if np.isnan(v):
                continue
            txt = str(int(v)) if v < 1000 else f"{int(v//1000)}k"
            fontsize = 7 if n_pw > 25 else 8
            color = "white" if v < vmax * 0.35 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=fontsize, color=color, fontweight="bold")

    ax.set_xticks(range(n_cfg))
    ax.set_xticklabels([c.replace(" (s1+s2)", "\n(s1+s2)").replace(" (s1)", "\n(s1)")
                        for c in cols_ordered],
                       fontsize=8, rotation=0, ha="center")
    ax.set_yticks(range(n_pw))
    ax.set_yticklabels([_clean_pw(pw) for pw in mat_plot.index], fontsize=8)

    dir_label = "Anti-sénescence" if direction == "anti-senescence" else "Pro-sénescence"
    color_title = "#1B5E20" if direction == "anti-senescence" else "#B71C1C"
    ax.set_title(f"Pathway drivers — {dir_label}\nRang cross-configs V5.4.1 (vert=top, gris=absent top-{vmax})",
                 fontsize=10, color=color_title, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("Rang pathway", fontsize=8)

    plt.tight_layout()
    stem = f"heatmap_pathway_{'anti' if direction=='anti-senescence' else 'pro'}sene"
    for ext in ("png", "svg"):
        fpath = out_dir / f"{stem}.{ext}"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"  Saved {fpath}")
    plt.close(fig)


def plot_dotplot(mat_anti: pd.DataFrame, configs: list, out_dir: Path,
                 top_n: int = 30, max_rank_show: int = 150):
    """Dot-plot : taille = score (inverted rank), couleur = config."""
    cols = [lbl for _, lbl, _ in configs if lbl in mat_anti.columns]
    cfg_colors = {lbl: col for _, lbl, col in configs}
    mat_top = mat_anti.head(top_n)[cols].copy()
    pw_labels = [_clean_pw(pw) for pw in mat_top.index]

    fig_w = max(10, len(cols) * 1.2 + 4)
    fig_h = max(7, top_n * 0.3 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for j, cfg_label in enumerate(cols):
        for i, pw in enumerate(mat_top.index):
            rank = mat_top.loc[pw, cfg_label]
            if np.isnan(rank) or rank > max_rank_show:
                # Point gris petit pour "absent du top"
                ax.scatter(j, top_n - 1 - i, s=20, color="#CCCCCC",
                           zorder=2, edgecolors="none")
                continue
            size = max(20, 400 * (1 - rank / max_rank_show))
            ax.scatter(j, top_n - 1 - i, s=size,
                       color=cfg_colors.get(cfg_label, "#888888"),
                       alpha=0.85, zorder=3, edgecolors="white", linewidth=0.5)
            if rank <= 10:
                ax.text(j, top_n - 1 - i, str(int(rank)),
                        ha="center", va="center", fontsize=6,
                        color="white", fontweight="bold", zorder=4)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.split(" (")[0] for c in cols],
                       rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(list(reversed(pw_labels)), fontsize=8)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(-0.5, top_n - 0.5)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    ax.set_title(f"Top-{top_n} pathways anti-sénescence × configs\n"
                 f"Taille ∝ rang (gros=top), gris=absent top-{max_rank_show}",
                 fontsize=10, fontweight="bold")

    handles = [Patch(facecolor=cfg_colors.get(lbl, "#888"), label=lbl.split(" (")[0])
               for lbl in cols]
    ax.legend(handles=handles, fontsize=8, loc="lower right",
              framealpha=0.8, ncol=2)

    plt.tight_layout()
    for ext in ("png", "svg"):
        fpath = out_dir / f"dotplot_top{top_n}_pathways.{ext}"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"  Saved {fpath}")
    plt.close(fig)


def plot_tier1_gene_ranks(reports_dir: Path, configs: list, out_dir: Path):
    """Barplot groupé : rang des gènes Tier-1 cross-config."""
    TIER1 = ["H2AFZ", "HMGB1", "HMGB2", "ENO1", "CYCS", "FHL2", "CD59", "TP53", "MYC", "ASNS"]
    ranks = {}
    cfg_labels = []
    for cfg_key, cfg_label, _ in configs:
        fpath = reports_dir / cfg_key / "cross_seed_gene_ranking.tsv"
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, sep="\t")
        df_sorted = df.sort_values("driver_score", ascending=False).reset_index(drop=True)
        df_sorted["rank"] = df_sorted.index + 1
        cfg_labels.append(cfg_label)
        for g in TIER1:
            r = df_sorted.loc[df_sorted["target"] == g, "rank"]
            ranks.setdefault(g, {})[cfg_label] = int(r.iloc[0]) if len(r) else None

    n_genes = len(TIER1)
    n_cfg = len(cfg_labels)
    fig, axes = plt.subplots(1, n_genes, figsize=(n_genes * 1.6, 5), sharey=False)
    cfg_colors = {lbl: col for _, lbl, col in configs}

    for ax, gene in zip(axes, TIER1):
        vals = [ranks.get(gene, {}).get(lbl) for lbl in cfg_labels]
        colors = [cfg_colors.get(lbl, "#888") for lbl in cfg_labels]
        ys = [v if v is not None else 10000 for v in vals]
        bars = ax.barh(range(n_cfg), ys, color=colors, alpha=0.85)
        ax.set_yticks(range(n_cfg))
        if ax == axes[0]:
            ax.set_yticklabels([lbl.split(" (")[0] for lbl in cfg_labels], fontsize=7)
        else:
            ax.set_yticklabels([""] * n_cfg)
        ax.set_title(gene, fontsize=9, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlim(0.5, 12000)
        ax.axvline(50, color="orange", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(200, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        for i, v in enumerate(ys):
            if v <= 200:
                ax.text(v + 0.5, i, str(v), va="center", fontsize=6)

    fig.suptitle("Rang des gènes Tier-1 cross-configs V5.4.1\n"
                 "(orange=top50, rouge=top200 — plus c'est à gauche, mieux c'est)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "svg"):
        fpath = out_dir / f"tier1_gene_ranks.{ext}"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"  Saved {fpath}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", type=Path,
                    default=Path("output/interpretation/V5.4.1/cross_seed"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("output/interpretation/V5.4.1/pathway_summary"))
    ap.add_argument("--top-n-pw", type=int, default=40,
                    help="Nb voies à afficher dans la heatmap (défaut 40)")
    ap.add_argument("--top-n-dot", type=int, default=30,
                    help="Nb voies dans le dot-plot (défaut 30)")
    ap.add_argument("--vmax", type=int, default=200,
                    help="Rang max affiché dans la heatmap (défaut 200)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pathway_heatmap] reports : {args.reports_dir}")
    print(f"[pathway_heatmap] output  : {args.out_dir}")

    dfs = load_rankings(args.reports_dir, GOOD_CONFIGS)
    if not dfs:
        print("Aucun ranking trouvé. Vérifier --reports-dir.")
        sys.exit(1)
    print(f"  {len(dfs)} configs chargées")

    # Matrices rang
    mat_anti = build_rank_matrix(dfs, "anti-senescence", top_n_select=80)
    mat_pro  = build_rank_matrix(dfs, "pro-senescence",  top_n_select=80)

    # Export TSV brut
    rank_table = pd.concat([
        mat_anti.assign(direction="anti-senescence"),
        mat_pro.assign(direction="pro-senescence"),
    ])
    rank_table.to_csv(args.out_dir / "pathway_rank_table.tsv", sep="\t")
    print(f"  Saved {args.out_dir}/pathway_rank_table.tsv")

    # Heatmaps
    plot_heatmap(mat_anti, GOOD_CONFIGS, "anti-senescence",
                 args.out_dir, max_rows=args.top_n_pw, vmax=args.vmax)
    plot_heatmap(mat_pro,  GOOD_CONFIGS, "pro-senescence",
                 args.out_dir, max_rows=min(25, args.top_n_pw), vmax=args.vmax)

    # Dot-plot
    plot_dotplot(mat_anti, GOOD_CONFIGS, args.out_dir,
                 top_n=args.top_n_dot, max_rank_show=args.vmax)

    # Tier-1 gene ranks
    plot_tier1_gene_ranks(args.reports_dir, GOOD_CONFIGS, args.out_dir)

    print("[pathway_heatmap] DONE")


if __name__ == "__main__":
    main()
