#!/usr/bin/env python3
"""
perturb_report.py — Aggregate & visualise GNN perturbation results.

Walks a perturbation directory (produced by gnn_perturbation.py or
perturb_top_genes.py) and produces a comparison table plus a handful of
figures.

Usage
-----
    python src/perturb_report.py \\
        --perturb-dir output/gnn_vgae/V3_Run3/perturbation

    # Restrict to a subset (glob patterns against sub-directory names)
    python src/perturb_report.py \\
        --perturb-dir output/gnn_vgae/V3_Run3/perturbation \\
        --include "knockdown_*" "overexpress_*"

Outputs (inside <perturb-dir>/report/)
--------------------------------------
    comparison_table.tsv        — headline stats per perturbation
    overview_movers.png         — max-up / max-down delta_rank per run
    overview_updown.png         — n_rising vs n_falling per run
    pathway_heatmap.png         — top pathways × perturbation (-log10 p.adj)
    top_risers_overlap.png      — Jaccard matrix of top-100 risers
    delta_rank_dist.png         — violin of |delta_rank| per run
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 10})


def iter_runs(perturb_dir: Path,
              includes: list[str] | None) -> list[Path]:
    runs = []
    for child in sorted(perturb_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "summary.json").exists():
            continue
        if includes and not any(fnmatch.fnmatch(child.name, p)
                                for p in includes):
            continue
        runs.append(child)
    return runs


def load_summary(run: Path) -> dict:
    with open(run / "summary.json") as f:
        s = json.load(f)
    s["tag"] = run.name
    return s


def load_top_ora(run: Path, k: int = 5) -> pd.DataFrame:
    fp = run / "delta_ora_top_up_reactome.tsv"
    if not fp.exists():
        return pd.DataFrame(columns=["pathway", "p_adj"])
    df = pd.read_csv(fp, sep="\t")
    return df.head(k)[["pathway", "p_adj"]]


def load_top_risers(run: Path, k: int = 100) -> set[str]:
    fp = run / "delta_ranking.csv"
    if not fp.exists():
        return set()
    df = pd.read_csv(fp, usecols=["gene", "delta_rank", "is_target"])
    df = df[df["is_target"] == 0].sort_values("delta_rank", ascending=False)
    return set(df["gene"].head(k).astype(str).tolist())


def load_delta_ranks(run: Path) -> np.ndarray:
    fp = run / "delta_ranking.csv"
    if not fp.exists():
        return np.array([])
    df = pd.read_csv(fp, usecols=["delta_rank", "is_target"])
    return df.loc[df["is_target"] == 0, "delta_rank"].to_numpy()


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_movers(table: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(table)), 5))
    x = np.arange(len(table))
    ax.bar(x, table["max_up_delta_rank"], width=0.4,
           color="#2a9d8f", label="max up (Δrank)")
    ax.bar(x, table["max_down_delta_rank"], width=0.4,
           color="#e76f51", label="max down (Δrank)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(table["tag"], rotation=90, fontsize=7)
    ax.set_ylabel("Δrank (signed)")
    ax.set_title("Top movers per perturbation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_updown(table: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(table)), 5))
    x = np.arange(len(table))
    w = 0.4
    ax.bar(x - w / 2, table["n_rising"], width=w,
           color="#2a9d8f", label="rising")
    ax.bar(x + w / 2, table["n_falling"], width=w,
           color="#e76f51", label="falling")
    ax.set_xticks(x)
    ax.set_xticklabels(table["tag"], rotation=90, fontsize=7)
    ax.set_ylabel("# genes")
    ax.set_title("Rising vs falling genes per perturbation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_pathway_heatmap(runs: list[Path], out: Path,
                        top_per_run: int = 5,
                        max_pathways: int = 30):
    # collect top pathways per run
    per_run = {run.name: load_top_ora(run, k=top_per_run) for run in runs}
    picked: list[str] = []
    for _, df in per_run.items():
        for pw in df["pathway"].tolist():
            if pw not in picked:
                picked.append(pw)
            if len(picked) >= max_pathways:
                break
        if len(picked) >= max_pathways:
            break

    if not picked:
        print("[warn] no pathways to plot")
        return

    mat = np.full((len(picked), len(per_run)), np.nan)
    for j, (tag, df) in enumerate(per_run.items()):
        lookup = dict(zip(df["pathway"], df["p_adj"]))
        for i, pw in enumerate(picked):
            if pw in lookup and lookup[pw] > 0:
                mat[i, j] = -np.log10(lookup[pw])

    labels = [p.replace("REACTOME_", "")[:55] for p in picked]

    fig, ax = plt.subplots(
        figsize=(max(6, 0.4 * len(per_run)),
                 max(4, 0.28 * len(picked))))
    sns.heatmap(mat, ax=ax,
                xticklabels=list(per_run.keys()),
                yticklabels=labels,
                cmap="viridis",
                cbar_kws={"label": "-log10 p.adj"},
                linewidths=0.3, linecolor="white")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    ax.set_title(f"Top-{top_per_run} pathways per perturbation "
                 f"(-log10 p.adj)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_riser_overlap(runs: list[Path], out: Path, k: int = 100):
    sets = {run.name: load_top_risers(run, k=k) for run in runs}
    names = list(sets.keys())
    n = len(names)
    m = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            sa, sb = sets[a], sets[b]
            union = sa | sb
            m[i, j] = (len(sa & sb) / len(union)) if union else 0.0

    fig, ax = plt.subplots(figsize=(max(6, 0.35 * n),
                                     max(5, 0.35 * n)))
    sns.heatmap(m, ax=ax,
                xticklabels=names, yticklabels=names,
                cmap="rocket_r", vmin=0, vmax=1,
                cbar_kws={"label": f"Jaccard (top-{k} risers)"},
                square=True, linewidths=0.3, linecolor="white")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    ax.set_title(f"Top-{k} riser overlap between perturbations")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_delta_dist(runs: list[Path], out: Path,
                   max_runs: int = 25):
    data = []
    for run in runs[:max_runs]:
        dr = load_delta_ranks(run)
        if dr.size:
            data.append((run.name, np.abs(dr)))
    if not data:
        return

    fig, ax = plt.subplots(
        figsize=(max(7, 0.35 * len(data)), 5))
    parts = ax.violinplot([d for _, d in data], showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#5e81ac")
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([n for n, _ in data], rotation=90, fontsize=7)
    ax.set_yscale("symlog")
    ax.set_ylabel("|Δrank| (symlog)")
    ax.set_title("Distribution of |Δrank| per perturbation")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perturb-dir", required=True, type=Path,
                    help="Directory holding one sub-directory per "
                         "perturbation run (each with summary.json).")
    ap.add_argument("--include", nargs="+", default=None,
                    help="Glob pattern(s) on run folder names to keep.")
    ap.add_argument("--top-per-run", type=int, default=5,
                    help="Top-k pathways per run used for the heatmap.")
    ap.add_argument("--top-k-risers", type=int, default=100)
    args = ap.parse_args()

    runs = iter_runs(args.perturb_dir, args.include)
    if not runs:
        print("No runs found.")
        return
    print(f"Found {len(runs)} perturbation run(s).")

    report_dir = args.perturb_dir / "report"
    report_dir.mkdir(exist_ok=True)

    # ---- comparison table ----
    rows = []
    for run in runs:
        s = load_summary(run)
        top_pw = s.get("top5_delta_pathways") or []
        rows.append({
            "tag": s["tag"],
            "mode": s.get("mode", ""),
            "n_targets": s.get("n_targets_in_graph", 0),
            "targets": ",".join((s.get("targets_in_graph") or [])[:5]),
            "n_rising": s.get("n_rising", 0),
            "n_falling": s.get("n_falling", 0),
            "median_abs_delta_rank": s.get("median_abs_delta_rank", 0),
            "max_up_gene": s.get("max_up_gene", ""),
            "max_up_delta_rank": s.get("max_up_delta_rank", 0),
            "max_down_gene": s.get("max_down_gene", ""),
            "max_down_delta_rank": s.get("max_down_delta_rank", 0),
            "n_sig_delta_pathways": s.get("n_sig_delta_pathways", 0),
            "top1_pathway": top_pw[0] if top_pw else "",
        })
    table = pd.DataFrame(rows)
    table.to_csv(report_dir / "comparison_table.tsv",
                 sep="\t", index=False)
    print(f"Wrote {report_dir / 'comparison_table.tsv'}")

    # ---- figures ----
    fig_movers(table, report_dir / "overview_movers.png")
    fig_updown(table, report_dir / "overview_updown.png")
    fig_pathway_heatmap(runs, report_dir / "pathway_heatmap.png",
                        top_per_run=args.top_per_run)
    fig_riser_overlap(runs, report_dir / "top_risers_overlap.png",
                      k=args.top_k_risers)
    fig_delta_dist(runs, report_dir / "delta_rank_dist.png")
    print(f"Wrote figures into {report_dir}")


if __name__ == "__main__":
    main()
