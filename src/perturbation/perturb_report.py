#!/usr/bin/env python3
"""
perturb_report.py — Aggregate & visualise GNN perturbation results.

Walks a perturbation directory (produced by gnn_perturbation.py or
perturb_top_genes.py) and produces a comparison table plus a set of
figures, split by perturbation mode (knockdown / knockout / overexpress).

Artifact filter
---------------
Some genes (often lncRNAs / pseudogenes like NPPA-AS1, RP1-140K8.5) move
up by hundreds of ranks across *many unrelated* perturbations, because
their baseline importance is dominated by components that don't change
under perturbation (e.g. low `vgae_specificity`) — this is a scoring
artifact, not a biological signal.

We build a data-driven blocklist **per mode**: any gene appearing in the
top-K risers of more than `--universal-threshold` (default 0.30 = 30%)
of the runs *of that mode* is considered a universal riser and filtered
out before computing max_up/max_down and the top-K riser sets.
Per-mode is important because artifacts tend to be mode-specific:
NPPA-AS1 rises in ~60% of knockout runs but is absent from knockdown /
overexpress runs, so a global threshold would miss it.

`--min-baseline-pct` (default 25) additionally drops bottom-percentile
importance genes.

Usage
-----
    python src/perturb_report.py \\
        --perturb-dir output/gnn_vgae/V3_Run3/perturbation

    # Looser / stricter artifact filter
    python src/perturb_report.py \\
        --perturb-dir output/gnn_vgae/V3_Run3/perturbation \\
        --min-baseline-pct 10

Outputs (inside <perturb-dir>/report/)
--------------------------------------
    comparison_table.tsv              — headline stats per perturbation
    comparison_table_filtered.tsv     — same, with max_up/down recomputed
                                         after the baseline-importance filter
    overview_movers_<mode>.png        — filtered max-up / max-down per mode
    overview_updown_<mode>.png        — rising vs falling per mode
    pathway_heatmap_<mode>.png        — top pathways × perturbation (-log10 p.adj)
    top_risers_overlap_<mode>.png     — Jaccard matrix (ordered + hierarchical)
    top_risers_clustermap_<mode>.png  — clustered Jaccard with dendrograms
    delta_rank_dist_<mode>.png        — violin of |Δrank| per run
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

MODES = ("knockdown", "knockout", "overexpress")


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
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
    empty = pd.DataFrame(columns=["pathway", "p_adj"])
    if not fp.exists() or fp.stat().st_size == 0:
        return empty
    try:
        df = pd.read_csv(fp, sep="\t")
    except pd.errors.EmptyDataError:
        return empty
    if df.empty or not {"pathway", "p_adj"}.issubset(df.columns):
        return empty
    return df.head(k)[["pathway", "p_adj"]]


def _load_delta(run: Path) -> pd.DataFrame:
    fp = run / "delta_ranking.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp, usecols=["gene", "baseline_importance",
                                   "delta_rank", "is_target"])
    return df[df["is_target"] == 0].copy()


def _apply_filters(df: pd.DataFrame,
                   min_baseline_pct: float,
                   blocklist: set[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df
    if min_baseline_pct > 0:
        thr = out["baseline_importance"].quantile(min_baseline_pct / 100.0)
        out = out[out["baseline_importance"] >= thr]
    if blocklist:
        out = out[~out["gene"].isin(blocklist)]
    return out


def build_universal_blocklist(runs: list[Path],
                              top_k: int = 200,
                              threshold: float = 0.20) -> tuple[set[str], pd.DataFrame]:
    """Genes appearing in top-K risers of > `threshold` fraction of runs."""
    from collections import Counter
    ctr: Counter = Counter()
    n = 0
    for run in runs:
        df = _load_delta(run)
        if df.empty:
            continue
        n += 1
        top = df.sort_values("delta_rank", ascending=False).head(top_k)
        ctr.update(top["gene"].astype(str).tolist())
    if n == 0:
        return set(), pd.DataFrame()
    rows = [(g, c, c / n) for g, c in ctr.items() if c / n >= threshold]
    rows.sort(key=lambda r: -r[1])
    report = pd.DataFrame(rows, columns=["gene", "n_runs", "frequency"])
    return set(report["gene"]), report


def filtered_movers(run: Path, min_baseline_pct: float,
                    blocklist: set[str]) -> dict:
    df = _apply_filters(_load_delta(run), min_baseline_pct, blocklist)
    if df.empty:
        return {"max_up_gene": "", "max_up_delta_rank": 0,
                "max_down_gene": "", "max_down_delta_rank": 0}
    up = df.sort_values("delta_rank", ascending=False).iloc[0]
    dn = df.sort_values("delta_rank").iloc[0]
    return {
        "max_up_gene": str(up["gene"]),
        "max_up_delta_rank": int(up["delta_rank"]),
        "max_down_gene": str(dn["gene"]),
        "max_down_delta_rank": int(dn["delta_rank"]),
    }


def load_top_risers(run: Path, k: int,
                    min_baseline_pct: float,
                    blocklist: set[str]) -> set[str]:
    df = _apply_filters(_load_delta(run), min_baseline_pct, blocklist)
    if df.empty:
        return set()
    df = df.sort_values("delta_rank", ascending=False)
    return set(df["gene"].head(k).astype(str).tolist())


def load_delta_ranks(run: Path, min_baseline_pct: float,
                     blocklist: set[str]) -> np.ndarray:
    df = _apply_filters(_load_delta(run), min_baseline_pct, blocklist)
    return df["delta_rank"].to_numpy() if not df.empty else np.array([])


def infer_mode_from_tag(tag: str) -> str:
    for m in MODES:
        if tag.startswith(m):
            return m
    return "other"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_movers(table: pd.DataFrame, out: Path, title: str):
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(table)), 5))
    x = np.arange(len(table))
    ax.bar(x, table["max_up_delta_rank"], width=0.5,
           color="#2a9d8f", label="max up (Δrank)")
    ax.bar(x, table["max_down_delta_rank"], width=0.5,
           color="#e76f51", label="max down (Δrank)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(table["tag"], rotation=90, fontsize=7)
    ax.set_ylabel("Δrank (signed)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_updown(table: pd.DataFrame, out: Path, title: str):
    if table.empty:
        return
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
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_pathway_heatmap(runs: list[Path], out: Path, title: str,
                        top_per_run: int = 5, max_pathways: int = 30):
    if not runs:
        return
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
        return

    mat = np.full((len(picked), len(per_run)), np.nan)
    for j, (_tag, df) in enumerate(per_run.items()):
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
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _jaccard_matrix(sets: dict[str, set[str]]) -> tuple[np.ndarray, list[str]]:
    names = list(sets.keys())
    n = len(names)
    m = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            sa, sb = sets[a], sets[b]
            union = sa | sb
            m[i, j] = (len(sa & sb) / len(union)) if union else 0.0
    return m, names


def fig_riser_overlap(runs: list[Path], out: Path, title: str,
                      k: int, min_baseline_pct: float,
                      blocklist: set[str]):
    if not runs:
        return
    sets = {run.name: load_top_risers(run, k, min_baseline_pct, blocklist)
            for run in runs}
    sets = {n: s for n, s in sets.items() if s}
    if len(sets) < 2:
        return
    mat, names = _jaccard_matrix(sets)

    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(names)),
                                     max(5, 0.35 * len(names))))
    sns.heatmap(mat, ax=ax,
                xticklabels=names, yticklabels=names,
                cmap="rocket_r", vmin=0, vmax=1,
                cbar_kws={"label": f"Jaccard (top-{k} risers)"},
                square=True, linewidths=0.3, linecolor="white")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_riser_clustermap(runs: list[Path], out: Path, title: str,
                         k: int, min_baseline_pct: float,
                         blocklist: set[str]) -> pd.DataFrame:
    """Hierarchical clustermap of top-k riser Jaccard. Returns the
    distance-based cluster label table."""
    if not runs:
        return pd.DataFrame()
    sets = {run.name: load_top_risers(run, k, min_baseline_pct, blocklist)
            for run in runs}
    sets = {n: s for n, s in sets.items() if s}
    if len(sets) < 3:
        return pd.DataFrame()
    mat, names = _jaccard_matrix(sets)

    # Distance = 1 - Jaccard; clustermap needs a square DataFrame.
    dist = 1.0 - mat
    df = pd.DataFrame(mat, index=names, columns=names)

    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    cond = squareform(dist, checks=False)
    Z = linkage(cond, method="average")

    # Auto cluster count: try k=min(6, n//3), min 2
    n_clusters = max(2, min(6, len(names) // 3))
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    g = sns.clustermap(df, row_linkage=Z, col_linkage=Z,
                       cmap="rocket_r", vmin=0, vmax=1,
                       figsize=(max(7, 0.38 * len(names)),
                                max(7, 0.38 * len(names))),
                       cbar_kws={"label": f"Jaccard (top-{k} risers)"},
                       xticklabels=True, yticklabels=True,
                       linewidths=0.2, linecolor="white")
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(),
                                 rotation=90, fontsize=7)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(),
                                 rotation=0, fontsize=7)
    g.figure.suptitle(title, y=1.02)
    g.savefig(out)
    plt.close(g.figure)

    return pd.DataFrame({"tag": names, "cluster": labels})


def cluster_signatures(runs: list[Path],
                       labels: dict[str, int],
                       k: int,
                       min_baseline_pct: float,
                       blocklist: set[str],
                       consensus_frac: float = 0.5,
                       top_genes: int = 15,
                       top_pathways: int = 5) -> pd.DataFrame:
    """For each cluster, return the consensus riser genes + pathways.

    A gene is in a cluster's signature if it appears in the top-k risers
    of at least `consensus_frac` of the cluster's runs.
    """
    from collections import Counter
    rows = []
    by_cluster: dict[int, list[Path]] = {}
    for run in runs:
        c = labels.get(run.name)
        if c is None:
            continue
        by_cluster.setdefault(c, []).append(run)

    for cluster, members in sorted(by_cluster.items()):
        n = len(members)
        gene_ctr: Counter = Counter()
        pw_ctr: Counter = Counter()
        for r in members:
            gene_ctr.update(load_top_risers(r, k, min_baseline_pct, blocklist))
            pw = load_top_ora(r, k=top_pathways)
            pw_ctr.update(pw["pathway"].tolist())
        min_hits = max(2, int(np.ceil(consensus_frac * n)))
        genes = [(g, c) for g, c in gene_ctr.most_common()
                 if c >= min_hits][:top_genes]
        pathways = [(p, c) for p, c in pw_ctr.most_common()
                    if c >= min_hits][:top_pathways]
        rows.append({
            "cluster": cluster,
            "n_members": n,
            "members": ", ".join(r.name.split("_", 1)[1] for r in members),
            "consensus_genes": "; ".join(f"{g}({c}/{n})"
                                          for g, c in genes),
            "consensus_pathways": "; ".join(
                f"{p.replace('REACTOME_','')}({c}/{n})"
                for p, c in pathways),
        })
    return pd.DataFrame(rows)


def fig_delta_dist(runs: list[Path], out: Path, title: str,
                   min_baseline_pct: float, blocklist: set[str],
                   max_runs: int = 30):
    if not runs:
        return
    data = []
    for run in runs[:max_runs]:
        dr = load_delta_ranks(run, min_baseline_pct, blocklist)
        if dr.size:
            data.append((run.name, np.abs(dr)))
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(data)), 5))
    parts = ax.violinplot([d for _, d in data], showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#5e81ac")
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([n for n, _ in data], rotation=90, fontsize=7)
    ax.set_yscale("symlog")
    ax.set_ylabel("|Δrank| (symlog)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perturb-dir", required=True, type=Path)
    ap.add_argument("--include", nargs="+", default=None,
                    help="Glob pattern(s) on run folder names to keep.")
    ap.add_argument("--top-per-run", type=int, default=5)
    ap.add_argument("--top-k-risers", type=int, default=100)
    ap.add_argument("--min-baseline-pct", type=float, default=25.0,
                    help="Drop genes whose baseline_importance is below "
                         "this percentile before computing max_up/down and "
                         "top-k risers (default 25).")
    ap.add_argument("--universal-threshold", type=float, default=0.45,
                    help="Fraction of runs (within a mode) above which a "
                         "gene is flagged as a universal riser (artifact) "
                         "and dropped. Tuned to catch NPPA-AS1-style "
                         "artifacts (~50–75%% plateau) while sparing "
                         "biological convergence like BBS/PEX (~30%%). "
                         "Default 0.45.")
    ap.add_argument("--universal-top-k", type=int, default=100,
                    help="Top-K risers scanned when computing the "
                         "universal-riser blocklist (default 100).")
    args = ap.parse_args()

    runs = iter_runs(args.perturb_dir, args.include)
    if not runs:
        print("No runs found.")
        return

    report_dir = args.perturb_dir / "report"
    report_dir.mkdir(exist_ok=True)

    # Per-mode blocklists — artifacts tend to be mode-specific.
    blocklists: dict[str, set[str]] = {}
    block_reports: list[pd.DataFrame] = []
    for m in MODES:
        m_runs = [r for r in runs if infer_mode_from_tag(r.name) == m]
        if not m_runs:
            blocklists[m] = set()
            continue
        bl, rep = build_universal_blocklist(
            m_runs, top_k=args.universal_top_k,
            threshold=args.universal_threshold)
        blocklists[m] = bl
        if not rep.empty:
            rep = rep.assign(mode=m)
            block_reports.append(rep)
    # Union for the global comparison table (mode-agnostic view).
    blocklist_global = set().union(*blocklists.values())
    if block_reports:
        pd.concat(block_reports, ignore_index=True).to_csv(
            report_dir / "universal_risers_blocklist.tsv",
            sep="\t", index=False)
    print(f"Found {len(runs)} perturbation run(s). "
          f"Baseline filter: pct ≥ {args.min_baseline_pct}  "
          f"| Blocklists ≥{args.universal_threshold:.0%} of runs per mode:")
    for m in MODES:
        bl = blocklists[m]
        shown = ', '.join(sorted(bl)[:8]) + ('...' if len(bl) > 8 else '')
        print(f"  [{m}] {len(bl)} genes{': ' + shown if bl else ''}")

    # ---- comparison tables ----
    raw_rows, filt_rows = [], []
    for run in runs:
        s = load_summary(run)
        top_pw = s.get("top5_delta_pathways") or []
        base = {
            "tag": s["tag"],
            "mode": s.get("mode", infer_mode_from_tag(s["tag"])),
            "n_targets": s.get("n_targets_in_graph", 0),
            "targets": ",".join((s.get("targets_in_graph") or [])[:5]),
            "n_rising": s.get("n_rising", 0),
            "n_falling": s.get("n_falling", 0),
            "median_abs_delta_rank": s.get("median_abs_delta_rank", 0),
            "n_sig_delta_pathways": s.get("n_sig_delta_pathways", 0),
            "top1_pathway": top_pw[0] if top_pw else "",
        }
        raw = dict(base)
        raw.update({
            "max_up_gene": s.get("max_up_gene", ""),
            "max_up_delta_rank": s.get("max_up_delta_rank", 0),
            "max_down_gene": s.get("max_down_gene", ""),
            "max_down_delta_rank": s.get("max_down_delta_rank", 0),
        })
        raw_rows.append(raw)

        run_mode = infer_mode_from_tag(run.name)
        mv = filtered_movers(run, args.min_baseline_pct,
                             blocklists.get(run_mode, set()))
        filt = dict(base)
        filt.update(mv)
        filt_rows.append(filt)

    raw_df = pd.DataFrame(raw_rows)
    filt_df = pd.DataFrame(filt_rows)
    raw_df.to_csv(report_dir / "comparison_table.tsv",
                  sep="\t", index=False)
    filt_df.to_csv(report_dir / "comparison_table_filtered.tsv",
                   sep="\t", index=False)
    print(f"Wrote {report_dir / 'comparison_table.tsv'}")
    print(f"Wrote {report_dir / 'comparison_table_filtered.tsv'}")

    # ---- per-mode figures ----
    cluster_frames: list[pd.DataFrame] = []
    for mode in MODES:
        mode_runs = [r for r in runs
                     if infer_mode_from_tag(r.name) == mode]
        mode_tab = filt_df[filt_df["tag"].apply(infer_mode_from_tag) == mode]
        if not mode_runs:
            continue
        bl = blocklists.get(mode, set())
        print(f"\n[{mode}] {len(mode_runs)} runs | blocklist={len(bl)}")

        fig_movers(mode_tab, report_dir / f"overview_movers_{mode}.png",
                   f"Top movers — {mode} (filtered pct≥{args.min_baseline_pct})")
        fig_updown(mode_tab, report_dir / f"overview_updown_{mode}.png",
                   f"Rising vs falling — {mode}")
        fig_pathway_heatmap(
            mode_runs, report_dir / f"pathway_heatmap_{mode}.png",
            f"Top pathways × perturbation — {mode} (-log10 p.adj)",
            top_per_run=args.top_per_run)
        fig_riser_overlap(
            mode_runs, report_dir / f"top_risers_overlap_{mode}.png",
            f"Top-{args.top_k_risers} riser Jaccard — {mode}",
            k=args.top_k_risers, min_baseline_pct=args.min_baseline_pct,
            blocklist=bl)
        clusters = fig_riser_clustermap(
            mode_runs, report_dir / f"top_risers_clustermap_{mode}.png",
            f"Clustered top-{args.top_k_risers} riser Jaccard — {mode}",
            k=args.top_k_risers, min_baseline_pct=args.min_baseline_pct,
            blocklist=bl)
        if not clusters.empty:
            clusters["mode"] = mode
            cluster_frames.append(clusters)
            labels = dict(zip(clusters["tag"], clusters["cluster"]))
            sig = cluster_signatures(
                mode_runs, labels, k=args.top_k_risers,
                min_baseline_pct=args.min_baseline_pct,
                blocklist=bl)
            sig["mode"] = mode
            sig.to_csv(report_dir / f"cluster_signatures_{mode}.tsv",
                       sep="\t", index=False)
        fig_delta_dist(
            mode_runs, report_dir / f"delta_rank_dist_{mode}.png",
            f"|Δrank| distribution — {mode}",
            min_baseline_pct=args.min_baseline_pct,
            blocklist=bl)

    if cluster_frames:
        out_clusters = pd.concat(cluster_frames, ignore_index=True)
        out_clusters.to_csv(report_dir / "riser_clusters.tsv",
                            sep="\t", index=False)
        print(f"\nWrote {report_dir / 'riser_clusters.tsv'}")

    print(f"\nAll figures written to {report_dir}")


if __name__ == "__main__":
    main()
