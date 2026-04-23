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
    shift_gene_weighted_<mode>.png    — Option 1: max shift at gene level (amplified)
    projection_signed_<mode>.png      — Option 2: signed direction on senescence axis
                                         (red = pro-senescence, green = anti-senescence)
    pathway_heatmap_<mode>.png        — top pathways × perturbation (-log10 p.adj)
    top_risers_overlap_<mode>.png     — Jaccard matrix (ordered + hierarchical)
    top_risers_clustermap_<mode>.png  — clustered Jaccard with dendrograms
    delta_rank_dist_<mode>.png        — violin of |Δrank| per run
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 10})

MODES = ("knockdown", "knockout", "overexpress")
CELL_GROUPS = ("P4", "P16_cluster_0", "P16_cluster_1",
               "P16_cluster_2", "P16_cluster_3")


# --------------------------------------------------------------------------- #
# Shortening helpers
# --------------------------------------------------------------------------- #
def shorten_mode(mode: str) -> str:
    """Convert mode names to short codes: knockdown→KD, knockout→KO, overexpress→OE."""
    mapping = {
        "knockdown": "KD",
        "knockout": "KO",
        "overexpress": "OE",
    }
    return mapping.get(mode, mode)


def shorten_pathway(pathway: str) -> str:
    """Convert REACTOME_WORD1_WORD2_... to REACTOME_pw_W1W2W3...

    Example: REACTOME_METABOLISM_OF_VITAMINS_AND_COFACTORS → REACTOME_pw_MOVAC
    """
    if not pathway or not pathway.startswith("REACTOME_"):
        return pathway

    # Remove REACTOME_ prefix
    words = pathway.replace("REACTOME_", "").split("_")
    # Take first letter of each word (uppercase already)
    initials = "".join(w[0] for w in words if w)
    return f"REACTOME_pw_{initials}"


_STOPWORDS_LC = {"of", "and", "the", "to", "by", "for", "in", "on", "at",
                 "with", "a", "an", "or", "from"}


def _abbreviate_words(words: list[str]) -> str:
    """Keep first letter of each word; lowercase for stopwords.

    Example: ["cytosolic", "sulfonation", "of", "small", "molecules"] → "CSoSM".
    """
    out = []
    for w in words:
        if not w:
            continue
        out.append(w[0].lower() if w.lower() in _STOPWORDS_LC else w[0].upper())
    return "".join(out)


def shorten_tag(tag: str) -> str:
    """Replace the leading perturbation mode, and abbreviate the pathway body
    when present, with short codes.

    Examples:
      knockout_TP53 → KO_TP53
      overexpress_FOXM1 → OE_FOXM1
      knockout_pw_cytosolic_sulfonation_of_small_molecules → KO_pw_CSoSM
    """
    mode = infer_mode_from_tag(tag)
    if mode == "other":
        return tag
    rest = tag[len(mode) + 1:] if tag.startswith(f"{mode}_") else tag
    if rest.startswith("pw_"):
        rest = f"pw_{_abbreviate_words(rest[len('pw_'):].split('_'))}"
    return f"{shorten_mode(mode)}_{rest}"


# --------------------------------------------------------------------------- #
# ALL-mode TSV → per-run folder materialization
# --------------------------------------------------------------------------- #
def _slugify_pathway(name: str, max_len: int = 60) -> str:
    """Mirror of perturb_top_genes.slugify_pathway (kept local to avoid a
    cross-module import from validation/ to perturbation/)."""
    slug = re.sub(r"^REACTOME_", "", name).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug[:max_len]


def _split_semi(val) -> list[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)) or val == "":
        return []
    return [x for x in str(val).split(";") if x != ""]


def _row_to_summary(row: pd.Series) -> dict:
    """Reconstruct a summary.json-equivalent dict from one aggregated TSV row."""
    def g(col, default=None):
        v = row.get(col, default)
        if isinstance(v, float) and np.isnan(v):
            return default
        return v

    target = str(row.get("target", ""))
    targets_missing = _split_semi(g("targets_missing", ""))
    summary = {
        "tag": "",  # set by caller via folder name
        "mode": g("mode"),
        "factor": g("factor"),
        "n_targets_in_graph": int(g("n_targets_in_graph", 0) or 0),
        "targets_in_graph": [target] if target else [],
        "targets_missing": targets_missing,
        "n_rising": int(g("n_rising", 0) or 0),
        "n_falling": int(g("n_falling", 0) or 0),
        "median_abs_delta_rank": float(g("median_abs_delta_rank", 0) or 0),
        "max_up_gene": g("max_up_gene", ""),
        "max_up_delta_rank": int(g("max_up_delta_rank", 0) or 0),
        "max_down_gene": g("max_down_gene", ""),
        "max_down_delta_rank": int(g("max_down_delta_rank", 0) or 0),
        "n_sig_delta_pathways": int(g("n_sig_delta_pathways", 0) or 0),
        "top5_delta_pathways": _split_semi(g("top5_delta_pathways", "")),
        "max_shift_gene_differential_group": g("max_shift_gene_differential_group", ""),
        "max_shift_gene_differential": float(g("max_shift_gene_differential", 0) or 0),
        "max_proj_signed_diff_group": g("max_proj_signed_diff_group", ""),
        "max_proj_signed_diff": float(g("max_proj_signed_diff", 0) or 0),
    }
    # Reconstruct per-group projection dicts from flat columns.
    proj_global = {}
    proj_cluster = {}
    for grp in CELL_GROUPS:
        ps = g(f"proj_signed_global_{grp}")
        pn = g(f"proj_signed_norm_global_{grp}")
        pd_ = g(f"proj_signed_diff_global_{grp}")
        if pd_ is not None:
            proj_global[grp] = {
                "proj_signed": float(ps or 0),
                "proj_signed_norm": float(pn or 0),
                "proj_signed_diff": float(pd_ or 0),
            }
        ps = g(f"proj_signed_cluster_{grp}")
        pn = g(f"proj_signed_norm_cluster_{grp}")
        pd_ = g(f"proj_signed_diff_cluster_{grp}")
        if pd_ is not None:
            proj_cluster[grp] = {
                "proj_signed": float(ps or 0),
                "proj_signed_norm": float(pn or 0),
                "proj_signed_diff": float(pd_ or 0),
            }
    if proj_global:
        summary["cell_group_shift_projected_global"] = proj_global
    if proj_cluster:
        summary["cell_group_shift_projected_cluster"] = proj_cluster
    return summary


def _write_delta_ranking(row: pd.Series, out: Path) -> None:
    """Synthesize a delta_ranking.csv from top_risers + top_fallers columns.

    Partial (only the top-K each side, not all genes), but enough for the
    riser-set / blocklist / movers pipeline. is_target is set to 0 since we
    never mark genes as targets here — this is the same filter used by
    perturb_report.
    """
    risers_g = _split_semi(row.get("top_risers_genes", ""))
    risers_d = _split_semi(row.get("top_risers_delta", ""))
    risers_b = _split_semi(row.get("top_risers_baseline", ""))
    fallers_g = _split_semi(row.get("top_fallers_genes", ""))
    fallers_d = _split_semi(row.get("top_fallers_delta", ""))
    if not risers_g and not fallers_g:
        return

    def _pad(xs, n, default):
        return xs + [default] * (n - len(xs)) if len(xs) < n else xs

    risers_b = _pad(risers_b, len(risers_g), "0.0")
    rows = []
    for g, d, b in zip(risers_g, risers_d, risers_b):
        rows.append({"gene": g, "baseline_importance": float(b),
                     "delta_rank": int(d), "is_target": 0})
    for g, d in zip(fallers_g, fallers_d):
        rows.append({"gene": g, "baseline_importance": 0.0,
                     "delta_rank": int(d), "is_target": 0})
    pd.DataFrame(rows).to_csv(out, index=False)


def _write_delta_ora(row: pd.Series, out: Path) -> None:
    """Synthesize delta_ora_top_up_reactome.tsv from top_delta_pathways cols."""
    names = _split_semi(row.get("top_delta_pathways", ""))
    padj = _split_semi(row.get("top_delta_pathways_padj", ""))
    if not names:
        # Fallback: legacy top5_delta_pathways (names only, no p_adj known).
        names = _split_semi(row.get("top5_delta_pathways", ""))
        if not names:
            return
        padj = ["1.0"] * len(names)
    n = min(len(names), len(padj))
    df = pd.DataFrame({
        "pathway": names[:n],
        "p_adj": [float(x) for x in padj[:n]],
    })
    df.to_csv(out, sep="\t", index=False)


def materialize_all_tsv(tsv_paths: list[Path], work_dir: Path) -> Path:
    """Convert one or more aggregated ALL-mode TSVs into a fake perturbation/
    directory that iter_runs() can walk. Returns the perturbation/ path.
    """
    perturb_dir = work_dir / "perturbation"
    perturb_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for p in tsv_paths:
        if not p.exists():
            print(f"[skip] missing: {p}")
            continue
        frames.append(pd.read_csv(p, sep="\t"))
    if not frames:
        return perturb_dir
    df = pd.concat(frames, ignore_index=True)

    n_written = 0
    for _, row in df.iterrows():
        mode = str(row.get("mode", ""))
        if mode not in MODES:
            continue
        target = str(row.get("target", "")).strip()
        target_type = str(row.get("target_type", "gene"))
        if not target:
            continue
        if target_type == "pathway":
            tag = f"{mode}_pw_{_slugify_pathway(target)}"
        else:
            tag = f"{mode}_{target}"
        run_dir = perturb_dir / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = _row_to_summary(row)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        _write_delta_ranking(row, run_dir / "delta_ranking.csv")
        _write_delta_ora(row, run_dir / "delta_ora_top_up_reactome.tsv")
        n_written += 1
    print(f"Materialized {n_written} perturbation runs from "
          f"{len(tsv_paths)} TSV(s) into {perturb_dir}")
    return perturb_dir


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


def load_projections(run: Path) -> dict:
    """Load proj_signed_diff from summary.json for global + cluster axes.
    
    Returns dict with keys: 'global', 'cluster' (dict[cluster_name, value]).
    """
    summary_path = run / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        s = json.load(f)
    proj = {}
    
    # Global projections (P4 + all P16 clusters)
    global_data = s.get("cell_group_shift_projected_global", {})
    if global_data:
        proj["global"] = {
            group: data.get("proj_signed_diff", 0.0)
            for group, data in global_data.items()
        }
    
    # Cluster-specific projections (only P16 clusters)
    cluster_data = s.get("cell_group_shift_projected_cluster", {})
    if cluster_data:
        proj["cluster"] = {
            group: data.get("proj_signed_diff", 0.0)
            for group, data in cluster_data.items()
        }
    
    return proj


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


def is_pathway_tag(tag: str) -> bool:
    """A perturbation tag targets a pathway (vs a single gene) iff its body
    starts with 'pw_' after the mode prefix."""
    mode = infer_mode_from_tag(tag)
    if mode == "other":
        return False
    return tag[len(mode) + 1:].startswith("pw_")


def split_gene_pw_runs(runs: list[Path]) -> tuple[list[Path], list[Path]]:
    genes = [r for r in runs if not is_pathway_tag(r.name)]
    pws = [r for r in runs if is_pathway_tag(r.name)]
    return genes, pws


def split_gene_pw_table(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if table.empty or "tag" not in table.columns:
        return table, table
    mask_pw = table["tag"].apply(is_pathway_tag)
    return table[~mask_pw].copy(), table[mask_pw].copy()


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
    ax.set_xticklabels([shorten_tag(t) for t in table["tag"]],
                       rotation=90, fontsize=7)
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
    ax.set_xticklabels([shorten_tag(t) for t in table["tag"]],
                       rotation=90, fontsize=7)
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

    labels = [shorten_pathway(p) for p in picked]
    fig, ax = plt.subplots(
        figsize=(max(6, 0.4 * len(per_run)),
                 max(4, 0.28 * len(picked))))
    sns.heatmap(mat, ax=ax,
                xticklabels=[shorten_tag(t) for t in per_run.keys()],
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
    short_names = [shorten_tag(n) for n in names]

    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(names)),
                                     max(5, 0.35 * len(names))))
    sns.heatmap(mat, ax=ax,
                xticklabels=short_names, yticklabels=short_names,
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
    short_names = [shorten_tag(n) for n in names]
    df = pd.DataFrame(mat, index=short_names, columns=short_names)

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


def fig_shift_gene_weighted(table: pd.DataFrame, out: Path, title: str):
    """Visualize max_shift_gene_differential (Option 1: amplified signal)."""
    if table.empty or "max_shift_gene_differential" not in table.columns:
        return
    # Filter out NaN/zero entries for cleaner viz
    tab = table.dropna(subset=["max_shift_gene_differential"])
    if tab.empty:
        return
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(tab)), 5))
    x = np.arange(len(tab))
    bars = ax.bar(x, tab["max_shift_gene_differential"],
                  color="#6a3d7d", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([shorten_tag(t) for t in tab["tag"]],
                       rotation=90, fontsize=7)
    ax.set_ylabel("max shift (gene-weighted, differential)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_projection_signed(table: pd.DataFrame, out: Path, title: str):
    """Visualize max_proj_signed_diff (Option 2: signed direction on senescence axis).
    
    Positive = pro-sénescence (KO is a brake → acceleration)
    Negative = anti-sénescence (KO is a motor → reversion)
    """
    if table.empty or "max_proj_signed_diff" not in table.columns:
        return
    tab = table.dropna(subset=["max_proj_signed_diff"])
    if tab.empty:
        return
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(tab)), 5))
    x = np.arange(len(tab))
    colors = ["#e76f51" if v > 0 else "#2a9d8f" for v in tab["max_proj_signed_diff"]]
    bars = ax.bar(x, tab["max_proj_signed_diff"],
                  color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([shorten_tag(t) for t in tab["tag"]],
                       rotation=90, fontsize=7)
    ax.set_ylabel("max proj_signed_diff (senescence axis)")
    ax.set_title(title)
    ax.text(0.02, 0.98, "red=pro-senescence | green=anti-senescence",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_scatter_global_vs_cluster(runs: list[Path], out: Path, title: str):
    """2D scatter: global projection vs cluster-specific projection divergence.
    
    Each point is a perturbation. X = mean(proj_signed_diff over all P16 clusters).
    Y = cluster_0 proj_signed_diff (the outlier cluster).
    Distance from diagonal = cluster_0 divergence from others.
    """
    if not runs:
        return
    data = []
    for run in runs:
        proj = load_projections(run)
        if not proj or "cluster" not in proj:
            continue
        cluster_proj = proj["cluster"]
        if "P16_cluster_0" not in cluster_proj:
            continue
        # Mean of clusters 1,2,3 vs cluster_0
        other_clusters = [v for k, v in cluster_proj.items() 
                         if k != "P16_cluster_0"]
        if not other_clusters:
            continue
        mean_others = np.mean(other_clusters)
        cluster_0_val = cluster_proj["P16_cluster_0"]
        data.append({
            "tag": shorten_tag(run.name),
            "cluster_0": cluster_0_val,
            "mean_others": mean_others,
            "divergence": cluster_0_val - mean_others,
        })
    
    if not data:
        return
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Scatter with color = divergence (red for large positive/negative, blue for low)
    divergence = df["divergence"].abs()
    scatter = ax.scatter(df["mean_others"], df["cluster_0"], 
                        c=divergence, cmap="RdYlBu_r", s=100, 
                        alpha=0.7, edgecolor="black", linewidth=0.5)
    
    # Diagonal line (where cluster_0 = mean_others)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k-', alpha=0.3, zorder=0, label="cluster_0 = mean(others)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Labels & colorbar
    ax.set_xlabel("Mean proj_signed_diff (clusters 1,2,3)")
    ax.set_ylabel("proj_signed_diff (cluster_0)")
    ax.set_title(title)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("|divergence|")
    ax.grid(True, alpha=0.3)
    
    # Annotate a few outliers
    for _, row in df.nlargest(3, "divergence").iterrows():
        ax.annotate(row["tag"].split("_", 1)[-1][:20], 
                   (row["mean_others"], row["cluster_0"]),
                   fontsize=7, alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_heatmap_projections_cluster(runs: list[Path], out: Path, title: str,
                                    max_runs: int = 200):
    """Heatmap: perturbations × P16_clusters, color = proj_signed_diff.

    Hierarchical clustering groups similar perturbations/clusters. For
    genome-wide inputs (thousands of runs) we top-200 by |max| signal and
    drop per-row labels to keep the figure legible.
    """
    if not runs:
        return

    cluster_names = ["P16_cluster_0", "P16_cluster_1",
                     "P16_cluster_2", "P16_cluster_3"]
    data = []
    tags = []
    for run in runs:
        proj = load_projections(run)
        if not proj or "cluster" not in proj:
            continue
        cluster_proj = proj["cluster"]
        row = [cluster_proj.get(c, np.nan) for c in cluster_names]
        if any(np.isnan(row)):
            continue
        data.append(row)
        tags.append(shorten_tag(run.name))
    if not data:
        return

    mat = np.array(data)
    df = pd.DataFrame(mat, columns=cluster_names, index=tags)

    # If genome-wide, restrict to top-max_runs by row-wise max |proj|.
    if len(df) > max_runs:
        keep = df.abs().max(axis=1).nlargest(max_runs).index
        df = df.loc[keep]
        n_hidden = len(mat) - len(df)
        title = f"{title}  (top {max_runs} of {len(mat)} by |max proj|)"
    else:
        n_hidden = 0

    show_y_labels = len(df) <= 60
    height = max(5, min(25, 0.15 * len(df)))

    g = sns.clustermap(df, cmap="RdBu_r", center=0,
                       method="average", metric="euclidean",
                       figsize=(6, height),
                       cbar_kws={"label": "proj_signed_diff"},
                       linewidths=0.2 if show_y_labels else 0,
                       linecolor="white",
                       yticklabels=show_y_labels, xticklabels=True)
    if show_y_labels:
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(),
                                     rotation=0, fontsize=6)
    g.figure.suptitle(title, y=1.01)
    g.savefig(out)
    plt.close(g.figure)


def fig_violin_projections_by_cluster(runs: list[Path], out: Path, title: str):
    """Violin plot: distribution of proj_signed_diff per cluster across all runs.
    
    Shows if cluster_0 is statistically different from others.
    """
    if not runs:
        return
    
    cluster_names = ["P16_cluster_0", "P16_cluster_1", "P16_cluster_2", "P16_cluster_3"]
    data_by_cluster = {c: [] for c in cluster_names}
    
    for run in runs:
        proj = load_projections(run)
        if not proj or "cluster" not in proj:
            continue
        cluster_proj = proj["cluster"]
        for c in cluster_names:
            if c in cluster_proj:
                data_by_cluster[c].append(cluster_proj[c])
    
    # Filter empty clusters
    data_by_cluster = {c: v for c, v in data_by_cluster.items() if v}
    if not data_by_cluster:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = range(1, len(data_by_cluster) + 1)
    parts = ax.violinplot([data_by_cluster[c] for c in data_by_cluster.keys()],
                          positions=positions, showmedians=True, showmeans=True)
    
    # Highlight cluster_0 differently
    for i, pc in enumerate(parts["bodies"]):
        if list(data_by_cluster.keys())[i] == "P16_cluster_0":
            pc.set_facecolor("#e74c3c")
            pc.set_alpha(0.7)
        else:
            pc.set_facecolor("#3498db")
            pc.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(data_by_cluster.keys(), rotation=45)
    ax.axhline(0, color="k", lw=0.5, linestyle="--", alpha=0.5)
    ax.set_ylabel("proj_signed_diff")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    
    # Add text annotation
    ax.text(0.02, 0.98, "red=cluster_0 | blue=other clusters",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Genome-wide helpers (used when #runs per mode is very large, e.g. ALL mode)
# --------------------------------------------------------------------------- #
def filter_top_polar(table: pd.DataFrame,
                     n_per_side: int,
                     sort_col: str = "max_proj_signed_diff") -> pd.DataFrame:
    """Return the top-n most positive + top-n most negative rows by `sort_col`.

    Used to cap bar-style figures at a legible size when the input has
    thousands of perturbations (--all mode).
    """
    if table.empty or sort_col not in table.columns:
        return table
    tab = table.dropna(subset=[sort_col]).copy()
    if len(tab) <= 2 * n_per_side:
        return tab.sort_values(sort_col, ascending=False)
    pos = tab[tab[sort_col] > 0].nlargest(n_per_side, sort_col)
    neg = tab[tab[sort_col] < 0].nsmallest(n_per_side, sort_col)
    return pd.concat([pos, neg], ignore_index=True).sort_values(
        sort_col, ascending=False)


def fig_projection_rank(table: pd.DataFrame, out: Path, title: str):
    """Rank plot of max_proj_signed_diff across all perturbations of a mode.

    Shape of the distribution matters more than individual labels when
    there are thousands of targets. Points are colored by sign.
    """
    if table.empty or "max_proj_signed_diff" not in table.columns:
        return
    tab = table.dropna(subset=["max_proj_signed_diff"]).copy()
    if tab.empty:
        return
    tab = tab.sort_values("max_proj_signed_diff", ascending=False).reset_index(drop=True)
    x = np.arange(len(tab))
    y = tab["max_proj_signed_diff"].to_numpy()
    colors = np.where(y > 0, "#e76f51", "#2a9d8f")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, c=colors, s=6, alpha=0.7, edgecolor="none")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(f"rank (sorted, n={len(tab)})")
    ax.set_ylabel("max_proj_signed_diff")
    ax.set_title(title)
    # Annotate the top 5 on each side (by name)
    head = tab.head(5)
    tail = tab.tail(5)
    for i, row in head.iterrows():
        ax.annotate(shorten_tag(row["tag"]),
                    (i, row["max_proj_signed_diff"]),
                    fontsize=7, alpha=0.85,
                    xytext=(4, 0), textcoords="offset points")
    for off, (_, row) in enumerate(tail.iterrows()):
        i = len(tab) - len(tail) + off
        ax.annotate(shorten_tag(row["tag"]),
                    (i, row["max_proj_signed_diff"]),
                    fontsize=7, alpha=0.85,
                    xytext=(4, 0), textcoords="offset points")
    ax.text(0.02, 0.98, "red=pro-senescence | green=anti-senescence",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_projection_hexbin(runs: list[Path], out: Path, title: str):
    """2D density (quiescent-like vs senescent projection) for genome-wide runs.

    Replaces per-label scatter when N >> 100 — individual annotations are
    useless but the shape tells you how many coherent drivers exist.
    """
    if not runs:
        return
    xs, ys = [], []
    for run in runs:
        proj = load_projections(run)
        g = proj.get("global") or {}
        need = {"P4", "P16_cluster_0", "P16_cluster_1",
                "P16_cluster_2", "P16_cluster_3"}
        if not need.issubset(g):
            continue
        xs.append(np.mean([g["P4"], g["P16_cluster_0"]]))
        ys.append(np.mean([g["P16_cluster_1"], g["P16_cluster_2"], g["P16_cluster_3"]]))
    if not xs:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    hb = ax.hexbin(xs, ys, gridsize=60, cmap="viridis", mincnt=1, bins="log")
    lims = [min(min(xs), min(ys)), max(max(xs), max(ys))]
    ax.plot(lims, lims, "w-", alpha=0.5, zorder=3, lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.axhline(0, color="w", lw=0.5, alpha=0.4)
    ax.axvline(0, color="w", lw=0.5, alpha=0.4)
    ax.set_xlabel("mean proj_signed_diff — P4 + P16_cluster_0 (quiescent-like)")
    ax.set_ylabel("mean proj_signed_diff — P16_cluster_{1,2,3} (senescent)")
    ax.set_title(f"{title}  (n={len(xs)})")
    plt.colorbar(hb, ax=ax, label="log10(count)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Cross-seed aggregation
# --------------------------------------------------------------------------- #
def aggregate_cross_seed(perturb_dirs: list[Path] | None = None,
                         seed_summaries: list[list[tuple[str, dict]]] | None = None,
                         min_robustness: float = 0.5,
                         min_stability: float = 0.7
                         ) -> pd.DataFrame:
    """Aggregate per-target robustness across seeds.

    Accepts either `perturb_dirs` (each a perturbation/ directory walked for
    summary.json) or pre-loaded `seed_summaries` (list of lists of
    `(tag, summary_dict)` pairs — produced by _collect_seed_summaries). The
    second path supports --all-mode TSV-backed seeds without materialisation.
    """
    from collections import Counter
    if seed_summaries is None:
        if not perturb_dirs:
            return pd.DataFrame()
        seed_summaries = [
            [(run.name, load_summary(run)) for run in iter_runs(pdir, None)]
            for pdir in perturb_dirs
        ]
    total_seeds = len(seed_summaries)
    if total_seeds == 0:
        return pd.DataFrame()

    # (mode, target) -> list[summary dicts]
    key_data: dict[tuple, list[dict]] = {}
    for seed in seed_summaries:
        for tag, s in seed:
            mode = s.get("mode") or infer_mode_from_tag(tag)
            if mode not in MODES:
                continue
            target = tag[len(mode) + 1:]
            key_data.setdefault((mode, target), []).append(s)

    rows = []
    for (mode, target), entries in key_data.items():
        projs = [float(e.get("max_proj_signed_diff") or 0.0) for e in entries]
        groups = [str(e.get("max_proj_signed_diff_group") or "") for e in entries]
        n_present = len(entries)

        signs = [np.sign(p) for p in projs if p != 0]
        if signs:
            pos = sum(1 for s in signs if s > 0)
            neg = sum(1 for s in signs if s < 0)
            stability = max(pos, neg) / len(signs)
        else:
            stability = 0.0

        avg_proj = float(np.mean(projs))
        std_proj = float(np.std(projs)) if len(projs) > 1 else 0.0
        group_counts = Counter(g for g in groups if g)
        consensus_group = group_counts.most_common(1)[0][0] if group_counts else ""

        if avg_proj > 0:
            direction = "pro-senescence"
        elif avg_proj < 0:
            direction = "anti-senescence"
        else:
            direction = "neutral"

        rows.append({
            "target": target,
            "mode": mode,
            "is_pathway": is_pathway_tag(f"{mode}_{target}"),
            "n_seeds_present": n_present,
            "robustness_score": n_present / total_seeds,
            "direction_stability": stability,
            "avg_proj_signed_diff": avg_proj,
            "std_proj_signed_diff": std_proj,
            "min_proj_signed_diff": float(min(projs)),
            "max_proj_signed_diff": float(max(projs)),
            "consensus_group": consensus_group,
            "direction": direction,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[(df["robustness_score"] >= min_robustness) &
            (df["direction_stability"] >= min_stability)].copy()
    df["abs_avg"] = df["avg_proj_signed_diff"].abs()
    df = df.sort_values("abs_avg", ascending=False).drop(columns=["abs_avg"])
    return df.reset_index(drop=True)


def fig_cross_seed_top_bars(df: pd.DataFrame, out: Path, title: str,
                             n_per_side: int = 10):
    """Top-n pro + top-n anti drivers with error bars = std across seeds."""
    if df.empty:
        return
    top = filter_top_polar(df, n_per_side, sort_col="avg_proj_signed_diff")
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(top)), 5))
    x = np.arange(len(top))
    colors = ["#e76f51" if v > 0 else "#2a9d8f"
              for v in top["avg_proj_signed_diff"]]
    ax.bar(x, top["avg_proj_signed_diff"], color=colors, alpha=0.8,
           edgecolor="black", linewidth=0.5,
           yerr=top["std_proj_signed_diff"], capsize=3,
           error_kw={"ecolor": "black", "elinewidth": 0.8, "alpha": 0.6})
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    labels = [f"{shorten_mode(r['mode'])}_{r['target']}"
              if not r["is_pathway"] else shorten_tag(f"{r['mode']}_{r['target']}")
              for _, r in top.iterrows()]
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("avg proj_signed_diff across seeds  (±std)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_cross_seed_volcano(df: pd.DataFrame, out: Path, title: str):
    """Per-driver plot of |avg| (effect size) vs std (instability).

    Robust drivers cluster in the upper-left (high |avg|, low std).
    """
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    abs_avg = df["avg_proj_signed_diff"].abs()
    colors = ["#e76f51" if v > 0 else "#2a9d8f"
              for v in df["avg_proj_signed_diff"]]
    sizes = 10 + 60 * df["robustness_score"]
    ax.scatter(df["std_proj_signed_diff"], abs_avg,
               c=colors, s=sizes, alpha=0.6,
               edgecolor="black", linewidth=0.3)
    ax.set_xlabel("std proj_signed_diff (instability across seeds)")
    ax.set_ylabel("|avg proj_signed_diff|  (effect size)")
    ax.set_title(title)
    # Annotate the 8 most robust (high avg, low std) drivers
    score = abs_avg / (df["std_proj_signed_diff"] + 0.1)
    for idx in score.nlargest(8).index:
        r = df.loc[idx]
        lbl = (f"{shorten_mode(r['mode'])}_{r['target']}" if not r["is_pathway"]
               else shorten_tag(f"{r['mode']}_{r['target']}"))
        ax.annotate(lbl, (r["std_proj_signed_diff"], abs(r["avg_proj_signed_diff"])),
                    fontsize=7, alpha=0.85,
                    xytext=(4, 2), textcoords="offset points")
    ax.text(0.02, 0.98,
            "red=pro-senescence | green=anti-senescence\n"
            "size ∝ robustness (fraction of seeds present)",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _normalize_seed_root(p: Path) -> Path:
    """Return the seed root directory for any of: seed root, perturbation/,
    or subdir containing perturbation_all_*.tsv.
    """
    p = p.resolve()
    if p.name == "perturbation" and p.parent.exists():
        return p.parent
    return p


def _common_seed_parent(seed_paths: list[Path]) -> Path:
    """Shared parent for a set of seed directories (the "version" level).

    Accepts seed roots or perturbation/ subdirs; normalises each to the seed
    root first, then returns the common parent across roots.
    """
    import os
    seed_roots = [_normalize_seed_root(p) for p in seed_paths]
    parents = {p.parent for p in seed_roots}
    if len(parents) == 1:
        return parents.pop()
    return Path(os.path.commonpath([str(p) for p in seed_roots]))


def _load_summaries_from_perturb_dir(p: Path) -> list[tuple[str, dict]]:
    """Per-target summary.json, one tuple per run folder."""
    return [(run.name, load_summary(run)) for run in iter_runs(p, None)]


def _load_summaries_from_tsvs(tsvs: list[Path]) -> list[tuple[str, dict]]:
    """Read ALL-mode TSVs and synthesise (tag, summary-dict) tuples.

    Reuses _row_to_summary / _slugify_pathway so the synthetic summaries are
    interchangeable with real summary.json content.
    """
    out: list[tuple[str, dict]] = []
    for tsv in tsvs:
        try:
            df = pd.read_csv(tsv, sep="\t")
        except Exception as e:
            print(f"[warn] failed to read {tsv}: {e}")
            continue
        for _, row in df.iterrows():
            target = str(row.get("target", "")).strip()
            mode = str(row.get("mode", "")).strip()
            if not target or mode not in MODES:
                continue
            target_type = str(row.get("target_type", "gene"))
            if target_type == "pathway":
                tag = f"{mode}_pw_{_slugify_pathway(target)}"
            else:
                tag = f"{mode}_{target}"
            out.append((tag, _row_to_summary(row)))
    return out


def _collect_seed_summaries(p: Path) -> list[tuple[str, dict]]:
    """Given a seed-ish path (seed root OR perturbation/ subdir), return
    (tag, summary) pairs.

    Priority: ALL-mode TSVs first (richer & avoids the 10k-folder materialise
    hit); per-target summary.json folders as fallback. Checks both `p` and
    `p/perturbation` to cover either layout convention.
    """
    if not p.exists():
        return []
    for candidate in (p, p / "perturbation"):
        if not candidate.is_dir():
            continue
        tsvs = sorted(candidate.glob("perturbation_all_*.tsv"))
        if tsvs:
            print(f"  {p.name}: TSV source ({len(tsvs)} file(s) in {candidate.name}/)")
            return _load_summaries_from_tsvs(tsvs)
    for candidate in (p, p / "perturbation"):
        if not candidate.is_dir():
            continue
        if list(candidate.glob("*/summary.json")):
            summaries = _load_summaries_from_perturb_dir(candidate)
            print(f"  {p.name}: per-target source ({len(summaries)} runs in {candidate.name}/)")
            return summaries
    return []


def run_cross_seed(args) -> None:
    """Handler for --cross-seed mode: aggregate across N seeds.

    Accepts seed roots (e.g. V3.3/run1) or perturbation/ subdirs. Each is
    probed for ALL-mode TSVs first, then per-target summary.json folders.
    """
    raw_paths = [Path(p) for p in args.cross_seed]
    seed_summaries: list[list[tuple[str, dict]]] = []
    kept_paths: list[Path] = []
    print(f"[CROSS-SEED] Collecting summaries for {len(raw_paths)} seed(s):")
    for p in raw_paths:
        sums = _collect_seed_summaries(p)
        if not sums:
            print(f"  {p.name}: [skip] no summaries or TSVs found")
            continue
        seed_summaries.append(sums)
        kept_paths.append(p)
    if len(seed_summaries) < 2:
        print(f"Need ≥2 seeds with data; got {len(seed_summaries)}.")
        return

    out_dir = args.report_dir or (_common_seed_parent(kept_paths) / "cross_seed_report")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CROSS-SEED] Aggregating {len(seed_summaries)} seeds → {out_dir}")

    df = aggregate_cross_seed(seed_summaries=seed_summaries,
                              min_robustness=args.min_robustness,
                              min_stability=args.min_stability)
    if df.empty:
        print("No drivers passed the robustness / stability thresholds.")
        return

    df.to_csv(out_dir / "cross_seed_drivers.tsv", sep="\t", index=False)
    df[~df["is_pathway"]].to_csv(
        out_dir / "cross_seed_drivers_genes.tsv", sep="\t", index=False)
    df[df["is_pathway"]].to_csv(
        out_dir / "cross_seed_drivers_pathways.tsv", sep="\t", index=False)
    print(f"Wrote cross_seed_drivers.tsv  ({len(df)} drivers passing filters)")

    for mode in MODES:
        d_mode = df[df["mode"] == mode]
        if d_mode.empty:
            continue
        d_genes = d_mode[~d_mode["is_pathway"]]
        d_pw = d_mode[d_mode["is_pathway"]]
        for suffix, sub in (("genes", d_genes), ("pw", d_pw)):
            if sub.empty:
                continue
            kind = "genes" if suffix == "genes" else "pathways"
            fig_cross_seed_top_bars(
                sub, out_dir / f"cross_seed_top_{mode}_{suffix}.png",
                f"Cross-seed top drivers — {shorten_mode(mode)} / {kind} "
                f"(top {args.top_per_side} pro + top {args.top_per_side} anti)",
                n_per_side=args.top_per_side)
            fig_cross_seed_volcano(
                sub, out_dir / f"cross_seed_volcano_{mode}_{suffix}.png",
                f"Effect vs stability — {shorten_mode(mode)} / {kind}")

    print(f"\nCross-seed figures written to {out_dir}")


def build_transition_drivers(table: pd.DataFrame) -> pd.DataFrame:
    """Rank targets by their impact on the P4 → P16 senescence axis.

    For each target (gene or pathway) that has been perturbed in ≥1 mode,
    pivot across {knockout, knockdown, overexpress} and expose:
      * OE_proj / KO_proj / KD_proj  — max_proj_signed_diff per mode
      * OE_group / KO_group / KD_group — which cell group showed the max shift
      * n_modes                      — coverage
      * max_abs                      — max |proj_signed_diff| across modes
      * sign_consistent              — True iff OE and KO/KD have opposite
                                        signs (expected for a causal driver:
                                        gain of function pushes one way,
                                        loss pushes the other).
    Rows are sorted by sign_consistent desc, then max_abs desc.
    """
    if table.empty or "tag" not in table.columns:
        return pd.DataFrame()
    df = table.copy()
    df["target"] = df["tag"].apply(
        lambda t: t[len(infer_mode_from_tag(t)) + 1:] if infer_mode_from_tag(t) != "other" else t)
    df["is_pathway"] = df["tag"].apply(is_pathway_tag)

    proj = df.pivot_table(index="target", columns="mode",
                          values="max_proj_signed_diff", aggfunc="first")
    grp = df.pivot_table(index="target", columns="mode",
                         values="max_proj_signed_diff_group", aggfunc="first")
    is_pw = df.drop_duplicates("target").set_index("target")["is_pathway"]

    out = pd.DataFrame(index=proj.index)
    for mode, short in (("overexpress", "OE"),
                        ("knockout", "KO"),
                        ("knockdown", "KD")):
        out[f"{short}_proj"] = proj[mode] if mode in proj.columns else np.nan
        out[f"{short}_group"] = grp[mode] if mode in grp.columns else ""

    proj_cols = [c for c in out.columns if c.endswith("_proj")]
    out["n_modes"] = out[proj_cols].notna().sum(axis=1)
    out["max_abs"] = out[proj_cols].abs().max(axis=1, skipna=True)

    def _sign_consistent(r):
        oe = r["OE_proj"]
        ko = r["KO_proj"]
        kd = r["KD_proj"]
        loss = ko if pd.notna(ko) else kd
        if pd.isna(oe) or pd.isna(loss):
            return False
        return (oe > 0 and loss < 0) or (oe < 0 and loss > 0)

    out["sign_consistent"] = out.apply(_sign_consistent, axis=1)
    out["is_pathway"] = out.index.map(is_pw)
    out = out.reset_index()
    out = out.sort_values(["sign_consistent", "max_abs"],
                          ascending=[False, False])
    return out


def fig_p4_vs_cluster0(runs: list[Path], out: Path, title: str,
                       annotate_n: int = 8):
    """Scatter of each perturbation's proj_signed_diff on P4 axis vs cluster_0.

    Diagonal = perturbations move P4 and cluster_0 similarly (expected if the
    two groups share a transcriptional profile). Off-diagonal points are
    where the two groups diverge.
    """
    if not runs:
        return
    rows = []
    for run in runs:
        proj = load_projections(run)
        g = proj.get("global") or {}
        if "P4" not in g or "P16_cluster_0" not in g:
            continue
        rows.append({
            "tag": shorten_tag(run.name),
            "P4": g["P4"],
            "cluster_0": g["P16_cluster_0"],
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    diff = (df["cluster_0"] - df["P4"]).abs()

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(df["P4"], df["cluster_0"], c=diff, cmap="RdYlBu_r",
                    s=90, alpha=0.8, edgecolor="black", linewidth=0.5)
    lims = [min(df["P4"].min(), df["cluster_0"].min()),
            max(df["P4"].max(), df["cluster_0"].max())]
    pad = 0.05 * (lims[1] - lims[0] + 1e-9)
    lims = [lims[0] - pad, lims[1] + pad]
    ax.plot(lims, lims, "k-", alpha=0.3, zorder=0, label="y = x (coherent)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax.axvline(0, color="k", lw=0.4, alpha=0.3)
    ax.set_xlabel("proj_signed_diff — P4 (on global senescence axis)")
    ax.set_ylabel("proj_signed_diff — P16_cluster_0 (on global senescence axis)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    plt.colorbar(sc, ax=ax, label="|cluster_0 − P4|")
    # Label the N largest divergences
    df_annot = df.assign(_d=diff).nlargest(annotate_n, "_d")
    for _, row in df_annot.iterrows():
        ax.annotate(row["tag"], (row["P4"], row["cluster_0"]),
                    fontsize=7, alpha=0.85,
                    xytext=(3, 3), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_quiescent_vs_senescent(runs: list[Path], out: Path, title: str):
    """Compare proj_signed_diff on (P4 + P16_cluster_0) vs (clusters 1,2,3).

    Hypothesis: P16_cluster_0 is transcriptionally closer to P4 (quiescent-
    like), while clusters 1-3 are the truly senescent phenotypes. For each
    perturbation, we plot the mean projection on the "quiescent-like" pair
    against the mean on the "senescent" trio; strong senescence modulators
    should concentrate off-diagonal.
    """
    if not runs:
        return
    rows = []
    for run in runs:
        proj = load_projections(run)
        g = proj.get("global") or {}
        need = {"P4", "P16_cluster_0", "P16_cluster_1",
                "P16_cluster_2", "P16_cluster_3"}
        if not need.issubset(g):
            continue
        quiescent = np.mean([g["P4"], g["P16_cluster_0"]])
        senescent = np.mean([g["P16_cluster_1"],
                             g["P16_cluster_2"],
                             g["P16_cluster_3"]])
        rows.append({
            "tag": shorten_tag(run.name),
            "quiescent_like": float(quiescent),
            "senescent": float(senescent),
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    # Modulation score = senescent - quiescent_like. Big |score| = perturbation
    # that acts preferentially on senescent clusters.
    df["score"] = df["senescent"] - df["quiescent_like"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                    gridspec_kw={"width_ratios": [1, 1.4]})

    # Left: scatter quiescent vs senescent
    sc = ax1.scatter(df["quiescent_like"], df["senescent"],
                     c=df["score"].abs(), cmap="viridis",
                     s=80, alpha=0.8, edgecolor="black", linewidth=0.4)
    lims = [min(df["quiescent_like"].min(), df["senescent"].min()),
            max(df["quiescent_like"].max(), df["senescent"].max())]
    pad = 0.05 * (lims[1] - lims[0] + 1e-9)
    lims = [lims[0] - pad, lims[1] + pad]
    ax1.plot(lims, lims, "k-", alpha=0.3, zorder=0, label="y = x")
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax1.axvline(0, color="k", lw=0.4, alpha=0.3)
    ax1.set_xlabel("mean proj_signed_diff — P4 + P16_cluster_0 (quiescent-like)")
    ax1.set_ylabel("mean proj_signed_diff — P16_cluster_{1,2,3} (senescent)")
    ax1.set_title("Per-perturbation split")
    ax1.legend(loc="lower right", fontsize=8)
    plt.colorbar(sc, ax=ax1, label="|senescent − quiescent_like|")
    for _, row in df.assign(_abs=df["score"].abs()).nlargest(8, "_abs").iterrows():
        ax1.annotate(row["tag"], (row["quiescent_like"], row["senescent"]),
                     fontsize=7, alpha=0.85,
                     xytext=(3, 3), textcoords="offset points")

    # Right: paired distribution (violin) of both groups
    parts = ax2.violinplot([df["quiescent_like"].values,
                            df["senescent"].values],
                           positions=[1, 2], showmedians=True, showmeans=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor("#3498db" if i == 0 else "#e74c3c")
        pc.set_alpha(0.7)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["P4 + cluster_0\n(quiescent-like)",
                         "clusters 1,2,3\n(senescent)"])
    ax2.axhline(0, color="k", lw=0.5, linestyle="--", alpha=0.5)
    ax2.set_ylabel("proj_signed_diff")
    ax2.set_title("Aggregate distribution across perturbations")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_transitions_scatter(runs: list[Path], out: Path, title: str,
                             n_annotate: int = 8):
    """Scatter matrix comparing proj_signed_diff across group transitions.

    Five panels (shared axes each):
      (1) P4 vs P16_cluster_0
      (2) P16_cluster_0 vs P16_cluster_1
      (3) P16_cluster_1 vs P16_cluster_2
      (4) P16_cluster_2 vs P16_cluster_3
      (5) mean(P4, P16_c0) vs mean(P16_c1,c2,c3)
    Each point = one perturbation. Color = |y - x| (off-diagonal distance).
    """
    if not runs:
        return
    need = ["P4", "P16_cluster_0", "P16_cluster_1",
            "P16_cluster_2", "P16_cluster_3"]
    rows = []
    for run in runs:
        proj = load_projections(run)
        g = proj.get("global") or {}
        if not set(need).issubset(g):
            continue
        rec = {"tag": shorten_tag(run.name), **{k: g[k] for k in need}}
        rec["quiescent_like"] = (rec["P4"] + rec["P16_cluster_0"]) / 2
        rec["senescent"] = (rec["P16_cluster_1"] + rec["P16_cluster_2"]
                            + rec["P16_cluster_3"]) / 3
        rows.append(rec)
    if not rows:
        return
    df = pd.DataFrame(rows)

    pairs = [
        ("P4", "P16_cluster_0", "P4 → c0 (quiescent→q-like)"),
        ("P16_cluster_0", "P16_cluster_1", "c0 → c1"),
        ("P16_cluster_1", "P16_cluster_2", "c1 → c2"),
        ("P16_cluster_2", "P16_cluster_3", "c2 → c3"),
        ("quiescent_like", "senescent",
         "mean(P4,c0) → mean(c1,c2,c3)"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))
    for ax, (a, b, sub) in zip(axes, pairs):
        x = df[a].to_numpy(); y = df[b].to_numpy()
        div = np.abs(y - x)
        sc = ax.scatter(x, y, c=div, cmap="viridis", s=10, alpha=0.6,
                        edgecolor="none")
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.05 * (hi - lo + 1e-9)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], "k-", alpha=0.3, lw=0.6, zorder=0)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.axhline(0, color="k", lw=0.3, alpha=0.3)
        ax.axvline(0, color="k", lw=0.3, alpha=0.3)
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title(sub, fontsize=10)
        # Annotate the top-N most divergent perturbations
        if n_annotate > 0 and len(df) > 0:
            order = np.argsort(div)[-n_annotate:]
            for idx in order:
                ax.annotate(df.iloc[idx]["tag"], (x[idx], y[idx]),
                            fontsize=6, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")
    fig.suptitle(f"{title}  (n={len(df)})", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap_projections_global(runs: list[Path], out: Path, title: str,
                                    max_runs: int = 200):
    """Heatmap: perturbations × [P4, P16_c0..c3] using the global-axis
    proj_signed_diff — extends fig_heatmap_projections_cluster by including P4.

    For genome-wide runs we keep the top-max_runs by row-wise max |proj| and
    drop per-row labels above a legibility threshold.
    """
    if not runs:
        return
    groups = ["P4", "P16_cluster_0", "P16_cluster_1",
              "P16_cluster_2", "P16_cluster_3"]
    data, tags = [], []
    for run in runs:
        proj = load_projections(run)
        g = proj.get("global") or {}
        row = [g.get(c, np.nan) for c in groups]
        if any(np.isnan(r) for r in row):
            continue
        data.append(row)
        tags.append(shorten_tag(run.name))
    if not data:
        return

    df = pd.DataFrame(np.array(data), columns=groups, index=tags)
    if len(df) > max_runs:
        keep = df.abs().max(axis=1).nlargest(max_runs).index
        df = df.loc[keep]
        title = f"{title}  (top {max_runs} of {len(tags)} by |max proj|)"
    show_y = len(df) <= 60
    height = max(5, min(25, 0.15 * len(df)))
    g = sns.clustermap(df, cmap="RdBu_r", center=0,
                       method="average", metric="euclidean",
                       figsize=(6, height),
                       cbar_kws={"label": "proj_signed_diff (global axis)"},
                       linewidths=0.2 if show_y else 0,
                       linecolor="white",
                       yticklabels=show_y, xticklabels=True,
                       col_cluster=False)
    if show_y:
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(),
                                     rotation=0, fontsize=6)
    g.figure.suptitle(title, y=1.01)
    g.savefig(out)
    plt.close(g.figure)


def fig_shift_methods_compare(table: pd.DataFrame, out: Path, title: str,
                               n_per_side: int = 10):
    """Heatmap comparing 3 shift metrics on the top-(n_per_side×2) drivers.

    Columns = [max_shift_relative, max_shift_gene_differential,
               max_proj_signed_diff]
    Rows    = top-n_per_side pro + top-n_per_side anti by max_proj_signed_diff
    Values  = z-scored per column so metrics with different scales align.
    """
    metrics = ["max_shift_relative",
               "max_shift_gene_differential",
               "max_proj_signed_diff"]
    missing = [m for m in metrics if m not in table.columns]
    if table.empty or missing:
        return
    top = filter_top_polar(table, n_per_side, sort_col="max_proj_signed_diff")
    if top.empty:
        return
    mat = top[metrics].astype(float).fillna(0.0)
    # z-score per column; guard against zero-std columns.
    std = mat.std().replace(0, 1.0)
    z = (mat - mat.mean()) / std
    labels = [shorten_tag(t) for t in top["tag"]]
    pretty = ["shift_relative\n(hidden)",
              "shift_gene_differential\n(amplified)",
              "proj_signed_diff\n(senescence axis)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(5, 0.3 * len(top))),
                                    gridspec_kw={"width_ratios": [1, 1.5]})
    # Left: z-scored heatmap for rank agreement
    sns.heatmap(z, ax=ax1, cmap="RdBu_r", center=0,
                xticklabels=pretty, yticklabels=labels,
                cbar_kws={"label": "z-score (per column)"},
                linewidths=0.3, linecolor="white")
    ax1.set_title("rank agreement (z-scored)")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=8)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=7)

    # Right: raw values as grouped bars for absolute-scale context
    x = np.arange(len(top))
    w = 0.25
    for i, (m, lbl) in enumerate(zip(metrics, pretty)):
        vals = top[m].astype(float).fillna(0.0).to_numpy()
        ax2.bar(x + (i - 1) * w, vals, width=w, label=lbl.split("\n")[0])
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90, fontsize=7)
    ax2.set_ylabel("raw value")
    ax2.set_title("raw magnitudes (different scales)")
    ax2.set_yscale("symlog")
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


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
    ax.set_xticklabels([shorten_tag(n) for n, _ in data],
                       rotation=90, fontsize=7)
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
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--perturb-dir", type=Path,
                     help="Existing perturbation/ directory with per-run "
                          "folders (TOP-N mode output).")
    src.add_argument("--all", type=Path, nargs="+", dest="all_tsv",
                     help="One or more aggregated per-mode TSVs from "
                          "perturb_top_genes --all-genes/--all-pathways "
                          "(e.g. perturbation_all_genes_knockout.tsv). "
                          "They are materialized into a tmp perturbation/ "
                          "dir and the rest of the pipeline runs normally. "
                          "Automatically enables genome-wide figure mode.")
    src.add_argument("--cross-seed", type=Path, nargs="+", dest="cross_seed",
                     help="Two or more perturbation/ directories (one per "
                          "seed). Aggregates per-target robustness and "
                          "direction stability across seeds, writes a "
                          "cross_seed_drivers.tsv and dedicated figures.")
    ap.add_argument("--report-dir", type=Path, default=None,
                    help="Where to write figures. Defaults: "
                         "<seed>/report (--perturb-dir or --all); "
                         "<version-parent>/cross_seed_report "
                         "(--cross-seed).")
    ap.add_argument("--top-per-side", type=int, default=10,
                    help="For genome-wide / cross-seed bar figures, keep "
                         "the top-N most positive AND top-N most negative "
                         "targets (default 10 → 20 bars).")
    ap.add_argument("--genome-wide-threshold", type=int, default=60,
                    help="Auto-apply --top-per-side filtering to bar "
                         "figures when a mode has more runs than this "
                         "(default 60).")
    ap.add_argument("--min-robustness", type=float, default=0.5,
                    help="--cross-seed only: min fraction of seeds the "
                         "target must appear in (default 0.5).")
    ap.add_argument("--min-stability", type=float, default=0.7,
                    help="--cross-seed only: min fraction of seeds with a "
                         "consistent sign of max_proj_signed_diff "
                         "(default 0.7).")
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

    # --- dispatch: cross-seed mode runs its own pipeline and returns early.
    if args.cross_seed:
        run_cross_seed(args)
        return

    # --- dispatch: --all materializes aggregate TSVs into a tmp perturb_dir,
    # then falls through to the single-seed pipeline with genome_wide=True.
    genome_wide = False
    if args.all_tsv:
        work_dir = Path(tempfile.mkdtemp(prefix="perturb_all_tsv_"))
        args.perturb_dir = materialize_all_tsv(args.all_tsv, work_dir)
        # Default: put the report under the seed folder (parent of the TSV).
        if args.report_dir is None:
            args.report_dir = args.all_tsv[0].resolve().parent / "report"
        genome_wide = True

    runs = iter_runs(args.perturb_dir, args.include)
    if not runs:
        print("No runs found.")
        return

    # Per-seed report lives next to perturbation/ as report/
    # (e.g. <seed>/perturbation/{runs...} + <seed>/report/).
    if args.report_dir is None:
        seed_dir = args.perturb_dir.resolve().parent
        report_dir = seed_dir / "report"
    else:
        report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

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
        print(f"  [{shorten_mode(m)}] {len(bl)} genes{': ' + shown if bl else ''}")

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
            # ---- Option 1 : gene-weighted shift (amplified signal) ----
            "max_shift_gene_differential_group": s.get("max_shift_gene_differential_group", ""),
            "max_shift_gene_differential": s.get("max_shift_gene_differential", 0.0),
            # ---- Option 2 : signed projection on senescence axis ----
            "max_proj_signed_diff_group": s.get("max_proj_signed_diff_group", ""),
            "max_proj_signed_diff": s.get("max_proj_signed_diff", 0.0),
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

    # ---- Transition driver ranking (genes and pathways) ----
    drivers = build_transition_drivers(filt_df)
    if not drivers.empty:
        d_genes = drivers[~drivers["is_pathway"]].drop(columns=["is_pathway"])
        d_pw = drivers[drivers["is_pathway"]].drop(columns=["is_pathway"])
        d_genes.to_csv(report_dir / "top_transition_drivers_genes.tsv",
                       sep="\t", index=False)
        d_pw.to_csv(report_dir / "top_transition_drivers_pathways.tsv",
                    sep="\t", index=False)
        drivers.to_csv(report_dir / "top_transition_drivers.tsv",
                       sep="\t", index=False)
        print(f"Wrote {report_dir / 'top_transition_drivers.tsv'} "
              f"(genes: {len(d_genes)}, pathways: {len(d_pw)}; "
              f"sign-consistent: {int(drivers['sign_consistent'].sum())})")

    # ---- per-mode figures ----
    cluster_frames: list[pd.DataFrame] = []
    for mode in MODES:
        mode_runs = [r for r in runs
                     if infer_mode_from_tag(r.name) == mode]
        mode_tab = filt_df[filt_df["tag"].apply(infer_mode_from_tag) == mode]
        if not mode_runs:
            continue
        bl = blocklists.get(mode, set())
        print(f"\n[{shorten_mode(mode)}] {len(mode_runs)} runs | blocklist={len(bl)}")

        tab_genes, tab_pw = split_gene_pw_table(mode_tab)
        # Bar / summary plots: gene-only and pathway-only panels written as
        # separate files (values span different orders of magnitude). When the
        # input is genome-wide (e.g. --all mode with thousands of targets),
        # filter to top_per_side pro + top_per_side anti to keep bar figures
        # legible; also emit full-distribution rank plots.
        for suffix, sub_full in (("genes", tab_genes), ("pw", tab_pw)):
            if sub_full.empty:
                continue
            kind_label = "genes" if suffix == "genes" else "pathways"
            too_many = (genome_wide or
                        len(sub_full) > args.genome_wide_threshold)
            if too_many:
                sub = filter_top_polar(sub_full, args.top_per_side)
                bar_suffix = (f" (top-{args.top_per_side} pro + top-"
                              f"{args.top_per_side} anti of {len(sub_full)})")
                fig_projection_rank(
                    sub_full,
                    report_dir / f"projection_rank_{mode}_{suffix}.png",
                    f"Projection rank plot — {shorten_mode(mode)} / "
                    f"{kind_label}  (all {len(sub_full)})")
            else:
                sub = sub_full
                bar_suffix = ""
            fig_movers(sub, report_dir / f"overview_movers_{mode}_{suffix}.png",
                       f"Top movers — {shorten_mode(mode)} / {kind_label}"
                       f"{bar_suffix} (filtered pct≥{args.min_baseline_pct})")
            fig_updown(sub, report_dir / f"overview_updown_{mode}_{suffix}.png",
                       f"Rising vs falling — {shorten_mode(mode)} / "
                       f"{kind_label}{bar_suffix}")
            fig_shift_gene_weighted(
                sub, report_dir / f"shift_gene_weighted_{mode}_{suffix}.png",
                f"Gene-weighted shift (Option 1) — {shorten_mode(mode)} / "
                f"{kind_label}{bar_suffix}")
            fig_projection_signed(
                sub, report_dir / f"projection_signed_{mode}_{suffix}.png",
                f"Signed projection on senescence axis (Option 2) — "
                f"{shorten_mode(mode)} / {kind_label}{bar_suffix}")
        fig_pathway_heatmap(
            mode_runs, report_dir / f"pathway_heatmap_{mode}.png",
            f"Top pathways × perturbation — {shorten_mode(mode)} (-log10 p.adj)",
            top_per_run=args.top_per_run)
        fig_riser_overlap(
            mode_runs, report_dir / f"top_risers_overlap_{mode}.png",
            f"Top-{args.top_k_risers} riser Jaccard — {shorten_mode(mode)}",
            k=args.top_k_risers, min_baseline_pct=args.min_baseline_pct,
            blocklist=bl)
        clusters = fig_riser_clustermap(
            mode_runs, report_dir / f"top_risers_clustermap_{mode}.png",
            f"Clustered top-{args.top_k_risers} riser Jaccard — {shorten_mode(mode)}",
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
        genes_runs, pw_runs = split_gene_pw_runs(mode_runs)
        for suffix, sub_runs in (("genes", genes_runs), ("pw", pw_runs)):
            if not sub_runs:
                continue
            kind_label = "genes" if suffix == "genes" else "pathways"
            fig_delta_dist(
                sub_runs, report_dir / f"delta_rank_dist_{mode}_{suffix}.png",
                f"|Δrank| distribution — {shorten_mode(mode)} / {kind_label}",
                min_baseline_pct=args.min_baseline_pct,
                blocklist=bl)

    # ---- cluster-specific analyses (across all modes) ----
    all_runs = runs
    genes_all, pw_all = split_gene_pw_runs(all_runs)
    for suffix, sub_runs in (("genes", genes_all), ("pw", pw_all)):
        if not sub_runs:
            continue
        kind_label = "genes" if suffix == "genes" else "pathways"
        fig_scatter_global_vs_cluster(
            sub_runs, report_dir / f"scatter_cluster0_divergence_{suffix}.png",
            f"Cluster_0 divergence: proj_signed_diff(cluster_0) vs mean(others) — {kind_label}")
    for suffix, sub_runs in (("genes", genes_all), ("pw", pw_all)):
        if not sub_runs:
            continue
        kind_label = "genes" if suffix == "genes" else "pathways"
        fig_heatmap_projections_cluster(
            sub_runs, report_dir / f"heatmap_projections_clusters_{suffix}.png",
            f"Senescence projection across clusters — {kind_label}")
        fig_violin_projections_by_cluster(
            sub_runs, report_dir / f"violin_projections_by_cluster_{suffix}.png",
            f"Senescence projection distribution per cluster — {kind_label}\n"
            "(red=cluster_0, blue=clusters 1-3)")
        fig_p4_vs_cluster0(
            sub_runs, report_dir / f"scatter_P4_vs_cluster0_{suffix}.png",
            f"P4 vs P16_cluster_0 — {kind_label}")
        fig_quiescent_vs_senescent(
            sub_runs, report_dir / f"quiescent_vs_senescent_{suffix}.png",
            f"Quiescent-like (P4 + cluster_0) vs senescent (clusters 1-3) — {kind_label}")
        # Genome-wide: labels are useless at N>>100, show density instead,
        # and expose extra comparison figures (transitions, P4+c0..c3
        # heatmap, shift-method comparison).
        if genome_wide or len(sub_runs) > args.genome_wide_threshold:
            fig_projection_hexbin(
                sub_runs,
                report_dir / f"quiescent_vs_senescent_density_{suffix}.png",
                f"Density: quiescent-like (P4+c0) vs senescent (c1,2,3) — {kind_label}")
            fig_transitions_scatter(
                sub_runs,
                report_dir / f"transitions_scatter_{suffix}.png",
                f"Transition scatter matrix — {kind_label}")
            fig_heatmap_projections_global(
                sub_runs,
                report_dir / f"heatmap_projections_global_{suffix}.png",
                f"Global-axis projection heatmap (P4 + c0..c3) — {kind_label}")

    # Shift-method comparison (top-20 drivers by max_proj_signed_diff).
    # filt_df holds rows for all runs this pass (all modes).
    if genome_wide or len(runs) > args.genome_wide_threshold:
        for mode in MODES:
            mode_tab = filt_df[filt_df["tag"].apply(infer_mode_from_tag) == mode]
            if mode_tab.empty:
                continue
            tab_genes, tab_pw = split_gene_pw_table(mode_tab)
            for suffix, sub in (("genes", tab_genes), ("pw", tab_pw)):
                if sub.empty:
                    continue
                kind = "genes" if suffix == "genes" else "pathways"
                fig_shift_methods_compare(
                    sub,
                    report_dir / f"shift_methods_compare_{mode}_{suffix}.png",
                    f"Shift methods comparison — {shorten_mode(mode)} / {kind} "
                    f"(top {args.top_per_side} pro + top {args.top_per_side} anti)",
                    n_per_side=args.top_per_side)

    # keep combined violin for global overview
    fig_violin_projections_by_cluster(
        all_runs, report_dir / "violin_projections_by_cluster.png",
        "Distribution of senescence projections per cluster\n(red=cluster_0, blue=clusters 1-3)")

    if cluster_frames:
        out_clusters = pd.concat(cluster_frames, ignore_index=True)
        out_clusters.to_csv(report_dir / "riser_clusters.tsv",
                            sep="\t", index=False)
        print(f"\nWrote {report_dir / 'riser_clusters.tsv'}")

    print(f"\nAll figures written to {report_dir}")


if __name__ == "__main__":
    main()
