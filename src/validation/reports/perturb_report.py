#!/usr/bin/env python3
"""
perturb_report.py — Aggregate & visualise GNN perturbation results.

Consumes the output of `gnn_perturbation.py` / `perturb_top_genes.py` and
emits comparison tables, ranking TSVs and figures, split by perturbation
mode (knockdown / knockout / overexpress). Three input modes are
supported, selected by mutually-exclusive CLI flags:

  - ``--perturb-dir <dir>`` — single seed, top-N layout (one sub-folder
    per perturbed target). Writes ``<dir>/report/``.
  - ``--all <tsv> [<tsv> ...]`` — one or more aggregated per-mode TSVs
    from ``perturb_top_genes --all-genes/--all-pathways``. They are
    materialised into a temporary ``perturbation/`` layout and the rest
    of the pipeline runs unchanged; genome-wide figure mode is
    auto-enabled.
  - ``--cross-seed <dir> [<dir> ...]`` — two or more per-seed
    perturbation directories. Aggregates per-target robustness, sign
    stability and the three composite scores (``driver_score``,
    ``validation_score``, ``discovery_score``); writes
    ``cross_seed_drivers.tsv``, ``cross_seed_gene_ranking.tsv`` and a
    dedicated figure set into the version-level ``cross_seed_report/``.

V5.4.1 — ``--driver-canon`` selects how ``driver_score`` aggregates the
perturbation modes (see ``_canonicalize_modes``): ``aligned`` (default)
sign-aligns and averages the coherent KO/KD/OE modes (true KO+KD+OE
scoring, gnn_futur §6.2); ``oe-only`` reproduces the legacy OE-anchored
behaviour (V3.4–V5.4).

Artifact filter
---------------
Some genes (often lncRNAs / pseudogenes like NPPA-AS1, RP1-140K8.5) move
up by hundreds of ranks across *many unrelated* perturbations because
their baseline importance is dominated by components that don't change
under perturbation (e.g. low ``vgae_specificity``) — this is a scoring
artifact, not a biological signal.

We build a data-driven blocklist **per mode**: any gene appearing in the
top-K risers of more than ``--universal-threshold`` (default 0.45) of
the runs *of that mode* is flagged as a universal riser and dropped
before computing max_up/max_down and the top-K riser sets. Per-mode
filtering matters because artifacts tend to be mode-specific (NPPA-AS1
rises in ~60% of knockout runs but is absent from knockdown / overexpress
runs, so a global threshold would miss it).

``--min-baseline-pct`` (default 25) additionally drops bottom-percentile
importance genes.

Usage
-----
    # Single-seed report (top-N layout)
    python src/validation/reports/perturb_report.py \\
        --perturb-dir output/gnn_vgae/V3_Run3/perturbation

    # Looser artifact filter
    python src/validation/reports/perturb_report.py \\
        --perturb-dir output/gnn_vgae/V3_Run3/perturbation \\
        --min-baseline-pct 10

    # Genome-wide (one TSV per mode from perturb_top_genes --all-genes)
    python src/validation/reports/perturb_report.py \\
        --all output/gnn_vgae/V4.1/full.s1/perturbation_all_genes_*.tsv

    # Cross-seed aggregation (V4 axis), 3 seeds
    python src/validation/reports/perturb_report.py \\
        --cross-seed output/gnn_vgae/V4.1/full.s{1,2,3}/perturbation \\
        --axis-tag axisV4 --top-per-side 3

Outputs
-------
``--perturb-dir`` / ``--all`` modes (inside ``<seed>/report/``):
    comparison_table.tsv              — headline stats per perturbation
    comparison_table_filtered.tsv     — same, with max_up/down recomputed
                                         after the baseline-importance filter
    overview_movers_<mode>.png        — filtered max-up / max-down per mode
    overview_updown_<mode>.png        — rising vs falling per mode
    shift_gene_weighted_<mode>.png    — Option 1: max shift at gene level
    projection_signed_<mode>.png      — Option 2: signed projection on the
                                         senescence axis (red = pro-sen,
                                         green = anti-sen)
    pathway_heatmap_<mode>.png        — top pathways × perturbation
                                         (-log10 p.adj)
    top_risers_overlap_<mode>.png     — Jaccard matrix (ordered + hierarchical)
    top_risers_clustermap_<mode>.png  — clustered Jaccard with dendrograms
    delta_rank_dist_<mode>.png        — violin of |Δrank| per run

``--cross-seed`` mode (inside ``<version>/cross_seed_report/``):
    cross_seed_drivers.tsv            — per-target robustness / stability /
                                         scores aggregated across seeds
    cross_seed_gene_ranking.tsv       — gene-level ranking with
                                         driver_score / validation_score /
                                         discovery_score, evidence_tier
                                         (A_confirmed / B_discovery /
                                         C_effector / D_hub / E_noise),
                                         DE / aging-DB annotations.
                                         + colonnes NON-rankantes du readout
                                         signé (gnn_futur §8.A/§9.2) si des
                                         *_signed_fanout.tsv sont présents :
                                         signed_readout / signed_coherence
                                         (role_pert=cosine_senescent headline)
                                         + variantes _de / _latent (diagnostic)
    cross_seed_pathway_ranking.tsv    — pathway-level analogue
    cross_seed_*.png                  — robustness, sign stability,
                                         tier distribution, driver figures
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
        "max_proj_signed_norm_group": g("max_proj_signed_norm_group", ""),
        "max_proj_signed_norm": float(g("max_proj_signed_norm", 0) or 0),
        "max_proj_signed_amplitude_group": g("max_proj_signed_amplitude_group", ""),
        "max_proj_signed_amplitude": float(g("max_proj_signed_amplitude", 0) or 0),
        "max_proj_signed_extent_group": g("max_proj_signed_extent_group", ""),
        "max_proj_signed_extent": float(g("max_proj_signed_extent", 0) or 0),
        "max_proj_signed_degree_group": g("max_proj_signed_degree_group", ""),
        "max_proj_signed_degree": float(g("max_proj_signed_degree", 0) or 0),
        "max_proj_signed_cosine_group": g("max_proj_signed_cosine_group", ""),
        "max_proj_signed_cosine": float(g("max_proj_signed_cosine", 0) or 0),
        "target_ppi_degree": int(g("target_ppi_degree", 0) or 0),
    }
    # Reconstruct per-group projection dicts from flat columns.
    # Toutes les métriques de proj_signed_* disponibles dans la flatten TSV.
    _metric_keys = ("proj_signed", "proj_signed_diff", "proj_signed_norm",
                    "proj_signed_amplitude", "proj_signed_extent",
                    "proj_signed_degree", "proj_signed_cosine")
    proj_global = {}
    proj_cluster = {}
    for grp in CELL_GROUPS:
        gdict = {k: g(f"{k}_global_{grp}") for k in _metric_keys}
        if gdict["proj_signed_diff"] is not None:
            proj_global[grp] = {k: float(v or 0) for k, v in gdict.items()}
        cdict = {k: g(f"{k}_cluster_{grp}") for k in _metric_keys}
        if cdict["proj_signed_diff"] is not None:
            proj_cluster[grp] = {k: float(v or 0) for k, v in cdict.items()}
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
    if len(table) > _HEATMAP_MAX_RUNS:
        # Bar plot avec 10k+ barres : figsize × 0.35 explose. Skip en
        # mode --all ; les bars top-N sont gérées par filter_top_polar.
        print(f"  [skip] fig_updown : {len(table)} entries > "
              f"{_HEATMAP_MAX_RUNS} (utiliser filter_top_polar en amont).")
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
    if len(runs) > _HEATMAP_MAX_RUNS:
        # Mode --all (10k+ runs) : figsize=0.4×N → matplotlib OOM. Le
        # ranking exhaustif vit déjà dans les TSV agrégés.
        print(f"  [skip] fig_pathway_heatmap : {len(runs)} runs > "
              f"{_HEATMAP_MAX_RUNS} (heatmap inutile à cette échelle).")
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


_HEATMAP_MAX_RUNS = 80  # Au-delà, figsize × cellules → OOM matplotlib
                         # (10504 runs × 0.35 inch × 100 DPI = 367k pixels carrés).


def fig_riser_overlap(runs: list[Path], out: Path, title: str,
                      k: int, min_baseline_pct: float,
                      blocklist: set[str]):
    if not runs:
        return
    if len(runs) > _HEATMAP_MAX_RUNS:
        # Mode --all (10k+ perturbations) : la heatmap N×N n'a pas de sens
        # à cette échelle, et matplotlib alloue ~540 Go pour la figure.
        # On skip proprement, le ranking exhaustif est déjà couvert par
        # top_transition_drivers.tsv et les figures bar/clustermap dédiées.
        print(f"  [skip] fig_riser_overlap : {len(runs)} runs > "
              f"{_HEATMAP_MAX_RUNS} (heatmap N×N inutile à cette échelle).")
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
    if len(runs) > _HEATMAP_MAX_RUNS:
        print(f"  [skip] fig_riser_clustermap : {len(runs)} runs > "
              f"{_HEATMAP_MAX_RUNS} (clustermap N×N → OOM matplotlib).")
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
# Cross-seed: optional figure filters + per-gene ranking aggregation.
# --------------------------------------------------------------------------- #
def apply_figure_filters(df: pd.DataFrame, args) -> pd.DataFrame:
    """Apply optional CLI filters to the cross-seed DataFrame BEFORE figures.

    All filters default to no-op (0 / False) so behaviour is unchanged
    unless the user explicitly passes the flags. Filters operate on the
    aggregated cross-seed metrics (avg_proj_signed_*) and on
    target_ppi_degree / is_hub_inflated.
    """
    if df.empty:
        return df
    out = df.copy()
    if args.min_abs_diff > 0 and "avg_proj_signed_diff" in out.columns:
        out = out[out["avg_proj_signed_diff"].abs() >= args.min_abs_diff]
    if args.min_abs_cosine > 0 and "avg_proj_signed_cosine" in out.columns:
        out = out[out["avg_proj_signed_cosine"].abs() >= args.min_abs_cosine]
    if args.min_abs_extent > 0 and "avg_proj_signed_extent" in out.columns:
        out = out[out["avg_proj_signed_extent"].abs() >= args.min_abs_extent]
    if args.min_abs_degree_metric > 0 and "avg_proj_signed_degree" in out.columns:
        out = out[out["avg_proj_signed_degree"].abs() >= args.min_abs_degree_metric]
    if args.min_ppi_degree > 0 and "target_ppi_degree" in out.columns:
        out = out[out["target_ppi_degree"] >= args.min_ppi_degree]
    if args.exclude_hubs and "is_hub_inflated" in out.columns:
        out = out[~out["is_hub_inflated"].astype(bool)]
    return out.reset_index(drop=True)


def _canonicalize_modes(oe, ko, kd,
                        mode_agg: str = "aligned"
                        ) -> tuple[float, float, float]:
    """Combine the up-to-3 perturbation modes into one canonical
    ``(sign, diff, cos)`` triple feeding the driver score.

    Two strategies (CLI ``--driver-canon``):

    * ``"oe-only"`` (legacy V3.4–V5.4) — anchor on **OE** (gain-of-function
      pushes the gene's natural phenotypic direction); fall back to −loss
      if OE is absent. KO/KD then contribute *only* via ``n_modes``
      (coverage) and ``sign_cons`` (coherence) downstream. This is what
      made the score effectively OE-driven (cf. gnn_futur §1.1).

    * ``"aligned"`` (V5.4.1, gnn_futur §6.2) — **sign-align and average all
      coherent modes**. OE keeps its sign; loss-of-function modes (KO/KD)
      are flipped (``−diff``, ``−cos``) since a real driver's loss reverses
      its natural action. A loss mode is **included only if it is coherent**
      with the OE anchor (opposite raw sign) — this implements
      "agréger KO+KD+OE si cohérent, sinon KD/OE" (§6.2) : an incoherent
      KO (e.g. residual topological-shock artefact) is dropped rather than
      allowed to cancel the signal. If OE is absent (or exactly 0), the
      available loss modes are flipped to the natural direction and
      averaged among themselves.

    Note : only ``canon_diff`` / ``canon_cos`` change between strategies.
    The per-mode ``KO_diff``/``KD_diff``/``OE_diff`` columns are exported
    separately for transparency, so the aggregation never hides a mode.
    """
    def _d(m):
        return float(m["avg_proj_signed_diff"]) if m is not None else None

    def _c(m):
        return float(m["avg_proj_signed_cosine"]) if m is not None else None

    if mode_agg == "oe-only":
        if oe is not None:
            d, c = _d(oe), _c(oe)
            return float(np.sign(d)), d, c
        loss = ko if ko is not None else kd
        if loss is not None:
            d, c = _d(loss), _c(loss)
            return float(-np.sign(d)), -d, -c
        return 0.0, 0.0, 0.0

    # mode_agg == "aligned" : sign-aligned mean over coherent modes.
    diffs: list[float] = []
    coss: list[float] = []
    oe_d = _d(oe)
    if oe is not None and oe_d != 0.0:
        anchor = np.sign(oe_d)
        diffs.append(oe_d)
        coss.append(_c(oe))
        for m in (ko, kd):
            if m is None:
                continue
            md = _d(m)
            if np.sign(md) == -anchor:        # loss opposes gain → coherent
                diffs.append(-md)
                coss.append(-_c(m))
    else:
        for m in (ko, kd):                    # no usable OE → loss-anchored
            if m is None:
                continue
            diffs.append(-_d(m))
            coss.append(-_c(m))
    if not diffs:
        return 0.0, 0.0, 0.0
    canon_diff = float(np.mean(diffs))
    canon_cos = float(np.mean(coss))
    return float(np.sign(canon_diff)), canon_diff, canon_cos


def _canon_metric(oe, ko, kd, col: str, mode_agg: str = "aligned") -> float:
    """Aligned-mean of an arbitrary per-mode metric column, with the same
    OE-anchored coherence rule as ``_canonicalize_modes`` (loss modes are
    included only if their diff opposes OE, and sign-flipped). Used for the
    visible degree-weighted diagnostic columns (e.g. amplitude). For
    ``avg_proj_signed_diff`` / ``avg_proj_signed_cosine`` it returns the same
    value as ``_canonicalize_modes`` — kept separate to canonicalize any
    metric without touching the main score path."""
    g = lambda m: float(m[col]) if m is not None else None
    gd = lambda m: float(m["avg_proj_signed_diff"]) if m is not None else None
    if mode_agg == "oe-only":
        if oe is not None:
            return g(oe)
        loss = ko if ko is not None else kd
        return -g(loss) if loss is not None else 0.0
    vals: list[float] = []
    oed = gd(oe)
    if oe is not None and oed != 0.0:
        anchor = np.sign(oed)
        vals.append(g(oe))
        for m in (ko, kd):
            if m is None:
                continue
            if np.sign(gd(m)) == -anchor:
                vals.append(-g(m))
    else:
        for m in (ko, kd):
            if m is None:
                continue
            vals.append(-g(m))
    return float(np.mean(vals)) if vals else 0.0


def _compute_driver_score(canon_diff: float, canon_cos: float, n_modes: int,
                           sign_cons: bool | None, hub: bool,
                           vgae_rank: int | None = None,
                           total_genes: int = 10500) -> float:
    """Continuous driver score ∈ [0, 1] — graph-intrinsic only.

    Aggregates only signals the GNN itself produces : amplitude
    (log-normalized) + purity (cosine alignment with senescence axis) +
    coverage (n_modes) + coherence (sign-consistency) + centrality
    (VGAE rank). External literature evidence (DE-significance, aging
    DBs) is **not** part of the driver score — it is exposed
    separately via `validation_score` (corroboration) and inverted in
    `discovery_score` (graph-only findings). This decoupling lets the
    user choose between confirmatory and exploratory ranking.

    Weights normalised to 1.0 with amplitude+purity dominant (0.65) so
    the score reflects the graph signal first, then coverage/coherence
    (sanity), then centrality (graph context).

    Hub-inflated genes : V3.4 attenuates rather than zeroing-out.
    Their amplitude is real but partly explained by PPI connectivity,
    so we down-weight by 0.5 to push them mid-rank — visible but not
    dominating. The [hub-inflated] tag in `interpretation` is the
    explicit caveat. Replaces the V3.3 killswitch which made ASNS &
    TP53 disappear entirely.
    """
    # log-normalized amplitude: log10(|x|+1) / log10(500+1) ≈ /2.7
    amp = float(min(np.log10(abs(canon_diff) + 1.0) / np.log10(501.0), 1.0))
    purity = float(min(abs(canon_cos), 1.0))
    coverage = float(n_modes / 3.0)
    if sign_cons is True:
        coherence = 1.0
    elif sign_cons is False:
        coherence = 0.3   # low but not 0 (keeps non-monotonic candidates visible)
    else:
        coherence = 0.5   # NaN (single mode) — neutral
    centrality = 0.0
    if vgae_rank is not None and vgae_rank > 0:
        centrality = max(0.0, 1.0 - vgae_rank / total_genes)
    # Weights sum to 1.0. Amplitude + purity = 0.65 (graph signal core),
    # coverage + coherence = 0.25 (sanity), centrality = 0.10 (context).
    score = (
        0.35 * amp
        + 0.30 * purity
        + 0.15 * coverage
        + 0.10 * coherence
        + 0.10 * centrality
    )
    if hub:
        score *= 0.9   # attenuation, not killswitch
    return float(min(max(score, 0.0), 1.0))


def _compute_discovery_score(canon_diff: float,
                              canon_cos: float,
                              n_modes: int,
                              is_de_significant: bool | None,
                              n_aging_dbs: int,
                              hub: bool,
                              low_purity: bool = False,
                              senescence_specificity: float | None = None,
                              mean_robustness: float | None = None,
                              mean_stability: float | None = None
                              ) -> float:
    """Discovery score (V3.5) — surfaces graph-only candidates the
    literature misses. **Independent** of driver_score : a strong
    driver may still not be a discovery (e.g. TP53 — known + corroborated).

    Hard gates (return 0.0) :
      * Literature evidence exists : `is_de_significant=True` OR
        `n_aging_dbs >= 2` → not a discovery (well-known).
      * Hub-inflated : a hub finding without literature is more likely
        a topology artefact than a real lead.

    Otherwise, the score combines three intrinsic graph signals
    (purity, amplitude, coverage) with a graph-quality modulator
    (robustness × stability) and a cluster-specificity bonus. The
    weights mirror the driver score conceptually, but are recomputed
    from raw inputs so neither score depends on the other.

    Formula (post-gate) :
        signal = 0.40·purity + 0.30·amp + 0.20·coverage + 0.10·spec_bonus
        score  = signal · graph_quality
    With :
        purity         = min(|canon_cos|, 1)
        amp            = log10(|canon_diff|+1) / log10(501)
        coverage       = n_modes / 3
        spec_bonus     = 1 if |senescence_specificity| > 0.3 else 0
        graph_quality  = 0.5 + 0.5·(robustness × stability)  ∈ [0.5, 1]

    Mild bonus (+0.05) if `n_aging_dbs == 1` — partial hint without
    being well-known. The result is clamped to [0, 1].
    """
    if is_de_significant is True or n_aging_dbs >= 2:
        return 0.0
    if hub or low_purity:
        return 0.0
    purity = float(min(abs(canon_cos), 1.0))
    amp = float(min(np.log10(abs(canon_diff) + 1.0) / np.log10(501.0), 1.0))
    coverage = float(min(n_modes / 3.0, 1.0))
    spec_bonus = 0.0
    if (senescence_specificity is not None
            and abs(senescence_specificity) > 0.3):
        spec_bonus = 1.0
    signal = 0.40 * purity + 0.30 * amp + 0.20 * coverage + 0.10 * spec_bonus
    quality = 1.0
    if mean_robustness is not None and mean_stability is not None:
        quality = 0.5 + 0.5 * float(mean_robustness) * float(mean_stability)
    score = signal * quality
    if n_aging_dbs == 1:
        score += 0.05
    return float(min(max(score, 0.0), 1.0))


def _compute_validation_score(is_de_significant: bool | None,
                               n_aging_dbs: int,
                               mean_robustness: float | None = None,
                               mean_stability: float | None = None) -> float:
    """Validation score (V3.5) — pure literature corroboration,
    **independent** of driver_score. A strong driver_score with no
    literature returns 0.0 (nothing to validate).

    The literature signal is modulated by graph-quality (robustness ×
    stability) so a literature-supported gene whose perturbation signal
    is unstable across seeds gets penalized — the validation only
    counts if the GNN's evidence is itself reliable.

    Formula :
        lit            = 0.5·is_de_significant + min(0.10·n_aging_dbs, 0.5)
        graph_quality  = 0.5 + 0.5·(robustness × stability)  ∈ [0.5, 1]
        score          = lit · graph_quality   (0 if lit == 0)

    Clamped to [0, 1].
    """
    lit = 0.0
    if is_de_significant is True:
        lit += 0.5
    lit += min(0.10 * max(n_aging_dbs, 0), 0.5)
    if lit == 0.0:
        return 0.0
    quality = 1.0
    if mean_robustness is not None and mean_stability is not None:
        quality = 0.5 + 0.5 * float(mean_robustness) * float(mean_stability)
    return float(min(max(lit * quality, 0.0), 1.0))


def _refine_weak_subtag(canon_diff: float, canon_cos: float,
                         sign_cons: bool | None, n_modes: int,
                         ko_diff: float | None, kd_diff: float | None,
                         oe_diff: float | None) -> str:
    """Sub-tag for the 'weak / noise' bucket (when no driver tag fires).

    Returns one of:
      * `noise`             — no signal at all
      * `subthreshold signal` — small but coherent driver-like
      * `mode-asymmetric weak` — single mode produces a real but small effect
      * `diffuse weak`      — amplitude but cascade dispersed (low cos)
      * `marginal`          — close to threshold, borderline
    """
    a, c = abs(canon_diff), abs(canon_cos)
    if a < 1.0 and c < 0.2:
        return "noise"
    # Sub-threshold but coherent driver-like signal.
    if 1.0 <= a < 20.0 and c >= 0.4 and sign_cons is True:
        return "subthreshold signal"
    # Mode asymmetric : at least one mode has a real effect, others nearly silent.
    per_mode_diffs = [abs(d) for d in (ko_diff, kd_diff, oe_diff) if d is not None]
    if per_mode_diffs:
        max_d = max(per_mode_diffs)
        n_strong = sum(1 for d in per_mode_diffs if d >= 5)
        if 5.0 <= max_d < 50.0 and c >= 0.4 and n_strong == 1:
            return "mode-asymmetric weak"
    # Diffuse but real amplitude.
    if a >= 20.0 and c < 0.3:
        return "diffuse weak"
    return "marginal"


def _gene_interpretation(canon_diff: float, canon_cos: float, ppi_deg: int,
                         hub: bool, sign_cons: bool | None, n_modes: int,
                         min_ppi_degree: int = 5,
                         senescence_specificity: float | None = None,
                         vgae_rank: int | None = None,
                         is_de_significant: bool | None = None,
                         n_aging_dbs: int = 0,
                         ko_diff: float | None = None,
                         kd_diff: float | None = None,
                         oe_diff: float | None = None,
                         oe_cos: float | None = None,
                         mean_abs_extent: float | None = None,
                         mean_abs_degree_metric: float | None = None,
                         driver_score: float | None = None,
                         is_tf: bool = False,
                         mean_robustness: float | None = None,
                         mean_stability: float | None = None,
                         min_robustness: float = 0.5,
                         min_stability: float = 0.7,
                         low_purity: bool = False) -> str:
    """One-line interpretation per gene.

    All genes are tagged (no early-exit filtering) — quality issues are
    surfaced as prefixes / suffixes so the user sees the verdict
    alongside the caveat.

    Prefixes (when applicable, applied in order):
      * `[unreliable] ...` — mean_robustness < min_robustness OR
        mean_stability < min_stability. The interpretation that follows
        is the *would-be* verdict if the signal were stable.
      * `[hub-inflated] ...` — `is_hub_inflated == True`. Verdict is
        kept but the user is warned the amplitude likely reflects PPI
        connectivity, not directional causation.
      * `[incoherent] ...` — `sign_consistent == False` and the gene
        does not match a non-monotonic pattern. The verdict shown is
        what the gene would be tagged if its OE/loss were coherent —
        offered as user-discretion.

    Tier-1 / Tier-2 extensions (preserved) : non-monotonic driver,
    borderline non-monotonic, weak non-monotonic (Tier-3, new — for
    very low |KD| like ANXA1), gain-of-function-only, adaptive cosine,
    TF-aware, low-PPI exception.

    New tags (V3.4) :
      * `low-amplitude direction-pure marker` — 0.5 < |cos| ≤ 0.7 AND
        |diff| < 5. Captures direction-pure but quasi-silent
        regulators (HELLS, DIAPH3 with |cos|≈0.7).
      * `[low PPI deg=N — high literature support]` suffix — for genes
        with `ppi_deg < min_ppi_degree` AND (DE-sig OR aging DBs ≥ 2).
        These genes were previously discarded with the generic
        'low PPI degree' tag — now they keep a proper verdict.
    """
    is_unreliable = (
        (mean_robustness is not None and mean_robustness < min_robustness) or
        (mean_stability is not None and mean_stability < min_stability)
    )

    base: str | None = None
    direction_word_global = "pro-senescence" if canon_diff > 0 else "anti-senescence"

    # Low-PPI handling : exception preserved for high-literature genes,
    # but instead of bailing out we just flag with a suffix and let the
    # rest of the logic compute a verdict.
    peripheral_warn = False
    low_ppi_high_lit = False
    if ppi_deg < min_ppi_degree:
        literature_support = bool(is_de_significant is True or n_aging_dbs >= 2)
        strong_signal = abs(canon_diff) > 50 and abs(canon_cos) > 0.5
        if strong_signal and literature_support:
            peripheral_warn = True
        elif literature_support:
            # New : low-PPI genes with literature support get a tag +
            # the would-be verdict. Surfaced separately in
            # cross_seed_gene_ranking_low_ppi_high_lit.tsv.
            low_ppi_high_lit = True
        else:
            # Genuine low-context : no PPI propagation, no literature.
            base = f"low PPI degree (<{min_ppi_degree}, insufficient context)"

    # Tier-1 extension : non-monotonic driver detection.
    # When sign_cons is False but the KD↔OE pair is itself coherent, the
    # incoherence comes from KO alone — likely cellular compensation or
    # the edge-cut artefact of the KO algorithm.
    if (base is None and sign_cons is False and n_modes == 3 and
            ko_diff is not None and kd_diff is not None and oe_diff is not None):
        ko_oe_same = (np.sign(ko_diff) == np.sign(oe_diff))
        kd_oe_opposite_sign = (np.sign(kd_diff) == -np.sign(oe_diff))
        # Strict (Tier-1) : |KD|>5, |OE|>30.
        if (kd_oe_opposite_sign and ko_oe_same
                and abs(kd_diff) > 5 and abs(oe_diff) > 30):
            d = "pro-senescence" if oe_diff > 0 else "anti-senescence"
            base = (f"non-monotonic {d} driver "
                    f"(KD/OE coherent, KO compensation-suspect)")
        # Relaxed (Tier-2) : |KD|>3, |OE|>15.
        elif (kd_oe_opposite_sign and ko_oe_same
                and abs(kd_diff) > 3 and abs(oe_diff) > 15):
            d = "pro-senescence" if oe_diff > 0 else "anti-senescence"
            base = (f"borderline non-monotonic {d} candidate "
                    f"(KD/OE coherent, KO unreliable)")
        # Tier-3 (V3.4, new) : very relaxed |KD|>1, |OE|>5 BUT requires
        # an additional pure-cosine constraint (|canon_cos|>0.6) to
        # avoid noise. Recovers ANXA1 (|KD|=2.3, |OE|=10.4, cos=−0.81).
        elif (kd_oe_opposite_sign and ko_oe_same
                and abs(kd_diff) > 1 and abs(oe_diff) > 5
                and abs(canon_cos) > 0.6):
            d = "pro-senescence" if oe_diff > 0 else "anti-senescence"
            base = (f"weak non-monotonic {d} candidate "
                    f"(KD/OE coherent, small KD effect)")

    # Tier-1 extension : gain-of-function-only candidate.
    if (base is None and sign_cons is True and
            oe_diff is not None and oe_cos is not None and
            ko_diff is not None and kd_diff is not None and
            abs(oe_diff) > 30 and abs(oe_cos) > 0.4 and
            abs(ko_diff) < 5 and abs(kd_diff) < 5):
        d = "pro-senescence" if oe_diff > 0 else "anti-senescence"
        base = f"gain-of-function-only {d} candidate"

    # Standard driver tags. Compute even when sign_cons is False (so
    # incoherent rows can carry a "would-be" verdict).
    pure = abs(canon_cos) > 0.5
    diffuse_amplitudinal = abs(canon_cos) > 0.3 and abs(canon_diff) > 80
    is_directional = pure or diffuse_amplitudinal
    relaxed_directional = (
        is_de_significant is False and
        abs(canon_cos) > 0.4 and abs(canon_diff) > 30
    )
    tf_directional = (
        is_tf and
        abs(canon_cos) > 0.25 and abs(canon_diff) > 20 and
        (is_de_significant is True or n_aging_dbs >= 2)
    )
    # Sign-coherence required for "true" driver tags ; for incoherent
    # genes we still compute the verdict (treated as if coherent) so
    # the user sees what the gene *would* have been tagged.
    cohere = (sign_cons is True) or (sign_cons is False)

    if base is None and n_modes >= 2 and cohere and is_directional:
        if abs(canon_diff) > 50:
            base = f"strong {direction_word_global} driver"
        elif abs(canon_diff) > 20:
            base = f"moderate {direction_word_global} driver"
    if base is None and n_modes >= 2 and cohere and relaxed_directional:
        if abs(canon_diff) > 50:
            base = f"strong {direction_word_global} driver"
        else:
            base = f"moderate {direction_word_global} driver"
    if base is None and n_modes >= 2 and cohere and tf_directional:
        base = f"moderate {direction_word_global} driver"
    if base is None and n_modes == 1 and abs(canon_cos) > 0.5 and abs(canon_diff) > 50:
        base = f"single-mode {direction_word_global} candidate"
    # Small-but-pure marker — high cosine, low amplitude. Previously
    # masked by the elif-chain ; now reachable as a fallthrough from
    # the directional branches when amplitude is too small to be a
    # driver. Catches TACC3 / DIAPH3 (|cos|≈0.74-0.75, |diff|≈0.1).
    if base is None and abs(canon_cos) > 0.7 and abs(canon_diff) < 20:
        base = "small but pure (potential marker / fine-tuned regulator)"
    # Low-amplitude direction-pure marker (V3.4, new) — bridges the
    # gap between 'small but pure' (|cos|>0.7) and 'marginal'.
    # Captures HELLS (|cos|=0.69, |diff|=0.1) and similar effectors.
    if base is None and 0.5 < abs(canon_cos) <= 0.7 and abs(canon_diff) < 5:
        base = "low-amplitude direction-pure marker"
    if base is None:
        base = _refine_weak_subtag(canon_diff, canon_cos, sign_cons,
                                    n_modes, ko_diff, kd_diff, oe_diff)

    # Enrichments — attach suffixes to any tag carrying a real signal :
    # drivers/candidates (strict + relaxed), 'subthreshold signal',
    # 'mode-asymmetric weak', 'small but pure', 'low-amplitude
    # direction-pure marker'. Excluded: noise, marginal, diffuse weak
    # (no/poor purity), low_ppi (no graph context).
    is_driver_tag = (
        ("driver" in base) or ("candidate" in base) or
        ("subthreshold signal" in base) or
        ("mode-asymmetric weak" in base) or
        ("small but pure" in base) or
        ("low-amplitude direction-pure marker" in base)
    )
    suffixes = []

    # Cluster specificity : senescence_specificity = cosine_senescent
    # (P16_c1+c2+c3) − cosine_quiescent_like (P4+P16_c0). |val| > 0.3 → spécifique.
    # NB: Tier-2 diagnostic showed all 5 cluster axes are highly colinear
    # (ρ > 0.92 across pairs) → 95% of drivers fall within |spec|<0.1
    # ('pan-cluster'). The pan-cluster suffix was therefore non-informative
    # and removed; only the rare 'senescence-cluster-specific' (1.6% of
    # drivers) is kept. See §10.13bisbis 'Limites du modèle'.
    if (senescence_specificity is not None and is_driver_tag
            and abs(senescence_specificity) > 0.3):
        suffixes.append("senescence-cluster-specific")

    # VGAE centrality enrichment (vgae_rank in 1..N, lower = more central).
    if vgae_rank is not None and vgae_rank > 0:
        if is_driver_tag:
            if vgae_rank <= 500:
                suffixes.append("canonical (high VGAE centrality)")
            elif vgae_rank > 1000:
                suffixes.append("novel/peripheral (VGAE-specific finding)")
        elif base == "weak / noise" and vgae_rank <= 100:
            return "housekeeping hub (high VGAE centrality, no causal impact)"

    # DE-significance enrichment (independent statistical evidence).
    if is_de_significant is True and is_driver_tag:
        suffixes.append("DE-significant (literature-aligned)")
    elif is_de_significant is False and is_driver_tag:
        suffixes.append("non-DE (graph-only finding)")
        # Discovery boost : non-DE + aging DBs hit → strong discovery signal
        # (gene known to be aging-relevant despite no DE in this dataset —
        # likely post-transcriptional or context-dependent regulation).
        if n_aging_dbs >= 2:
            suffixes.append("aging-DB-anchored discovery")

    # Tier-1.5 : cascade architecture (extent → concentrated/diffuse).
    if mean_abs_extent is not None and is_driver_tag:
        if mean_abs_extent > 0.01:
            suffixes.append("concentrated cascade")
        elif mean_abs_extent < 0.002:
            suffixes.append("diffuse cascade")

    # Tier-1.5 : impact-per-connection (degree metric → "small but mighty").
    if (mean_abs_degree_metric is not None
            and is_driver_tag and mean_abs_degree_metric > 5.0):
        suffixes.append("high impact-per-connection")

    # Tier-1.5 : peripheral PPI warning (deg < min_ppi_degree exception).
    if peripheral_warn and is_driver_tag:
        suffixes.append(f"peripheral PPI deg={ppi_deg} — WARN")

    # V3.4 : low-PPI high-literature genes (no PPI context but DE/aging
    # support). Different from peripheral_warn (which already requires
    # strong signal). Surfaces metallothionein-type markers (MT1E, etc.).
    if low_ppi_high_lit and is_driver_tag:
        suffixes.append(f"low PPI deg={ppi_deg} — high literature support")

    # TF marker — pleiotropy expected, low cosine is normal here.
    if is_tf and is_driver_tag:
        suffixes.append("transcription factor (pleiotropy expected)")

    # Compose the verdict body.
    body = f"{base} [{' | '.join(suffixes)}]" if suffixes else base

    # Quality prefixes — applied last, in order from most-blocking to
    # least. The user sees the would-be verdict alongside the caveat
    # rather than a hard-filtered nothing.
    prefixes = []
    if is_unreliable:
        prefixes.append("unreliable")
    if hub:
        prefixes.append("hub-inflated")
    elif low_purity:
        # V3.5 : amplitude forte mais cosine < 0.3 sans être un hub PPI
        # (deg ≤ 200). Le signal est réel (pas explicable par la
        # connectivité), juste à direction faible — laisser le verdict
        # mais avertir que la pureté directionnelle est limite.
        prefixes.append("low-purity signal")
    if sign_cons is False and not (
            base.startswith("non-monotonic")
            or base.startswith("borderline non-monotonic")
            or base.startswith("weak non-monotonic")):
        prefixes.append("incoherent")
    if prefixes:
        return f"[{' | '.join(prefixes)}] {body}"
    return body


def _pathway_interpretation(canon_diff: float, canon_cos: float,
                             hub: bool, sign_cons: bool | None, n_modes: int,
                             ko_diff: float | None, kd_diff: float | None,
                             oe_diff: float | None,
                             mean_abs_extent: float | None = None) -> str:
    """Pathway-specific interpretation.

    Same logic as `_gene_interpretation` but with thresholds adjusted to
    the pathway-scale of canon_diff (P95 ≈ 420 vs ≈ 30 for genes).

    Tags:
      * `strong {pro/anti}-senescence pathway` — n_modes ≥ 2,
        sign-consistent, |cos|>0.4 (pathways have lower cos on average),
        |diff|>100.
      * `moderate {pro/anti}-senescence pathway` — |diff| ∈ ]30, 100].
      * `non-monotonic {direction} pathway (KD/OE coherent, KO compensation-suspect)`
      * `single-mode {direction} candidate pathway`
      * `weak / noise`, `hub-inflated`, `incoherent` as for genes.
    """
    if hub:
        return "hub-inflated (filter out)"

    base: str | None = None

    # Non-monotonic detection (same logic, pathway-scaled thresholds).
    if (sign_cons is False and n_modes == 3 and
            ko_diff is not None and kd_diff is not None and oe_diff is not None):
        ko_oe_same = (np.sign(ko_diff) == np.sign(oe_diff))
        kd_oe_opposite_sign = (np.sign(kd_diff) == -np.sign(oe_diff))
        if (kd_oe_opposite_sign and ko_oe_same
                and abs(kd_diff) > 20 and abs(oe_diff) > 100):
            d = "pro-senescence" if oe_diff > 0 else "anti-senescence"
            base = (f"non-monotonic {d} pathway "
                    f"(KD/OE coherent, KO compensation-suspect)")
    if base is None and sign_cons is False:
        return "incoherent (OE and loss-of-function point same direction)"

    direction_word = "pro-senescence" if canon_diff > 0 else "anti-senescence"
    if base is None:
        base = "weak / noise"
        # Pathway-scaled thresholds (P95 of |canon_diff| ≈ 420).
        pure = abs(canon_cos) > 0.4
        diffuse_amplitudinal = abs(canon_cos) > 0.3 and abs(canon_diff) > 200
        is_directional = pure or diffuse_amplitudinal
        if n_modes >= 2 and sign_cons is True and is_directional:
            if abs(canon_diff) > 100:
                base = f"strong {direction_word} pathway"
            elif abs(canon_diff) > 30:
                base = f"moderate {direction_word} pathway"
        elif n_modes == 1 and abs(canon_cos) > 0.4 and abs(canon_diff) > 100:
            base = f"single-mode {direction_word} candidate pathway"

    is_driver_tag = ("pathway" in base and
                      ("driver" not in base or "non-monotonic" in base) and
                      "incoherent" not in base and
                      "hub" not in base and
                      "weak" not in base)
    suffixes = []
    if mean_abs_extent is not None and is_driver_tag:
        if mean_abs_extent > 0.05:
            suffixes.append("concentrated cascade")
        elif mean_abs_extent < 0.005:
            suffixes.append("diffuse cascade")
    if suffixes:
        return f"{base} [{' | '.join(suffixes)}]"
    return base


def build_pathway_ranking(df: pd.DataFrame,
                           min_robustness: float = 0.7,
                           min_stability: float = 0.7,
                           mode_agg: str = "aligned") -> pd.DataFrame:
    """Per-pathway cross-mode ranking (pathways only).

    Mirrors `build_gene_ranking` but on the is_pathway==True subset and
    with pathway-scaled interpretation thresholds. No PPI degree filter,
    no VGAE/DE lookup (those are gene-level concepts).
    """
    if df.empty:
        return pd.DataFrame()
    pw = df[df["is_pathway"] == True].copy()  # noqa: E712
    if pw.empty:
        return pd.DataFrame()

    rows = []
    for target, sub in pw.groupby("target"):
        modes = {r["mode"]: r for _, r in sub.iterrows()}
        oe = modes.get("overexpress")
        ko = modes.get("knockout")
        kd = modes.get("knockdown")
        loss = ko if ko is not None else kd

        n_modes = sum(1 for m in (oe, ko, kd) if m is not None)
        sign_cons: bool | None = None
        if oe is not None and loss is not None:
            sign_cons = bool(np.sign(oe["avg_proj_signed_diff"])
                              == -np.sign(loss["avg_proj_signed_diff"]))

        max_abs_diff = float(sub["avg_proj_signed_diff"].abs().max())
        max_abs_cos = float(sub["avg_proj_signed_cosine"].abs().max())
        mean_extent = float(sub["avg_proj_signed_extent"].abs().mean())
        mean_robustness = float(sub["robustness_score"].mean())
        mean_stability = float(sub["direction_stability"].mean())
        any_hub = bool(sub["is_hub_inflated"].any())

        # Canonical (sign, diff, cos) over modes (see _canonicalize_modes).
        canon_sign, canon_diff, canon_cos = _canonicalize_modes(
            oe, ko, kd, mode_agg=mode_agg)

        if canon_sign > 0:
            direction = "pro-senescence"
        elif canon_sign < 0:
            direction = "anti-senescence"
        else:
            direction = "neutral"
        if sign_cons is False:
            direction += " (mixed)"

        ko_diff_val = float(ko["avg_proj_signed_diff"]) if ko is not None else None
        kd_diff_val = float(kd["avg_proj_signed_diff"]) if kd is not None else None
        oe_diff_val = float(oe["avg_proj_signed_diff"]) if oe is not None else None

        interp = _pathway_interpretation(
            canon_diff, canon_cos, any_hub, sign_cons, n_modes,
            ko_diff_val, kd_diff_val, oe_diff_val,
            mean_abs_extent=mean_extent,
        )

        rec = {
            "pathway": target,
            # Pathway driver_score : pathway-scaled. Amplitude normalised
            # by P95(|diff|) ≈ 420; no DE/VGAE/aging-DB lit boost.
            "driver_score": round(_compute_pathway_driver_score(
                canon_diff, canon_cos, n_modes, sign_cons, any_hub), 3),
            "n_modes_present": n_modes,
            "mean_robustness": round(mean_robustness, 2),
            "mean_stability": round(mean_stability, 2),
            "canon_diff": round(canon_diff, 1),
            "canon_cosine": round(canon_cos, 3),
            "max_abs_diff": round(max_abs_diff, 1),
            "max_abs_cosine": round(max_abs_cos, 3),
            "mean_abs_extent": round(mean_extent, 4),
            "sign_consistent": sign_cons if sign_cons is not None else "",
            "is_hub_inflated": any_hub,
            "direction": direction,
            "interpretation": interp,
            "KO_diff": round(float(ko["avg_proj_signed_diff"]), 1) if ko is not None else None,
            "KD_diff": round(float(kd["avg_proj_signed_diff"]), 1) if kd is not None else None,
            "OE_diff": round(float(oe["avg_proj_signed_diff"]), 1) if oe is not None else None,
            "KO_cos": round(float(ko["avg_proj_signed_cosine"]), 3) if ko is not None else None,
            "KD_cos": round(float(kd["avg_proj_signed_cosine"]), 3) if kd is not None else None,
            "OE_cos": round(float(oe["avg_proj_signed_cosine"]), 3) if oe is not None else None,
        }
        rows.append(rec)

    out = pd.DataFrame(rows)
    out = out[(out["mean_robustness"] >= min_robustness) &
              (out["mean_stability"] >= min_stability) &
              (~out["is_hub_inflated"])]
    out = out.sort_values(["driver_score", "max_abs_diff"],
                          ascending=[False, False]).reset_index(drop=True)
    return out


def _compute_pathway_driver_score(canon_diff: float, canon_cos: float,
                                    n_modes: int, sign_cons: bool | None,
                                    hub: bool) -> float:
    """Pathway-scaled driver score ∈ [0, 1].

    Same structure as gene driver_score but pathway-scaled :
    amplitude denominator = log10(500+1), no DE/VGAE/aging boost (those
    are gene-level concepts).
    """
    if hub:
        return 0.0
    amp = float(min(np.log10(abs(canon_diff) + 1.0) / np.log10(501.0), 1.0))
    purity = float(min(abs(canon_cos), 1.0))
    coverage = float(n_modes / 3.0)
    if sign_cons is True:
        coherence = 1.0
    elif sign_cons is False:
        coherence = 0.3
    else:
        coherence = 0.5
    score = 0.40 * amp + 0.30 * purity + 0.15 * coverage + 0.15 * coherence
    return float(min(max(score, 0.0), 1.0))


def _load_is_tf(seed_paths: list[Path]) -> pd.Series:
    """Extract `is_tf` (feature index 0) from any seed's hetero_graph_vgae.pt.

    Source : pySCENIC-detected TFs in HUVEC (regulons with motif support,
    ~62 genes — restricted set). Returns Series indexed by gene symbol,
    values ∈ {0.0, 1.0}.

    Post-hoc augmentation (V4.1+) : if the OmniPath CollecTRI cache exists
    at `data/omnipath/tf_collectri.tsv.gz`, the returned series is OR-ed
    with the CollecTRI source_symbol set (monomeric TFs only, ~1186) so
    that the TF-aware downstream logic (interpretation suffix, B_discovery
    threshold relaxation) covers all curated TFs, not only the pySCENIC
    HUVEC-restricted subset. This does NOT modify the trained graph — the
    GNN itself still sees only the original 62 TFs as 1.0 in feature[:,0].
    Fix at the training level is planned for V5 (cf. §17 V4.1 du rapport).
    """
    series = pd.Series(dtype=float)
    for seed in seed_paths:
        graph_p = seed / "hetero_graph_vgae.pt"
        emb_p = seed / "gene_embeddings_vgae.csv"
        if not graph_p.exists() or not emb_p.exists():
            continue
        try:
            import torch
            data = torch.load(graph_p, weights_only=False)
            emb = pd.read_csv(emb_p, index_col=0)
            genes = list(emb.index.astype(str))
            x = data["gene"].x.numpy()
            series = pd.Series(x[:, 0], index=genes, name="is_tf")
            break
        except Exception:
            continue
    if series.empty:
        return series

    cache_candidates = [
        Path("data/omnipath/tf_collectri.tsv.gz"),
        Path(__file__).resolve().parents[2] / "data" / "omnipath" / "tf_collectri.tsv.gz",
    ]
    for cache in cache_candidates:
        if not cache.exists():
            continue
        try:
            import gzip
            collectri_tfs: set[str] = set()
            with gzip.open(cache, "rt") as f:
                next(f, None)  # header
                for line in f:
                    src = line.split("\t", 1)[0]
                    if src and "_" not in src:  # drop heterodimers like NFKB1_REL
                        collectri_tfs.add(src)
            if not collectri_tfs:
                break
            extra = collectri_tfs & set(series.index)
            n_before = int(series.sum())
            series = series.copy()
            series.loc[list(extra)] = 1.0
            n_after = int(series.sum())
            print(f"  [is_tf] pySCENIC = {n_before}, "
                  f"+CollecTRI ∩ available = {len(extra)}, "
                  f"union = {n_after}")
        except Exception as e:
            print(f"  [is_tf] CollecTRI augmentation skipped: {e}")
        break
    return series


def _load_reactome_pathways() -> dict[str, set[str]]:
    """Load REACTOME pathway → gene members mapping. Reuse ora_consensus loader."""
    try:
        # Import locally to avoid circular dependency at module level.
        from ora_consensus import load_reactome_gmt
        return load_reactome_gmt()
    except Exception:
        try:
            import sys
            here = Path(__file__).resolve()
            # src/validation/reports/ → src/validation/ora/ + flat fallback src/
            for cand in [here.parent.parent / "ora", here.parent.parent.parent]:
                if cand.exists() and str(cand) not in sys.path:
                    sys.path.insert(0, str(cand))
            from ora_consensus import load_reactome_gmt
            return load_reactome_gmt()
        except Exception as e:
            print(f"[INFO] Could not load REACTOME GMT: {e}")
            return {}


def _build_gene_to_strong_pathways(
        gene_rank: pd.DataFrame,
        pw_rank: pd.DataFrame,
        reactome: dict[str, set[str]]) -> dict[str, list[str]]:
    """For each gene in `gene_rank`, list strong/moderate REACTOME pathways
    that contain it AND are flagged driver/candidate in `pw_rank`.

    Returns dict {gene_symbol: ["pw_slug:tag", ...]} with up to 5 pathways
    per gene (most relevant — strong > moderate > non-monotonic > single-mode).
    """
    if not reactome or pw_rank.empty:
        return {}
    # Map slug -> original REACTOME name (slugify is approximate, do exact lookup).
    # The slug used in pw_rank is `pw_<slugified>` (cf. _slugify_pathway).
    # We rebuild reverse lookup : strong_pw_slugs -> original REACTOME members.
    strong_tags = ("strong", "moderate", "non-monotonic", "single-mode")
    pw_filter = pw_rank[pw_rank["interpretation"].astype(str).str.startswith(
        strong_tags)].copy()
    if pw_filter.empty:
        return {}
    # Build slug -> members mapping. Slug = "pw_" + slugify(REACTOME_name).
    slug_to_members: dict[str, set[str]] = {}
    for orig_name, members in reactome.items():
        slug = "pw_" + _slugify_pathway(orig_name)
        slug_to_members[slug] = members
    # For each strong pathway in the ranking, get its members.
    out: dict[str, list[tuple[str, str]]] = {}
    for _, row in pw_filter.iterrows():
        slug = str(row["pathway"])
        tag = str(row["interpretation"]).split(" [")[0]
        members = slug_to_members.get(slug, set())
        for g in members:
            out.setdefault(g, []).append((slug, tag))
    # Compress to top 5 per gene (sort by tag rank: strong > moderate > rest).
    rank_order = {"strong": 0, "moderate": 1, "non-monotonic": 2, "single-mode": 3}
    def _key(t):
        return rank_order.get(t[1].split()[0], 9)
    out_clean: dict[str, list[str]] = {}
    for g, lst in out.items():
        lst_sorted = sorted(lst, key=_key)[:5]
        out_clean[g] = [f"{slug.replace('pw_','')} ({tag.split()[0]})"
                        for slug, tag in lst_sorted]
    return out_clean


def _load_de_magnitude(de_path: Path) -> pd.DataFrame:
    """Load MAST DE magnitudes (P4 vs P16) for the per-gene ranking.

    Expects columns ``gene``, ``avg_log2FC``, ``p_val_adj``. Returns DataFrame
    indexed by gene with two numeric columns:
      * ``de_log2fc_p4_vs_p16`` — signed log2 fold-change. Sign convention
        follows the source file (typically positive = up in P4 vs P16, or
        the reverse depending on the contrast direction; the value is
        propagated as-is so the reader knows the absolute magnitude).
      * ``de_neglog10_padj`` — −log10(p_val_adj), clipped to 50 to keep
        the column readable when padj == 0 in the source.

    Returns empty DataFrame on missing/invalid file (caller handles).
    """
    if not de_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(de_path)
    except Exception:
        return pd.DataFrame()
    cols = {c.lower(): c for c in df.columns}
    gene_col = cols.get("gene") or cols.get("hgnc_symbol")
    lfc_col = cols.get("avg_log2fc") or cols.get("log2foldchange")
    padj_col = cols.get("p_val_adj") or cols.get("padj")
    if not (gene_col and lfc_col and padj_col):
        return pd.DataFrame()
    out = pd.DataFrame({
        "gene": df[gene_col].astype(str),
        "de_log2fc_p4_vs_p16": pd.to_numeric(df[lfc_col], errors="coerce"),
        "de_neglog10_padj": -np.log10(
            pd.to_numeric(df[padj_col], errors="coerce").clip(lower=1e-50)
        ).clip(upper=50.0),
    }).dropna(subset=["de_log2fc_p4_vs_p16", "de_neglog10_padj"])
    out = out.drop_duplicates(subset="gene", keep="first").set_index("gene")
    return out


def _resolve_signed_fanout_paths(args, seed_paths: list[Path]) -> list[Path]:
    """Résout les tables long-format du readout signé.

    Priorité : ``--signed-fanout`` explicite ; sinon auto-découverte de
    ``*_signed_fanout.tsv`` à côté des TSV --all dans chaque chemin de seed.
    """
    explicit = getattr(args, "signed_fanout", None)
    if explicit:
        return [Path(p) for p in explicit if Path(p).exists()]
    found: list[Path] = []
    for p in seed_paths or []:
        if p.is_file():
            p = p.parent
        found += sorted(p.glob("*_signed_fanout.tsv"))
        found += sorted(p.glob("*/*_signed_fanout.tsv"))
    # Dédup en gardant l'ordre.
    seen, out = set(), []
    for f in found:
        if f not in seen:
            seen.add(f); out.append(f)
    return out


def _aggregate_signed_fanout(fanout_paths: list[Path]) -> pd.DataFrame:
    """Concatène + replie par (source, target) les tables long-format du
    readout signé (``*_signed_fanout.tsv``) sur seeds & modes.

    La direction du readout ne dépend que de role × sign (invariants
    mode/seed) ; seul ``|proj_target|`` varie par mode (OE > KO/KD) → on
    moyenne ``|proj|`` et on prend le signe MAJORITAIRE de ``sign_pred`` /
    ``sign_known`` cross-seed. Une ligne par arête (source, target).
    """
    frames = []
    for p in fanout_paths:
        try:
            frames.append(pd.read_csv(p, sep="\t"))
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] signed-fanout illisible {p} : {e}")
    if not frames:
        return pd.DataFrame()
    f = pd.concat(frames, ignore_index=True)
    f["abs_proj"] = f["proj_target"].abs()
    agg = f.groupby(["source", "target"], sort=False).agg(
        abs_proj=("abs_proj", "mean"),
        sign_pred=("sign_pred", lambda s: float(np.sign(s.mean()))),
        sign_known=("sign_known", lambda s: float(np.sign(s.mean()))),
        role_latent=("role_latent_sign", lambda s: float(np.sign(s.mean()))),
        n_obs=("abs_proj", "size"),
    ).reset_index()
    return agg


def compute_signed_readout_columns(fanout_agg: pd.DataFrame,
                                   role_pert_map: dict,
                                   de_role_map: dict | None = None
                                   ) -> pd.DataFrame:
    """Readout signé + cohérence de fan-out par source (gnn_futur §8.A / §9.2).

    Pour chaque source S et chaque cible T de son fan-out signé 1-hop :
        contrib(T) = |proj_T| · sgn(role_T) · sign_pred_{S→T}
    readout(S)   = Σ_T contrib(T)   (>0 pro-sén, <0 anti-sén ; degré-chargé)
    coherence(S) = |Σ_T sgn(role_T)·sign_pred| / N_T  ∈ [0,1]  (degree-free)

    Inhiber-un-pro-sén (−·−) et activer-un-anti-sén (+·+→ via sign) reçoivent
    le même signe → s'additionnent au lieu de s'annuler (cf. driver_score).

    Trois sources de rôle de la CIBLE, **en colonnes séparées, AUCUNE
    headline** (décision 2026-06-09 : comparatif n=19 gènes à rôle connu →
    role_de 16/19, role_latent 15/19, role_pert 10/19 ; aucun uniformément
    bon, cf. design_log §14bis.6quatervicies). Chaque rôle a un angle mort
    distinct :
      * ``role_pert``  = cosine_senescent de T (effet causal) — fiable sur les
        DRIVERS, **aveugle aux effecteurs** (perturber p16/p21 ne déplace pas
        l'état car ils sont aval → mauvais signe).
      * ``role_de``    = signe DE de T — fiable sur effecteurs/marqueurs, **aveugle
        aux compensatoires** (FHL2 up-en-P16 mais anti-sén → mal étiqueté). =
        l'objection « marqueur ≠ rôle » §1.4.
      * ``role_latent``= position au repos latente — biaisée (signe ~uniforme).
    Le **désaccord role_de↔role_pert** est le signal des cas marqueur/driver
    (``conflict_frac``). Arbitrage final = validation cross-dataset (§0/§5-2b).

    Returns: DataFrame, colonne ``gene`` = source, readout/coherence × 3 rôles.
    """
    if fanout_agg.empty:
        return pd.DataFrame()
    a = fanout_agg.copy()
    a["role_pert"] = a["target"].map(
        lambda t: float(np.sign(role_pert_map[t]))
        if role_pert_map.get(t) is not None
        and np.isfinite(role_pert_map.get(t, np.nan)) else np.nan)
    if de_role_map is not None:
        a["role_de"] = a["target"].map(
            lambda t: float(de_role_map.get(t, np.nan)))
    else:
        a["role_de"] = np.nan

    def _score(sub: pd.DataFrame, role_col: str, sign_col: str = "sign_pred"):
        role = sub[role_col].to_numpy(dtype=float)
        sp = sub[sign_col].to_numpy(dtype=float)
        ap = sub["abs_proj"].to_numpy(dtype=float)
        valid = np.isfinite(role) & (role != 0) & (sp != 0)
        n = int(valid.sum())
        if n == 0:
            return np.nan, np.nan, 0
        align = np.sign(role[valid]) * sp[valid]
        return float((ap[valid] * align).sum()), float(abs(align.sum()) / n), n

    rows = []
    for src, sub in a.groupby("source", sort=False):
        ro_p, coh_p, n_p = _score(sub, "role_pert")
        ro_l, coh_l, _ = _score(sub, "role_latent")
        # Fraction des cibles où role_de et role_pert se contredisent (= cibles
        # marqueur/driver-ambiguës dans le fan-out de S, type FHL2).
        rp = sub["role_pert"].to_numpy(dtype=float)
        rd = sub["role_de"].to_numpy(dtype=float)
        both = np.isfinite(rp) & np.isfinite(rd) & (rp != 0) & (rd != 0)
        conflict_frac = (float((np.sign(rp[both]) != np.sign(rd[both])).mean())
                         if both.any() else np.nan)
        rec = {
            "gene": src,
            "signed_fanout_n": int(len(sub)),
            # 3 rôles symétriques — AUCUN headline (cf. docstring).
            "signed_readout_pert": ro_p,
            "signed_coherence_pert": coh_p,
            "signed_n_role_pert": n_p,
            "signed_readout_latent": ro_l,
            "signed_coherence_latent": coh_l,
            "signed_fanout_conflict_frac": conflict_frac,
            "signed_pred_known_agree": float(
                (sub["sign_pred"] == sub["sign_known"]).mean()),
        }
        if de_role_map is not None:
            ro_d, coh_d, n_d = _score(sub, "role_de")
            rec["signed_readout_de"] = ro_d
            rec["signed_coherence_de"] = coh_d
            rec["signed_n_role_de"] = n_d
        rows.append(rec)
    return pd.DataFrame(rows)


def _compute_evidence_tier(driver_score: float,
                           canon_cosine: float,
                           is_de_significant: bool | None,
                           n_aging_dbs: int,
                           is_hub_inflated: bool,
                           is_low_purity_signal: bool,
                           min_driver_score: float = 0.5,
                           min_cosine_purity: float = 0.4,
                           weak_driver_score: float = 0.3) -> str:
    """Single-letter evidence tier (column placed before `interpretation`).

    Priority order (first match wins) :
      D — hub : `is_hub_inflated` ∨ `is_low_purity_signal` ∨
          (driver_score ≥ 0.5 ∧ |cos| < 0.4). Score élevé porté par la
          connectivité plutôt que par une direction cohérente : artefact.
      A — confirmé : driver pur (driver_score ≥ 0.5 ∧ |cos| ≥ 0.4) ET
          littérature (DE-sig OU ≥2 aging DBs).
      B — découverte : driver pur, sans littérature → finding graph-only,
          prioritaire pour validation expérimentale.
      C — effecteur : littérature présente mais driver pas pur (faible
          score OU faible cosine). Marqueur clinique probable, peu
          actionnable comme cible mécanistique.
      E — bruit : ni littérature ni driver pur (signal de fond).

    Returns one of {"A_confirmed", "B_discovery", "C_effector", "D_hub",
    "E_noise"}.
    """
    cos_abs = abs(canon_cosine) if canon_cosine is not None else 0.0
    has_lit = bool(is_de_significant is True) or n_aging_dbs >= 2
    is_pure_driver = (driver_score >= min_driver_score
                      and cos_abs >= min_cosine_purity)
    has_strong_amplitude = driver_score >= min_driver_score

    # D-hub : flagged hub, low-purity, OR strong-amplitude-but-low-purity.
    # Le 3e cas attrape les gènes type EHMT2 (driver_score haut mais
    # |cos|<0.4) que le hub-flag explicite (ppi>200 ∧ |cos|<0.3) rate.
    if (is_hub_inflated or is_low_purity_signal
            or (has_strong_amplitude and cos_abs < min_cosine_purity)):
        return "D_hub"

    if is_pure_driver and has_lit:
        return "A_confirmed"
    if is_pure_driver and not has_lit:
        return "B_discovery"
    if has_lit:
        return "C_effector"
    return "E_noise"


def _load_vgae_training_metrics(seed_paths: list[Path]) -> list[dict]:
    """Load vgae_metrics.json from each seed root.

    Returns one dict per seed (older runs without the JSON sidecar are
    skipped silently — this keeps the cross-seed report backward-compatible
    with V3.3 / V3.6 runs that predate the metrics export). Each dict has
    an extra ``seed_dir`` key pointing to the run folder name for plotting.
    """
    import json as _json
    out: list[dict] = []
    for p in seed_paths:
        f = p / "vgae_metrics.json"
        if not f.exists():
            continue
        try:
            with open(f) as fh:
                m = _json.load(fh)
            m["seed_dir"] = p.name
            out.append(m)
        except Exception as e:
            print(f"[WARN] Failed to read {f}: {e}")
    return out


def _save_vgae_training_summary(metrics: list[dict], out_dir: Path) -> None:
    """Write per-seed scalars to TSV + bundle full history into one JSON.

    Two files in ``out_dir``:
      * vgae_training_summary.tsv — one row per seed, scalar metrics
        (best_epoch, best_auc, best_ap, mlp_auc, delta_auc, n_epochs_run,
        early_stopped). Easy to read in spreadsheets / pandas.
      * vgae_training_history.json — bundled per-epoch arrays for every
        seed. Consumed by ``--figures-only`` so the curves can be redrawn
        without re-loading every seed root.
    """
    import json as _json
    if not metrics:
        return
    rows = []
    for m in metrics:
        rows.append({
            "seed_dir": m.get("seed_dir"),
            "seed": m.get("seed"),
            "run_tag": m.get("run_tag", ""),
            "best_epoch": m.get("best_epoch"),
            "best_auc": m.get("best_auc"),
            "best_ap": m.get("best_ap"),
            "mlp_auc": m.get("mlp_auc"),
            "mlp_ap": m.get("mlp_ap"),
            "delta_auc_vgae_minus_mlp": m.get("delta_auc_vgae_minus_mlp"),
            "n_epochs_run": m.get("n_epochs_run"),
            "n_epochs_planned": m.get("n_epochs_planned"),
            "early_stopped": m.get("early_stopped"),
        })
    pd.DataFrame(rows).to_csv(out_dir / "vgae_training_summary.tsv",
                              sep="\t", index=False)
    with open(out_dir / "vgae_training_history.json", "w") as fh:
        _json.dump(metrics, fh)


def _load_vgae_training_history(out_dir: Path) -> list[dict]:
    """Re-read the bundled history written by _save_vgae_training_summary.

    Used by --figures-only so the curves can be redrawn without revisiting
    every seed root.
    """
    import json as _json
    f = out_dir / "vgae_training_history.json"
    if not f.exists():
        return []
    try:
        with open(f) as fh:
            return _json.load(fh)
    except Exception as e:
        print(f"[WARN] Failed to read {f}: {e}")
        return []


def fig_cross_seed_training_curves(metrics: list[dict], out: Path,
                                    title: str) -> None:
    """4-panel cross-seed training summary.

    Panels:
      (1) train loss vs epoch — one curve per seed, low alpha to show spread.
      (2) test AUC vs epoch — markers at each eval point + best-epoch dot.
      (3) test AP vs epoch — same layout as AUC.
      (4) bar: best AUC per seed vs MLP baseline (mean + per-seed scatter).
    """
    if not metrics:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    cmap = plt.get_cmap("tab10")
    # Panel 1 — train loss
    ax = axes[0, 0]
    for i, m in enumerate(metrics):
        h = m.get("history", {})
        ep = h.get("epoch", [])
        tl = h.get("train_loss", [])
        if ep and tl:
            ax.plot(ep, tl, color=cmap(i % 10), alpha=0.7, lw=1.2,
                    label=m.get("seed_dir", f"seed{i}"))
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.set_title("Train loss vs epoch")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # Panel 2 — test AUC
    ax = axes[0, 1]
    for i, m in enumerate(metrics):
        eh = m.get("eval_history", {})
        ep = eh.get("epoch", [])
        au = eh.get("test_auc", [])
        if ep and au:
            ax.plot(ep, au, marker="o", ms=3, color=cmap(i % 10), alpha=0.75,
                    lw=1.2, label=m.get("seed_dir", f"seed{i}"))
        be = m.get("best_epoch")
        ba = m.get("best_auc")
        if be is not None and ba is not None:
            ax.scatter([be], [ba], color=cmap(i % 10), s=70,
                       edgecolor="black", lw=0.8, zorder=5)
    # MLP baseline (mean across seeds)
    mlp_aucs = [m.get("mlp_auc") for m in metrics if m.get("mlp_auc") is not None]
    if mlp_aucs:
        ax.axhline(np.mean(mlp_aucs), color="#2ECC71", ls="--", lw=1.5,
                   label=f"MLP baseline (mean={np.mean(mlp_aucs):.3f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test AUC")
    ax.set_title("Test AUC vs epoch (• = best)")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # Panel 3 — test AP
    ax = axes[1, 0]
    for i, m in enumerate(metrics):
        eh = m.get("eval_history", {})
        ep = eh.get("epoch", [])
        ap = eh.get("test_ap", [])
        if ep and ap:
            ax.plot(ep, ap, marker="o", ms=3, color=cmap(i % 10), alpha=0.75,
                    lw=1.2, label=m.get("seed_dir", f"seed{i}"))
        be = m.get("best_epoch")
        bp = m.get("best_ap")
        if be is not None and bp is not None:
            ax.scatter([be], [bp], color=cmap(i % 10), s=70,
                       edgecolor="black", lw=0.8, zorder=5)
    mlp_aps = [m.get("mlp_ap") for m in metrics if m.get("mlp_ap") is not None]
    if mlp_aps:
        ax.axhline(np.mean(mlp_aps), color="#2ECC71", ls="--", lw=1.5,
                   label=f"MLP baseline (mean={np.mean(mlp_aps):.3f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test AP")
    ax.set_title("Test AP vs epoch (• = best)")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # Panel 4 — bar: VGAE best AUC per seed vs MLP
    ax = axes[1, 1]
    labels = [m.get("seed_dir", f"seed{i}") for i, m in enumerate(metrics)]
    vgae = [m.get("best_auc", 0.0) for m in metrics]
    mlp = [m.get("mlp_auc", 0.0) for m in metrics]
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w / 2, vgae, w, label="VGAE", color="#3498DB")
    ax.bar(x + w / 2, mlp, w, label="MLP", color="#2ECC71")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("test AUC")
    ax.set_ylim(min(min(vgae + mlp) - 0.05, 0.5), 1.0)
    ax.set_title("Best test AUC per seed vs MLP")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    if vgae:
        ax.axhline(np.mean(vgae), color="#3498DB", ls=":", lw=1,
                   alpha=0.6)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  → {out.name}")


def _load_vgae_baselines(seed_paths: list[Path]) -> pd.DataFrame:
    """Load + average gene_ranking_vgae.csv across seeds.

    Returns DataFrame indexed by gene with cross-seed averaged columns:
    vgae_importance, rank_vgae, stat_score, rank_stat, plus the
    in_{genage,cellage,msigdb_aging,ageanno,aging_local} flags as max
    (binary OR across seeds).
    """
    frames = []
    for p in seed_paths:
        rk = p / "gene_ranking_vgae.csv"
        if not rk.exists():
            continue
        df = pd.read_csv(rk)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    cat = pd.concat(frames, ignore_index=True)
    num_cols = ["vgae_importance", "rank_vgae", "stat_score", "rank_stat"]
    bool_cols = ["in_genage", "in_cellage", "in_msigdb_aging",
                 "in_ageanno", "in_aging_local"]
    num_cols = [c for c in num_cols if c in cat.columns]
    bool_cols = [c for c in bool_cols if c in cat.columns]
    agg = {c: "mean" for c in num_cols}
    agg.update({c: "max" for c in bool_cols})
    out = cat.groupby("gene").agg(agg).reset_index()
    return out


def build_gene_ranking(df: pd.DataFrame,
                        min_robustness: float = 0.7,
                        min_stability: float = 0.7,
                        min_ppi_degree: int = 5,
                        vgae_baseline: pd.DataFrame | None = None,
                        de_top_n: int = 1000,
                        de_sig_mode: str = "magnitude-rank",
                        de_padj_max: float = 0.05,
                        de_abs_lfc_min: float = 0.5,
                        is_tf_series: pd.Series | None = None,
                        gene_to_pathways: dict | None = None,
                        de_magnitude: pd.DataFrame | None = None,
                        mode_agg: str = "aligned",
                        coexpr_degree_map: dict | None = None) -> pd.DataFrame:
    """Per-gene cross-mode ranking (genes only).

    For each gene, aggregates the up to 3 perturbation modes (KO/KD/OE)
    into a single row, and enriches with optional baselines:
      * vgae_baseline → vgae_importance, vgae_rank columns + interpretation
        suffixes (canonical / novel / housekeeping).
      * de_top_n → is_de_significant flag (rank_stat ≤ de_top_n).
      * Cluster-level cosine: senescence_specificity = mean(c1,c2,c3) cos
        − mean(P4,c0) cos. Reflects that c0 is quiescent-like.

    Defaults filter to robustness ≥ 0.7, stability ≥ 0.7, NOT
    hub-inflated. The min_ppi_degree only affects the interpretation
    label (not a hard filter). Sort: max_abs_diff desc.
    """
    if df.empty:
        return pd.DataFrame()
    g = df[~df["is_pathway"]].copy()
    if g.empty:
        return pd.DataFrame()

    # Build per-gene baseline lookup if provided.
    vgae_lookup: dict[str, dict] = {}
    if vgae_baseline is not None and not vgae_baseline.empty:
        for _, r in vgae_baseline.iterrows():
            vgae_lookup[str(r["gene"])] = r.to_dict()

    rows = []
    for target, sub in g.groupby("target"):
        modes = {r["mode"]: r for _, r in sub.iterrows()}
        oe = modes.get("overexpress")
        ko = modes.get("knockout")
        kd = modes.get("knockdown")
        loss = ko if ko is not None else kd

        n_modes = sum(1 for m in (oe, ko, kd) if m is not None)
        # Sign-consistency: OE and loss-of-function should oppose each other
        # for a real causal driver (gain pushes one way, loss the other).
        sign_cons: bool | None = None
        if oe is not None and loss is not None:
            sign_cons = bool(np.sign(oe["avg_proj_signed_diff"])
                              == -np.sign(loss["avg_proj_signed_diff"]))

        # Magnitude metrics (max_abs across modes) — capture the strongest
        # signal regardless of which mode triggered it.
        max_abs_diff = float(sub["avg_proj_signed_diff"].abs().max())
        max_abs_cos = float(sub["avg_proj_signed_cosine"].abs().max())
        mean_extent = float(sub["avg_proj_signed_extent"].abs().mean())
        mean_degree = float(sub["avg_proj_signed_degree"].abs().mean())
        mean_robustness = float(sub["robustness_score"].mean())
        mean_stability = float(sub["direction_stability"].mean())
        ppi_degree = int(sub["target_ppi_degree"].iloc[0])
        any_hub = bool(sub["is_hub_inflated"].any())
        # V3.5 : low-purity signal flag (high amplitude, low cosine,
        # but PPI deg ≤ 200 — not mechanically a hub artefact). Backward
        # compat : older runs without the column → False.
        if "is_low_purity_signal" in sub.columns:
            any_low_purity = bool(sub["is_low_purity_signal"].any())
        else:
            any_low_purity = False

        # Canonical (sign, diff, cos) over the available modes. With
        # mode_agg="oe-only" this is OE-anchored (legacy). With
        # mode_agg="aligned" (default, §6.2) KO/KD are sign-aligned and
        # averaged with OE when coherent → driver_score becomes KO+KD+OE.
        canon_sign, canon_diff, canon_cos = _canonicalize_modes(
            oe, ko, kd, mode_agg=mode_agg)
        # Canonical amplitude (hub-corrected directional fraction ∈[−1,1]) —
        # only for the VISIBLE degree-weighted diagnostic columns below, not
        # for driver_score. See gnn_futur §7.11.
        canon_amp = _canon_metric(oe, ko, kd, "avg_proj_signed_amplitude",
                                  mode_agg=mode_agg)

        if canon_sign > 0:
            direction = "pro-senescence"
        elif canon_sign < 0:
            direction = "anti-senescence"
        else:
            direction = "neutral"
        if sign_cons is False:
            direction += " (mixed)"

        # Cluster specificity (Q1.E) : P16_c0 est quiescent-like (proche de
        # P4) — un vrai driver de sénescence devrait toucher c1/c2/c3 dans le
        # même sens, et pas (P4, c0). On utilise les cosines OE-canonicalisés.
        # Les colonnes <grp>_cosine viennent de aggregate_cross_seed.
        def _grab_cluster_cos(mode_row, group: str) -> float | None:
            if mode_row is None:
                return None
            v = mode_row.get(f"{group}_cosine")
            if v is None or pd.isna(v):
                return None
            return float(v)

        # Canonicalise par mode : prendre OE si présent, sinon −loss.
        if oe is not None:
            cluster_signer = 1.0
            cluster_src = oe
        elif loss is not None:
            cluster_signer = -1.0
            cluster_src = loss
        else:
            cluster_signer = 0.0
            cluster_src = None

        cos_p4 = _grab_cluster_cos(cluster_src, "P4")
        cos_c0 = _grab_cluster_cos(cluster_src, "P16_cluster_0")
        cos_c1 = _grab_cluster_cos(cluster_src, "P16_cluster_1")
        cos_c2 = _grab_cluster_cos(cluster_src, "P16_cluster_2")
        cos_c3 = _grab_cluster_cos(cluster_src, "P16_cluster_3")
        senescence_specificity: float | None = None
        cosine_quiescent: float | None = None
        cosine_senescent: float | None = None
        if cos_c1 is not None and cos_c2 is not None and cos_c3 is not None:
            qvals = [v for v in (cos_p4, cos_c0) if v is not None]
            if qvals:
                cosine_quiescent = cluster_signer * float(np.mean(qvals))
                cosine_senescent = cluster_signer * float(np.mean([cos_c1, cos_c2, cos_c3]))
                senescence_specificity = cosine_senescent - cosine_quiescent

        # Baseline VGAE / DE lookup (Q1.F + Q2).
        vgae_row = vgae_lookup.get(target, {})
        vgae_importance = vgae_row.get("vgae_importance")
        vgae_rank_v = vgae_row.get("rank_vgae")
        rank_stat_v = vgae_row.get("rank_stat")
        # is_de_significant — deux modes (cf. --de-significance) :
        #   'magnitude-rank' (défaut, backward-compat) : rang |ΔExpr P16-P4| ≤ de_top_n
        #     (rank_stat issu de gene_ranking_vgae.csv ; couvre tous les gènes du graphe).
        #   'pvalue' : test MAST padj < de_padj_max ET |avg_log2FC| ≥ de_abs_lfc_min.
        #     (padj seul ne discrimine pas en scRNA → exiger AUSSI |logFC|.) Gène absent
        #     du MAST → None (inconnu), PAS de fallback rank_stat (comparaison propre).
        is_de_significant: bool | None = None
        if de_sig_mode == "pvalue":
            if de_magnitude is not None and target in de_magnitude.index:
                _lfc = de_magnitude.at[target, "de_log2fc_p4_vs_p16"]
                _nlp = de_magnitude.at[target, "de_neglog10_padj"]  # = -log10(padj)
                if pd.notna(_lfc) and pd.notna(_nlp):
                    is_de_significant = bool(
                        float(_nlp) >= -np.log10(de_padj_max)
                        and abs(float(_lfc)) >= de_abs_lfc_min
                    )
        else:  # 'magnitude-rank'
            if rank_stat_v is not None and not (isinstance(rank_stat_v, float) and np.isnan(rank_stat_v)):
                is_de_significant = bool(int(rank_stat_v) <= de_top_n)
        n_aging_dbs = sum(int(vgae_row.get(c, 0) or 0) for c in
                           ("in_genage", "in_cellage", "in_msigdb_aging",
                            "in_ageanno", "in_aging_local"))

        # Per-mode raw diffs/cosines (passed to _gene_interpretation for
        # Tier-1 tags : non-monotonic, gain-of-function-only).
        ko_diff_val = float(ko["avg_proj_signed_diff"]) if ko is not None else None
        kd_diff_val = float(kd["avg_proj_signed_diff"]) if kd is not None else None
        oe_diff_val = float(oe["avg_proj_signed_diff"]) if oe is not None else None
        oe_cos_val = float(oe["avg_proj_signed_cosine"]) if oe is not None else None

        # is_tf flag (pySCENIC-detected TF in HUVEC). For TFs, we expect
        # pleiotropic cascades → relaxed cosine threshold downstream.
        is_tf_v = False
        if is_tf_series is not None and target in is_tf_series.index:
            is_tf_v = bool(int(is_tf_series.loc[target]) == 1)

        # Pathways the gene is a member of, restricted to strong/moderate
        # pathways from pw_ranking (informative cross-reference).
        member_of = (gene_to_pathways or {}).get(target, [])

        # MAST DE magnitudes (signed log2FC + −log10 padj, both clipped).
        # Source : data/gnn_data/DEGs_P4_vs_P16_MAST.csv. Decoupled from
        # `is_de_significant` (which uses rank_stat from the pseudo-LFC
        # baseline); see report §10.13decies.
        de_lfc: float | None = None
        de_neglog10_padj: float | None = None
        if de_magnitude is not None and target in de_magnitude.index:
            de_lfc = float(de_magnitude.at[target, "de_log2fc_p4_vs_p16"])
            de_neglog10_padj = float(de_magnitude.at[target, "de_neglog10_padj"])

        vgae_rank_int = (int(vgae_rank_v)
                         if vgae_rank_v is not None and not (
                             isinstance(vgae_rank_v, float) and np.isnan(vgae_rank_v))
                         else None)

        # Compute continuous scores (V3.5). The three scores are
        # **independent** : driver = graph-intrinsic, validation = pure
        # literature × graph-quality, discovery = graph-novelty for
        # genes the literature ignores. A high driver_score does NOT
        # propagate into the other two.
        driver_score = _compute_driver_score(
            canon_diff, canon_cos, n_modes, sign_cons, any_hub,
            vgae_rank=vgae_rank_int)
        discovery_score = _compute_discovery_score(
            canon_diff, canon_cos, n_modes,
            is_de_significant, n_aging_dbs, any_hub,
            low_purity=any_low_purity,
            senescence_specificity=senescence_specificity,
            mean_robustness=mean_robustness,
            mean_stability=mean_stability)
        validation_score = _compute_validation_score(
            is_de_significant, n_aging_dbs,
            mean_robustness=mean_robustness,
            mean_stability=mean_stability)

        evidence_tier = _compute_evidence_tier(
            driver_score=driver_score,
            canon_cosine=canon_cos,
            is_de_significant=is_de_significant,
            n_aging_dbs=n_aging_dbs,
            is_hub_inflated=any_hub,
            is_low_purity_signal=any_low_purity,
        )

        interp = _gene_interpretation(
            canon_diff, canon_cos, ppi_degree, any_hub, sign_cons, n_modes,
            min_ppi_degree=min_ppi_degree,
            senescence_specificity=senescence_specificity,
            vgae_rank=vgae_rank_int,
            is_de_significant=is_de_significant,
            n_aging_dbs=n_aging_dbs,
            ko_diff=ko_diff_val, kd_diff=kd_diff_val,
            oe_diff=oe_diff_val, oe_cos=oe_cos_val,
            mean_abs_extent=mean_extent,
            mean_abs_degree_metric=mean_degree,
            driver_score=driver_score,
            is_tf=is_tf_v,
            mean_robustness=mean_robustness,
            mean_stability=mean_stability,
            min_robustness=min_robustness,
            min_stability=min_stability,
            low_purity=any_low_purity,
        )

        rec = {
            "target": target,
            # Continuous scores (V3.4) — sort by driver_score (graph-only).
            # Use discovery_score for graph-only candidates, validation_score
            # for literature-backed shortlists.
            "driver_score": round(driver_score, 3),
            "discovery_score": round(discovery_score, 3),
            "validation_score": round(validation_score, 3),
            "is_tf": is_tf_v,
            "n_modes_present": n_modes,
            "mean_robustness": round(mean_robustness, 2),
            "mean_stability": round(mean_stability, 2),
            # Canonical metrics (signed by OE if present, else by −loss).
            "canon_diff": round(canon_diff, 1),
            "canon_cosine": round(canon_cos, 3),
            "canon_amplitude": round(canon_amp, 3),
            "max_abs_diff": round(max_abs_diff, 1),
            "max_abs_cosine": round(max_abs_cos, 3),
            "mean_abs_extent": round(mean_extent, 4),
            "mean_abs_degree": round(mean_degree, 2),
            "target_ppi_degree": ppi_degree,
            # Cluster specificity (Q1.E)
            "cosine_quiescent_like": round(cosine_quiescent, 3) if cosine_quiescent is not None else None,
            "cosine_senescent": round(cosine_senescent, 3) if cosine_senescent is not None else None,
            "senescence_specificity": round(senescence_specificity, 3) if senescence_specificity is not None else None,
            # Baseline cross-checks
            "vgae_importance": round(float(vgae_importance), 4) if vgae_importance is not None else None,
            "vgae_rank": vgae_rank_int,
            "is_de_significant": is_de_significant,
            "de_log2fc_p4_vs_p16": de_lfc,
            "de_neglog10_padj": de_neglog10_padj,
            "n_aging_dbs": n_aging_dbs,
            "sign_consistent": sign_cons if sign_cons is not None else "",
            "is_hub_inflated": any_hub,
            "is_low_purity_signal": any_low_purity,
            "direction": direction,
            "evidence_tier": evidence_tier,
            "interpretation": interp,
            # Per-mode breakdown
            "KO_diff": round(float(ko["avg_proj_signed_diff"]), 1) if ko is not None else None,
            "KD_diff": round(float(kd["avg_proj_signed_diff"]), 1) if kd is not None else None,
            "OE_diff": round(float(oe["avg_proj_signed_diff"]), 1) if oe is not None else None,
            "KO_cos": round(float(ko["avg_proj_signed_cosine"]), 3) if ko is not None else None,
            "KD_cos": round(float(kd["avg_proj_signed_cosine"]), 3) if kd is not None else None,
            "OE_cos": round(float(oe["avg_proj_signed_cosine"]), 3) if oe is not None else None,
            # Pathway membership — moved to last column (long string,
            # better at the end for readability of the TSV).
            "member_of_strong_pathways": ";".join(member_of) if member_of else "",
        }
        rows.append(rec)

    out = pd.DataFrame(rows)

    # --- Degree-weighted DIAGNOSTIC columns (V5.4.1, visible, NON-ranking) ---
    # Aucune ne pilote le tri (toujours `driver_score`). Elles exposent la
    # même information sous différentes corrections de degré pour suivre
    # l'évolution des scores cross-ablation/version (gnn_futur §7.11, §5).
    # Rappel des conclusions : `diff×cos` aggrave le biais PPI ; `amplitude`
    # ≈ pureté (perd la force) ; `diff/PPIdeg` laisse le hub coexpr ;
    # `diff/(PPI+coexpr)` peut sur-corriger. → diagnostic, pas décision.
    if not out.empty:
        def _p99norm(x: pd.Series) -> pd.Series:
            lx = np.log10(x.abs() + 1.0)
            p = float(np.nanpercentile(lx, 99))
            return (lx / (p + 1e-9)).clip(0.0, 1.0)

        purity = out["canon_cosine"].abs()
        ppideg = out["target_ppi_degree"].clip(lower=1)
        out["coexpr_degree"] = (out["target"].map(coexpr_degree_map).fillna(0.0)
                                if coexpr_degree_map else np.nan)
        # force × pureté (degré-biaisé PPI)
        out["ds_diffcos"] = (_p99norm(out["canon_diff"]) * purity).round(4)
        # pureté directionnelle seule (≈ cosinus, degree-free total)
        out["ds_amp"] = out["canon_amplitude"].abs().round(4)
        # force/degré PPI × pureté (dé-biaise le hub PPI, laisse le hub coexpr)
        out["ds_ppideg"] = (_p99norm(out["canon_diff"] / ppideg) * purity).round(4)
        # force/(degré PPI+coexpr) × pureté (si coexpr_degree fourni)
        if coexpr_degree_map:
            totdeg = (out["target_ppi_degree"] + out["coexpr_degree"]).clip(lower=1)
            out["ds_totdeg"] = (_p99norm(out["canon_diff"] / totdeg) * purity).round(4)

    # V3.4 : do NOT filter. Quality issues (low robustness/stability,
    # hub-inflated, incoherent, low-PPI) are surfaced in the
    # `interpretation` column as `[unreliable]`, `[hub-inflated]`,
    # `[incoherent]` prefixes / `low PPI ...` suffix. The user can
    # filter post-hoc on `mean_robustness` / `is_hub_inflated` /
    # `sign_consistent` / `target_ppi_degree`.
    # Sort by driver_score (graph-only signal) ; tie-break on max_abs_diff.
    out = out.sort_values(["driver_score", "max_abs_diff"],
                          ascending=[False, False]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Cross-seed aggregation
# --------------------------------------------------------------------------- #
def aggregate_cross_seed(perturb_dirs: list[Path] | None = None,
                         seed_summaries: list[list[tuple[str, dict]]] | None = None,
                         min_robustness: float = 0.5,
                         min_stability: float = 0.7
                         ) -> pd.DataFrame:
    """
    Agrège les résultats de plusieurs seeds pour identifier les drivers robustes.
    
    Nouveautés de cette version :
    1. Agrégation des projections par cluster (P4, P16_c0, etc.) pour fig_transitions_scatter.
    2. Agrégation des métriques de shift (relative/gene_diff) pour fig_shift_methods_compare.
    3. Gestion robuste des NaNs si une seed n'a pas calculé certains clusters.
    """
    from collections import Counter
    
    # 1. Collecte des données (si non fournies via --all)
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

    # 2. Groupement par cible : (mode, target) -> list[summary_dicts]
    key_data: dict[tuple, list[dict]] = {}
    for seed in seed_summaries:
        for tag, s in seed:
            mode = s.get("mode") or infer_mode_from_tag(tag)
            if mode not in MODES:
                continue
            # On extrait le nom du gène/pathway sans le préfixe du mode
            target = tag[len(mode) + 1:] if tag.startswith(f"{mode}_") else tag
            key_data.setdefault((mode, target), []).append(s)

    rows = []
    # Toutes les variantes de projection à agréger en cross-seed.
    _proj_variants = ("diff", "norm", "amplitude", "extent", "degree", "cosine")

    # 3. Boucle d'agrégation statistique
    for (mode, target), entries in key_data.items():
        # --- Métriques de projection (Senescence Axis) ---
        # On collecte les séries multi-seeds pour toutes les variantes.
        projs_by_variant: dict[str, list[float]] = {
            v: [float(e.get(f"max_proj_signed_{v}") or 0.0) for e in entries]
            for v in _proj_variants
        }
        projs = projs_by_variant["diff"]   # réf. pour la stabilité de signe
        degrees = [int(e.get("target_ppi_degree") or 0) for e in entries]

        # --- Métriques de Shift (pour fig_shift_methods_compare) ---
        shifts_diff = [float(e.get("max_shift_gene_differential") or 0.0) for e in entries]
        shifts_rel = [float(e.get("max_shift_relative") or 0.0) for e in entries]

        # --- Données par Cluster (pour fig_transitions_scatter / heatmap_global) ---
        # On définit les groupes qu'on veut récupérer
        target_groups = ["P4", "P16_cluster_0", "P16_cluster_1", "P16_cluster_2", "P16_cluster_3"]
        cluster_vals = {grp: [] for grp in target_groups}
        cluster_vals_cosine = {grp: [] for grp in target_groups}

        for e in entries:
            # On cherche dans le dictionnaire global du summary.json
            g_data = e.get("cell_group_shift_projected_global", {})
            for grp in target_groups:
                val = g_data.get(grp, {}).get("proj_signed_diff")
                if val is not None:
                    cluster_vals[grp].append(float(val))
                val_c = g_data.get(grp, {}).get("proj_signed_cosine")
                if val_c is not None:
                    cluster_vals_cosine[grp].append(float(val_c))

        # Calcul de la stabilité du signe (basé sur proj_signed_diff, métrique de référence).
        signs = [np.sign(p) for p in projs if p != 0]
        if signs:
            pos = sum(1 for s in signs if s > 0)
            neg = sum(1 for s in signs if s < 0)
            stability = max(pos, neg) / len(signs)
        else:
            stability = 0.0

        # --- Construction du dictionnaire de résultats ---
        n_present = len(entries)
        avg_proj = float(np.mean(projs))

        res = {
            "target": target,
            "mode": mode,
            "is_pathway": is_pathway_tag(f"{mode}_{target}"),
            # Tag reconstitué pour les figures qui l'attendent (shorten_tag, etc.)
            "tag": f"{mode}_{target}",
            "n_seeds_present": n_present,
            "robustness_score": n_present / total_seeds,
            "direction_stability": stability,
            # Moyennes pour les figures de barres et volcans (métrique principale).
            "avg_proj_signed_diff": avg_proj,
            "std_proj_signed_diff": float(np.std(projs)) if n_present > 1 else 0.0,
            "max_proj_signed_diff": avg_proj, # Alias pour compatibilité avec fonctions existantes
            # Degré PPI moyen (utile pour figures correctives hub).
            "target_ppi_degree": float(np.mean(degrees)) if degrees else 0.0,
            # Moyennes des autres shifts (Option 1)
            "max_shift_gene_differential": float(np.mean(shifts_diff)),
            "max_shift_relative": float(np.mean(shifts_rel)),
        }
        # Variantes additionnelles (norm, amplitude, extent, degree, cosine) :
        # moyennes + std. On expose aussi max_* = avg_* par convention
        # (alias pour les figures existantes).
        for v in _proj_variants:
            if v == "diff":
                continue
            vs = projs_by_variant[v]
            mean_v = float(np.mean(vs)) if vs else 0.0
            std_v = float(np.std(vs)) if len(vs) > 1 else 0.0
            res[f"avg_proj_signed_{v}"] = mean_v
            res[f"std_proj_signed_{v}"] = std_v
            res[f"max_proj_signed_{v}"] = mean_v

        # Ajout des moyennes par cluster (pour les scatters de transition).
        # Colonnes <grp> = proj_signed_diff (legacy), <grp>_cosine = cosine.
        for grp, vals in cluster_vals.items():
            res[grp] = float(np.mean(vals)) if vals else np.nan
        for grp, vals in cluster_vals_cosine.items():
            res[f"{grp}_cosine"] = float(np.mean(vals)) if vals else np.nan

        # Consensus de la direction (pour le texte du TSV)
        if avg_proj > 0:
            res["direction"] = "pro-senescence"
        elif avg_proj < 0:
            res["direction"] = "anti-senescence"
        else:
            res["direction"] = "neutral"

        # Hub-inflated flag (V3.5) : effet absolu fort + cosine faible +
        # **degré PPI élevé**. Sans la condition de degré, des gènes
        # borderline (cos ≈ 0.29, deg ≈ 48 — ASNS) étaient flaggés à tort
        # comme hubs alors que l'inflation par connectivité ne tient pas
        # mécaniquement. Le seuil 200 sépare proprement les vrais hubs
        # (UBC, RPS27A, UBA52 — deg > 1000) des gènes simplement à
        # purety modérée.
        # Pour les borderline (forte amplitude, cos faible, deg≤200) on
        # ajoute `is_low_purity_signal` qui nourrit le tag
        # `[low-purity signal]` dans l'interprétation, sans pénaliser
        # le driver_score (l'amplitude n'est pas explicable par le hub).
        avg_cos = res.get("avg_proj_signed_cosine", 0.0)
        ppi_deg = res.get("target_ppi_degree", 0.0)
        is_low_purity = bool(abs(avg_proj) > 50.0 and abs(avg_cos) < 0.3)
        res["is_hub_inflated"] = bool(is_low_purity and ppi_deg > 200)
        res["is_low_purity_signal"] = bool(is_low_purity and ppi_deg <= 200)

        rows.append(res)

    # 4. Conversion en DataFrame et filtrage
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Application des seuils de rigueur
    df = df[(df["robustness_score"] >= min_robustness) &
            (df["direction_stability"] >= min_stability)].copy()

    # Tri par importance absolue du signal moyen
    df["abs_avg"] = df["avg_proj_signed_diff"].abs()
    df = df.sort_values("abs_avg", ascending=False).drop(columns=["abs_avg"])
    
    return df.reset_index(drop=True)


def fig_gene_ranking_top_bars(df: pd.DataFrame, out: Path, title: str,
                               direction: str = "pro",
                               n: int = 30,
                               sort_by: str = "driver_score") -> None:
    """Horizontal bar plot of top-N genes from cross_seed_gene_ranking.

    `direction`: 'pro' or 'anti' — the function filters the corresponding
    rows of the `direction` column (split on the parenthetical suffix used
    by build_gene_ranking, e.g. "anti-senescence (mixed)").

    Bar height = `sort_by` (default driver_score). Bars are colored by
    direction (red=pro / green=anti) and alpha-modulated by |canon_cosine|
    so direction-pure drivers appear saturated and hub-flavoured rows
    appear pale.

    Each bar is annotated, on the right, with:
      - DE-significance (DE✓ / ✗)
      - n_aging_dbs
      - |canon_cosine| (= purity of the senescence direction)
      - target_ppi_degree
      - quality / status tags drawn from the columns produced by
        build_gene_ranking : [HUB] [low-purity] [incoherent] [unreliable]
        [TF]. Alarming tags switch the annotation colour to dark red so
        the eye picks them up before the score is read.
    """
    if df.empty or "direction" not in df.columns or sort_by not in df.columns:
        return
    work = df.copy()
    work["direction_clean"] = (work["direction"].astype(str)
                               .str.split(r"\s*\(", regex=True).str[0])
    keep = "pro-senescence" if direction == "pro" else "anti-senescence"
    sub = work[work["direction_clean"] == keep].copy()
    if sub.empty:
        return
    sub = sub.sort_values(sort_by, ascending=False).head(n)
    if sub.empty:
        return
    sub = sub.iloc[::-1].reset_index(drop=True)  # top driver at top of barh

    fig_h = max(4, 0.35 * len(sub) + 1.2)
    fig, ax = plt.subplots(figsize=(13, fig_h))

    base_color = "#e76f51" if direction == "pro" else "#2a9d8f"
    cos_abs = (sub["canon_cosine"].abs().fillna(0.0).values
               if "canon_cosine" in sub.columns
               else np.full(len(sub), 1.0))
    alphas = np.clip(0.30 + 0.65 * cos_abs, 0.30, 0.95)

    y = np.arange(len(sub))
    scores = sub[sort_by].values.astype(float)
    for yi, s, a in zip(y, scores, alphas):
        ax.barh(yi, s, color=base_color, alpha=a,
                edgecolor="black", linewidth=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(sub["target"].astype(str), fontsize=9)
    ax.set_xlabel(sort_by.replace("_", " "))
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    xmax = float(np.nanmax(scores)) if len(scores) and np.isfinite(scores).any() else 1.0
    if xmax <= 0:
        xmax = max(1.0, float(np.nanmax(np.abs(scores))) if np.isfinite(scores).any() else 1.0)
    # Reserve a wide right margin for the annotation strings.
    ax.set_xlim(0, xmax * 1.75)

    alarm_tags = {"HUB", "low-purity", "incoherent", "unreliable"}
    for yi, (_, r) in zip(y, sub.iterrows()):
        s = float(r[sort_by]) if pd.notna(r[sort_by]) else 0.0

        de_flag = bool(r.get("is_de_significant", False))
        de_str = "DE✓" if de_flag else "DE✗"
        n_db = int(r["n_aging_dbs"]) if pd.notna(r.get("n_aging_dbs", np.nan)) else 0
        cos = r.get("canon_cosine", np.nan)
        cos_str = f"|cos|={abs(cos):.2f}" if pd.notna(cos) else "|cos|=–"
        ppi = int(r["target_ppi_degree"]) if pd.notna(r.get("target_ppi_degree", np.nan)) else 0
        info = f"{de_str}  aging_DB={n_db}  {cos_str}  deg={ppi}"

        tags: list[str] = []
        if bool(r.get("is_hub_inflated", False)):
            tags.append("HUB")
        if bool(r.get("is_low_purity_signal", False)):
            tags.append("low-purity")
        if r.get("sign_consistent") is False:
            tags.append("incoherent")
        rob = r.get("mean_robustness", np.nan)
        stab = r.get("mean_stability", np.nan)
        if (pd.notna(rob) and rob < 0.5) or (pd.notna(stab) and stab < 0.7):
            tags.append("unreliable")
        if bool(r.get("is_tf", 0)):
            tags.append("TF")
        tag_str = " ".join(f"[{t}]" for t in tags)

        text = f"{info}   {tag_str}".rstrip()
        color = "darkred" if any(t in alarm_tags for t in tags) else "black"
        ax.text(s + xmax * 0.015, yi, text,
                va="center", ha="left", fontsize=7.5, color=color,
                family="monospace")

    legend = ("alpha ∝ |canon_cosine|  (saturated = direction-pure)\n"
              "tags : [HUB] hub-inflated · [low-purity] |cos|<0.3 ·\n"
              "       [incoherent] OE/loss same sign · [unreliable] rob<0.5 / stab<0.7 · [TF]")
    ax.text(0.99, 0.01, legend, transform=ax.transAxes,
            fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_cross_seed_top_bars(df: pd.DataFrame, out: Path, title: str,
                             n_per_side: int = 10):
    """Top-n pro + top-n anti drivers with bar height = avg_proj_signed_diff,
    error = std across seeds. Bars are colored by direction (red=pro-,
    green=anti-senescence) **and** alpha-modulated by |avg_proj_signed_cosine|
    so hub-inflated drivers (low cosine) appear pale grey, real directional
    drivers (cosine high) appear saturated. A grey overlay marks |cos|<0.3.
    """
    if df.empty:
        return
    top = filter_top_polar(df, n_per_side, sort_col="avg_proj_signed_diff")
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(top)), 5))
    x = np.arange(len(top))
    has_cos = "avg_proj_signed_cosine" in top.columns
    cos_abs = top["avg_proj_signed_cosine"].abs() if has_cos else np.full(len(top), 1.0)
    # Alpha = 0.25 (hub-inflated, |cos|≤0.1) → 0.95 (pure directional, |cos|≥0.8).
    alphas = np.clip(0.25 + 0.875 * cos_abs, 0.25, 0.95)
    base_colors = ["#e76f51" if v > 0 else "#2a9d8f"
                   for v in top["avg_proj_signed_diff"]]
    for xi, h, c, a, e in zip(x,
                              top["avg_proj_signed_diff"],
                              base_colors,
                              alphas,
                              top["std_proj_signed_diff"]):
        ax.bar(xi, h, color=c, alpha=a, edgecolor="black", linewidth=0.5,
               yerr=e, capsize=3,
               error_kw={"ecolor": "black", "elinewidth": 0.8, "alpha": 0.6})
    # Grey overlay around hub-inflated bars.
    if has_cos:
        for xi, c in zip(x, cos_abs):
            if c < 0.3:
                ax.axvspan(xi - 0.45, xi + 0.45, color="lightgrey", alpha=0.3, zorder=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    labels = [f"{shorten_mode(r['mode'])}_{r['target']}"
              if not r["is_pathway"] else shorten_tag(f"{r['mode']}_{r['target']}")
              for _, r in top.iterrows()]
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("avg proj_signed_diff  (±std)   |   alpha ∝ |cosine|")
    ax.set_title(title)
    ax.text(0.01, 0.99,
            "red=pro / green=anti  |  pale = hub-inflated (|cos|<0.3)\n"
            "saturated = intrinsically directional",
            transform=ax.transAxes, fontsize=7, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


PROJ_VARIANTS = ("diff", "norm", "amplitude", "extent", "degree", "cosine")
PROJ_VARIANT_LABELS = {
    "diff":      "diff (raw sum)",
    "norm":      "norm (÷ Σw_diff)",
    "amplitude": "amplitude (÷ weighted ‖Δz‖)",
    "extent":    "extent (÷ n_affected)",
    "degree":    "degree (÷ PPI degree)",
    "cosine":    "cosine(Δz̄, axis)",
}


def _variant_cols(df: pd.DataFrame, prefix: str = "avg_proj_signed_") -> list[str]:
    """Return the <prefix><v> columns that exist in df, in stable order.

    `prefix` is typically 'avg_proj_signed_' (cross-seed) or
    'max_proj_signed_' (single-seed).
    """
    return [f"{prefix}{v}" for v in PROJ_VARIANTS
            if f"{prefix}{v}" in df.columns]


def fig_cross_seed_quadrant_diff_cosine(df: pd.DataFrame, out: Path, title: str,
                                         n_annotate: int = 15):
    """Quadrant scatter: avg_proj_signed_diff (X) vs avg_proj_signed_cosine (Y).

    Reveals 4 driver categories at a glance:
      * NE  (diff>0, cos>0)  — pro-senescence, coherent.
      * SW  (diff<0, cos<0)  — anti-senescence, coherent.
      * NW/SE (sign mismatch) — incoherent (rare; usually noise).
      * Around X-axis (|cos|<0.3) — hub-inflated diffuse cascade.
    The grey vertical band marks the hub-inflated zone (|cos|<0.3).
    Marker color = sign of diff. Size ∝ robustness. Annotates the top
    `n_annotate` by |diff|·|cos| (best compromise) plus the top-5 by |cos|
    that aren't already in the diff·cos top.
    """
    if df.empty or "avg_proj_signed_cosine" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    diff = df["avg_proj_signed_diff"].to_numpy()
    cos = df["avg_proj_signed_cosine"].to_numpy()
    colors = ["#e76f51" if v > 0 else "#2a9d8f" for v in diff]
    sizes = 10 + 60 * df["robustness_score"].to_numpy()
    # Background shaded band: hub-inflated zone (|cos| < 0.3 — small purity).
    ax.axhspan(-0.3, 0.3, color="lightgrey", alpha=0.25, zorder=0,
               label="hub-inflated zone (|cos|<0.3)")
    ax.scatter(diff, cos, c=colors, s=sizes, alpha=0.55,
               edgecolor="black", linewidth=0.3, zorder=2)
    ax.axhline(0, color="k", lw=0.6, linestyle="--", alpha=0.5)
    ax.axvline(0, color="k", lw=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("avg proj_signed_diff  (effect amplitude, signed)")
    ax.set_ylabel("avg proj_signed_cosine  (purity, ∈ [−1, +1])")
    ax.set_title(title)
    ax.set_ylim(-1.05, 1.05)
    # Annotate top-N by |diff·cos| (compromise) + top-5 by |cos| not yet labeled.
    score_combo = np.abs(diff) * np.abs(cos)
    idx_combo = np.argsort(-score_combo)[:n_annotate]
    seen = set(idx_combo.tolist())
    score_pure = np.abs(cos)
    idx_pure = [i for i in np.argsort(-score_pure) if i not in seen][:5]
    for k in list(idx_combo) + list(idx_pure):
        r = df.iloc[k]
        lbl = (f"{shorten_mode(r['mode'])}_{r['target']}" if not r["is_pathway"]
               else shorten_tag(f"{r['mode']}_{r['target']}"))
        ax.annotate(lbl, (diff[k], cos[k]),
                    fontsize=7, alpha=0.9,
                    xytext=(4, 2), textcoords="offset points")
    # Quadrant text overlays (subtle).
    ax.text(0.99, 0.99, "REAL pro-senescence drivers\n(high diff, high +cos)",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.55))
    ax.text(0.01, 0.01, "REAL anti-senescence drivers\n(low diff, low −cos)",
            transform=ax.transAxes, fontsize=8, ha="left", va="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.55))
    ax.text(0.99, 0.5, "← diffuse cascade  →",
            transform=ax.transAxes, fontsize=7, color="grey",
            ha="right", va="center", alpha=0.85)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_heatmap_projections_cosine_df(df: pd.DataFrame, out: Path, title: str,
                                       n_per_side: int = 25):
    """Per-cluster cosine heatmap. Colors are intrinsically bounded to
    [−1, 1] so the figure is immune to hub-inflation issues that plague
    the proj_signed_diff heatmap. Rows = top drivers (selected by |diff|
    for direct comparability), columns = P4 + P16_cluster_0..3.
    """
    grp_cols = ["P4_cosine", "P16_cluster_0_cosine", "P16_cluster_1_cosine",
                "P16_cluster_2_cosine", "P16_cluster_3_cosine"]
    have_cos_cols = [c for c in grp_cols if c in df.columns]
    if df.empty or not have_cos_cols:
        return
    top = filter_top_polar(df, n_per_side, sort_col="avg_proj_signed_diff")
    if top.empty:
        return
    M = top[have_cos_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6, max(4, 0.28 * len(top))))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(have_cos_cols)))
    ax.set_xticklabels([c.replace("_cosine", "") for c in have_cos_cols],
                       rotation=35, ha="right", fontsize=8)
    labels = [f"{shorten_mode(r['mode'])}_{r['target']}"
              if not r["is_pathway"] else shorten_tag(f"{r['mode']}_{r['target']}")
              for _, r in top.iterrows()]
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(labels, fontsize=6)
    plt.colorbar(im, ax=ax, label="cosine(Δz̄, axis)  ∈ [−1, 1]")
    ax.set_title(title)
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


def _matches_axis_suffix(tsv_name: str, axis_tag: str | None) -> bool:
    """Filtre un nom de TSV par axe de sénescence (V3 vs V4).

    Convention nommage produite par perturb_top_genes.py :
      sans suffixe (V3) : perturbation_all_<target>_<mode>.tsv
      avec --out-suffix _axisV4 : perturbation_all_<target>_axisV4_<mode>.tsv

    Args:
        axis_tag : None  → match tout (legacy)
                   ""    → seulement les TSV sans suffixe d'axe (V3)
                   "axisV4" → seulement les TSV avec _axisV4_

    Returns: True si le TSV passe le filtre.
    """
    if axis_tag is None:
        return True
    stem = tsv_name[:-len(".tsv")] if tsv_name.endswith(".tsv") else tsv_name
    parts = stem.split("_")
    # Schéma : ['perturbation', 'all', '<target>', ('<axis>',) '<mode>']
    if len(parts) < 4 or parts[0] != "perturbation" or parts[1] != "all":
        return False
    if len(parts) == 4:
        actual = ""              # pas de suffixe d'axe
    elif len(parts) == 5:
        actual = parts[3]        # axisV3 / axisV4 / autre
    else:
        return False             # nom inattendu, on ne risque pas
    return actual == axis_tag


def _collect_seed_summaries(
    p: Path,
    axis_tag: str | None = None,
) -> list[tuple[str, dict]]:
    """Given a seed-ish path (seed root OR perturbation/ subdir), return
    (tag, summary) pairs.

    Priority: ALL-mode TSVs first (richer & avoids the 10k-folder materialise
    hit); per-target summary.json folders as fallback. Checks both `p` and
    `p/perturbation` to cover either layout convention.

    Args:
        axis_tag : None=tout, ""=V3 (sans suffixe), "axisV4"=V4 only.
            Évite de mélanger V3 et V4 dans un même cross-seed quand un run
            contient les deux côte à côte (--axis both).
    """
    if not p.exists():
        return []
    for candidate in (p, p / "perturbation"):
        if not candidate.is_dir():
            continue
        all_tsvs = sorted(candidate.glob("perturbation_all_*.tsv"))
        tsvs = [t for t in all_tsvs if _matches_axis_suffix(t.name, axis_tag)]
        if tsvs:
            filt_msg = f" [axis={axis_tag!r}]" if axis_tag is not None else ""
            print(f"  {p.name}: TSV source ({len(tsvs)} file(s) in "
                  f"{candidate.name}/){filt_msg}")
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
    """
    Handler principal pour le mode --cross-seed.

    Cette fonction pilote l'agrégation de N seeds (répétitions) et génère
    un rapport de robustesse incluant les drivers stables, les analyses de
    transition et les comparaisons de méthodes de calcul de shift.

    Avec ``--figures-only`` on saute toute la phase d'agrégation et
    d'écriture des TSV : les fichiers ``cross_seed_drivers.tsv`` et
    ``cross_seed_gene_ranking.tsv`` sont lus depuis ``--report-dir`` (ou le
    dossier ``cross_seed_report`` par défaut) et seules les figures sont
    régénérées.
    """
    # --- Branche --figures-only : on saute l'agrégation, on lit les TSV ---
    if getattr(args, "figures_only", False):
        if args.report_dir is not None:
            out_dir = Path(args.report_dir)
        else:
            raw_paths = [Path(p) for p in args.cross_seed]
            out_dir = _common_seed_parent(raw_paths) / "cross_seed_report"
        drivers_path = out_dir / "cross_seed_drivers.tsv"
        gene_rank_path = out_dir / "cross_seed_gene_ranking.tsv"
        if not drivers_path.exists():
            print(f"[figures-only] Missing {drivers_path} — cannot regenerate "
                  f"figures without an existing cross_seed_drivers.tsv.")
            return
        print(f"[figures-only] Loading TSVs from {out_dir}")
        df = pd.read_csv(drivers_path, sep="\t")
        gene_rank = (pd.read_csv(gene_rank_path, sep="\t")
                     if gene_rank_path.exists() else pd.DataFrame())
        if gene_rank.empty:
            print(f"[figures-only] {gene_rank_path.name} not found — "
                  f"gene-ranking bar figures will be skipped.")
        _render_cross_seed_figures(df, gene_rank, args, out_dir)
        print(f"\n[SUCCESS] Figures regenerated in {out_dir}")
        return

    raw_paths = [Path(p) for p in args.cross_seed]
    seed_summaries: list[list[tuple[str, dict]]] = []
    kept_paths: list[Path] = []

    # Filtre d'axe : permet de séparer V3 (suffix vide) et V4 (axisV4) quand
    # les runs contiennent les deux côte à côte (--axis both côté grid).
    axis_tag = getattr(args, "axis_tag", None)

    # 1. Collecte des données à travers les différentes seeds
    print(f"[CROSS-SEED] Collecting summaries for {len(raw_paths)} seed(s)"
          + (f" [axis={axis_tag!r}]" if axis_tag is not None else "") + ":")
    for p in raw_paths:
        # collect_seed_summaries cherche soit des dossiers individuels, soit des TSV --all
        sums = _collect_seed_summaries(p, axis_tag=axis_tag)
        if not sums:
            print(f" {p.name}: [skip] no summaries or TSVs found")
            continue
        seed_summaries.append(sums)
        kept_paths.append(p)

    # Vérification du quorum
    if len(seed_summaries) < 1:
        print(f"Error: Need ≥1 seed with data; got {len(seed_summaries)}.")
        return
    if len(seed_summaries) < 2:
        print(f"Warning: Only 1 seed — robustness/stability will be 1.0 (trivial).")

    # 2. Définition du répertoire de sortie
    # Par défaut : un dossier 'cross_seed_report' au niveau parent des seeds
    out_dir = args.report_dir or (_common_seed_parent(kept_paths) / "cross_seed_report")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Agrégation statistique. V3.4 : on garde toutes les lignes
    # (min_robustness/stability=0) pour produire un gene_ranking
    # exhaustif. Le filtre args.min_robustness / args.min_stability
    # est ré-appliqué juste après pour les TSV "drivers" et les figures
    # — comportement identique à V3.3 sur ces sorties.
    print(f"[CROSS-SEED] Aggregating {len(seed_summaries)} seeds → {out_dir}")
    df_all = aggregate_cross_seed(
        seed_summaries=seed_summaries,
        min_robustness=0.0,
        min_stability=0.0,
    )
    if df_all.empty:
        print("No drivers passed the robustness / stability thresholds.")
        return
    # Strict view : same filter as V3.3 — used for cross_seed_drivers*.tsv,
    # the cross-seed figures, and the pathway ranking input.
    df = df_all[(df_all["robustness_score"] >= args.min_robustness) &
                (df_all["direction_stability"] >= args.min_stability)].copy()

    if df.empty:
        print("[INFO] No drivers passed the strict robustness / stability "
              "thresholds — drivers TSV / figures will be empty. The "
              "exhaustive gene_ranking is still produced from df_all.")

    # 4. Export des tables de résultats (TSV) — table strict (drivers only).
    df.to_csv(out_dir / "cross_seed_drivers.tsv", sep="\t", index=False)
    # Séparation gènes vs pathways pour faciliter la lecture bio
    df[~df["is_pathway"]].to_csv(out_dir / "cross_seed_drivers_genes.tsv", sep="\t", index=False)
    df[df["is_pathway"]].to_csv(out_dir / "cross_seed_drivers_pathways.tsv", sep="\t", index=False)
    print(f"Wrote cross_seed_drivers.tsv ({len(df)} drivers passing filters)")

    # 4b. Per-gene cross-mode ranking (aggregation over KO/KD/OE per gene).
    # Cross with VGAE baseline + DE info from gene_ranking_vgae.csv (per seed).
    seed_roots = [_normalize_seed_root(p) for p in kept_paths]

    # Persist VGAE training metrics (loss / AUC / AP histories) as a sidecar
    # so cross-seed figures (and --figures-only re-runs) can plot training
    # curves without revisiting each seed root. Silent skip if every seed
    # predates the vgae_metrics.json export (V3.6 and earlier).
    vgae_metrics = _load_vgae_training_metrics(seed_roots)
    if vgae_metrics:
        _save_vgae_training_summary(vgae_metrics, out_dir)
        print(f"Wrote vgae_training_summary.tsv "
              f"({len(vgae_metrics)} seeds with metrics)")
    else:
        print("[INFO] No vgae_metrics.json found in seed roots — training "
              "curve figures will be skipped (run gnn_vgae.py to (re)generate).")

    vgae_baseline = _load_vgae_baselines(seed_roots)
    if vgae_baseline.empty:
        print("[INFO] No gene_ranking_vgae.csv found in seed roots — "
              "vgae_importance / is_de_significant columns will be empty.")
    is_tf_series = _load_is_tf(seed_roots)
    if is_tf_series.empty:
        print("[INFO] Could not extract is_tf from any seed graph.")
    de_magnitude = _load_de_magnitude(args.de_magnitude_csv)
    if de_magnitude.empty:
        print(f"[INFO] DE magnitude file not loaded ({args.de_magnitude_csv}); "
              f"de_log2fc_p4_vs_p16 / de_neglog10_padj will be empty.")
    else:
        print(f"[INFO] Loaded DE magnitudes for {len(de_magnitude)} genes "
              f"from {args.de_magnitude_csv}.")
    # Pre-compute pathway ranking (we'll re-compute it below for the TSV;
    # here just to seed the gene-pathway cross-reference). The duplicated
    # call has cost ~2-3s and keeps the call sites independent.
    pw_rank_for_xref = build_pathway_ranking(
        df,
        min_robustness=args.gene_ranking_min_robustness,
        min_stability=args.gene_ranking_min_stability,
        mode_agg=args.driver_canon,
    )
    reactome = _load_reactome_pathways()
    gene_to_pathways = _build_gene_to_strong_pathways(
        df_all[~df_all["is_pathway"]], pw_rank_for_xref, reactome)
    # V3.4 : the per-gene ranking sees ALL rows (df_all), not just the
    # robust ones. Quality is encoded via [unreliable]/[hub-inflated]/
    # [incoherent] prefixes in `interpretation`.
    # Optional coexpr degree map for the VISIBLE diagnostic columns
    # (coexpr_degree, ds_totdeg). Built from a coexpr edge TSV (TF, target).
    coexpr_degree_map = None
    if getattr(args, "coexpr_degree_file", None) is not None:
        try:
            _cx = pd.read_csv(args.coexpr_degree_file, sep="\t",
                              usecols=["TF", "target"])
            _deg = pd.concat([_cx["TF"], _cx["target"]]).value_counts()
            coexpr_degree_map = _deg.to_dict()
            print(f"[INFO] coexpr_degree from {args.coexpr_degree_file.name} "
                  f"({len(coexpr_degree_map)} genes) → ds_totdeg enabled.")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] could not load --coexpr-degree-file "
                  f"({e}); ds_totdeg/coexpr_degree skipped.")

    gene_rank = build_gene_ranking(
        df_all,
        min_robustness=args.gene_ranking_min_robustness,
        min_stability=args.gene_ranking_min_stability,
        min_ppi_degree=args.gene_ranking_min_ppi_degree,
        vgae_baseline=vgae_baseline,
        de_top_n=args.gene_ranking_de_top_n,
        de_sig_mode=args.de_significance,
        de_padj_max=args.de_padj_max,
        de_abs_lfc_min=args.de_abs_lfc_min,
        is_tf_series=is_tf_series,
        gene_to_pathways=gene_to_pathways,
        de_magnitude=de_magnitude,
        mode_agg=args.driver_canon,
        coexpr_degree_map=coexpr_degree_map,
    )

    # --- Readout signé (gnn_futur §8.A / §9.2) : colonnes NON-rankantes
    # (tri driver_score inchangé). 3 rôles de cible SANS headline (role_pert
    # /role_de/role_latent ont chacun un angle mort, cf. §14bis.6quatervicies) ;
    # arbitrage = cross-dataset. role_de optionnel via --de-magnitude-csv.
    if not gene_rank.empty:
        # Rôle DE par gène (signe corrigé : +1 = logFC>0 = up-P16 = pro-sén).
        de_role_map = None
        if de_magnitude is not None and not de_magnitude.empty:
            _ds = int(getattr(args, "signed_de_sign", 1))
            de_role_map = {
                str(g): _ds * float(np.sign(v))
                for g, v in de_magnitude["de_log2fc_p4_vs_p16"].items()
                if np.isfinite(v) and v != 0}

        # Flag par gène marqueur/driver (FHL2-type) : DE-marqueur vs effet
        # causal (cosine_senescent) se contredisent. Indépendant du fan-out.
        if de_role_map is not None and "cosine_senescent" in gene_rank.columns:
            def _mdc(row):
                rp = row.get("cosine_senescent")
                rd = de_role_map.get(str(row["target"]))
                if rd is None or rp is None or not np.isfinite(rp) or rp == 0:
                    return False
                return bool(np.sign(rp) != np.sign(rd))
            gene_rank["marker_driver_conflict"] = gene_rank.apply(_mdc, axis=1)
            n_conf = int(gene_rank["marker_driver_conflict"].sum())
            print(f"[INFO] flag marker_driver_conflict : {n_conf} gènes "
                  f"(DE-marqueur ≠ effet causal, type FHL2).")

        # Readout signé fan-out (3 rôles) si tables fan-out présentes.
        fanout_paths = ([] if getattr(args, "no_signed_fanout", False)
                        else _resolve_signed_fanout_paths(args, kept_paths))
        if fanout_paths and "cosine_senescent" in gene_rank.columns:
            fanout_agg = _aggregate_signed_fanout(fanout_paths)
            if not fanout_agg.empty:
                role_pert_map = dict(zip(gene_rank["target"].astype(str),
                                         gene_rank["cosine_senescent"]))
                sig_cols = compute_signed_readout_columns(
                    fanout_agg, role_pert_map, de_role_map)
                if not sig_cols.empty:
                    # sig_cols est clé "gene" (gène source) ; gene_rank est clé
                    # "target" → merge croisé puis on retire la colonne "gene"
                    # dupliquée (fix bug 'gene'≠'target', branche fan-out oubliée
                    # du fix _mdc 2026-06-09).
                    gene_rank = gene_rank.merge(
                        sig_cols, left_on="target", right_on="gene", how="left"
                    ).drop(columns=["gene"])
                    print(f"[INFO] readout signé : {len(sig_cols)} sources avec "
                          f"fan-out signé ({len(fanout_paths)} fichier(s)) → "
                          f"signed_readout_{{pert,de,latent}} + cohérences.")

    if not gene_rank.empty:
        # V3.4 : exhaustive ranking — all genes present, quality
        # surfaced via [unreliable]/[hub-inflated]/[incoherent] prefixes
        # in the `interpretation` column. The user can post-filter on
        # `mean_robustness`, `is_hub_inflated`, `sign_consistent`,
        # `target_ppi_degree` if they prefer the V3.3 strict view.
        gene_rank.to_csv(out_dir / "cross_seed_gene_ranking.tsv",
                         sep="\t", index=False)
        n_unreliable = gene_rank["interpretation"].astype(str).str.contains(
            r"\[unreliable", regex=True).sum()
        n_hub = int(gene_rank["is_hub_inflated"].sum())
        n_inc = int((gene_rank["sign_consistent"] == False).sum())  # noqa: E712
        print(f"Wrote cross_seed_gene_ranking.tsv ({len(gene_rank)} genes, "
              f"exhaustive: {n_unreliable} [unreliable], {n_hub} [hub-inflated], "
              f"{n_inc} [incoherent] — flagged in `interpretation`)")
        # Companion export — incoherent-but-tagged subset for users who
        # want to inspect the non-canonical patterns. Contains TFDP1,
        # PARD3, etc., that previously fell into a separate file.
        is_inc = (gene_rank["sign_consistent"] == False)  # noqa: E712
        interp_str = gene_rank["interpretation"].astype(str)
        is_non_monotonic = (
            interp_str.str.contains("non-monotonic", regex=False) |
            interp_str.str.contains("borderline non-monotonic", regex=False) |
            interp_str.str.contains("weak non-monotonic", regex=False)
        )
        incoh = gene_rank[is_inc & ~is_non_monotonic].copy()
        if not incoh.empty:
            incoh.to_csv(out_dir / "cross_seed_gene_ranking_incoherent.tsv",
                         sep="\t", index=False)
            print(f"Wrote cross_seed_gene_ranking_incoherent.tsv "
                  f"({len(incoh)} genes with OE/loss same-direction; "
                  f"{int(is_non_monotonic.sum())} non-monotonic kept in main "
                  f"ranking with proper tag)")

        # V3.4 : low-PPI high-literature companion export.
        # Surfaces metallothionein-type markers (MT1E, etc.) that are
        # disconnected in STRING@900 but anchored in aging DBs / DE.
        is_low_ppi = gene_rank["target_ppi_degree"] < args.gene_ranking_min_ppi_degree
        has_lit = (
            (gene_rank["is_de_significant"] == True) |  # noqa: E712
            (gene_rank["n_aging_dbs"] >= 2)
        )
        low_ppi_lit = gene_rank[is_low_ppi & has_lit].copy()
        if not low_ppi_lit.empty:
            low_ppi_lit = low_ppi_lit.sort_values(
                ["validation_score", "max_abs_cosine"],
                ascending=[False, False]).reset_index(drop=True)
            low_ppi_lit.to_csv(
                out_dir / "cross_seed_gene_ranking_low_ppi_high_lit.tsv",
                sep="\t", index=False)
            print(f"Wrote cross_seed_gene_ranking_low_ppi_high_lit.tsv "
                  f"({len(low_ppi_lit)} genes with PPI deg < "
                  f"{args.gene_ranking_min_ppi_degree} but DE-sig OR ≥2 aging DBs)")

        # Discovery-focused subset : non-DE genes ranked by discovery_score.
        # Surfaces the graph-only findings (the GNN's value-add over DE).
        if "is_de_significant" in gene_rank.columns:
            non_de = gene_rank[gene_rank["is_de_significant"] == False].copy()  # noqa: E712
            if not non_de.empty:
                non_de = non_de.sort_values(
                    ["discovery_score", "driver_score"],
                    ascending=[False, False]).reset_index(drop=True)
                # Save top-300 by discovery_score (configurable cutoff).
                top_disc = non_de.head(300)
                top_disc.to_csv(out_dir / "cross_seed_gene_ranking_discoveries.tsv",
                                sep="\t", index=False)
                print(f"Wrote cross_seed_gene_ranking_discoveries.tsv "
                      f"(top-{len(top_disc)} non-DE genes by discovery_score)")

        # V3.4 : validation-focused subset — DE-sig OR aging-DB anchored,
        # ranked by validation_score (literature-corroborated targets).
        has_validation = (
            (gene_rank["is_de_significant"] == True) |  # noqa: E712
            (gene_rank["n_aging_dbs"] >= 1)
        )
        validated = gene_rank[has_validation].copy()
        if not validated.empty:
            validated = validated.sort_values(
                ["validation_score", "driver_score"],
                ascending=[False, False]).reset_index(drop=True)
            top_val = validated.head(300)
            top_val.to_csv(out_dir / "cross_seed_gene_ranking_validation.tsv",
                           sep="\t", index=False)
            print(f"Wrote cross_seed_gene_ranking_validation.tsv "
                  f"(top-{len(top_val)} literature-corroborated genes "
                  f"by validation_score)")

    # 4d. Per-pathway cross-mode ranking (similaire au gene_ranking, mais
    # sur les pathways = REACTOME slug, sans VGAE/DE/PPI gene-level).
    pw_rank = build_pathway_ranking(
        df,
        min_robustness=args.gene_ranking_min_robustness,
        min_stability=args.gene_ranking_min_stability,
        mode_agg=args.driver_canon,
    )
    if not pw_rank.empty:
        pw_rank.to_csv(out_dir / "cross_seed_pathway_ranking.tsv",
                       sep="\t", index=False)
        print(f"Wrote cross_seed_pathway_ranking.tsv ({len(pw_rank)} pathways "
              f"after default filters: robustness≥{args.gene_ranking_min_robustness}, "
              f"stability≥{args.gene_ranking_min_stability}, NOT hub-inflated)")
        # Separate incoherent pathway export (mirror gene logic).
        is_inc_pw = (pw_rank["sign_consistent"] == False)  # noqa: E712
        interp_str_pw = pw_rank["interpretation"].astype(str)
        is_kept_pw = interp_str_pw.str.startswith("non-monotonic")
        incoh_pw = pw_rank[is_inc_pw & ~is_kept_pw].copy()
        if not incoh_pw.empty:
            incoh_pw.to_csv(out_dir / "cross_seed_pathway_ranking_incoherent.tsv",
                            sep="\t", index=False)
            print(f"Wrote cross_seed_pathway_ranking_incoherent.tsv "
                  f"({len(incoh_pw)} pathways; "
                  f"{int(is_kept_pw.sum())} non-monotonic kept in main)")

    _render_cross_seed_figures(df, gene_rank, args, out_dir)

    print(f"\n[SUCCESS] Cross-seed report and figures written to {out_dir}")


def _render_cross_seed_figures(df: pd.DataFrame,
                               gene_rank: pd.DataFrame,
                               args,
                               out_dir: Path) -> None:
    """Generate every cross-seed figure given the aggregated DataFrames.

    Split out of run_cross_seed so ``--figures-only`` can call it after
    loading the existing TSVs without redoing the per-seed aggregation.
    """
    # Training-curves figure (loss / AUC / AP per seed). Reads the bundled
    # history JSON written during run_cross_seed; silent skip if absent
    # (older runs without vgae_metrics.json sidecars).
    metrics = _load_vgae_training_history(out_dir)
    if metrics:
        fig_cross_seed_training_curves(
            metrics,
            out_dir / "cross_seed_vgae_training.png",
            f"Cross-seed VGAE training — {len(metrics)} seed(s)",
        )

    # Filtres optionnels appliqués AVANT figures uniquement.
    df_fig = apply_figure_filters(df, args)
    if len(df_fig) < len(df):
        print(f"[FILTER] {len(df) - len(df_fig)} rows removed for figures "
              f"(min_abs_diff={args.min_abs_diff}, min_abs_cosine="
              f"{args.min_abs_cosine}, min_ppi_degree={args.min_ppi_degree}, "
              f"exclude_hubs={args.exclude_hubs}). Now plotting {len(df_fig)} rows.")

    # Per-mode figures sur df_fig (drivers TSV).
    for mode in MODES:
        d_mode = df_fig[df_fig["mode"] == mode]
        if d_mode.empty:
            continue

        d_genes = d_mode[~d_mode["is_pathway"]]
        d_pw = d_mode[d_mode["is_pathway"]]

        for suffix, sub in (("genes", d_genes), ("pw", d_pw)):
            if sub.empty:
                continue

            kind = "genes" if suffix == "genes" else "pathways"
            tag_mode = shorten_mode(mode)

            # Barplot des top drivers avec barres d'erreur (std entre seeds)
            fig_cross_seed_top_bars(
                sub, out_dir / f"cross_seed_top_{mode}_{suffix}.png",
                f"Cross-seed top drivers — {tag_mode} / {kind}",
                n_per_side=args.top_per_side
            )

            sub_fig = sub.copy()
            if "tag" not in sub_fig.columns:
                sub_fig["tag"] = sub_fig["mode"].astype(str) + "_" + sub_fig["target"].astype(str)

            fig_transitions_scatter_df(
                sub_fig, out_dir / f"cross_seed_transitions_{mode}_{suffix}.png",
                f"Cross-seed transitions — {tag_mode} / {kind}")
            fig_heatmap_projections_global_df(
                sub_fig, out_dir / f"cross_seed_heatmap_diff_{mode}_{suffix}.png",
                f"Cross-seed heatmap (diff) — {tag_mode} / {kind}",
                max_runs=args.top_per_side * 2)
            fig_heatmap_projections_cosine_df(
                sub_fig, out_dir / f"cross_seed_heatmap_cosine_{mode}_{suffix}.png",
                f"Cross-seed heatmap (cosine) — {tag_mode} / {kind}",
                n_per_side=args.top_per_side)
            fig_cross_seed_quadrant_diff_cosine(
                sub_fig, out_dir / f"cross_seed_quadrant_diff_cosine_{mode}_{suffix}.png",
                f"Quadrant diff × cosine — {tag_mode} / {kind}",
                n_annotate=args.top_per_side)

    # Gene-ranking bar figures (per direction × per score), built from
    # cross_seed_gene_ranking.tsv. Replaces the legacy display_top_genes.py
    # barplot but with quality flags + literature context inline.
    if gene_rank is not None and not gene_rank.empty:
        n_top = max(args.top_per_side * 3, 20)
        sort_specs = [("driver_score", "driver_score")]
        if "discovery_score" in gene_rank.columns:
            sort_specs.append(("discovery_score", "discovery_score"))
        for sort_col, fname in sort_specs:
            for direction, dir_label in (("pro", "pro_senescence"),
                                         ("anti", "anti_senescence")):
                fig_gene_ranking_top_bars(
                    gene_rank,
                    out_dir / f"gene_ranking_top_{dir_label}_by_{fname}.png",
                    title=(f"Top {n_top} {dir_label.replace('_', '-')} "
                           f"genes — sorted by {sort_col}"),
                    direction=direction,
                    n=n_top,
                    sort_by=sort_col,
                )


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


def fig_transitions_scatter_df(df: pd.DataFrame, out: Path, title: str,
                                n_annotate: int = 8):
    """DataFrame-backed version of fig_transitions_scatter — used by
    --cross-seed where per-run summary paths aren't available.

    Expects columns: tag, P4, P16_cluster_0, P16_cluster_1, P16_cluster_2,
    P16_cluster_3. Values are assumed to be on the global senescence axis
    (e.g. cross-seed averaged proj_signed_diff per group).
    """
    need = ["P4", "P16_cluster_0", "P16_cluster_1",
            "P16_cluster_2", "P16_cluster_3"]
    if df.empty or any(c not in df.columns for c in need):
        return
    df = df.dropna(subset=need).copy()
    if df.empty:
        return
    if "tag" not in df.columns:
        df["tag"] = df["mode"].astype(str) + "_" + df["target"].astype(str)
    df["quiescent_like"] = (df["P4"] + df["P16_cluster_0"]) / 2
    df["senescent"] = (df["P16_cluster_1"] + df["P16_cluster_2"]
                       + df["P16_cluster_3"]) / 3

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
        ax.scatter(x, y, c=div, cmap="viridis", s=12, alpha=0.6,
                   edgecolor="none")
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.05 * (hi - lo + 1e-9)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], "k-", alpha=0.3, lw=0.6, zorder=0)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.axhline(0, color="k", lw=0.3, alpha=0.3)
        ax.axvline(0, color="k", lw=0.3, alpha=0.3)
        ax.set_xlabel(a); ax.set_ylabel(b)
        ax.set_title(sub, fontsize=10)
        if n_annotate > 0:
            order = np.argsort(div)[-n_annotate:]
            for idx in order:
                ax.annotate(shorten_tag(df.iloc[idx]["tag"]),
                            (x[idx], y[idx]),
                            fontsize=6, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")
    fig.suptitle(f"{title}  (n={len(df)})", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap_projections_global_df(df: pd.DataFrame, out: Path, title: str,
                                       max_runs: int = 50):
    """DataFrame-backed variant of fig_heatmap_projections_global for
    --cross-seed mode. Same behaviour, same column requirements."""
    groups = ["P4", "P16_cluster_0", "P16_cluster_1",
              "P16_cluster_2", "P16_cluster_3"]
    if df.empty or any(c not in df.columns for c in groups):
        return
    sub = df.dropna(subset=groups).copy()
    if sub.empty:
        return
    if "tag" not in sub.columns:
        sub["tag"] = sub["mode"].astype(str) + "_" + sub["target"].astype(str)
    sub = sub.set_index(sub["tag"].map(shorten_tag))[groups]

    if len(sub) > max_runs:
        keep = sub.abs().max(axis=1).nlargest(max_runs).index
        sub = sub.loc[keep]
        title = f"{title}  (top {max_runs} of {len(df)} by |max proj|)"
    show_y = len(sub) <= 60
    height = max(5, min(25, 0.15 * len(sub)))
    g = sns.clustermap(sub, cmap="RdBu_r", center=0,
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
    ap.add_argument("--figures-only", action="store_true",
                    help="--cross-seed only: skip per-seed aggregation and "
                         "TSV writes; load existing cross_seed_drivers.tsv "
                         "and cross_seed_gene_ranking.tsv from --report-dir "
                         "(or the default cross_seed_report folder) and "
                         "regenerate figures only.")
    ap.add_argument("--driver-canon", choices=["aligned", "oe-only"],
                    default="aligned",
                    help="How driver_score aggregates perturbation modes "
                         "(gnn_futur §6.2). 'aligned' (default, V5.4.1) : "
                         "sign-align KO/KD to the OE gain-of-function "
                         "direction and average the coherent modes → "
                         "driver_score becomes KO+KD+OE. 'oe-only' (legacy "
                         "V3.4–V5.4) : anchor on OE alone (KO/KD feed only "
                         "coverage/coherence). Use 'oe-only' to reproduce "
                         "pre-V5.4.1 rankings.")
    ap.add_argument("--coexpr-degree-file", type=Path, default=None,
                    help="Optionnel (cross-seed) : TSV coexpr (colonnes TF, "
                         "target — ex. coexpr_diff*.tsv) pour les colonnes "
                         "diagnostiques VISIBLES `coexpr_degree` + `ds_totdeg` "
                         "dans cross_seed_gene_ranking.tsv (non-rankantes, "
                         "gnn_futur §7.11). Absent → ds_totdeg/coexpr_degree omis.")
    ap.add_argument("--signed-fanout", type=Path, nargs="*", default=None,
                    help="Optionnel (cross-seed) : tables long-format "
                         "`*_signed_fanout.tsv` (readout signé §8.A/§9.2). Si "
                         "omis, auto-découvertes à côté des TSV --all de chaque "
                         "seed. Produit les colonnes NON-rankantes "
                         "signed_readout/signed_coherence (role_pert headline) "
                         "+ variantes _de/_latent dans cross_seed_gene_ranking.tsv.")
    ap.add_argument("--signed-de-sign", type=int, default=1, choices=[-1, 1],
                    help="Convention du --de-magnitude-csv pour le rôle DE du "
                         "readout signé. +1 (DÉFAUT, vérifié : logFC>0 = UP en P16 "
                         "= pro-sén, CDKN2A +4.3) ; -1 si l'inverse.")
    ap.add_argument("--top-per-side", type=int, default=10,
                    help="For genome-wide / cross-seed bar figures, keep "
                         "the top-N most positive AND top-N most negative "
                         "targets (default 10 → 20 bars).")
    ap.add_argument("--genome-wide-threshold", type=int, default=60,
                    help="Auto-apply --top-per-side filtering to bar "
                         "figures when a mode has more runs than this "
                         "(default 60).")
    ap.add_argument("--axis-tag", default=None,
                    help="--cross-seed only : filtre les TSV consommés selon "
                         "l'axe de sénescence. Empty string '' = V3 (suffixe "
                         "absent dans le nom de TSV, défaut historique). "
                         "'axisV4' = lit uniquement perturbation_all_*_axisV4_*.tsv. "
                         "None (défaut) = aucun filtre, lit tout (legacy).")
    ap.add_argument("--min-robustness", type=float, default=0.5,
                    help="--cross-seed only: min fraction of seeds the "
                         "target must appear in (default 0.5).")
    ap.add_argument("--min-stability", type=float, default=0.7,
                    help="--cross-seed only: min fraction of seeds with a "
                         "consistent sign of max_proj_signed_diff "
                         "(default 0.7).")
    # --- Filtres optionnels appliqués AVANT génération des figures
    # cross-seed. Tous default 0/False → aucun filtre additionnel.
    ap.add_argument("--min-abs-diff", type=float, default=0.0,
                    help="Cross-seed: min |avg_proj_signed_diff| (default 0).")
    ap.add_argument("--min-abs-cosine", type=float, default=0.0,
                    help="Cross-seed: min |avg_proj_signed_cosine| (default 0).")
    ap.add_argument("--min-abs-extent", type=float, default=0.0,
                    help="Cross-seed: min |avg_proj_signed_extent| (default 0).")
    ap.add_argument("--min-abs-degree-metric", type=float, default=0.0,
                    help="Cross-seed: min |avg_proj_signed_degree| (default 0). "
                         "Distinct de --min-ppi-degree (filtre topologique).")
    ap.add_argument("--min-ppi-degree", type=int, default=0,
                    help="Cross-seed: min target_ppi_degree to keep a target "
                         "(default 0; recommended 5 for interpretation).")
    ap.add_argument("--exclude-hubs", action="store_true",
                    help="Cross-seed: drop is_hub_inflated rows before figures.")
    # --- Per-gene ranking (cross-mode aggregation, written as TSV)
    ap.add_argument("--gene-ranking-min-robustness", type=float, default=0.7,
                    help="Per-gene ranking TSV: default 0.7.")
    ap.add_argument("--gene-ranking-min-stability", type=float, default=0.7,
                    help="Per-gene ranking TSV: default 0.7.")
    ap.add_argument("--gene-ranking-min-ppi-degree", type=int, default=5,
                    help="Per-gene ranking TSV: filter target_ppi_degree<X "
                         "for the interpretation column (default 5).")
    ap.add_argument("--gene-ranking-de-top-n", type=int, default=1000,
                    help="Per-gene ranking TSV: a gene is tagged "
                         "is_de_significant=True if its rank_stat (from "
                         "gene_ranking_vgae.csv) is ≤ this value (default 1000). "
                         "Only used in --de-significance magnitude-rank mode.")
    ap.add_argument("--de-significance", choices=["magnitude-rank", "pvalue"],
                    default="pvalue",
                    help="Définition de is_de_significant. **'pvalue' (DÉFAUT depuis "
                         "2026-06-15, plus reproductible/défendable)** = MAST padj < "
                         "--de-padj-max ET |avg_log2FC| ≥ --de-abs-lfc-min (gène absent "
                         "du MAST → None). 'magnitude-rank' (legacy) = rang |ΔExpr "
                         "P16-P4| ≤ --gene-ranking-de-top-n.")
    ap.add_argument("--de-padj-max", type=float, default=0.05,
                    help="Mode pvalue : seuil padj MAST (défaut 0.05).")
    ap.add_argument("--de-abs-lfc-min", type=float, default=0.5,
                    help="Mode pvalue : seuil |avg_log2FC| MAST (défaut 0.5). "
                         "⚠️ padj seul ne discrimine pas en scRNA → |logFC| requis.")
    ap.add_argument("--no-signed-fanout", action="store_true", default=False,
                    help="Désactive l'auto-découverte des tables *_signed_fanout.tsv "
                         "(colonnes signed_readout_*). Recommandé pour le pipeline "
                         "standard afin d'éviter d'agréger des fan-out d'autres axes/"
                         "expériences (ex. concordDE/v6smoke) traînant dans un run-dir.")
    ap.add_argument("--de-magnitude-csv", type=Path,
                    default=Path("data/gnn_data/DEGs_P4_vs_P16_MAST.csv"),
                    help="MAST DE table (gene, avg_log2FC, p_val_adj). Used to "
                         "populate de_log2fc_p4_vs_p16 and de_neglog10_padj in "
                         "cross_seed_gene_ranking.tsv. Empty/missing → columns "
                         "left empty (no error).")
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
            # ---- Variantes normalisées (comparaison des corrections hub) ----
            "max_proj_signed_norm_group": s.get("max_proj_signed_norm_group", ""),
            "max_proj_signed_norm": s.get("max_proj_signed_norm", 0.0),
            "max_proj_signed_amplitude_group": s.get("max_proj_signed_amplitude_group", ""),
            "max_proj_signed_amplitude": s.get("max_proj_signed_amplitude", 0.0),
            "max_proj_signed_extent_group": s.get("max_proj_signed_extent_group", ""),
            "max_proj_signed_extent": s.get("max_proj_signed_extent", 0.0),
            "max_proj_signed_degree_group": s.get("max_proj_signed_degree_group", ""),
            "max_proj_signed_degree": s.get("max_proj_signed_degree", 0.0),
            "max_proj_signed_cosine_group": s.get("max_proj_signed_cosine_group", ""),
            "max_proj_signed_cosine": s.get("max_proj_signed_cosine", 0.0),
            "target_ppi_degree": s.get("target_ppi_degree", 0),
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
