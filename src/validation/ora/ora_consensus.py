#!/usr/bin/env python3
"""
Hypergeometric ORA on one or more gene lists, with optional consensus-building
from VGAE run outputs.

Supports two gene-set "databases":
  * ``--db reactome`` (default): REACTOME pathways from MSigDB
    (``data/databases/c2.cp.reactome.symbols.gmt``). One hypergeometric test
    per pathway, BH-FDR on all tested pathways.
  * ``--db aging``: four aging gene catalogues -- SenMayo (local aging list),
    CellAge, Fridman (MSigDB hallmarks filtered on aging keywords), GenAge.
    One hypergeometric test per catalogue, BH-FDR across the four.

Input gene lists come either from ``--genes FILE...`` (plain text, one symbol
per line) or are built on the fly from VGAE runs via ``--consensus-runs``
(consensus = genes appearing in the top-N of at least ``--consensus-min-shared``
runs, ranked by average ``rank_vgae``).

Rationale
---------
- The Enrichr REST API was unreliable in our environment (response parsing
  errors), and gseapy was blocked by PEP 668 on the cluster. A self-contained,
  dependency-light ORA avoids both.
- BH-FDR is reimplemented inline (no statsmodels) -- ~6 lines.
- Hypergeometric survival function is computed in log space for numerical
  stability on large populations (~20k genes).

Output TSV columns (matches format of pre-existing ora_v3 files):
    pathway, k/K, pw_size, p, p_adj, genes

Usage
-----
    # REACTOME ORA on an explicit gene list
    python src/validation/ora_consensus.py \\
        --genes docs/novel_genes_consensus_3of3.txt \\
        --out output/ora_v3/consensus_3of3_reactome.tsv

    # Aging-database ORA with consensus built from three runs
    python src/validation/ora_consensus.py \\
        --db aging \\
        --consensus-runs output/gnn_vgae/V3.3/run1 \\
                         output/gnn_vgae/V3.3/run2 \\
                         output/gnn_vgae/V3.3/run3 \\
        --consensus-top-n 95 --consensus-min-shared 2 \\
        --consensus-save data/pathway_gene_list/consensus_top95_2of3.txt \\
        --out output/ora_v3/consensus_top95_2of3_aging.tsv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import exp, lgamma
from pathlib import Path

import numpy as np
import pandas as pd

def _find_project_root(start: Path, fallback_levels: int = 2) -> Path:
    """Remonte les ancêtres jusqu'à trouver `data/databases/`.

    Robuste aux deux layouts du projet :
        local   : src/validation/ora_consensus.py → parents[2] = projet
        cluster : src/ora_consensus.py            → parents[1] = projet
    Un override env GNN_PROJECT_ROOT court-circuite la détection (utile en
    cas de structure exotique). Fallback : parents[fallback_levels] si
    rien n'est trouvé sur la remontée.
    """
    import os
    env = os.environ.get("GNN_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    start_resolved = start.resolve()
    for p in [start_resolved] + list(start_resolved.parents):
        if (p / "data" / "databases").is_dir():
            return p
    return start_resolved.parents[fallback_levels]


ROOT = _find_project_root(Path(__file__))
# Override possible : GNN_PROJECT_ROOT=/LAB-DATA/GLiCID/users/.../gnn/ python ...
GMT_PATH = ROOT / "data/databases/c2.cp.reactome.symbols.gmt"
DE_PATH = ROOT / "data/RNAseq/GSE98440_diff_expr_analysis_afterNorm_HUVEC_2reps.txt"
DATA_DIR = ROOT / "data"
DB_DIR = DATA_DIR / "databases"

# Aging catalogues -- file paths and the column holding the gene symbol.
AGING_DB_FILES = {
    "GenAge": (DB_DIR / "genage_human.csv", "symbol", ","),
    "CellAge": (DB_DIR / "cellage3.tsv", "Gene symbol", "\t"),
    "SenMayo": (DATA_DIR / "human_age_related_gene.csv", "Symbol", ","),
}
FRIDMAN_GMT = DB_DIR / "h.all.symbols.gmt"
FRIDMAN_KEYWORDS = (
    "SENESCENCE", "P53", "APOPTOSIS", "INFLAMMATORY", "TNFA", "IL6",
    "KRAS", "MTORC", "REACTIVE_OXYGEN", "DNA_REPAIR", "OXIDATIVE", "AGING",
)


# --------------------------------------------------------------------------- #
# Hypergeometric ORA (self-contained).
# --------------------------------------------------------------------------- #
def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def hypergeom_sf(k: int, M: int, n: int, N: int) -> float:
    """P(X >= k) for X ~ Hypergeom(M, n, N). M=population, n=successes, N=draws."""
    if k <= 0:
        return 1.0
    k_max = min(n, N)
    if k > k_max:
        return 0.0
    log_denom = _log_comb(M, N)
    log_terms = [
        _log_comb(n, i) + _log_comb(M - n, N - i) - log_denom
        for i in range(k, k_max + 1)
    ]
    m = max(log_terms)
    return exp(m) * sum(exp(t - m) for t in log_terms)


def bh_fdr(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    prev = 1.0
    for rank_idx, orig_idx in enumerate(reversed(order)):
        r = n - rank_idx
        val = min(prev, pvals[orig_idx] * n / r)
        adj[orig_idx] = val
        prev = val
    return adj


# --------------------------------------------------------------------------- #
# Loading helpers.
# --------------------------------------------------------------------------- #
def load_gene_list(path: Path) -> set[str]:
    genes: set[str] = set()
    for line in path.read_text().splitlines():
        g = line.strip()
        if g and not g.startswith("#"):
            genes.add(g)
    return genes


def load_background(path: Path | None = None) -> set[str]:
    """Background universe of gene symbols.

    Default: the DE table at ``DE_PATH``. ``path`` may point to a ``gene_ranking_vgae.csv``
    (uses the ``gene`` column) or a plain text file with one symbol per line.
    """
    if path is None:
        df = pd.read_csv(DE_PATH, sep="\t")
        syms = df["hgnc_symbol"].dropna().astype(str)
        return {s for s in syms if s and s != "NA"}
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        col = "gene" if "gene" in df.columns else df.columns[0]
        return {str(s) for s in df[col].dropna() if str(s) and str(s) != "NA"}
    return load_gene_list(path)


def load_reactome_gmt() -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    with open(GMT_PATH) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name, _url, *genes = parts
            sets[name] = {g.strip() for g in genes if g.strip()}
    return sets


def load_aging_databases() -> dict[str, set[str]]:
    """Four aging-gene catalogues as ``{name: set_of_symbols}``.

    Shape matches :func:`load_reactome_gmt` so the dict can be fed directly to
    :func:`run_ora`.
    """
    sets: dict[str, set[str]] = {}
    for name, (path, col, sep) in AGING_DB_FILES.items():
        df = pd.read_csv(path, sep=sep)
        sets[name] = {s.strip() for s in df[col].dropna().astype(str) if s.strip()}

    # Fridman = union of MSigDB hallmarks whose name matches an aging keyword.
    fridman: set[str] = set()
    if FRIDMAN_GMT.exists():
        with open(FRIDMAN_GMT) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                name_up = parts[0].upper()
                if any(kw in name_up for kw in FRIDMAN_KEYWORDS):
                    fridman |= {g.strip() for g in parts[2:] if g.strip()}
    else:
        print(f"Warning: {FRIDMAN_GMT} not found -- Fridman set is empty")
    sets["Fridman"] = fridman
    return sets


def build_consensus_genes(ranking_paths: list[Path],
                          top_n: int = 95,
                          min_shared: int = 2) -> list[str]:
    """Consensus top genes across several VGAE runs.

    A gene is kept if it appears in the top ``top_n`` of at least ``min_shared``
    rankings. Kept genes are then ordered by their mean ``rank_vgae`` across
    runs (lower is better) and truncated to ``top_n`` entries.
    """
    per_run_top: list[set[str]] = []
    per_run_ranks: list[dict[str, float]] = []
    for p in ranking_paths:
        df = pd.read_csv(p).sort_values("rank_vgae")
        per_run_top.append(set(df.head(top_n)["gene"].tolist()))
        per_run_ranks.append(dict(zip(df["gene"].astype(str), df["rank_vgae"])))

    candidates = set().union(*per_run_top)
    kept = [g for g in candidates
            if sum(g in s for s in per_run_top) >= min_shared]

    mean_rank = {
        g: float(np.mean([r[g] for r in per_run_ranks if g in r]))
        for g in kept
    }
    ordered = sorted(mean_rank.items(), key=lambda kv: kv[1])
    return [g for g, _ in ordered[:top_n]]


# --------------------------------------------------------------------------- #
# ORA core.
# --------------------------------------------------------------------------- #
@dataclass
class OraRow:
    pathway: str
    k: int
    pw_size: int
    k_over_K: str
    p: float
    p_adj: float
    genes: str


def run_ora(gene_list: set[str],
            background: set[str],
            reactome: dict[str, set[str]],
            min_overlap: int = 3,
            min_pw_size: int = 5,
            max_pw_size: int = 500) -> list[OraRow]:
    gl = gene_list & background
    M = len(background)
    N = len(gl)
    rows: list[OraRow] = []
    for name, members in reactome.items():
        members_bg = members & background
        n = len(members_bg)
        if n < min_pw_size or n > max_pw_size:
            continue
        overlap = gl & members_bg
        k = len(overlap)
        if k < min_overlap:
            continue
        rows.append(OraRow(
            pathway=name, k=k, pw_size=n,
            k_over_K=f"{k}/{N}",
            p=hypergeom_sf(k, M, n, N),
            p_adj=0.0,
            genes=",".join(sorted(overlap)),
        ))
    if rows:
        adj = bh_fdr([r.p for r in rows])
        for r, a in zip(rows, adj):
            r.p_adj = a
        rows.sort(key=lambda r: r.p_adj)
    return rows


def write_tsv(rows: list[OraRow], path: Path) -> None:
    df = pd.DataFrame([
        {"pathway": r.pathway, "k/K": r.k_over_K, "pw_size": r.pw_size,
         "p": r.p, "p_adj": r.p_adj, "genes": r.genes}
        for r in rows
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def _resolve_ranking_path(p: Path) -> Path:
    """Accept either a run directory or the ranking CSV directly."""
    p = Path(p)
    if p.is_dir():
        return p / "gene_ranking_vgae.csv"
    return p


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--genes", nargs="+",
                     help="One or more plain-text gene list files.")
    src.add_argument("--consensus-runs", nargs="+",
                     help="Run directories (or gene_ranking_vgae.csv paths) to "
                          "build a consensus gene list from.")

    ap.add_argument("--db", choices=("reactome", "aging"), default="reactome",
                    help="Gene-set collection to test against (default: reactome).")
    ap.add_argument("--background", type=str, default=None,
                    help="Optional background file (gene_ranking_vgae.csv or plain "
                         "text). Default: DE table; if --consensus-runs is used and "
                         "no --background is given, the first run's ranking is used.")

    ap.add_argument("--out", type=str, default=None,
                    help="Output TSV path (required for single input).")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Output directory (used when multiple --genes files are given).")
    ap.add_argument("--fdr", type=float, default=0.05,
                    help="Significance threshold on BH-adjusted p (default: 0.05).")

    ap.add_argument("--consensus-top-n", type=int, default=95)
    ap.add_argument("--consensus-min-shared", type=int, default=2)
    ap.add_argument("--consensus-save", type=str, default=None,
                    help="Optional path to save the built consensus gene list.")

    ap.add_argument("--min-overlap", type=int, default=None)
    ap.add_argument("--min-pw-size", type=int, default=None)
    ap.add_argument("--max-pw-size", type=int, default=None)
    args = ap.parse_args()

    # Looser defaults for aging (4 small named sets; we want all of them reported).
    if args.db == "aging":
        min_overlap = args.min_overlap if args.min_overlap is not None else 0
        min_pw_size = args.min_pw_size if args.min_pw_size is not None else 1
        max_pw_size = args.max_pw_size if args.max_pw_size is not None else 10**9
    else:
        min_overlap = args.min_overlap if args.min_overlap is not None else 3
        min_pw_size = args.min_pw_size if args.min_pw_size is not None else 5
        max_pw_size = args.max_pw_size if args.max_pw_size is not None else 500

    # Resolve the gene set dictionary.
    print(f"Loading {args.db} gene sets and background ...")
    if args.db == "reactome":
        gene_sets = load_reactome_gmt()
    else:
        gene_sets = load_aging_databases()

    # Build gene lists to test.
    if args.consensus_runs:
        ranking_paths = [_resolve_ranking_path(Path(p)) for p in args.consensus_runs]
        missing = [str(p) for p in ranking_paths if not p.exists()]
        if missing:
            ap.error(f"ranking files not found: {missing}")
        # Clip min_shared to n runs to avoid "≥2 in 1 run" empty-consensus bug.
        effective_min_shared = min(args.consensus_min_shared, len(ranking_paths))
        if effective_min_shared < args.consensus_min_shared:
            print(f"[INFO] --consensus-min-shared={args.consensus_min_shared} "
                  f"clipped to {effective_min_shared} (only {len(ranking_paths)} "
                  f"run(s) provided).")
        consensus = build_consensus_genes(
            ranking_paths,
            top_n=args.consensus_top_n,
            min_shared=effective_min_shared,
        )
        print(f"  consensus: {len(consensus)} genes "
              f"(top-{args.consensus_top_n} in >= {effective_min_shared}/"
              f"{len(ranking_paths)} runs)")
        if args.consensus_save:
            save_path = Path(args.consensus_save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text("\n".join(consensus) + "\n")
            print(f"  consensus list -> {save_path}")

        # Default background = first run's ranking (preserves old behavior).
        bg_path = Path(args.background) if args.background else ranking_paths[0]
        background = load_background(bg_path)
        jobs = [("consensus", set(consensus))]
    else:
        background = load_background(Path(args.background) if args.background else None)
        gene_paths = [Path(p) for p in args.genes]
        if len(gene_paths) == 1 and args.out is None:
            ap.error("--out must be provided when exactly one --genes file is given.")
        if len(gene_paths) > 1 and args.out_dir is None:
            ap.error("--out-dir must be provided when multiple --genes files are given.")
        jobs = [(gp.stem, load_gene_list(gp)) for gp in gene_paths]

    print(f"  gene sets: {len(gene_sets)} | background: {len(background)} genes.")

    # Run ORA on each job.
    all_set_genes: set[str] = set()
    for members in gene_sets.values():
        all_set_genes |= (members & background)

    for job_name, genes in jobs:
        rows = run_ora(
            genes, background, gene_sets,
            min_overlap=min_overlap,
            min_pw_size=min_pw_size,
            max_pw_size=max_pw_size,
        )
        sig = sum(1 for r in rows if r.p_adj < args.fdr)

        if args.out and len(jobs) == 1:
            out_path = Path(args.out)
        elif args.out_dir is not None:
            out_path = Path(args.out_dir) / f"{job_name}_{args.db}.tsv"
        elif args.consensus_runs:
            # Default for consensus mode: write next to the first run.
            default_dir = ranking_paths[0].parent / "ora_consensus"
            default_dir.mkdir(parents=True, exist_ok=True)
            out_path = default_dir / f"{job_name}_{args.db}.tsv"
        else:
            ap.error("Specify --out (single input) or --out-dir (multiple).")
        write_tsv(rows, out_path)

        in_bg = genes & background
        novel = len(in_bg - all_set_genes)
        novel_pct = (100.0 * novel / len(in_bg)) if in_bg else 0.0
        print(f"  {job_name}: {len(genes)} input genes, "
              f"{len(rows)} tested sets, "
              f"{sig} significant at FDR<{args.fdr}, "
              f"novel (not in any {args.db} set): {novel}/{len(in_bg)} "
              f"({novel_pct:.1f}%)  ->  {out_path}")


if __name__ == "__main__":
    main()
