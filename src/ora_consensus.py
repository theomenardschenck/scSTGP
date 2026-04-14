#!/usr/bin/env python3
"""
Hypergeometric ORA (REACTOME MSigDB) on one or more gene lists.

This is the script used to produce the V3 consensus ORA outputs
(consensus_3of3_reactome.tsv, consensus_2of3_reactome.tsv) in
output/ora_v3/. It is generic: give it one or more plain-text files
(one gene symbol per line) and it runs a local hypergeometric test
against the REACTOME gene sets in data/databases/c2.cp.reactome.symbols.gmt.

Rationale
---------
- The Enrichr REST API was unreliable in our environment
  (response parsing errors), and gseapy was blocked by PEP 668 on
  the cluster. A self-contained, dependency-light ORA avoids both.
- BH-FDR is reimplemented inline (no statsmodels) — ~6 lines.
- Hypergeometric survival function is computed in log space for numerical
  stability on large populations (~20k genes).

Output TSV columns (matches format of pre-existing ora_v3 files):
    pathway, k/K, pw_size, p, p_adj, genes

Usage
-----
    # Single list (classic V3 novel-genes ORA)
    python src/ora_consensus.py \\
        --genes docs/novel_genes_consensus_3of3.txt \\
        --out output/ora_v3/consensus_3of3_reactome.tsv

    # Several lists at once
    python src/ora_consensus.py \\
        --genes docs/novel_genes_consensus_2of3.txt \\
                docs/novel_genes_consensus_3of3.txt \\
        --out-dir output/ora_v3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import exp, lgamma
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GMT_PATH = ROOT / "data/databases/c2.cp.reactome.symbols.gmt"
DE_PATH = ROOT / "data/RNAseq/GSE98440_diff_expr_analysis_afterNorm_HUVEC_2reps.txt"


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


def load_background() -> set[str]:
    """Background = universe of gene symbols in the DE table."""
    df = pd.read_csv(DE_PATH, sep="\t")
    syms = df["hgnc_symbol"].dropna().astype(str)
    return {s for s in syms if s and s != "NA"}


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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", nargs="+", required=True,
                    help="One or more plain-text gene list files.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output TSV path (required when exactly one --genes file is given).")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Output directory (used when multiple --genes files are given).")
    ap.add_argument("--fdr", type=float, default=0.05,
                    help="Significance threshold on BH-adjusted p (default: 0.05).")
    args = ap.parse_args()

    gene_paths = [Path(p) for p in args.genes]
    if len(gene_paths) == 1 and args.out is None:
        ap.error("--out must be provided when exactly one --genes file is given.")
    if len(gene_paths) > 1 and args.out_dir is None:
        ap.error("--out-dir must be provided when multiple --genes files are given.")

    print("Loading REACTOME GMT and background ...")
    reactome = load_reactome_gmt()
    background = load_background()
    print(f"  pathways: {len(reactome)} | background: {len(background)} genes.")

    for gp in gene_paths:
        genes = load_gene_list(gp)
        rows = run_ora(genes, background, reactome)
        sig = sum(1 for r in rows if r.p_adj < args.fdr)

        if args.out and len(gene_paths) == 1:
            out_path = Path(args.out)
        else:
            out_path = Path(args.out_dir) / f"{gp.stem}_reactome.tsv"

        write_tsv(rows, out_path)
        print(f"  {gp.name}: {len(genes)} input genes, "
              f"{len(rows)} tested pathways, "
              f"{sig} significant at FDR<{args.fdr}  ->  {out_path}")


if __name__ == "__main__":
    main()
