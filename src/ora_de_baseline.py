#!/usr/bin/env python3
"""
Ablation: differential expression (DE) baseline vs GNN VGAE consensus.

Goal
----
Answer the question: does the VGAE bring structure that a plain |t-stat|
ranking + REACTOME ORA would also uncover? If the DE baseline already lands
on peroxisomes / GPI-anchor / ESCRT-III / RNA-Pol-III, the GNN is redundant.
If it does not, the GNN reveals pathway structure invisible to gene-by-gene
DE analysis.

Pipeline
--------
  1. Load DE results (GSE98440 limma-style output).
  2. Rank genes by |stat|, filter on padj < PADJ_THRESH, keep top K.
  3. Load V3_Run{1,2,3} VGAE rankings, take top K of each,
     intersect (2/3) to build the GNN consensus list.
  4. Run hypergeometric REACTOME ORA on both lists against the same
     background (union of gene symbols in the DE table).
  5. Compute gene overlap (Jaccard, exact count) and pathway overlap.
  6. Write side-by-side TSV outputs and a short verdict file.

Usage
-----
    python src/ora_de_baseline.py --top-k 95
    python src/ora_de_baseline.py --top-k 500 --padj 0.05

Outputs (in gnn_huvec/output/ora_ablation/):
    de_top{K}_genes.txt
    gnn_consensus2of3_top{K}_genes.txt
    de_top{K}_reactome.tsv
    gnn_consensus2of3_top{K}_reactome.tsv
    comparison_summary.tsv
    verdict.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import lgamma, log
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DE_PATH = ROOT / "data/RNAseq/GSE98440_diff_expr_analysis_afterNorm_HUVEC_2reps.txt"
GMT_PATH = ROOT / "data/databases/c2.cp.reactome.symbols.gmt"
VGAE_DIR = ROOT / "output/gnn_vgae"
V3_RUNS = ["V3_Run1", "V3_Run2", "V3_Run3"]
OUT_DIR = ROOT / "output/ora_ablation"


# --------------------------------------------------------------------------- #
# Hypergeometric ORA (self-contained — no scipy/statsmodels dependency).
# --------------------------------------------------------------------------- #
def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def hypergeom_sf(k: int, M: int, n: int, N: int) -> float:
    """P(X >= k) for X ~ Hypergeom(M, n, N). M=pop, n=successes, N=draws."""
    if k <= 0:
        return 1.0
    k_max = min(n, N)
    if k > k_max:
        return 0.0
    log_denom = _log_comb(M, N)
    log_terms = []
    for i in range(k, k_max + 1):
        lt = _log_comb(n, i) + _log_comb(M - n, N - i) - log_denom
        log_terms.append(lt)
    m = max(log_terms)
    s = sum(pow(2.718281828459045, t - m) for t in log_terms)
    from math import exp
    return exp(m) * s


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
def load_de(padj_thresh: float) -> pd.DataFrame:
    df = pd.read_csv(DE_PATH, sep="\t")
    df = df[df["hgnc_symbol"].notna() & (df["hgnc_symbol"] != "NA")]
    df = df[df["stat"].notna() & df["padj"].notna()]
    df["abs_stat"] = df["stat"].abs()
    df = df[df["padj"] < padj_thresh]
    df = df.sort_values("abs_stat", ascending=False)
    df = df.drop_duplicates(subset="hgnc_symbol", keep="first")
    return df.reset_index(drop=True)


def load_de_background() -> set[str]:
    df = pd.read_csv(DE_PATH, sep="\t")
    syms = df["hgnc_symbol"].dropna().astype(str)
    return {s for s in syms if s and s != "NA"}


def load_gnn_consensus(top_k: int, min_votes: int = 2) -> tuple[set[str], dict]:
    """Return the set of genes that rank in top-K in >= min_votes of 3 V3 runs."""
    votes: dict[str, int] = {}
    per_run_top: dict[str, set[str]] = {}
    for run in V3_RUNS:
        path = VGAE_DIR / run / "gene_ranking_vgae.csv"
        df = pd.read_csv(path)
        df = df.sort_values("rank_vgae", ascending=True).head(top_k)
        genes = set(df["gene"].astype(str))
        per_run_top[run] = genes
        for g in genes:
            votes[g] = votes.get(g, 0) + 1
    consensus = {g for g, v in votes.items() if v >= min_votes}
    info = {"per_run_sizes": {r: len(s) for r, s in per_run_top.items()}}
    return consensus, info


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
# ORA.
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


def run_ora(gene_list: set[str], background: set[str],
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
        p = hypergeom_sf(k, M, n, N)
        rows.append(OraRow(
            pathway=name, k=k, pw_size=n,
            k_over_K=f"{k}/{N}",
            p=p, p_adj=0.0,
            genes=",".join(sorted(overlap)),
        ))
    if not rows:
        return rows
    adj = bh_fdr([r.p for r in rows])
    for r, a in zip(rows, adj):
        r.p_adj = a
    rows.sort(key=lambda r: r.p_adj)
    return rows


def write_ora_tsv(rows: list[OraRow], path: Path, fdr_thresh: float = 0.05) -> int:
    df = pd.DataFrame([
        {"pathway": r.pathway, "k/K": r.k_over_K, "pw_size": r.pw_size,
         "p": r.p, "p_adj": r.p_adj, "genes": r.genes}
        for r in rows
    ])
    df.to_csv(path, sep="\t", index=False)
    return int((df["p_adj"] < fdr_thresh).sum()) if not df.empty else 0


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=95,
                    help="Number of top genes taken per method (default: 95).")
    ap.add_argument("--padj", type=float, default=0.05,
                    help="DE padj threshold before |stat| ranking (default: 0.05).")
    ap.add_argument("--fdr", type=float, default=0.05,
                    help="BH-FDR threshold for pathway significance (default: 0.05).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    K = args.top_k

    print(f"[1/5] Loading DE (padj<{args.padj}) ...")
    de = load_de(args.padj)
    de_top = set(de.head(K)["hgnc_symbol"].astype(str))
    print(f"      DE significant: {len(de)} genes, keeping top {len(de_top)}.")

    print(f"[2/5] Loading GNN V3 consensus top-{K} (>=2/3 runs) ...")
    gnn_top, info = load_gnn_consensus(top_k=K, min_votes=2)
    print(f"      per-run top-{K}: {info['per_run_sizes']} "
          f"-> consensus 2/3 = {len(gnn_top)} genes.")

    print(f"[3/5] Loading REACTOME GMT ...")
    reactome = load_reactome_gmt()
    background = load_de_background()
    print(f"      pathways: {len(reactome)} | background: {len(background)} genes.")

    (OUT_DIR / f"de_top{K}_genes.txt").write_text(
        "\n".join(sorted(de_top)) + "\n")
    (OUT_DIR / f"gnn_consensus2of3_top{K}_genes.txt").write_text(
        "\n".join(sorted(gnn_top)) + "\n")

    print(f"[4/5] Running hypergeometric ORA on both lists ...")
    rows_de = run_ora(de_top, background, reactome)
    rows_gnn = run_ora(gnn_top, background, reactome)
    sig_de = write_ora_tsv(rows_de, OUT_DIR / f"de_top{K}_reactome.tsv",
                           fdr_thresh=args.fdr)
    sig_gnn = write_ora_tsv(rows_gnn, OUT_DIR / f"gnn_consensus2of3_top{K}_reactome.tsv",
                            fdr_thresh=args.fdr)
    print(f"      significant pathways (padj<{args.fdr}): DE={sig_de}, GNN={sig_gnn}.")

    print(f"[5/5] Comparing gene and pathway overlap ...")
    gene_inter = de_top & gnn_top
    gene_union = de_top | gnn_top
    jacc = len(gene_inter) / len(gene_union) if gene_union else 0.0

    top_de_pw = {r.pathway for r in rows_de if r.p_adj < args.fdr}
    top_gnn_pw = {r.pathway for r in rows_gnn if r.p_adj < args.fdr}
    pw_inter = top_de_pw & top_gnn_pw
    pw_union = top_de_pw | top_gnn_pw
    pw_jacc = len(pw_inter) / len(pw_union) if pw_union else 0.0

    summary = pd.DataFrame([
        {"metric": "top_k",                           "value": K},
        {"metric": "DE_genes",                        "value": len(de_top)},
        {"metric": "GNN_consensus2of3_genes",         "value": len(gnn_top)},
        {"metric": "gene_overlap",                    "value": len(gene_inter)},
        {"metric": "gene_jaccard",                    "value": round(jacc, 4)},
        {"metric": "DE_sig_pathways",                 "value": len(top_de_pw)},
        {"metric": "GNN_sig_pathways",                "value": len(top_gnn_pw)},
        {"metric": "pathway_overlap",                 "value": len(pw_inter)},
        {"metric": "pathway_jaccard",                 "value": round(pw_jacc, 4)},
    ])
    summary.to_csv(OUT_DIR / "comparison_summary.tsv", sep="\t", index=False)

    keywords = ["PEROXISOM", "GPI_ANCHOR", "ESCRT", "RNA_POLYMERASE_III",
                "GLYCOSYLPHOSPHATIDYLINOSITOL"]
    def hits(rows, kw):
        return [r.pathway for r in rows
                if r.p_adj < args.fdr and kw in r.pathway.upper()]

    verdict_lines = [
        "=== DE baseline vs GNN consensus 2/3 — ablation verdict ===",
        f"top_k={K}, padj<{args.padj}, FDR<{args.fdr}",
        "",
        f"Gene overlap: {len(gene_inter)} / {len(gene_union)} "
        f"(Jaccard={jacc:.3f})",
        f"Pathway overlap: {len(pw_inter)} / {len(pw_union)} "
        f"(Jaccard={pw_jacc:.3f})",
        "",
        "Key pathways — hits per method:",
    ]
    for kw in keywords:
        verdict_lines.append(
            f"  {kw:40s}  DE={len(hits(rows_de, kw)):2d}  "
            f"GNN={len(hits(rows_gnn, kw)):2d}"
        )
    verdict_lines.append("")
    pw_only_gnn = sorted(top_gnn_pw - top_de_pw)[:15]
    pw_only_de = sorted(top_de_pw - top_gnn_pw)[:15]
    verdict_lines.append(f"Pathways unique to GNN (showing {len(pw_only_gnn)}):")
    for p in pw_only_gnn:
        verdict_lines.append(f"  + {p}")
    verdict_lines.append("")
    verdict_lines.append(f"Pathways unique to DE (showing {len(pw_only_de)}):")
    for p in pw_only_de:
        verdict_lines.append(f"  - {p}")
    verdict_lines.append("")
    verdict_lines.append("Interpretation:")
    verdict_lines.append(
        "  - If DE already hits PEROXISOM/GPI/ESCRT: GNN is redundant with DE + ORA."
    )
    verdict_lines.append(
        "  - If pathway_jaccard < 0.3 and GNN captures families DE misses: "
        "GNN reveals structure invisible to gene-by-gene DE."
    )
    (OUT_DIR / "verdict.txt").write_text("\n".join(verdict_lines) + "\n")
    print("\n" + "\n".join(verdict_lines))
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
