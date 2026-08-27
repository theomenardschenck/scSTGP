#!/usr/bin/env python3
"""gene_reference_table.py — one row per gene: rank in every view, DE status,
and degree decomposed by edge type.

Why: whenever the thesis names a gene in a table, the reader needs the two
numbers that let them judge it without trusting the score — how differential it
is, and how connected it is. Reproducing those by hand in every table is
error-prone; this builds the reference table once, and the per-edge-type
breakdown becomes an appendix.

Degrees are read from the built graph tensor (`hetero_graph_vgae.pt`), so they
are the degrees the encoder actually saw, not a recomputed approximation.
Reverse edge types (`*_by`, `regulated_by`) are folded into their forward type:
they encode the same relation, and counting both would double every degree.

Usage
-----
    python scripts/gene_reference_table.py \
        --out output/ora_memoire/gene_reference.tsv
    python scripts/gene_reference_table.py --genes KMT2A OCRL SERPINE1 --latex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/gnn_vgae/V6.1.3"
VIEWS = {
    "T-o": "output_fi/rfi2.pure-dir",
    "T-c": "output_fi/rfi2.pure-legacy",
    "S-o": "output_fi/rfi2.rich-dir",
    "S-c": "output_fi/rfi2.rich-legacy",
}
# reverse types folded into their forward counterpart (same relation, counted once)
FOLD = {"tf_curated_by": "tf_curated", "regulated_by": "regulates",
        "expressed_in": "expresses"}
SKIP = {"expresses"}          # gene<->cell_group, not a gene-gene degree


def edge_type_degrees(view: str, seed: str = "s1") -> pd.DataFrame:
    """Per-gene degree for each gene-gene edge type of one view's graph."""
    import torch
    p = BASE / VIEWS[view] / seed / "hetero_graph_vgae.pt"
    if not p.exists():
        return pd.DataFrame()
    g = torch.load(p, map_location="cpu", weights_only=False)
    n = g["gene"].num_nodes
    # The tensor stores no symbols; the node order is the one of
    # gene_embeddings_vgae.csv (first column), written by the same run.
    emb = p.parent / "gene_embeddings_vgae.csv"
    names = None
    if emb.exists():
        names = pd.read_csv(emb, usecols=[0]).iloc[:, 0].astype(str).tolist()
        if len(names) != n:
            print(f"[deg] {view}: {len(names)} symboles pour {n} nœuds -> ignoré")
            names = None
    deg: dict[str, np.ndarray] = {}
    for (src, rel, dst) in g.edge_types:
        if src != "gene" or dst != "gene":
            continue
        key = FOLD.get(rel, rel)
        if key in SKIP:
            continue
        ei = g[(src, rel, dst)].edge_index.numpy()
        d = np.bincount(ei[0], minlength=n) + np.bincount(ei[1], minlength=n)
        deg[key] = deg.get(key, np.zeros(n, dtype=np.int64)) + d
    df = pd.DataFrame(deg)
    # a fold pair counts each undirected edge twice -> halve the folded types
    for rel in set(FOLD.values()):
        if rel in df.columns:
            df[rel] = (df[rel] / 2).round().astype(int)
    df.index = names if names and len(names) == n else range(n)
    df.index.name = "gene"
    return df


def build() -> pd.DataFrame:
    out = None
    for v, rel in VIEWS.items():
        p = BASE / rel / "analysis/cross_seed_gene_ranking.tsv"
        if not p.exists():
            print(f"[skip] {v}: {p} absent"); continue
        d = pd.read_csv(p, sep="\t").set_index("target")
        d[f"rang_{v}"] = d.driver_score.rank(ascending=False, method="min").astype(int)
        cols = {f"rang_{v}": f"rang_{v}", "driver_score": f"score_{v}"}
        sub = d[list(cols)].rename(columns=cols)
        out = sub if out is None else out.join(sub, how="outer")
        if v == "S-c":
            ref = d
    # DE block, from the widest universe
    ref = ref.assign(abs_lfc=ref.de_log2fc_p4_vs_p16.abs())
    ref["rang_DE"] = ref.abs_lfc.rank(ascending=False, method="min",
                                      na_option="bottom").astype(int)
    de = ref[["de_log2fc_p4_vs_p16", "de_neglog10_padj", "rang_DE",
              "is_de_significant", "n_aging_dbs"]].rename(columns={
        "de_log2fc_p4_vs_p16": "log2FC", "de_neglog10_padj": "neglog10_padj",
        "is_de_significant": "DE_significatif", "n_aging_dbs": "n_bases_sen"})
    out = out.join(de, how="left")
    # per-database membership, from any run's gene_ranking_vgae.csv
    gr = BASE / VIEWS["S-c"] / "s1/gene_ranking_vgae.csv"
    if gr.exists():
        gd = pd.read_csv(gr).set_index("gene")
        keep = [c for c in gd.columns if c.startswith("in_")]
        if keep:
            out = out.join(gd[keep], how="left")
    # degree decomposition, per view (they differ: T views have no PPI/coexpr)
    for v in VIEWS:
        dd = edge_type_degrees(v)
        if dd.empty:
            print(f"[deg] {v}: graphe absent"); continue
        dd = dd.add_prefix(f"deg_{v}_")
        dd[f"deg_{v}_TOTAL"] = dd.sum(axis=1)
        out = out.join(dd, how="left")
    for v in VIEWS:
        c = f"deg_{v}_TOTAL"
        if c in out.columns:
            out[f"rang_deg_{v}"] = out[c].rank(ascending=False, method="min")
    return out.sort_values("rang_S-c")


def to_latex(df: pd.DataFrame, genes: list[str]) -> str:
    """Compact LaTeX row set: rank per view + DE + total degree."""
    sel = df.reindex([g for g in genes if g in df.index])
    lines = [r"\begin{tabular}{@{}l r r r r r r r@{}}", r"\toprule",
             r"\textbf{Gène} & \textbf{T-o} & \textbf{T-c} & \textbf{S-o} & "
             r"\textbf{S-c} & \textbf{log2FC} & \textbf{rang DE} & "
             r"\textbf{degré} \\", r"\midrule"]
    for g, r in sel.iterrows():
        lfc = "n.t." if pd.isna(r.log2FC) else f"{r.log2FC:+.2f}"
        rde = "---" if pd.isna(r.log2FC) else f"{int(r.rang_DE)}"
        dv = r.get("deg_S-c_TOTAL", np.nan)
        deg = "---" if pd.isna(dv) else str(int(dv))
        lines.append(f"{g} & {int(r['rang_T-o'])} & {int(r['rang_T-c'])} & "
                     f"{int(r['rang_S-o'])} & {int(r['rang_S-c'])} & {lfc} & "
                     f"{rde} & {deg} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "output/ora_memoire/gene_reference.tsv")
    ap.add_argument("--genes", nargs="*", default=None,
                    help="restrict the printed view to these genes")
    ap.add_argument("--latex", action="store_true",
                    help="print a LaTeX row set for --genes")
    a = ap.parse_args()
    df = build()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, sep="\t")
    print(f"{len(df)} gènes, {len(df.columns)} colonnes -> {a.out}")
    if a.genes:
        show = [c for c in df.columns
                if c.startswith("rang_") or c in ("log2FC", "n_bases_sen")
                or c.endswith("_TOTAL")]
        print(df.reindex(a.genes)[show].to_string())
        if a.latex:
            print("\n" + to_latex(df, a.genes))
