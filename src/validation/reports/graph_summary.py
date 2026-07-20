#!/usr/bin/env python
"""graph_summary.py — per-gene connectivity summary of a built hetero graph.

WHY (2026-07-16, LOG §23-24). The ranking exposes a single connectivity column,
`target_ppi_degree`, which counts **STRING PPI edges only**. It reads **0** for
every graph-native target (OCRL / SYNJ2 / SMPD1) whose real degree is 65-170,
carried by reactome_fi and HuMess. Any degree-aware reading of the ranking was
therefore blind for exactly the genes under investigation.

This script emits the missing table: for each gene, the degree **per edge_type**,
the **total** degree, the directed in/out split, and the **local clustering
coefficient** — the statistic that separates a genuine driver from a member of a
dense module.

Clustering coefficient (Watts & Strogatz 1998): fraction of a node's neighbour
pairs that are themselves connected, i.e. `2·e_i / (k_i·(k_i−1))`. It is already
degree-normalised, so it is NOT a degree proxy — a hub wired to mutually
unconnected partners scores ~0, while a member of a clique scores ~1. Measured
on the ribosome subgraph of reactome_fi it reaches **0.909** (538x the graph
density) whereas OCRL/SYNJ2 score **0.000** — the discriminator the additive
`driver_score` lacks. It is reported on the union gene-gene graph AND per source
(`--per-source-clustering`), because the clique is source-specific.

Usage
-----
    python src/validation/reports/graph_summary.py \
        --run-dir output/gnn_vgae/V6.1.3/output_op/op.all.s1 \
        --out     output/gnn_vgae/V6.1.3/output_op/op.all.s1/graph_summary.tsv

Outputs `graph_summary.tsv` with one row per gene:
    gene, degree_total, degree_out, degree_in, n_edge_types,
    clustering_coeff, clustering_<source>…, deg_<edge_type>…
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_graph(run_dir: Path):
    """Return (HeteroData, gene_names). Names come from gene_embeddings_vgae.csv,
    whose row order is the gene node index order."""
    import torch
    g = torch.load(run_dir / "hetero_graph_vgae.pt", map_location="cpu",
                   weights_only=False)
    emb = run_dir / "gene_embeddings_vgae.csv"
    if not emb.exists():
        sys.exit(f"[graph_summary] {emb} absent — impossible de nommer les nœuds.")
    names = pd.read_csv(emb, usecols=[0]).iloc[:, 0].astype(str).tolist()
    n = int(g["gene"].num_nodes)
    if len(names) != n:
        sys.exit(f"[graph_summary] {len(names)} noms ≠ {n} nœuds gène.")
    return g, names


def _gene_edge_types(g):
    return [et for et in g.edge_types if et[0] == "gene" and et[2] == "gene"]


def _clustering(adj: dict[int, set[int]], nodes) -> np.ndarray:
    """Local clustering coefficient (Watts-Strogatz) on an undirected adj map."""
    out = np.zeros(len(nodes), dtype=float)
    for i in nodes:
        nb = adj.get(i)
        if not nb or len(nb) < 2:
            continue
        k = len(nb)
        # count links among neighbours (each pair once)
        links = 0
        nb_list = list(nb)
        for a_i, a in enumerate(nb_list):
            na = adj.get(a)
            if not na:
                continue
            for b in nb_list[a_i + 1:]:
                if b in na:
                    links += 1
        out[i] = 2.0 * links / (k * (k - 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="run containing hetero_graph_vgae.pt + gene_embeddings_vgae.csv")
    ap.add_argument("--out", type=Path, default=None,
                    help="output TSV (default <run-dir>/graph_summary.tsv)")
    ap.add_argument("--per-source-clustering", default="reactome_fi",
                    help="comma-separated edge_types to also get their own "
                         "clustering column (default reactome_fi ; '' to skip). "
                         "The clique artefact is source-specific.")
    ap.add_argument("--no-clustering", action="store_true",
                    help="skip clustering coefficients (much faster on dense graphs)")
    args = ap.parse_args()

    g, names = _load_graph(args.run_dir)
    n = len(names)
    ets = _gene_edge_types(g)
    if not ets:
        sys.exit("[graph_summary] aucun edge_type gene↔gene.")

    df = pd.DataFrame({"gene": names})
    total = np.zeros(n, dtype=int)
    deg_out = np.zeros(n, dtype=int)
    deg_in = np.zeros(n, dtype=int)
    present = np.zeros(n, dtype=int)
    union_adj: dict[int, set[int]] = {}
    per_src = [s.strip() for s in args.per_source_clustering.split(",") if s.strip()]
    src_adj: dict[str, dict[int, set[int]]] = {s: {} for s in per_src}

    for et in ets:
        rel = et[1]
        ei = g[et].edge_index.numpy()
        d_o = np.bincount(ei[0], minlength=n)
        d_i = np.bincount(ei[1], minlength=n)
        deg = d_o + d_i                      # endpoint count (matches earlier audits)
        df[f"deg_{rel}"] = deg
        total += deg
        deg_out += d_o
        deg_in += d_i
        present += (deg > 0).astype(int)
        if not args.no_clustering:
            for a, b in zip(ei[0].tolist(), ei[1].tolist()):
                if a == b:
                    continue
                union_adj.setdefault(a, set()).add(b)
                union_adj.setdefault(b, set()).add(a)
                if rel in src_adj:
                    src_adj[rel].setdefault(a, set()).add(b)
                    src_adj[rel].setdefault(b, set()).add(a)

    df.insert(1, "degree_total", total)
    df.insert(2, "degree_out", deg_out)
    df.insert(3, "degree_in", deg_in)
    df.insert(4, "n_edge_types", present)

    if not args.no_clustering:
        nodes = range(n)
        df.insert(5, "clustering_coeff", _clustering(union_adj, nodes))
        for s, adj in src_adj.items():
            if adj:
                df[f"clustering_{s}"] = _clustering(adj, nodes)

    out = args.out or (args.run_dir / "graph_summary.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"[graph_summary] {out}  ({n} gènes, {len(ets)} edge_types gene↔gene)")
    print(f"  degré total : médiane={np.median(total):.0f} "
          f"p99={np.quantile(total, .99):.0f} max={total.max()}")
    if not args.no_clustering:
        cc = df["clustering_coeff"].to_numpy()
        print(f"  clustering  : médiane={np.median(cc):.3f} moyenne={cc.mean():.3f}")


if __name__ == "__main__":
    main()
