"""
build_supervised_labels.py — Multi-label DEG targets + DE node features for the
supervised (circular ceiling) mode of gnn_vgae.

CONTEXT
-------
The VGAE is *unsupervised* and deliberately excludes differential-expression
(DE) features (log2FC/padj/Δ%) to break circularity. The supervised mode
("V-sup", circular ceiling) does the opposite on purpose: it uses the DEG calls
as labels and the DE statistics as node features, training the encoder
end-to-end to classify each gene's senescence-DE status. The gap between this
circular upper bound and the VGAE quantifies how much the topology alone
captures (cf. roadmap "supervisé circulaire").

This module produces, aligned to a given `gene_symbols` ordering:
  - 5 binary multi-labels : {P4_vs_P16, cluster_0, cluster_1, cluster_2, cluster_3}
  - a per-gene confidence weight (bootstrap × consensus) for loss weighting
  - the DE feature matrix (global log2FC, -log10 padj, Δ%, per-cluster log2FC)

The global P4-vs-P16 call comes from `DEGs_P4_vs_P16.csv` (Seurat output) and
the confidence from `consensus_P4_vs_P16.csv`. The per-cluster DEGs
(`DEGs_P16_cluster_{c}.csv`) are no longer shipped in `data/gnn_data/`, so they
are **recomputed in Python** (scanpy Wilcoxon, cluster-c vs P4) from
`merged_P4_P16_normalized.csv` + the `cluster_P16` column, then cached.

CLI
---
    python build_supervised_labels.py --emit /tmp/sup_labels.tsv [--recompute]

Author: Théo Ménard — CRCI2NA. Created 2026-06-30.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Seurat-style significance thresholds for the recomputed per-cluster DE.
DEFAULT_LFC_THRESH = 0.25
DEFAULT_PADJ_THRESH = 0.05
DEFAULT_CLUSTERS = (0, 1, 2, 3)

_META_COLS = ("barcode", "passage", "cluster_P16", "cell_state")


@dataclass
class SupervisedLabels:
    """Container for supervised targets + DE node features, aligned to genes."""
    gene_symbols: np.ndarray            # (n_genes,) order of reference
    label_names: list                   # len 5
    labels: np.ndarray                  # (n_genes, 5) float32 in {0,1}
    confidence: np.ndarray              # (n_genes,) float32 in [0,1]
    global_lfc: np.ndarray              # (n_genes,) raw avg_log2FC P4→P16
    neg_log_padj: np.ndarray            # (n_genes,) -log10 padj P4 vs P16
    delta_pct: np.ndarray               # (n_genes,) pct.1 - pct.2
    cluster_lfc: np.ndarray             # (n_genes, 4) raw avg_log2FC per cluster

    def de_feature_matrix(self) -> np.ndarray:
        """(n_genes, 7) circular DE node features (normalised where unbounded)."""
        glfc = self.global_lfc / (np.abs(self.global_lfc).max() + 1e-8)
        npadj = self.neg_log_padj / (self.neg_log_padj.max() + 1e-8)
        clfc = self.cluster_lfc / (np.abs(self.cluster_lfc).max(axis=0) + 1e-8)
        return np.column_stack([glfc, npadj, self.delta_pct, clfc]).astype(np.float32)

    @staticmethod
    def de_feature_names() -> list:
        return ["de_log2fc_global", "de_neg_log_padj", "de_delta_pct",
                "de_log2fc_c0", "de_log2fc_c1", "de_log2fc_c2", "de_log2fc_c3"]


def _aligned(mapping: dict, gene_symbols, default=0.0) -> np.ndarray:
    return np.array([mapping.get(g, default) for g in gene_symbols], dtype=np.float32)


def compute_cluster_de(gnn_data_dir: str,
                       clusters=DEFAULT_CLUSTERS,
                       lfc_thresh: float = DEFAULT_LFC_THRESH,
                       padj_thresh: float = DEFAULT_PADJ_THRESH,
                       cache: bool = True,
                       verbose: bool = True) -> dict:
    """Recompute per-cluster DE (cluster-c P16 vs P4) with scanpy Wilcoxon.

    Reads `merged_P4_P16_normalized.csv` (cells × genes, log-normalised) +
    `cluster_P16`. Writes `DEGs_P16_cluster_{c}.csv` (gene, avg_log2FC,
    p_val_adj, significant) when `cache=True`. Returns {c: DataFrame}.
    """
    import anndata as ad
    import scanpy as sc

    norm_path = os.path.join(gnn_data_dir, "merged_P4_P16_normalized.csv")
    if verbose:
        print(f"  [sup-labels] recompute per-cluster DE from {os.path.basename(norm_path)}")
    # Memory-lean read (host may have ~4 GB free): float32 gene matrix via
    # usecols+dtype (~0.8 GB) instead of the default float64 full read (~1.6 GB
    # + copy → swap death). Meta columns read separately and cheaply.
    meta = pd.read_csv(norm_path, usecols=["passage", "cluster_P16"])
    header = pd.read_csv(norm_path, nrows=0).columns
    gene_cols = [c for c in header if c not in _META_COLS]
    df = pd.read_csv(norm_path, usecols=gene_cols,
                     dtype={c: "float32" for c in gene_cols})
    X = df[gene_cols].to_numpy(dtype=np.float32, copy=False)
    del df

    obs = pd.DataFrame({
        "passage": meta["passage"].astype(str).values,
        "cluster_P16": meta["cluster_P16"].values,
    })
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = gene_cols

    # Group label : "P4" for proliferative cells, "c<k>" for each P16 cluster.
    grp = np.where(
        obs["passage"].values == "P4",
        "P4",
        np.array(["c" + str(v).split(".")[0] for v in obs["cluster_P16"].values]),
    )
    adata.obs["grp"] = pd.Categorical(grp)
    target_groups = [f"c{c}" for c in clusters]

    sc.tl.rank_genes_groups(
        adata, "grp", groups=target_groups, reference="P4",
        method="wilcoxon", pts=False,
    )

    out = {}
    for c in clusters:
        res = sc.get.rank_genes_groups_df(adata, group=f"c{c}")
        res = res.rename(columns={
            "names": "gene", "logfoldchanges": "avg_log2FC",
            "pvals_adj": "p_val_adj",
        })[["gene", "avg_log2FC", "p_val_adj"]]
        res["significant"] = (
            (res["avg_log2FC"].abs() > lfc_thresh)
            & (res["p_val_adj"] < padj_thresh)
        )
        out[c] = res
        if cache:
            cache_path = os.path.join(gnn_data_dir, f"DEGs_P16_cluster_{c}.csv")
            res.to_csv(cache_path, index=False)
        if verbose:
            print(f"    cluster {c}: {int(res['significant'].sum())} DEGs "
                  f"(|log2FC|>{lfc_thresh}, padj<{padj_thresh})")
    return out


def _load_cluster_de(gnn_data_dir, clusters, recompute, lfc_thresh,
                     padj_thresh, verbose) -> dict:
    """Load cached per-cluster DEGs, recomputing/caching if any file is missing."""
    paths = {c: os.path.join(gnn_data_dir, f"DEGs_P16_cluster_{c}.csv")
             for c in clusters}
    if recompute or not all(os.path.exists(p) for p in paths.values()):
        return compute_cluster_de(gnn_data_dir, clusters, lfc_thresh,
                                  padj_thresh, cache=True, verbose=verbose)
    if verbose:
        print("  [sup-labels] per-cluster DEGs loaded from cache")
    out = {}
    for c, p in paths.items():
        d = pd.read_csv(p)
        if "significant" not in d.columns:
            d["significant"] = ((d["avg_log2FC"].abs() > lfc_thresh)
                                & (d["p_val_adj"] < padj_thresh))
        out[c] = d
    return out


def build_supervised_labels(gene_symbols,
                            gnn_data_dir: str,
                            clusters=DEFAULT_CLUSTERS,
                            recompute: bool = False,
                            lfc_thresh: float = DEFAULT_LFC_THRESH,
                            padj_thresh: float = DEFAULT_PADJ_THRESH,
                            verbose: bool = True) -> SupervisedLabels:
    """Assemble 5 multi-labels + confidence + DE features aligned to gene_symbols."""
    gene_symbols = np.asarray(gene_symbols)
    n = len(gene_symbols)

    # ── Global P4-vs-P16 (Seurat DEGs) ──────────────────────────────────────
    g = pd.read_csv(os.path.join(gnn_data_dir, "DEGs_P4_vs_P16.csv"))
    sig_map = dict(zip(g["gene"], g["significant"].astype(bool)))
    lfc_map = dict(zip(g["gene"], g["avg_log2FC"]))
    padj_map = dict(zip(g["gene"], g["p_val_adj"]))
    pct1_map = dict(zip(g["gene"], g["pct.1"]))
    pct2_map = dict(zip(g["gene"], g["pct.2"]))

    global_label = np.array([1.0 if sig_map.get(gn, False) else 0.0
                             for gn in gene_symbols], dtype=np.float32)
    global_lfc = _aligned(lfc_map, gene_symbols, 0.0)
    neg_log_padj = np.array(
        [-np.log10(max(padj_map.get(gn, 1.0), 1e-300)) for gn in gene_symbols],
        dtype=np.float32)
    delta_pct = np.array(
        [pct1_map.get(gn, 0.0) - pct2_map.get(gn, 0.0) for gn in gene_symbols],
        dtype=np.float32)

    # ── Confidence : bootstrap × consensus ──────────────────────────────────
    consensus = np.full(n, 0.5, dtype=np.float32)
    cons_path = os.path.join(gnn_data_dir, "consensus_P4_vs_P16.csv")
    if os.path.exists(cons_path):
        cdf = pd.read_csv(cons_path)
        cmap = dict(zip(cdf["gene"], cdf["consensus_score"].astype(float)))
        consensus = _aligned(cmap, gene_symbols, 0.5)
    elif verbose:
        print(f"  [sup-labels] {os.path.basename(cons_path)} absent — consensus=0.5")

    bootstrap = np.full(n, 0.5, dtype=np.float32)
    boot_path = os.path.join(gnn_data_dir, "bootstrap_stability_P4_vs_P16.csv")
    if os.path.exists(boot_path):
        bdf = pd.read_csv(boot_path)
        bmap = dict(zip(bdf["gene"], bdf["bootstrap_stability"].astype(float)))
        bootstrap = _aligned(bmap, gene_symbols, 0.5)
    elif verbose:
        print(f"  [sup-labels] {os.path.basename(boot_path)} absent — bootstrap=0.5")

    confidence = (bootstrap * consensus).astype(np.float32)
    # Floor so a confident-zero never fully zeroes the loss term.
    confidence = np.clip(confidence, 0.05, 1.0)

    # ── Per-cluster DE (recomputed/cached) ──────────────────────────────────
    cluster_de = _load_cluster_de(gnn_data_dir, clusters, recompute,
                                  lfc_thresh, padj_thresh, verbose)
    cluster_labels = np.zeros((n, len(clusters)), dtype=np.float32)
    cluster_lfc = np.zeros((n, len(clusters)), dtype=np.float32)
    for j, c in enumerate(clusters):
        d = cluster_de[c]
        slfc = dict(zip(d["gene"], d["avg_log2FC"]))
        ssig = dict(zip(d["gene"], d["significant"].astype(bool)))
        cluster_lfc[:, j] = _aligned(slfc, gene_symbols, 0.0)
        cluster_labels[:, j] = np.array(
            [1.0 if ssig.get(gn, False) else 0.0 for gn in gene_symbols],
            dtype=np.float32)

    labels = np.column_stack([global_label, cluster_labels]).astype(np.float32)
    label_names = ["P4_vs_P16"] + [f"cluster_{c}" for c in clusters]

    if verbose:
        pos = labels.sum(axis=0).astype(int)
        print(f"  [sup-labels] positives per label: "
              f"{dict(zip(label_names, pos.tolist()))}")
        print(f"  [sup-labels] confidence mean={confidence.mean():.3f}")

    return SupervisedLabels(
        gene_symbols=gene_symbols, label_names=label_names, labels=labels,
        confidence=confidence, global_lfc=global_lfc, neg_log_padj=neg_log_padj,
        delta_pct=delta_pct, cluster_lfc=cluster_lfc,
    )


def _cli():
    ap = argparse.ArgumentParser(description="Build/inspect supervised DEG labels.")
    ap.add_argument("--gnn-data-dir", default=None,
                    help="data/gnn_data dir (default: repo data/gnn_data)")
    ap.add_argument("--recompute", action="store_true",
                    help="force recompute per-cluster DE (ignore cache)")
    ap.add_argument("--emit", default=None, help="write the assembled table to TSV")
    args = ap.parse_args()

    gnn_data_dir = args.gnn_data_dir
    if gnn_data_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))         # src/data/preprocess
        repo = os.path.dirname(os.path.dirname(os.path.dirname(here)))
        gnn_data_dir = os.path.join(repo, "data", "gnn_data")

    # Use the global DEG gene list as the reference ordering for a standalone run.
    genes = pd.read_csv(os.path.join(gnn_data_dir, "DEGs_P4_vs_P16.csv"))["gene"].values
    sl = build_supervised_labels(genes, gnn_data_dir, recompute=args.recompute)

    if args.emit:
        out = pd.DataFrame({"gene": sl.gene_symbols})
        for j, name in enumerate(sl.label_names):
            out[name] = sl.labels[:, j]
        out["confidence"] = sl.confidence
        for name, col in zip(sl.de_feature_names(),
                             sl.de_feature_matrix().T):
            out[name] = col
        out.to_csv(args.emit, sep="\t", index=False)
        print(f"  → wrote {args.emit} ({len(out)} genes)")


if __name__ == "__main__":
    _cli()
