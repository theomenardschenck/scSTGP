#!/usr/bin/env python
"""readout_specificity.py — degree-free readout diagnostics from the Δz cache.

WHY (2026-07-20, LOG §25). On the stripped graph (`op.all`) the perturbation
signal DECREASES monotonically with degree: |amp| median goes 0.046 (deg 4-10)
-> 0.001 (deg >1000); TP53 (deg 4228) and MYC (deg 4121) read amp = 0.000. The
cause is mechanical -- `dz_mean` is a weighted MEAN over affected nodes
(gnn_perturbation.py:995), so a hub that nudges 4000 nodes by eps averages to
eps. On a dense graph the same readout inflates hubs instead. Both regimes are
the same defect: the statistic is degree-confounded, only the sign of the
confound changes. `driver_score` then correlates +0.451 with degree while its
own `amp` term anti-correlates -- an additive chimera of two opposite regimes.

This script computes two statistics that are degree-free BY CONSTRUCTION, both
post-hoc on artefacts already on disk (no re-perturbation):

1. CONCENTRATION -- `w_motion / n_affected`, the mean latent motion per node
   actually displaced. A hub spreads its budget thinly (diffuse); a specific
   driver moves few nodes hard (concentrated). Both operands are recovered
   exactly by inverting the stored ratio columns:
       n_affected = proj_signed_diff / proj_signed_extent
       w_motion   = proj_signed_diff / proj_signed_amplitude
   This is the participation-ratio idea (effective number of moved nodes)
   adapted to what the cache retains; the exact PR would need per-node |Δz|,
   which `dz_mean` has already collapsed.

2. SUBSPACE DECOMPOSITION (F2 of axis_idea.md) -- the senescence subspace S is
   spanned by the displacement vectors of DE-anchor genes (SVD, k dims). Each
   gene's displacement V is split on an orthonormal basis [u, e_1..e_k]:
       on_axis  = |V·u|                       (what driver_score already sees)
       off_axis = ||P_{S ⟂ u} V||             structured, but NOT colinear to DE
       residual = sqrt(||V||² - on² - off²)   outside S -> noise/idiopathic
   Falsifiable prediction: if the graph-native thesis holds, the metabolic
   targets (OCRL/SYNJ2/SMPD1) are enriched in the off_axis bin -- they move
   coherently with senescence without being DE-colinear. If they only populate
   the residual bin, the pure graph carries no signal for them.

Both are ranked degree-blind, so a hit is not a degree artefact.

Refs: Maslov & Sneppen 2002 Science (degree-preserving null); Milo 2002 Science
(z-score against a degree-matched null); La Manno 2018 Nature (latent
displacement fields).

Usage
-----
    python src/validation/explain/readout_specificity.py \
        --run-dir output/gnn_vgae/V6.1.3/output_op/op.all.s1 \
        --graph-summary /path/to/graph_summary.tsv \
        --mode knockout --subspace-k 6 \
        --out output/.../readout_specificity.tsv

Outputs one row per gene:
    gene, degree_total, norm_disp, on_axis, off_axis, residual,
    off_axis_ratio, n_affected, w_motion, concentration,
    rank_off_axis, rank_concentration
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS_FALLBACK = ["P4", "P16_cluster_0", "P16_cluster_1",
                   "P16_cluster_2", "P16_cluster_3"]


# ---------------------------------------------------------------- Δz cache ---
def load_cache(run_dir: Path, mode: str):
    """Return (genes, V, groups, axis_u) where V is the w_diff-weighted mean
    displacement per gene, aggregated over cell groups exactly the way
    `proj_signed_diff` sums them (reproject_axes.py:12-14)."""
    fp = run_dir / f"perturbation_all_genes_dz_cache_{mode}.npz"
    if not fp.exists():
        sys.exit(f"[specificity] Δz cache absent : {fp}")
    z = np.load(fp, allow_pickle=True)
    genes = z["genes"].astype(str)
    dz = z["dz_mean"].astype(np.float64)          # (N, n_groups, latent)
    w = z["w_diff_sum"].astype(np.float64)        # (N, n_groups)
    groups = (z["group_names"].astype(str).tolist()
              if "group_names" in z.files else GROUPS_FALLBACK)
    axis = z["axis_global"].astype(np.float64) if "axis_global" in z.files else None
    # Weighted mean across groups -> one displacement vector per gene.
    wsum = w.sum(axis=1, keepdims=True)
    V = (w[:, :, None] * dz).sum(axis=1) / np.where(wsum == 0, 1.0, wsum)
    return genes, V, groups, axis


# ------------------------------------------------------------ concentration --
def concentration_table(run_dir: Path, mode: str, groups) -> pd.DataFrame:
    """Recover n_affected and w_motion by inverting the stored ratio columns.

    proj_signed_extent    = proj_signed_diff / n_affected
    proj_signed_amplitude = proj_signed_diff / w_motion
    Summed over cell groups, so `concentration` is a per-gene scalar.
    """
    fp = run_dir / f"perturbation_all_genes_{mode}.tsv"
    if not fp.exists():
        sys.exit(f"[specificity] perturbation TSV absent : {fp}")
    df = pd.read_csv(fp, sep="\t", low_memory=False)
    df = df[df.get("target_type", "gene") == "gene"] if "target_type" in df else df

    n_aff = np.zeros(len(df))
    w_mot = np.zeros(len(df))
    used = 0
    for g in groups:
        c_diff = f"proj_signed_diff_global_{g}"
        c_ext = f"proj_signed_extent_global_{g}"
        c_amp = f"proj_signed_amplitude_global_{g}"
        if not all(c in df.columns for c in (c_diff, c_ext, c_amp)):
            continue
        used += 1
        d = df[c_diff].to_numpy(float)
        e = df[c_ext].to_numpy(float)
        a = df[c_amp].to_numpy(float)
        # ratios are exact inverses; guard the 0/0 cells (gene moved nothing)
        with np.errstate(divide="ignore", invalid="ignore"):
            n_aff += np.nan_to_num(np.abs(d / e), nan=0.0, posinf=0.0)
            w_mot += np.nan_to_num(np.abs(d / a), nan=0.0, posinf=0.0)
    if used == 0:
        sys.exit("[specificity] aucune colonne proj_signed_*_global_<group>.")
    conc = np.divide(w_mot, n_aff, out=np.zeros_like(w_mot), where=n_aff > 0)
    return pd.DataFrame({"gene": df["target"].astype(str),
                         "n_affected": n_aff,
                         "w_motion": w_mot,
                         "concentration": conc})


# ----------------------------------------------------------------- subspace --
def build_basis(V: np.ndarray, anchor_idx: np.ndarray, axis_u: np.ndarray,
                k: int):
    """Orthonormal basis [u, e_1..e_k] : u = senescence axis, e_i = top SVD
    directions of the ANCHOR displacements after removing the u component.
    Returns (u, E) with E of shape (k_eff, latent)."""
    u = axis_u / (np.linalg.norm(axis_u) + 1e-12)
    A = V[anchor_idx]
    A = A[np.linalg.norm(A, axis=1) > 0]
    if len(A) < 3:
        sys.exit(f"[specificity] seulement {len(A)} ancres mobiles — sous-espace "
                 "non identifiable.")
    # scale-free: each anchor contributes a DIRECTION, not its magnitude,
    # otherwise a single hub anchor would define S on its own.
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    A = A - np.outer(A @ u, u)                    # deflate the axis
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    k_eff = int(min(k, (sv > 1e-10).sum()))
    E = Vt[:k_eff]
    # re-orthonormalise against u (numerical hygiene)
    E = E - np.outer(E @ u, u)
    Q, _ = np.linalg.qr(E.T)
    E = Q.T[:k_eff]
    var = float((sv[:k_eff] ** 2).sum() / (sv ** 2).sum())
    print(f"[specificity] sous-espace S : k={k_eff}, {len(A)} ancres, "
          f"variance hors-axe captée = {var:.1%}")
    return u, E


def decompose(V: np.ndarray, u: np.ndarray, E: np.ndarray) -> pd.DataFrame:
    norm = np.linalg.norm(V, axis=1)
    on = np.abs(V @ u)
    off = np.linalg.norm(V @ E.T, axis=1)
    res2 = norm ** 2 - on ** 2 - off ** 2
    res = np.sqrt(np.clip(res2, 0.0, None))
    ratio = np.divide(off, norm, out=np.zeros_like(off), where=norm > 0)
    return pd.DataFrame({"norm_disp": norm, "on_axis": on, "off_axis": off,
                         "residual": res, "off_axis_ratio": ratio})


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--mode", default="knockout",
                    choices=["knockout", "knockdown", "overexpress"],
                    help="driver_score is built on KO/KD; OE is marker-broadcast")
    ap.add_argument("--ranking", type=Path, default=None,
                    help="cross_seed_gene_ranking.tsv (DE anchors + comparison). "
                         "Default <run-dir>/xseed/cross_seed_gene_ranking.tsv")
    ap.add_argument("--graph-summary", type=Path, default=None,
                    help="graph_summary.tsv for the degree-independence check")
    ap.add_argument("--subspace-k", type=int, default=6,
                    help="dimension of S beyond the axis (axis_idea.md: 3-8)")
    ap.add_argument("--n-anchors", type=int, default=150,
                    help="top-|logFC| significant DE genes spanning S")
    ap.add_argument("--targets", default="OCRL,SYNJ2,SMPD1,NAMPT,GCLC,"
                                         "HMGB1,HMGB2,H2AFZ,TP53,MYC,ASNS",
                    help="genes to report explicitly")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    genes, V, groups, axis_u = load_cache(args.run_dir, args.mode)
    if axis_u is None:
        sys.exit("[specificity] axis_global absent du cache.")
    print(f"[specificity] {len(genes)} gènes, mode={args.mode}, "
          f"{len(groups)} groupes, latent={V.shape[1]}")

    rk_fp = args.ranking or (args.run_dir / "xseed" / "cross_seed_gene_ranking.tsv")
    if not rk_fp.exists():
        sys.exit(f"[specificity] ranking absent : {rk_fp}")
    rk = pd.read_csv(rk_fp, sep="\t", low_memory=False)
    rk["rank_driver"] = np.arange(1, len(rk) + 1)

    # ── DE anchors : significant, ranked by |logFC| (de_schema convention) ──
    need = {"target", "is_de_significant", "de_log2fc_p4_vs_p16"}
    if not need.issubset(rk.columns):
        sys.exit(f"[specificity] colonnes DE manquantes dans {rk_fp}")
    de = rk[rk["is_de_significant"].astype(str).str.lower().isin(("true", "1"))]
    de = de.assign(_a=de["de_log2fc_p4_vs_p16"].abs()).nlargest(args.n_anchors, "_a")
    pos = {g: i for i, g in enumerate(genes)}
    anchor_idx = np.array([pos[g] for g in de["target"] if g in pos], dtype=int)
    print(f"[specificity] ancres DE : {len(anchor_idx)}/{args.n_anchors} mappées")

    u, E = build_basis(V, anchor_idx, axis_u, args.subspace_k)
    out = decompose(V, u, E)
    out.insert(0, "gene", genes)

    out = out.merge(concentration_table(args.run_dir, args.mode, groups),
                    on="gene", how="left")
    if args.graph_summary and args.graph_summary.exists():
        gs = pd.read_csv(args.graph_summary, sep="\t")
        out = out.merge(gs[["gene", "degree_total"]], on="gene", how="left")
    out = out.merge(rk[["target", "rank_driver", "driver_score"]],
                    left_on="gene", right_on="target", how="left").drop(columns="target")

    out["rank_off_axis"] = out["off_axis"].rank(ascending=False, method="min")
    out["rank_concentration"] = out["concentration"].rank(ascending=False,
                                                          method="min")
    out = out.sort_values("off_axis", ascending=False).reset_index(drop=True)

    dest = args.out or (args.run_dir / f"readout_specificity_{args.mode}.tsv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, sep="\t", index=False)
    print(f"[specificity] → {dest}")

    # ── degree-independence check (the whole point) ───────────────────────
    if "degree_total" in out.columns:
        from scipy.stats import spearmanr
        v = out[["degree_total", "off_axis", "concentration", "driver_score",
                 "off_axis_ratio"]].dropna()
        print("\n  ρ(degree_total, ·) — proche de 0 = métrique degree-free")
        for c in ["driver_score", "off_axis", "off_axis_ratio", "concentration"]:
            print(f"    {c:16s} {spearmanr(v['degree_total'], v[c])[0]:+.3f}")

    print("\n  TOP-15 off_axis (mouvement structuré NON colinéaire au DE)")
    cols = [c for c in ["gene", "off_axis", "off_axis_ratio", "on_axis",
                        "residual", "concentration", "degree_total",
                        "rank_driver"] if c in out.columns]
    print(out[cols].head(15).to_string(index=False))

    tg = [t.strip() for t in args.targets.split(",") if t.strip()]
    sub = out[out["gene"].isin(tg)].sort_values("rank_off_axis")
    if len(sub):
        print("\n  CIBLES")
        print(sub[cols + ["rank_off_axis", "rank_concentration"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
