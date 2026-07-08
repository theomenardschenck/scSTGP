#!/usr/bin/env python3
"""
reproject_axes.py — Re-project a cached Δz onto NEW senescence axes without
re-perturbing (no forward pass, no SLURM).

The perturbation step can persist, per (gene, mode, cell_group), the
AXIS-INDEPENDENT quantities:
  * dz_mean_c   = (Σ w_diff·Δz) / Σ w_diff   — weighted-mean latent shift
  * w_diff_sum_c = Σ w_diff                  — the scale (added 2026-07)
via `perturb_top_genes.py --cache-delta-z` → `*_dz_cache.npz`. From those, the
per-axis metrics of ANY axis u are exactly reconstructable:
    proj_signed_cosine = cos(dz_mean_c, u)
    proj_signed_norm   = dz_mean_c · u
    proj_signed_diff   = w_diff_sum_c · (dz_mean_c · u)
So a new cluster / transition axis (or new anchors) is a pure re-projection —
seconds per run, no re-encoding.

This tool builds the P4→cluster and transition axes with the SAME machinery as
the perturbation (`compute_senescence_axes` on the run's μ + group_expression +
anchors), projects the cached dz_mean, canonicalises KO/KD/OE per axis, and
writes per-axis driver rankings in the SAME schema as
`perturb_report.write_per_axis_rankings`.

Caveat vs the full report: the driver_score here uses the graph-signal core
(amplitude × purity × coverage × coherence [+ centrality if a VGAE baseline is
present]) but omits the hub down-weight (`is_hub_inflated`), which needs the PPI
degree from the graph. Rankings are otherwise identical to the report's per-axis
driver_score. For the headline global axis, use the normal `perturb_report`.

Usage
-----
    python src/perturbation/reproject_axes.py \\
        --run-dirs 'output/gnn_vgae/V5.4.1/v5.4.baseline.s*' \\
        --quiescent-groups P4,P16_cluster_0 \\
        --transition-axes default --cluster-anchor-mode manual \\
        --out output/gnn_vgae/V5.4.1/reproject_baseline
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import siblings (perturb_top_genes loaders) + gnn_perturbation (axes) + report.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))                        # src/perturbation
sys.path.insert(0, str(_THIS.parent.parent / "gnn"))        # src/gnn
sys.path.insert(0, str(_THIS.parent.parent / "validation" / "reports"))

from perturb_top_genes import (_norm_group, _load_manual_anchors,      # noqa: E402
                               _load_de_marker_anchors, _build_transition_pairs,
                               GNN_DATA_DIR)
from perturb_report import (_canon_axis, _compute_driver_score,        # noqa: E402
                            write_per_axis_rankings)


def _compute_axes(run_dir: Path, quiescent, p16, cluster_anchors, transition_pairs):
    """Rebuild axis vectors from the run's μ + group_expression (no torch)."""
    from gnn_perturbation import compute_senescence_axes
    emb = pd.read_csv(run_dir / "gene_embeddings_vgae.csv", index_col=0)
    gene_symbols = emb.index.to_numpy().astype(str)
    mu = emb.to_numpy().astype(np.float32)
    group_expr = pd.read_csv(run_dir / "group_expression.tsv", sep="\t")
    # Auto-dérivation identique à prepare_baseline : si p16 non fourni, on RETIRE
    # les groupes quiescents (ex. c0 en V4) du côté sénescent → axes_cluster =
    # {c1,c2,c3} et non {c0..c3}.
    if p16:
        p16_list = list(p16)
    else:
        default_p16 = ["P16_cluster_0", "P16_cluster_1",
                       "P16_cluster_2", "P16_cluster_3"]
        p16_list = ([g for g in default_p16 if g not in set(quiescent)]
                    if quiescent else default_p16)
    _ag, axes_cluster, axes_transition, _c = compute_senescence_axes(
        mu, group_expr, gene_symbols,
        quiescent_groups=(tuple(quiescent) if quiescent else None),
        p16_groups=tuple(p16_list),
        cluster_anchors=cluster_anchors, transition_pairs=transition_pairs)
    # Axis registry : name -> (unit vector, weighting cell_group).
    axes: dict[str, tuple[np.ndarray, str]] = {}
    for grp, u in axes_cluster.items():
        axes[grp] = (np.asarray(u, np.float32), grp)             # weight = ck
    for (src, dst), u in axes_transition.items():
        axes[f"trans_{src}->{dst}"] = (np.asarray(u, np.float32), dst)  # weight = dst
    return axes


def _load_caches(run_dir: Path, cache_filter: str | None):
    """Load every *_dz_cache*.npz in run_dir. Returns list of dicts."""
    out = []
    for p in sorted(run_dir.glob("*_dz_cache*.npz")):
        if cache_filter and cache_filter not in p.name:
            continue
        z = np.load(p, allow_pickle=True)
        wsum = z["w_diff_sum"] if "w_diff_sum" in z.files else None
        out.append(dict(
            genes=z["genes"].astype(str), modes=z["modes"].astype(str),
            dz_mean=z["dz_mean"].astype(np.float32),
            wsum=(wsum.astype(np.float32) if wsum is not None else None),
            group_names=list(z["group_names"].astype(str)), path=p.name))
    return out


def _load_vgae_rank(run_dirs) -> dict:
    """Mean rank_vgae across runs (centrality term of driver_score). Optional."""
    frames = []
    for d in run_dirs:
        f = Path(d) / "gene_ranking_vgae.csv"
        if f.exists():
            df = pd.read_csv(f)
            if {"gene", "rank_vgae"} <= set(df.columns):
                frames.append(df[["gene", "rank_vgae"]])
    if not frames:
        return {}
    m = pd.concat(frames).groupby("gene")["rank_vgae"].mean()
    return {str(k): float(v) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dirs", nargs="+", required=True,
                    help="dossiers de run (globs autorisés) contenant "
                         "*_dz_cache*.npz + gene_embeddings_vgae.csv + group_expression.tsv")
    ap.add_argument("--out", type=Path, required=True,
                    help="dossier de sortie des rankings par axe")
    ap.add_argument("--cache-filter", default=None,
                    help="sous-chaîne pour filtrer les npz (ex. 'axisMULTI').")
    ap.add_argument("--quiescent-groups", default="P4,P16_cluster_0")
    ap.add_argument("--p16-groups", default=None,
                    help="clusters côté sénescent (défaut auto = c1..c3 hors quiescent).")
    ap.add_argument("--cluster-anchor-mode",
                    choices=["none", "de-markers", "manual"], default="none")
    ap.add_argument("--cluster-anchors-file", type=Path,
                    default=GNN_DATA_DIR / "ahn_cluster_anchors.tsv")
    ap.add_argument("--de-cluster-dir", type=Path, default=GNN_DATA_DIR)
    ap.add_argument("--de-cluster-top-n", type=int, default=50)
    ap.add_argument("--de-cluster-weight", choices=["equal", "logfc"], default="logfc")
    ap.add_argument("--transition-axes",
                    choices=["none", "default", "all-pairs"], default="default")
    ap.add_argument("--transition-pairs", default=None)
    ap.add_argument("--mode-agg", choices=["aligned", "oe-only"], default="aligned")
    args = ap.parse_args()

    run_dirs = []
    for pat in args.run_dirs:
        run_dirs += [Path(p) for p in sorted(glob.glob(pat))]
    run_dirs = [d for d in run_dirs if (d / "gene_embeddings_vgae.csv").exists()]
    if not run_dirs:
        sys.exit("[reproject] aucun run avec gene_embeddings_vgae.csv trouvé.")

    quiescent = [s.strip() for s in args.quiescent_groups.split(",") if s.strip()]
    p16 = ([s.strip() for s in args.p16_groups.split(",") if s.strip()]
           if args.p16_groups else None)
    anchors = None
    if args.cluster_anchor_mode == "manual":
        anchors = _load_manual_anchors(args.cluster_anchors_file)
    elif args.cluster_anchor_mode == "de-markers":
        anchors = _load_de_marker_anchors(args.de_cluster_dir, args.de_cluster_top_n,
                                          args.de_cluster_weight)
    trans = _build_transition_pairs(args.transition_axes, args.transition_pairs,
                                    p16_groups=p16)

    # Accumulate per (gene, mode, axis) across all runs/caches: mean cos + diff.
    acc: dict[tuple, dict[str, list]] = {}
    axis_names: set[str] = set()
    n_cache = 0
    for rd in run_dirs:
        caches = _load_caches(rd, args.cache_filter)
        if not caches:
            print(f"[reproject] {rd.name}: aucun dz_cache — sauté.")
            continue
        axes = _compute_axes(rd, quiescent, p16, anchors, trans)
        axis_names |= set(axes)
        for c in caches:
            n_cache += 1
            gidx = {g: i for i, g in enumerate(c["group_names"])}
            for name, (u, wgrp) in axes.items():
                if wgrp not in gidx:
                    continue
                gi = gidx[wgrp]
                dzc = c["dz_mean"][:, gi, :]                  # (N, latent)
                dots = dzc @ u                                # (N,)
                norms = np.linalg.norm(dzc, axis=1) + 1e-8
                cos = dots / norms
                diff = (c["wsum"][:, gi] * dots) if c["wsum"] is not None \
                    else np.full(len(dots), np.nan, np.float32)
                for j, (g, m) in enumerate(zip(c["genes"], c["modes"])):
                    d = acc.setdefault((str(g), str(m)), {})
                    d.setdefault(f"{name}::cos", []).append(float(cos[j]))
                    d.setdefault(f"{name}::diff", []).append(float(diff[j]))
    if not acc:
        sys.exit("[reproject] rien à projeter (pas de cache exploitable).")
    axis_names = sorted(axis_names)
    print(f"[reproject] {len(run_dirs)} run(s), {n_cache} cache(s), "
          f"{len(axis_names)} axe(s) : {', '.join(axis_names)}")

    # Mean across runs → mode-row dicts per gene → canonicalise per axis.
    genes = sorted({g for (g, _m) in acc})
    vgae_rank = _load_vgae_rank(run_dirs)
    total = len(genes)
    rows = []
    for g in genes:
        modes = {}
        for m in ("overexpress", "knockout", "knockdown"):
            key = (g, m)
            if key not in acc:
                continue
            modes[m] = {k: float(np.mean(v)) for k, v in acc[key].items()}
        oe, ko, kd = modes.get("overexpress"), modes.get("knockout"), modes.get("knockdown")
        n_modes = sum(x is not None for x in (oe, ko, kd))
        rec = {"target": g, "n_modes_present": n_modes}
        for name in axis_names:
            _s, cd, cc, sc = _canon_axis(oe, ko, kd, f"{name}::diff",
                                         f"{name}::cos", args.mode_agg)
            ds = _compute_driver_score(cd, cc, n_modes, sc, hub=False,
                                       vgae_rank=vgae_rank.get(g), total_genes=total)
            rec[f"driver_score_{name}"] = round(ds, 3)
            rec[f"canon_diff_{name}"] = round(cd, 3)
            rec[f"canon_cos_{name}"] = round(cc, 3)
        rows.append(rec)

    gene_rank = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    gene_rank.to_csv(args.out / "reproject_gene_ranking.tsv", sep="\t", index=False)
    write_per_axis_rankings(gene_rank, args.out)
    print(f"[reproject] écrit dans {args.out} "
          f"({len(gene_rank)} gènes, {len(axis_names)} axes) — aucune re-perturbation.")


if __name__ == "__main__":
    main()
