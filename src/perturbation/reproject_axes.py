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

F1 axis estimators (2026-07-22, FUT §2.8 / BIB §21.13)
------------------------------------------------------
`--axis-method diff,lda,cav,pca` registers the GLOBAL axis once per estimator on
the SAME poles, so one pass yields the robustness table "N axis definitions x
same drivers" — still with zero re-perturbation. Extra outputs:
`axis_method_diagnostics.tsv`, `axis_method_comparison.tsv` (Spearman + top-50
overlap per pair) and `axis_method_gene_ranks.tsv` (`rank_spread` = per-gene
axis instability). `--axis-lda-shrinkage 1.0` provably collapses back onto
`diff` (cos = 1.000), which is the analytic check of the implementation.

Usage
-----
    python src/perturbation/reproject_axes.py \\
        --run-dirs 'output/gnn_vgae/V5.4.1/v5.4.baseline.s*' \\
        --quiescent-groups P4,P16_cluster_0 \\
        --transition-axes default --cluster-anchor-mode manual \\
        --out output/gnn_vgae/V5.4.1/reproject_baseline

    # F1 robustness pass
    python src/perturbation/reproject_axes.py \\
        --run-dirs 'output/gnn_vgae/V5.4.1/v5.4.baseline.s*' \\
        --quiescent-groups P4,P16_cluster_0 \\
        --axis-method diff,lda,cav,pca --axis-cav-permutations 100 \\
        --out output/gnn_vgae/V5.4.1/reproject_axisF1
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


def _compute_axes(run_dir: Path, quiescent, p16, cluster_anchors, transition_pairs,
                  n_random=0, random_seed=0, effector_genes=None,
                  global_axis_name=None, axis_methods=(), axis_opts=None,
                  axis_diag=None):
    """Rebuild axis vectors from the run's μ + group_expression (no torch).

    Si n_random>0, ajoute N axes UNITAIRES ALÉATOIRES (decoy de spécificité
    d'axe) — pondérés par le dernier groupe sénescent. La projection est
    identique à celle des vrais axes (dz_mean·u) → le null de spécificité est
    entièrement reconstructible depuis le cache, sans re-perturber. Graine fixe
    ⇒ mêmes directions aléatoires entre runs (moyennables cross-seed)."""
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
        effector_axis_genes=effector_genes,
        cluster_anchors=cluster_anchors, transition_pairs=transition_pairs)
    # Axis registry : name -> (unit vector, weighting cell_group).
    axes: dict[str, tuple[np.ndarray, str]] = {}
    # Global axis, exposed as a NAMED axis only when explicitly anchored
    # (DE poles or effector gene lists). Weighted by the most senescent group,
    # like the random decoys. Un-anchored global stays implicit (the headline
    # ranking already carries it) to keep the axis registry unambiguous.
    _wref = p16_list[-1] if p16_list else "P4"
    if global_axis_name is not None:
        axes[global_axis_name] = (np.asarray(_ag, np.float32), _wref)
    # F1 (FUT §2.8): alternative estimators of the SAME pole contrast. Each is
    # registered as its own named axis so one pass yields the robustness figure
    # "N axis definitions x same drivers" — no re-perturbation, the Δz cache is
    # axis-independent. `axis_diff` is the historical centroid-difference axis.
    _opts = axis_opts or {}
    for _m in axis_methods:
        _info: dict = {}
        _ag_m, _c, _t, _ = compute_senescence_axes(
            mu, group_expr, gene_symbols,
            quiescent_groups=(tuple(quiescent) if quiescent else None),
            p16_groups=tuple(p16_list),
            effector_axis_genes=effector_genes,
            cluster_anchors=cluster_anchors, transition_pairs=transition_pairs,
            axis_method=_m, axis_info=_info, **_opts)
        _name = f"{global_axis_name}_{_m}" if global_axis_name else f"axis_{_m}"
        axes[_name] = (np.asarray(_ag_m, np.float32), _wref)
        if axis_diag is not None:
            axis_diag.append({"run": run_dir.name, "axis": _name, **_info})
    for grp, u in axes_cluster.items():
        axes[grp] = (np.asarray(u, np.float32), grp)             # weight = ck
    for (src, dst), u in axes_transition.items():
        axes[f"trans_{src}->{dst}"] = (np.asarray(u, np.float32), dst)  # weight = dst
    # Decoy d'axe : N directions aléatoires dans l'espace latent, pondérées par le
    # groupe le plus sénescent présent (référence de comparaison aux vrais axes).
    if n_random > 0:
        _rng = np.random.default_rng(random_seed)
        _latent = mu.shape[1]
        _ref = p16_list[-1] if p16_list else axes and next(iter(axes.values()))[1]
        for _k in range(n_random):
            _u = _rng.standard_normal(_latent).astype(np.float32)
            _u /= (np.linalg.norm(_u) + 1e-8)
            axes[f"random_{_k}"] = (_u, _ref)
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
    ap.add_argument("--random-axis", type=int, default=0,
                    help="N axes aléatoires (decoy de spécificité d'axe), "
                         "reconstruits depuis le cache Δz — aucune re-perturbation.")
    ap.add_argument("--random-axis-seed", type=int, default=0)
    # ── Anchored GLOBAL axis (V6) — reproject the headline axis from the Δz
    #    cache with poles fixed by MEASUREMENT (DE) or by curated effector
    #    lists, instead of the latent cell_group contrast. logFC stays a
    #    readout signal (never an encoder feature). Both routes feed
    #    `effector_axis_genes` of compute_senescence_axes.
    ap.add_argument("--de-axis-file", type=Path, default=None,
                    help="DE file -> anchored global axis (+ pole = top-N up "
                         "= pro-senescent, - pole = top-N down). Mutually "
                         "exclusive with --effector-axis.")
    ap.add_argument("--de-axis-label", default="sen_vs_pro")
    ap.add_argument("--de-axis-rank", default="stat")
    ap.add_argument("--de-axis-top-n", type=int, default=150)
    ap.add_argument("--de-axis-padj", type=float, default=0.05)
    ap.add_argument("--effector-axis", action="store_true",
                    help="anchored global axis = unit(mean(mu[pro]) - mean(mu[anti])).")
    ap.add_argument("--effector-pro", default="",
                    help="comma-separated pro-senescent effector symbols.")
    ap.add_argument("--effector-anti", default="",
                    help="comma-separated anti-senescent effector symbols.")
    # ── F1 (FUT §2.8 / BIB §21.13): alternative estimators of the global axis.
    #    Same poles, different direction estimator -> robustness control.
    ap.add_argument("--axis-method", default="",
                    help="comma list among diff,lda,cav,pca — register the "
                         "GLOBAL axis once per estimator (ex. 'diff,lda,cav,pca') "
                         "to get the 'N axis definitions x same drivers' table. "
                         "Empty = off (historical behaviour).")
    ap.add_argument("--axis-lda-shrinkage", type=float, default=0.05,
                    help="ridge blend of the within-class scatter towards a "
                         "scaled identity (0 = plain LDA, 1 = back to 'diff').")
    ap.add_argument("--axis-cav-top-n", type=int, default=500,
                    help="genes per pole for the logistic probe (CAV).")
    ap.add_argument("--axis-cav-seed", type=int, default=0)
    ap.add_argument("--axis-cav-permutations", type=int, default=0,
                    help="TCAV-style label-permutation null on probe accuracy "
                         "(0 = off). 100 is cheap and gives a p-value.")
    args = ap.parse_args()

    axis_methods = [s.strip() for s in args.axis_method.split(",") if s.strip()]
    _bad = [m for m in axis_methods if m not in ("diff", "lda", "cav", "pca")]
    if _bad:
        sys.exit(f"[reproject] --axis-method inconnu(s) : {_bad} "
                 "(attendu diff,lda,cav,pca)")
    axis_opts = dict(axis_lda_shrinkage=args.axis_lda_shrinkage,
                     axis_cav_top_n=args.axis_cav_top_n,
                     axis_cav_seed=args.axis_cav_seed,
                     axis_cav_permutations=args.axis_cav_permutations)

    if args.de_axis_file and args.effector_axis:
        sys.exit("[reproject] --de-axis-file et --effector-axis sont exclusifs.")
    effector_genes, global_axis_name = None, None
    if args.de_axis_file:
        from perturb_top_genes import _load_de_axis_anchors     # noqa: E402
        effector_genes = _load_de_axis_anchors(
            args.de_axis_file, args.de_axis_label, args.de_axis_top_n,
            args.de_axis_rank, args.de_axis_padj)
        global_axis_name = "de_anchored"
    elif args.effector_axis:
        effector_genes = (
            tuple(s.strip() for s in args.effector_pro.split(",") if s.strip()),
            tuple(s.strip() for s in args.effector_anti.split(",") if s.strip()))
        if not effector_genes[0] or not effector_genes[1]:
            sys.exit("[reproject] --effector-axis exige --effector-pro ET --effector-anti.")
        global_axis_name = "effector_anchored"

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
    axis_diag: list[dict] = []
    n_cache = n_no_wsum = 0
    for rd in run_dirs:
        caches = _load_caches(rd, args.cache_filter)
        if not caches:
            print(f"[reproject] {rd.name}: aucun dz_cache — sauté.")
            continue
        axes = _compute_axes(rd, quiescent, p16, anchors, trans,
                             n_random=args.random_axis,
                             random_seed=args.random_axis_seed,
                             effector_genes=effector_genes,
                             global_axis_name=global_axis_name,
                             axis_methods=axis_methods, axis_opts=axis_opts,
                             axis_diag=axis_diag)
        axis_names |= set(axes)
        for c in caches:
            n_cache += 1
            n_no_wsum += (c["wsum"] is None)
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
    if n_no_wsum:
        print(f"[warn] {n_no_wsum}/{n_cache} cache(s) SANS 'w_diff_sum' (produits "
              "avant 2026-07) : leur `diff` est NaN, or `_canon_axis` exige diff "
              "→ driver_score ET canon_cos retombent à 0 pour TOUS les axes. "
              "Re-perturber avec --cache-delta-z pour exploiter ces runs.")
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

    # ── F1 deliverable: are the drivers stable across axis definitions? ──────
    if axis_diag:
        pd.DataFrame(axis_diag).to_csv(args.out / "axis_method_diagnostics.tsv",
                                       sep="\t", index=False)
    if len(axis_methods) >= 2:
        _write_axis_method_comparison(gene_rank, axis_methods, global_axis_name,
                                      args.out)

    print(f"[reproject] écrit dans {args.out} "
          f"({len(gene_rank)} gènes, {len(axis_names)} axes) — aucune re-perturbation.")


def _write_axis_method_comparison(gene_rank: pd.DataFrame, axis_methods,
                                  global_axis_name, out: Path, top_n: int = 50):
    """Pairwise agreement between the F1 axis estimators (FUT §2.8).

    Spearman on the full driver_score plus top-N overlap: a driver set that
    survives whitening (lda), a logistic probe (cav) and the unsupervised PC1
    (pca) is not an artefact of the difference-of-means estimator. Low
    agreement is itself the result — it localises the floating-axis problem.
    """
    from itertools import combinations
    from scipy.stats import spearmanr

    def _col(m):
        name = f"{global_axis_name}_{m}" if global_axis_name else f"axis_{m}"
        return f"driver_score_{name}"

    present = [m for m in axis_methods if _col(m) in gene_rank.columns]
    if len(present) < 2:
        return
    rows = []
    for a, b in combinations(present, 2):
        ca, cb = gene_rank[_col(a)], gene_rank[_col(b)]
        ok = ca.notna() & cb.notna()
        rho = spearmanr(ca[ok], cb[ok]).statistic if ok.sum() > 2 else float("nan")
        ta = set(gene_rank.loc[ok].nlargest(top_n, _col(a))["target"])
        tb = set(gene_rank.loc[ok].nlargest(top_n, _col(b))["target"])
        rows.append({"axis_a": a, "axis_b": b, "n_genes": int(ok.sum()),
                     "spearman_driver_score": round(float(rho), 4),
                     f"top{top_n}_overlap": len(ta & tb),
                     f"top{top_n}_jaccard": round(len(ta & tb) / len(ta | tb), 4)
                     if (ta | tb) else float("nan")})
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(out / "axis_method_comparison.tsv", sep="\t", index=False)

    # Per-gene rank across estimators: max rank spread = instability flag.
    rk = pd.DataFrame({"target": gene_rank["target"]})
    for m in present:
        rk[m] = gene_rank[_col(m)].rank(ascending=False, method="min")
    rk["rank_min"] = rk[present].min(axis=1)
    rk["rank_max"] = rk[present].max(axis=1)
    rk["rank_spread"] = rk["rank_max"] - rk["rank_min"]
    rk.sort_values("rank_min").to_csv(out / "axis_method_gene_ranks.tsv",
                                      sep="\t", index=False)
    print(f"[reproject] F1 comparaison ({', '.join(present)}) → "
          f"axis_method_comparison.tsv + axis_method_gene_ranks.tsv")
    print(cmp_df.to_string(index=False))


if __name__ == "__main__":
    main()
