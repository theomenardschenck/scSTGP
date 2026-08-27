#!/usr/bin/env python3
"""score_variants.py — what happens to the ranking if the centrality term goes?

The headline `driver_score` is a weighted sum whose last term is
`max(0, 1 - vgae_rank/N)`, i.e. an explicit centrality prior. Since the whole
chapter on confounds argues that connectivity is the main nuisance, keeping a
centrality term *inside* the score is hard to defend. This script rebuilds the
ranking under several alternatives and reports, for each, the correlation to
degree, the correlation to the reference, and the rank of every module gene.

Variants
--------
V0  reference          0.35 amp + 0.30 cos + 0.15 cov + 0.10 coh + 0.10 centralité, hub x0.9
V1  sans centralité    same, centrality dropped, weights renormalised to 1
V2  sans centralité ni atténuation de hub
V3  multiplicatif      amp x |cos|            (the "log|effet| x cosinus" of the backlog)
V4  pureté seule       |cos|
V6  signal - malus     0.5 amp + 0.5 |cos| - 0.15 (1-cov) - 0.10 (1-coh)
                       The 2026-08-13 headline. Same four ingredients as V2, but
                       coverage and coherence can only ever COST points: a gene
                       gets nothing for being well-covered, it only loses for
                       being thin. Since n_modes = 3 for every gene of every
                       view, V2's coverage term was a strictly constant +0.167
                       on the whole ranking (and coherence +0.033 to +0.111 on
                       top): ~0.28 of the 0.32 median score was floor.
V5  produit par groupe mean_g( |amp_g| x cos_g ) read from the delta-z cache,
                       i.e. the product taken BEFORE aggregating the five cell
                       groups, instead of after. cos(mean) != mean(cos): a target
                       that pushes several groups in mutually cancelling
                       directions scores low under V0-V4 and high under V5.

The per-affected-gene version of V5 is NOT computable: the cache stores the mean
displacement per (target, group), not the 13 173 x 13 173 per-gene matrix. It
would need a re-run of the perturbation with per-gene retention.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/gnn_vgae/V6.1.3"
VIEWS = {"T-o": "output_fi/rfi2.pure-dir", "T-c": "output_fi/rfi2.pure-legacy",
         "S-o": "output_fi/rfi2.rich-dir", "S-c": "output_fi/rfi2.rich-legacy"}
MODULE_GENES = {
    "écriture chromatinienne": ["KMT2A", "KMT2C", "DOT1L", "EHMT2", "SETD8"],
    "architecture chromatinienne": ["HMGB1", "HMGB2", "H2AFZ"],
    "phospho-inositides": ["SYNJ2", "OCRL", "PIP4K2A", "PIKFYVE"],
    "sphingolipides": ["SMPD1", "UGCG"],
    "interface endothéliale": ["CD40", "TNFRSF1B", "ITGB1", "KDR"],
    "hubs (contrôle)": ["TP53", "MYC", "AKT1"],
    "effecteurs (contrôle)": ["CDKN2A", "SERPINE1", "IL6"],
}


def _amp(x):
    return np.minimum(np.log10(np.abs(x) + 1.0) / np.log10(501.0), 1.0)


def variants(view: str) -> pd.DataFrame:
    d = pd.read_csv(BASE / VIEWS[view] / "analysis/cross_seed_gene_ranking.tsv",
                    sep="\t").set_index("target")
    n = len(d)
    amp = _amp(d.canon_diff.fillna(0.0))
    cos = d.canon_cosine.fillna(0.0).abs().clip(upper=1.0)
    cov = (d.n_modes_present.fillna(1) / 3.0).clip(upper=1.0)
    coh = d.sign_consistent.map({True: 1.0, False: 0.3}).fillna(0.5)
    vr = pd.to_numeric(d.vgae_rank, errors="coerce")
    cen = (1.0 - vr / n).clip(lower=0.0).fillna(0.0)
    hub = d.is_hub_indexed if "is_hub_indexed" in d else d.get("is_hub_inflated", False)
    hub = pd.Series(hub, index=d.index).astype(str).str.lower().isin(["true", "1", "1.0"])
    hub_f = np.where(hub, 0.9, 1.0)

    out = pd.DataFrame(index=d.index)
    out["V0_reference"] = (0.35*amp + 0.30*cos + 0.15*cov + 0.10*coh + 0.10*cen) * hub_f
    out["V1_sans_centralite"] = ((0.35*amp + 0.30*cos + 0.15*cov + 0.10*coh) / 0.90) * hub_f
    out["V2_sans_cent_ni_hub"] = (0.35*amp + 0.30*cos + 0.15*cov + 0.10*coh) / 0.90
    out["V3_multiplicatif"] = amp * cos
    out["V4_purete_seule"] = cos
    coh6 = d.sign_consistent.map({True: 1.0, False: 0.0}).fillna(0.5)
    out["V6_signal_malus"] = (0.5*amp + 0.5*cos
                              - 0.15*(1-cov) - 0.10*(1-coh6)).clip(0.0, 1.0)
    out["_degre"] = d.get("target_total_degree", d.get("target_ppi_degree"))
    out["_ref_publie"] = d.driver_score
    return out


def variant_V5(view: str, seed: str = "s1") -> pd.Series | None:
    """mean_g(|amp_g| x cos_g): product before aggregating the cell groups."""
    p = BASE / VIEWS[view] / seed / "perturbation_all_genes_dz_cache_knockout.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    dz, u, genes = z["dz_mean"], z["axis_global"], z["genes"]
    u = u / (np.linalg.norm(u) + 1e-12)
    nrm = np.linalg.norm(dz, axis=2)                      # (targets, groups)
    cos = (dz @ u) / (nrm + 1e-12)                        # (targets, groups)
    prod = _amp(nrm * z["w_diff_sum"]) * np.abs(cos)      # produit PAR GROUPE
    return pd.Series(np.nanmean(prod, axis=1), index=genes, name="V5_produit_par_groupe")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", nargs="*", default=list(VIEWS))
    ap.add_argument("--out", type=Path, default=ROOT / "output/ora_memoire/score_variants")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    for v in a.views:
        df = variants(v)
        v5 = variant_V5(v)
        if v5 is not None:
            df = df.join(v5, how="left")
        cols = [c for c in df.columns if not c.startswith("_")]
        deg, ref = df["_degre"], df["_ref_publie"]

        print(f"\n{'='*88}\n{v}\n{'='*88}")
        print(f"{'variante':24s} {'rho degré':>10s} {'rho V0':>8s} {'rho publié':>11s} {'top-100 ∩ V0':>13s}")
        base_top = set(df["V0_reference"].nlargest(100).index)
        for c in cols:
            rd = spearmanr(df[c], deg, nan_policy="omit").statistic
            r0 = spearmanr(df[c], df["V0_reference"], nan_policy="omit").statistic
            rp = spearmanr(df[c], ref, nan_policy="omit").statistic
            ov = len(base_top & set(df[c].nlargest(100).index))
            print(f"{c:24s} {rd:>+10.3f} {r0:>+8.3f} {rp:>+11.3f} {ov:>10d}/100")

        rk = df[cols].rank(ascending=False, method="min").astype("Int64")
        print(f"\n{'gène':12s}" + "".join(f"{c.split('_')[0]:>8s}" for c in cols))
        for mod, gs in MODULE_GENES.items():
            print(f"  -- {mod}")
            for g in gs:
                if g in rk.index:
                    print(f"{g:12s}" + "".join(f"{rk.loc[g, c]:>8}" for c in cols))
        df.to_csv(a.out / f"score_variants_{v}.tsv", sep="\t")
    print(f"\n-> {a.out}")
