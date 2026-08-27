#!/usr/bin/env python3
"""rescore_headline.py — rebuild the headline ranking without the centrality prior.

Why
---
The published `driver_score` is
    0.35 amp + 0.30 purity + 0.15 coverage + 0.10 coherence + 0.10 centrality
with a x0.9 factor on hubs. Two of those terms are indefensible once the thesis
argues that connectivity is the main confound:

  * `centrality = max(0, 1 - vgae_rank/N)` is an explicit degree prior INSIDE a
    score whose whole point is to be read against a degree control;
  * the x0.9 hub attenuation is a post-hoc patch on the same confound, and it is
    measurably inert (V1 and V2 of score_variants.py are rank-identical).

v2 headline score, weights renormalised to 1:
    (0.35 amp + 0.30 purity + 0.15 coverage + 0.10 coherence) / 0.90

2026-08-13 — v3, signal minus malus
-----------------------------------
v2 was still a convex combination, so `coverage` and `coherence` handed out
points for being *checkable* rather than for being a driver. Since `n_modes`
is 3 for every gene of every view, coverage was a strictly constant +0.167 on
the whole ranking, and coherence added +0.033 to +0.111 on top: median 0.32,
of which ~0.28 pure floor. v3 keeps the same four ingredients and the same
relative sanity weights, but only amplitude and purity can earn points (50/50)
and the sanity terms only ever subtract:

    signal = 0.50 amp + 0.50 purity
    malus  = 0.15 (1 - coverage) + 0.10 (1 - coherence)
    v3     = clip(signal - malus, 0, 1)

with coherence 1 / 0.5 / 0 for sign-consistent / single-mode / inconsistent
(v2 used 1 / 0.5 / 0.3, i.e. it still paid the inconsistent ones).

This is a pure post-processing of `cross_seed_gene_ranking.tsv`: the encoder,
the perturbation and the axis are untouched, only the aggregation of the four
surviving ingredients changes. No re-training is involved, so it applies to
every config already on disk, including the PPI-fixed arm. `--apply` is
re-runnable: the ingredients it reads are never rewritten, and the published
v0 column is preserved across successive patches.

Output
------
`output/ora_memoire/rescore/<config>.tsv` — one row per gene with the old and
new score and rank, plus the ingredients, so any downstream table can be
rebuilt from it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/gnn_vgae/V6.1.3"

# The four views of the thesis, plus the PPI-fixed twin of S-c and its own
# control. `mirror.rich-legacy` is the same graph as `rfi2.rich-legacy` and is
# kept so the PPI effect is measured within one wave rather than across two.
CONFIGS = {
    "T-o": "output_fi/rfi2.pure-dir",
    "T-c": "output_fi/rfi2.pure-legacy",
    "S-o": "output_fi/rfi2.rich-dir",
    "S-c": "output_fi/rfi2.rich-legacy",
    "S-c ppi-legacy": "output_ppi/mirror.rich-legacy",
    "S-c ppi-fixed": "output_ppi/mirror.rich-fixed",
    "T-c ppi-ctrl": "output_ppi/mirror.pure-fixed",
}

MODULES = {
    "écriture chromatinienne": ["KMT2A", "KMT2B", "KMT2C", "KMT2D", "KMT2E",
                                "DOT1L", "EHMT1", "EHMT2", "SETD8", "SETDB1"],
    "architecture chromatinienne": ["HMGB1", "HMGB2", "H2AFZ"],
    "phospho-inositides": ["OCRL", "SYNJ2", "PI4K2A", "PIP4K2A", "PIP4K2B",
                           "PIP4K2C", "PIKFYVE"],
    "sphingolipides": ["SMPD1", "UGCG"],
    "interface endothéliale": ["CD40", "TNFRSF1A", "TNFRSF1B", "TRAF2", "LTBR",
                               "ITGB1", "ITGB3", "ITGAV", "ITGA5", "SDC2"],
    "cycle et mitose": ["CDK1", "CDK2", "CCNB1", "AURKB", "HMMR", "CHEK1"],
    "interféron": ["ISG15", "IRF1", "STAT3"],
    "hubs (contrôle)": ["TP53", "MYC", "AKT1"],
    "effecteurs (contrôle)": ["CDKN1A", "CDKN2A", "SERPINE1", "IL6", "IL1B"],
}


def _amp(x: pd.Series) -> pd.Series:
    return np.minimum(np.log10(np.abs(x) + 1.0) / np.log10(501.0), 1.0)


def rescore(cfg_dir: Path) -> pd.DataFrame | None:
    """Old and new score side by side, from the cross-seed ingredients."""
    p = cfg_dir / "analysis/cross_seed_gene_ranking.tsv"
    if not p.exists():
        return None
    d = pd.read_csv(p, sep="\t").set_index("target")
    n = len(d)

    amp = _amp(d.canon_diff.fillna(0.0))
    cos = d.canon_cosine.fillna(0.0).abs().clip(upper=1.0)
    cov = (d.n_modes_present.fillna(1) / 3.0).clip(upper=1.0)
    coh_v2 = d.sign_consistent.map({True: 1.0, False: 0.3}).fillna(0.5)
    coh = d.sign_consistent.map({True: 1.0, False: 0.0}).fillna(0.5)
    vr = pd.to_numeric(d.vgae_rank, errors="coerce")
    cen = (1.0 - vr / n).clip(lower=0.0).fillna(0.0)
    hub = d.get("is_hub_indexed", d.get("is_hub_inflated", False))
    hub = pd.Series(hub, index=d.index).astype(str).str.lower().isin(["true", "1", "1.0"])

    out = pd.DataFrame(index=d.index)
    out["amp"], out["purity"], out["coverage"], out["coherence"] = amp, cos, cov, coh
    out["centralite"], out["is_hub"] = cen, hub
    # v0 = as published. Read from the legacy column once --apply has run,
    # otherwise from driver_score itself (pre-patch config).
    out["score_v0"] = (d.driver_score_v0_legacy if "driver_score_v0_legacy" in d
                       else d.driver_score)
    out["score_v2"] = (0.35 * amp + 0.30 * cos + 0.15 * cov + 0.10 * coh_v2) / 0.90
    # v3 : signal (50/50 amplitude/purity) minus malus for missing evidence.
    out["signal"] = 0.50 * amp + 0.50 * cos
    out["malus"] = 0.15 * (1.0 - cov) + 0.10 * (1.0 - coh)
    out["score_v3"] = (out.signal - out.malus).clip(lower=0.0, upper=1.0)
    for v in ("v0", "v2", "v3"):
        out[f"rang_{v}"] = out[f"score_{v}"].rank(ascending=False,
                                                  method="min").astype(int)
    out["degre"] = d.get("target_total_degree", d.get("target_ppi_degree"))
    return out


def apply_axes(cfg_dir: Path, variant: str = "v3") -> str:
    """Same rescoring, applied to the per-axis and per-transition rankings.

    The cluster and transition files are reprojections of a cached displacement
    (`reproject_axes.py`): they carry `canon_cos` rather than `canon_cosine`,
    and they have no `sign_consistent` column at all, because the sign of a
    reprojected axis is not compared across modes. The coherence term is
    therefore set to the "unknown" value 0.5 for every gene of such a file --
    a CONSTANT, so it shifts all scores equally and leaves the ranking, which
    is the only thing the thesis reads from these files, untouched.

    Without this pass the transition table of chapter 4 would compare a v3
    global axis against transition axes still scored by the published formula.
    """
    seen = 0
    for p in sorted(cfg_dir.glob("analysis/**/cross_seed_gene_ranking__*.tsv")):
        d = pd.read_csv(p, sep="\t")
        cos_col = "canon_cosine" if "canon_cosine" in d.columns else "canon_cos"
        if "canon_diff" not in d.columns or cos_col not in d.columns:
            continue
        if d.get("driver_score_variant", pd.Series([""])).iloc[0] == variant:
            continue
        orig = p.with_suffix(".tsv.orig")
        if not orig.exists():
            orig.write_bytes(p.read_bytes())
        if "driver_score_v0_legacy" not in d.columns:
            d["driver_score_v0_legacy"] = d.driver_score.values
        amp = _amp(d.canon_diff.fillna(0.0))
        cos = d[cos_col].fillna(0.0).abs().clip(upper=1.0)
        cov = (d.n_modes_present.fillna(1) / 3.0).clip(upper=1.0)
        coh = (d.sign_consistent.map({True: 1.0, False: 0.0}).fillna(0.5)
               if "sign_consistent" in d.columns else pd.Series(0.5, index=d.index))
        d["driver_score"] = ((0.50 * amp + 0.50 * cos)
                             - (0.15 * (1.0 - cov) + 0.10 * (1.0 - coh))
                             ).clip(lower=0.0, upper=1.0)
        d["driver_score_variant"] = variant
        d = d.sort_values("driver_score", ascending=False, kind="mergesort")
        d.to_csv(p, sep="\t", index=False)
        seen += 1
    return f"{seen} axes patchés"


def _median_rank(df: pd.DataFrame, genes: list[str], col: str) -> float:
    g = [x for x in genes if x in df.index]
    return float(df.loc[g, col].median()) if g else float("nan")


def apply_in_place(cfg_dir: Path, variant: str = "v3") -> str:
    """Rewrite `driver_score` inside the cross-seed ranking of one config.

    Everything downstream (gene_reference_table, ora_consensus,
    memoire_figures, build_annexes) reads `driver_score` from this one file,
    so patching it here propagates the new score through the whole thesis with
    no other change. The original is kept twice over: a `.orig` copy of the
    file, and a `driver_score_v0_legacy` column inside the new one.

    Re-runnable across variants: the ingredient columns (`canon_diff`,
    `canon_cosine`, `n_modes_present`, `sign_consistent`) are never rewritten,
    so a second `--apply` recomputes from the same inputs. `driver_score_v0_legacy`
    is written once and then left alone — it must keep holding the *published*
    score, not the previous patch. `driver_score_variant` records which formula
    the file currently carries.
    """
    p = cfg_dir / "analysis/cross_seed_gene_ranking.tsv"
    if not p.exists():
        return "absent"
    d = pd.read_csv(p, sep="\t")
    if d.get("driver_score_variant", pd.Series([""])).iloc[0] == variant:
        return f"déjà patché ({variant})"
    orig = p.with_suffix(".tsv.orig")
    if not orig.exists():
        orig.write_bytes(p.read_bytes())
    t = rescore(cfg_dir)
    if "driver_score_v0_legacy" in d.columns:
        # already patched at least once; unmarked files carry v2 by history
        was = d.get("driver_score_variant", pd.Series(["v2"])).iloc[0] or "v2"
    else:
        d["driver_score_v0_legacy"] = d.driver_score.values
        was = "v0"
    d["driver_score"] = t.loc[d.target, f"score_{variant}"].values
    d["driver_score_variant"] = variant
    d = d.sort_values("driver_score", ascending=False, kind="mergesort")
    if "rank" in d.columns:
        d["rank"] = np.arange(1, len(d) + 1)
    d.to_csv(p, sep="\t", index=False)
    return f"patché ({was} -> {variant})"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "output/ora_memoire/rescore")
    ap.add_argument("--apply", action="store_true",
                    help="réécrit driver_score dans les cross_seed_gene_ranking.tsv "
                         "(sauvegarde .orig + colonne driver_score_v0_legacy)")
    ap.add_argument("--variant", choices=["v2", "v3"], default="v3",
                    help="formule écrite par --apply (défaut v3 = signal-malus) ; "
                         "le tableau comparatif affiche toujours les trois")
    ap.add_argument("--from", dest="ref", choices=["v0", "v2"], default="v2",
                    help="référence des colonnes de comparaison (défaut v2, "
                         "la formule en place sur disque)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    if a.apply:
        for lab, rel in CONFIGS.items():
            print(f"  {lab:16s} {apply_in_place(BASE / rel, a.variant)}"
                  f"  |  {apply_axes(BASE / rel, a.variant)}")
        print("Ranking patché. Relancer gene_reference_table.py puis build_annexes.py.")
        raise SystemExit(0)

    ref, new = a.ref, a.variant
    tables: dict[str, pd.DataFrame] = {}
    print(f"{'config':16s} {'n':>6s} {'rho(deg) '+ref:>12s} {'rho(deg) '+new:>12s} "
          f"{'rho '+ref+'-'+new:>10s} {'top100 gardé':>13s} {'plancher 0':>11s}")
    for lab, rel in CONFIGS.items():
        t = rescore(BASE / rel)
        if t is None:
            print(f"{lab:16s} <absent: {rel}>")
            continue
        tables[lab] = t
        t.to_csv(a.out / f"{lab.replace(' ', '_')}.tsv", sep="\t")
        r0 = spearmanr(t[f"score_{ref}"], t.degre, nan_policy="omit").statistic
        r2 = spearmanr(t[f"score_{new}"], t.degre, nan_policy="omit").statistic
        rr = spearmanr(t[f"score_{ref}"], t[f"score_{new}"],
                       nan_policy="omit").statistic
        keep = len(set(t.nsmallest(100, f"rang_{ref}").index)
                   & set(t.nsmallest(100, f"rang_{new}").index))
        floored = int((t[f"score_{new}"] <= 0).sum())
        print(f"{lab:16s} {len(t):>6,} {r0:>+12.3f} {r2:>+12.3f} {rr:>+10.3f} "
              f"{keep:>10d}/100 {floored:>11,}".replace(",", " "))

    print("\n" + "=" * 100)
    print(f"MÉDIANE DE RANG PAR MODULE — {ref} → {new}")
    print("=" * 100)
    labs = [l for l in CONFIGS if l in tables]
    print(f"{'module':30s}" + "".join(f"{l:>18s}" for l in labs))
    for mod, genes in MODULES.items():
        cells = []
        for l in labs:
            m0 = _median_rank(tables[l], genes, f"rang_{ref}")
            m2 = _median_rank(tables[l], genes, f"rang_{new}")
            cells.append(f"{m0:>7.0f}→{m2:<9.0f}")
        print(f"{mod:30s}" + "".join(f"{c:>18s}" for c in cells))
    print(f"\n-> {a.out}")
