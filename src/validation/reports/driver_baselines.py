#!/usr/bin/env python3
"""
driver_baselines.py — Le driver_score apporte-t-il quelque chose au-delà des
baselines triviales (degré, coexpr) ? (gnn_futur §5 action 3 / §7.10).

Sur un `cross_seed_gene_ranking.tsv` :
  1. ρ(driver_score, coexpr_degree) et ρ(driver_score, ppi_degree)
     → si ρ_coexpr ≈ 0 : le score n'est PAS la centralité coexpr.
  2. AUROC de rappel CellAge **brut** vs **intra-degré** (degré contrôlé)
     → l'AUROC brut est confondu par le biais d'étude (DB↔degré) ; l'AUROC
        intra-strate de degré mesure le signal AU-DELÀ du degré (>0.5 = oui).
  3. Rangs des drivers Tier-1 + part du top-50 robuste (rob=1 ∧ stab=1).

Sortie : `driver_baselines.tsv` (métriques) + impression lisible.

Usage
-----
    python src/validation/reports/driver_baselines.py \\
        --ranking output/.../cross_seed_gene_ranking.tsv \\
        --coexpr-file data/pyscenic/diff_coexpr/coexpr_diff.tsv \\
        --out output/.../driver_baselines.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_THIS = Path(__file__).resolve()
ROOT = next((p for p in _THIS.parents if (p / "data" / "databases").is_dir()),
            _THIS.parents[3])
TIER1 = ["H2AFZ", "HMGB1", "HMGB2", "ENO1", "FHL2", "CD59", "TP53", "MYC", "ASNS"]


def _auc(score, label):
    """AUROC via rang de Mann-Whitney (sans sklearn)."""
    s = np.asarray(score, float); y = np.asarray(label, int)
    ok = ~np.isnan(s); s, y = s[ok], y[ok]
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    df = pd.DataFrame({"s": s, "r": r}); df["r"] = df.groupby("s")["r"].transform("mean")
    return float((df["r"][y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _coexpr_degree(coexpr_file: Path) -> pd.Series:
    cx = pd.read_csv(coexpr_file, sep="\t", usecols=["TF", "target"])
    return pd.concat([cx["TF"], cx["target"]]).value_counts().rename("coexpr_degree")


def run(ranking: Path, coexpr_file: Path | None, cellage: Path,
        out: Path, score_col: str = "driver_score") -> pd.DataFrame:
    d = pd.read_csv(ranking, sep="\t").sort_values(score_col, ascending=False).reset_index(drop=True)
    d["rank"] = d.index + 1
    metrics = {}

    # 1. corrélations degré
    if "coexpr_degree" not in d.columns and coexpr_file and Path(coexpr_file).exists():
        d = d.merge(_coexpr_degree(coexpr_file), left_on="target", right_index=True, how="left")
        d["coexpr_degree"] = d["coexpr_degree"].fillna(0)
    if "coexpr_degree" in d.columns:
        metrics["rho_coexpr_degree"] = round(float(spearmanr(d[score_col], d["coexpr_degree"], nan_policy="omit")[0]), 3)
    if "target_ppi_degree" in d.columns:
        metrics["rho_ppi_degree"] = round(float(spearmanr(d[score_col], d["target_ppi_degree"], nan_policy="omit")[0]), 3)

    # 2. CellAge brut + intra-degré
    pos = set(pd.read_csv(cellage, sep="\t")["Gene symbol"].astype(str))
    d["_y"] = d["target"].isin(pos).astype(int)
    metrics["auroc_cellage_raw"] = round(_auc(d[score_col], d["_y"]), 3)
    if "target_ppi_degree" in d.columns:
        d["_db"] = pd.qcut(d["target_ppi_degree"].rank(method="first"), 6, labels=False)
        wa = wt = 0.0
        for _, sub in d.groupby("_db"):
            a = _auc(sub[score_col], sub["_y"])
            if not np.isnan(a):
                wa += a * len(sub); wt += len(sub)
        metrics["auroc_cellage_degctrl"] = round(wa / wt, 3) if wt else np.nan

    # 3. robustesse top-50 + Tier-1
    if {"mean_robustness", "mean_stability"}.issubset(d.columns):
        top = d.head(50)
        metrics["top50_robust"] = int(((top.mean_robustness >= 0.99) & (top.mean_stability >= 0.99)).sum())
    tier1_rank = {g: (int(d.loc[d.target == g, "rank"].iloc[0]) if (d.target == g).any() else None) for g in TIER1}

    # sortie
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"metric": k, "value": v} for k, v in metrics.items()]
    rows += [{"metric": f"rank_{g}", "value": r} for g, r in tier1_rank.items()]
    res = pd.DataFrame(rows)
    res.to_csv(out, sep="\t", index=False)

    print(f"=== driver_baselines ({ranking.name}) ===")
    print(f"  ρ(driver, coexpr_degree) = {metrics.get('rho_coexpr_degree','—')}  "
          f"(≈0 ⇒ ≠ centralité coexpr)")
    print(f"  ρ(driver, ppi_degree)    = {metrics.get('rho_ppi_degree','—')}")
    print(f"  AUROC CellAge brut       = {metrics.get('auroc_cellage_raw','—')}  (confondu degré↔étude)")
    print(f"  AUROC CellAge intra-degré= {metrics.get('auroc_cellage_degctrl','—')}  "
          f"(>0.5 ⇒ signal au-delà du degré)")
    if "top50_robust" in metrics:
        print(f"  top-50 robustes (2 seeds)= {metrics['top50_robust']}/50")
    print("  Tier-1 : " + " ".join(f"{g}#{r}" for g, r in tier1_rank.items() if r))
    print(f"[wrote] {out}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranking", type=Path, required=True)
    ap.add_argument("--coexpr-file", type=Path,
                    default=ROOT / "data/pyscenic/diff_coexpr/coexpr_diff.tsv")
    ap.add_argument("--cellage", type=Path, default=ROOT / "data/databases/cellage3.tsv")
    ap.add_argument("--score-col", default="driver_score")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run(args.ranking, args.coexpr_file, args.cellage, args.out, args.score_col)


if __name__ == "__main__":
    main()
