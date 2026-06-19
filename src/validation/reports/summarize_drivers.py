#!/usr/bin/env python3
# =============================================================================
# summarize_drivers.py — Synthèse lisible d'un (ou plusieurs) cross_seed_gene_
# ranking.tsv : ranking_final.tsv (colonnes simplifiées) + recap.md (top-N
# interprété : rôle driver/cible/marqueur, sens, force, cos, DE, DBs, confiance).
# =============================================================================
# Cleanup V6 : remplace la lecture des 42 colonnes du cross-seed par une vue
# courte et défendable. Avec plusieurs axes (--also), ajoute une colonne de
# CONCORDANCE (rang par axe + nb d'axes où le gène est top-K) = driver
# système-indépendant (cf. S4).
#
# Rôle (heuristique v1, documentée) :
#   hub        : is_hub_inflated → à arbitrer (degré-gonflé)
#   marqueur   : DE-significatif + tier effecteur (C*) → effet, pas cause
#   cible       : pro-sénescence + driver_score élevé + non-hub → actionnable (KD réduit la sén.)
#   driver     : anti-sénescence + driver_score élevé → causal (confirmation, cf. actionabilité)
#   candidat   : sinon
# =============================================================================
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SENS = {"anti-senescence": "anti", "pro-senescence": "pro"}


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    """Table markdown sans dépendance externe (tabulate non requis)."""
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(r[c]) for c in cols) + " |"
            for _, r in df.iterrows()]
    return "\n".join([head, sep, *rows])


def _role(r, drv_hi: float) -> str:
    if bool(r.get("is_hub_inflated", False)):
        return "hub"
    tier = str(r.get("evidence_tier", ""))
    if bool(r.get("is_de_significant", False)) and tier.startswith("C"):
        return "marqueur"
    hi = r["driver_score"] >= drv_hi
    if r.get("direction") == "pro-senescence" and hi:
        return "cible"
    if r.get("direction") == "anti-senescence" and hi:
        return "driver"
    return "candidat"


def load_simplified(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t")
    padj = 10 ** (-d["de_neglog10_padj"]) if "de_neglog10_padj" in d else np.nan
    out = pd.DataFrame({
        "gene": d["target"],
        "driver_score": d["driver_score"].round(3),
        "force": d.get("canon_amplitude", np.nan).round(3),
        "cos": d.get("canon_cosine", np.nan).round(3),
        "sens": d.get("direction", "").map(lambda x: SENS.get(x, "?")),
        "is_DE": d.get("is_de_significant", False).fillna(False).astype(bool)
                 if "is_de_significant" in d else False,
        "DE_log2fc": d.get("de_log2fc_p4_vs_p16", np.nan).round(2),
        "DE_padj": pd.Series(padj).round(4) if np.ndim(padj) else np.nan,
        "n_DBs": d.get("n_aging_dbs", 0),
        "tier": d.get("evidence_tier", ""),
        "ppi_degree": d.get("target_ppi_degree", np.nan),
        "hub_inflated": d.get("is_hub_inflated", False),
        "robustness": d.get("mean_robustness", np.nan),
        "stability": d.get("mean_stability", np.nan),
        "_interp": d.get("interpretation", ""),
    })
    drv_hi = out["driver_score"].quantile(0.99)   # seuil "élevé" = top 1 %
    out["role"] = [_role(r, drv_hi) for _, r in d.iterrows()]
    return out.sort_values("driver_score", ascending=False).reset_index(drop=True)


def add_concordance(primary: pd.DataFrame, others: dict[str, Path], k: int) -> pd.DataFrame:
    """Ajoute rang par axe + nb d'axes où le gène est top-K (driver système-indép.)."""
    df = primary.copy()
    df["rank"] = np.arange(1, len(df) + 1)
    in_topk = (df["rank"] <= k).astype(int)
    for name, p in others.items():
        o = pd.read_csv(p, sep="\t")[["target", "driver_score"]].copy()
        o["rk"] = o["driver_score"].rank(ascending=False, method="min").astype(int)
        m = df["gene"].map(dict(zip(o["target"], o["rk"])))
        df[f"rank_{name}"] = m
        in_topk += (m <= k).fillna(False).astype(int)
    df["n_axes_topK"] = in_topk
    return df


def write_recap(df: pd.DataFrame, out: Path, axis: str, n_samples: int | None,
                top_n: int, n_axes: int, k: int):
    L = [f"# Récapitulatif drivers — {axis}", ""]
    if n_samples is not None and n_samples < 20:
        L += [f"> ⚠️ **{n_samples} échantillons** : coexpr/SCENIC peu fiables "
              f"(p≫n) ; lire le réseau data-dérivé avec prudence.", ""]
    if n_axes > 1:
        L += [f"`n_axes_topK` = nb d'axes (sur {n_axes}) où le gène est top-{k} "
              f"→ **{int((df['n_axes_topK'] == n_axes).sum())} drivers concordants "
              f"sur tous les axes** (système-indépendants).", ""]
    cols = ["gene", "driver_score", "role", "sens", "force", "cos", "is_DE",
            "n_DBs", "tier"] + (["n_axes_topK"] if n_axes > 1 else [])
    L += ["## Top drivers", "", _md_table(df.head(top_n), cols), ""]
    L += ["## Lecture par gène", ""]
    for _, r in df.head(top_n).iterrows():
        tag = {"driver": "🟢 driver", "cible": "🎯 cible", "marqueur": "🔵 marqueur",
               "hub": "⚪ hub", "candidat": "· candidat"}.get(r["role"], r["role"])
        de = f"DE {r['DE_log2fc']:+.1f}" if pd.notna(r["DE_log2fc"]) else "non-DE"
        L.append(f"- **{r['gene']}** [{tag}, {r['sens']}-sén] "
                 f"score={r['driver_score']} cos={r['cos']} force={r['force']} | "
                 f"{de}, {int(r['n_DBs'])} DB aging, tier {r['tier']}"
                 + (f", concord {int(r['n_axes_topK'])}/{n_axes}" if n_axes > 1 else "")
                 + (f"\n  > {r['_interp']}" if str(r['_interp']).strip() else ""))
    out.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranking", type=Path, required=True, help="cross_seed_gene_ranking.tsv (axe principal)")
    ap.add_argument("--name", default=None, help="label de l'axe principal")
    ap.add_argument("--also", action="append", default=[], metavar="NAME=PATH",
                    help="axe supplémentaire pour la concordance (répétable)")
    ap.add_argument("--decoy", type=Path, default=None, help="decoy_confidence.tsv (target, decoy_confidence)")
    ap.add_argument("--n-samples", type=int, default=None, help="nb d'échantillons du DE (caveat coexpr si <20)")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--concord-k", type=int, default=50)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    axis = args.name or args.ranking.parent.name
    df = load_simplified(args.ranking)
    others = {kv.split("=", 1)[0]: Path(kv.split("=", 1)[1]) for kv in args.also}
    df = add_concordance(df, others, args.concord_k)
    n_axes = 1 + len(others)
    if args.decoy and args.decoy.exists():
        dc = pd.read_csv(args.decoy, sep="\t")
        col = next((c for c in dc.columns if "decoy" in c.lower() and "conf" in c.lower()), None)
        if col:
            df["decoy_conf"] = df["gene"].map(dict(zip(dc["target"], dc[col]))).round(3)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tsv = args.out_dir / "ranking_final.tsv"
    df.drop(columns=["_interp"]).to_csv(tsv, sep="\t", index=False)
    write_recap(df, args.out_dir / "recap.md", axis, args.n_samples,
                args.top_n, n_axes, args.concord_k)
    print(f"[summarize] {len(df)} gènes → {tsv}")
    print(f"[summarize] recap → {args.out_dir / 'recap.md'}")
    if n_axes > 1:
        print(f"[summarize] {int((df['n_axes_topK'] == n_axes).sum())} drivers "
              f"concordants sur {n_axes} axes (top-{args.concord_k})")


if __name__ == "__main__":
    main()
