#!/usr/bin/env python3
# =============================================================================
# probe.py — Probe linéaire diagnostique V6 (Tête B), stop-gradient par
#            construction (lit un embedding figé ; ne rétropropage jamais).
# =============================================================================
# Mesure COMBIEN les représentations topologiques (μ post-encoder) capturent de
# la séparation portée par chaque couche omique (RNA bulk, protéomique, …),
# SANS contaminer l'encoder. C'est le mécanisme "stop-grad probe" de v6.md
# (§design probe) : on entraîne une régression linéaire μ → cible_omique en
# cross-validation et on rapporte l'AUROC (binaire) / R² (continu).
#
# Lecture :
#   AUROC haute  ⇒ la topologie suffit à prédire cette couche (axe DE
#                  potentiellement redondant au readout) ;
#   AUROC ~0.5   ⇒ la topologie ne capte pas ce signal ⇒ il FAUT l'axe DE.
#
# ⚠️ Sur un encoder ENCORE CIRCULAIRE (V5.4.1 : cell_group edges +
#    variance_across_groups), l'AUROC est CONTAMINÉE — c'est un test de
#    plomberie. Le chiffre défendable attend l'encoder topologie-seule (S3).
#
# Cibles construites par couche omique (présence dépend des fichiers fournis) :
#   <layer>_is_DE  : binaire  |log_fc|>τ & padj<seuil      → AUROC (clf)
#   <layer>_sign   : binaire  sign(log_fc) parmi les DE     → AUROC (clf, sens)
#   <layer>_logfc  : continu  log_fc (gènes mesurés)        → R²    (reg)
#
# stop-grad : trivialement satisfait (entrée = CSV figé). Aucun chemin V5 touché.
# =============================================================================
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# --- loaders V6 : src/data sur le path (imports relatifs internes) -----------
_SRC = Path(__file__).resolve().parents[2]          # …/src
_DATA = _SRC / "data"
if str(_DATA) not in sys.path:
    sys.path.insert(0, str(_DATA))


def load_embedding(emb_path: Path) -> pd.DataFrame:
    """Charge gene_embeddings_vgae.csv → DataFrame indexé par gène (n × d)."""
    df = pd.read_csv(emb_path, index_col=0)
    df.index = df.index.astype(str)
    # colonnes = dimensions latentes (entiers en str) ; on garde tout numérique
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="any")
    return df


def _build_targets(de_df: pd.DataFrame, layer: str,
                   logfc_thresh: float, padj_thresh: float) -> dict:
    """À partir d'une table DE canonique (gene_symbol, log_fc, padj), construit
    les cibles per-gene. Retourne {nom: (Series indexée gène, kind)}."""
    d = de_df.copy()
    d = d[d["gene_symbol"].astype(str).str.len() > 0]
    d = d.dropna(subset=["log_fc"])
    d = d.groupby(d["gene_symbol"].astype(str))["log_fc"].mean()  # dédup → moyenne
    base = de_df.dropna(subset=["log_fc"]).copy()
    gsym = base["gene_symbol"].astype(str)
    # Significativité : padj si présent, sinon fallback pvalue brute (ex. protéo
    # serie B = pvalue binomiale sans ajustement multiple).
    sig_col = "padj"
    if base["padj"].notna().sum() == 0 and "pvalue" in base.columns:
        sig_col = "pvalue"
        warnings.warn(f"{layer}: padj absent → fallback significativité sur pvalue.")
    sig = base.groupby(gsym)[sig_col].min()
    out: dict = {}
    # is_DE : significatif ET amplitude (négatifs = mesurés-mais-pas-DE)
    is_de = ((d.abs() > logfc_thresh) & (sig.reindex(d.index) < padj_thresh)).astype(int)
    out[f"{layer}_is_DE"] = (is_de, "clf")
    # sign : sens parmi les DE seulement
    de_genes = is_de[is_de == 1].index
    sign = (d.loc[de_genes] > 0).astype(int)
    out[f"{layer}_sign"] = (sign, "clf")
    # logfc continu : tous les gènes mesurés
    out[f"{layer}_logfc"] = (d, "reg")
    return out


def _run_probe(X: np.ndarray, y: np.ndarray, kind: str,
               n_splits: int, seed: int) -> dict:
    """Régression linéaire CV μ→cible. Retourne {score, score_std, metric, n,
    n_pos, prevalence, status}."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

    n = len(y)
    res = {"n": n, "n_pos": "", "prevalence": "", "metric": "", "score": np.nan,
           "score_std": np.nan, "status": "ok"}
    if kind == "clf":
        n_pos = int(y.sum())
        res["n_pos"] = n_pos
        res["prevalence"] = round(n_pos / n, 4) if n else np.nan
        res["metric"] = "auroc"
        if n_pos < n_splits or (n - n_pos) < n_splits or n_pos == 0:
            res["status"] = f"skip (n_pos={n_pos}, n_neg={n-n_pos} < n_splits={n_splits})"
            return res
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        sc = cross_val_score(clf, X, y, scoring="roc_auc", cv=cv)
    else:  # reg
        res["metric"] = "r2"
        if n < n_splits + 1:
            res["status"] = f"skip (n={n} < n_splits+1)"
            return res
        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        sc = cross_val_score(reg, X, y, scoring="r2", cv=cv)
    res["score"] = round(float(np.mean(sc)), 4)
    res["score_std"] = round(float(np.std(sc)), 4)
    return res


def run(emb_path: Path, rna_de: Path | None, proteo: Path | None,
        out_path: Path, *, rna_label: str, proteo_label: str,
        logfc_thresh: float, padj_thresh: float,
        n_splits: int, seed: int) -> pd.DataFrame:
    emb = load_embedding(emb_path)
    print(f"[probe] embedding : {emb.shape[0]} gènes × {emb.shape[1]} dims "
          f"({emb_path})")

    layers: dict = {}
    if rna_de is not None:
        from loaders.bulk_rna import load_bulk_rna_de
        de = load_bulk_rna_de(rna_de, condition_label=rna_label)
        layers.update(_build_targets(de, "rna", logfc_thresh, padj_thresh))
        print(f"[probe] RNA bulk : {de['gene_symbol'].astype(str).str.len().gt(0).sum()} "
              f"gènes annotés ({rna_de.name})")
    if proteo is not None:
        from loaders.proteomics import load_proteomics_de
        pr = load_proteomics_de(proteo, condition_label=proteo_label)
        layers.update(_build_targets(pr, "proteo", logfc_thresh, padj_thresh))
        print(f"[probe] protéo : {pr['gene_symbol'].astype(str).str.len().gt(0).sum()} "
              f"gènes annotés ({proteo.name})")
    if not layers:
        raise SystemExit("Aucune couche omique fournie (--rna-de / --proteo).")

    rows = []
    for name, (target, kind) in layers.items():
        common = emb.index.intersection(target.dropna().index)
        if len(common) < n_splits + 1:
            rows.append({"target": name, "kind": kind, "n_common": len(common),
                         "status": f"skip (n_common={len(common)})"})
            print(f"  - {name:16s} : skip (n_common={len(common)})")
            continue
        X = emb.loc[common].to_numpy()
        y = target.loc[common].to_numpy()
        r = _run_probe(X, y, kind, n_splits, seed)
        row = {"target": name, "kind": kind, "n_common": len(common), **r}
        rows.append(row)
        sc = f"{r['metric']}={r['score']}±{r['score_std']}" if r["status"] == "ok" else r["status"]
        extra = f" (n_pos={r['n_pos']})" if kind == "clf" else ""
        print(f"  - {name:16s} : {sc}{extra}")

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)
    print(f"[probe] écrit → {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", type=Path, required=True,
                    help="gene_embeddings_vgae.csv (μ topology-only post-encoder)")
    ap.add_argument("--rna-de", type=Path, default=None,
                    help="DE bulk RNA (schéma DESeq2 logFC/padj)")
    ap.add_argument("--proteo", type=Path, default=None,
                    help="table protéomique différentielle")
    ap.add_argument("--rna-label", default="sen_vs_pro",
                    help="condition_label <A>_vs_<B> pour le loader RNA")
    ap.add_argument("--proteo-label", default="mutant_vs_wt",
                    help="condition_label <A>_vs_<B> pour le loader protéo")
    ap.add_argument("--logfc-thresh", type=float, default=1.0,
                    help="|log_fc| seuil pour la cible is_DE")
    ap.add_argument("--padj-thresh", type=float, default=0.1,
                    help="padj seuil pour la cible is_DE")
    ap.add_argument("--n-splits", type=int, default=5, help="folds CV")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True,
                    help="probe_diagnostic.tsv")
    args = ap.parse_args()
    if args.rna_de is None and args.proteo is None:
        ap.error("fournir au moins --rna-de ou --proteo.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run(args.emb, args.rna_de, args.proteo, args.out,
            rna_label=args.rna_label, proteo_label=args.proteo_label,
            logfc_thresh=args.logfc_thresh, padj_thresh=args.padj_thresh,
            n_splits=args.n_splits, seed=args.seed)


if __name__ == "__main__":
    main()
