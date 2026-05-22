#!/usr/bin/env python3
"""build_diff_coexpr.py — Coexpression différentielle P4 vs P16 (V4.2, option A).

Contexte
--------
Jusqu'en V4.1, GRNBoost2 était inféré sur les cellules **P16 uniquement**
(`output/pyscenic/adjacencies.csv`). Cela introduit un biais de
circularité : les arêtes coexpression encodent « ce qui co-varie dans
l'état sénescent », partiellement colinéaire avec l'axe P4→P16 qu'on
cherche à mesurer (cf. §14bis.1-2 du rapport).

V4.2 — option A : on infère GRNBoost2 SÉPARÉMENT sur P4 et P16, puis on
construit un edge_attr enrichi `[imp_p4, imp_p16, delta, cat_shared,
cat_p4, cat_p16]` (edge_dim=6) qui laisse le modèle apprendre quoi
pondérer plutôt que de figer un prior P16-only.

Workflow (3 étapes — GRNBoost2 doit tourner sur le cluster)
-----------------------------------------------------------
1. `extract-matrices` (LOCAL) : depuis
   `data/gnn_data/merged_P4_P16_normalized.csv`, produit
   `expr_matrix_P4.csv` et `expr_matrix_P16.csv` avec le MÊME jeu de
   gènes (top-N HVG calculés sur toutes les cellules → comparabilité
   stricte du delta). Sortie : `data/pyscenic/diff_coexpr/`.

2. GRNBoost2 (CLUSTER) — hors de ce script (arboreto non installé en
   local). Commandes affichées par `print-grn-cmds`.

3. `merge-adjacencies` (LOCAL) : fusionne les réseaux COMPLETS P4+P16
   sur (TF,target), calcule delta + imp_max, PUIS élague (per-target
   top-K sur imp_max par défaut) → `coexpr_diff.tsv` (option A,
   edge_dim=6). Merge-first : évite l'explosion par union de top-K
   disjoints (bug V4.2 → AUC 0.97→0.65 ; fix §14bis.6terdecies).

Usage
-----
    python src/preprocess/build_diff_coexpr.py extract-matrices \\
        --merged data/gnn_data/merged_P4_P16_normalized.csv \\
        --n-hvg 5000 \\
        --out-dir data/pyscenic/diff_coexpr

    python src/preprocess/build_diff_coexpr.py print-grn-cmds \\
        --out-dir data/pyscenic/diff_coexpr

    python src/preprocess/build_diff_coexpr.py merge-adjacencies \\
        --adj-p4  data/pyscenic/diff_coexpr/adjacencies_P4.csv \\
        --adj-p16 data/pyscenic/diff_coexpr/adjacencies_P16.csv \\
        --prune-mode per-target-topk --per-target-k 5 \\
        --out data/pyscenic/diff_coexpr/coexpr_diff.tsv
    # merge-first : réseaux COMPLETS fusionnés PUIS élagage top-K sur
    # imp_max=max(imp_P4,imp_P16) → K arêtes/cible exactement (pas
    # d'explosion par union de top-K disjoints). Cf. §14bis.6terdecies.

Référence : Tesson 2010 *Bioinformatics* DiffCoEx ; Anglani 2014 *BMC
Syst Biol* (coexpression différentielle WGCNA).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

META_COLS = ("passage", "cluster_P16", "cell_state")


# ---------------------------------------------------------------------------
# Étape 1 — extract-matrices
# ---------------------------------------------------------------------------
def cmd_extract_matrices(args: argparse.Namespace) -> None:
    merged_path = Path(args.merged)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    universe = args.gene_universe
    chunksize = 500

    # En-tête : gènes mesurés (= post-QC, c'est déjà la matrice
    # normalisée filtrée) + nom de la colonne d'index.
    with open(merged_path) as _fh:
        _header = _fh.readline().rstrip("\n").split(",")
    index_name = _header[0].strip().strip('"')
    all_genes = [c.strip().strip('"') for c in _header[1:]
                 if c.strip().strip('"') not in META_COLS]
    print(f"[extract] {merged_path.name} : {len(all_genes)} gènes "
          f"mesurés (post-QC) | mode --gene-universe={universe}")

    # ----- Sélection du jeu de gènes cible selon le mode -----
    if universe == "all":
        selected = list(all_genes)
        sel_label = "all-QC"
        print(f"[extract] mode all : {len(selected)} gènes (tous les QC)")
    elif universe == "graph":
        if not args.graph_genes:
            sys.exit("[extract] --gene-universe graph exige --graph-genes "
                     "<fichier> (cross_seed_gene_ranking.tsv ou "
                     "gene_ranking_vgae.csv ou 1 gène/ligne).")
        gp = Path(args.graph_genes)
        if gp.suffix in (".tsv", ".csv"):
            sep = "\t" if gp.suffix == ".tsv" else ","
            gdf = pd.read_csv(gp, sep=sep)
            col = ("target" if "target" in gdf.columns
                   else gdf.columns[0])
            graph_set = set(gdf[col].astype(str))
        else:
            graph_set = set(x.strip() for x in open(gp) if x.strip())
        selected = [g for g in all_genes if g in graph_set]
        sel_label = "graph"
        print(f"[extract] mode graph : {len(graph_set)} gènes graphe → "
              f"{len(selected)} ∩ mesurés "
              f"({100*len(selected)/max(1,len(graph_set)):.0f}% du graphe)")
    else:  # hvg
        # Variance incrémentale (Welford-like) par chunks pour ranker.
        n_total = 0
        sum_x = np.zeros(len(all_genes), dtype=np.float64)
        sum_x2 = np.zeros(len(all_genes), dtype=np.float64)
        reader = pd.read_csv(merged_path, index_col=0, chunksize=chunksize)
        for ci, chunk in enumerate(reader):
            if "passage" not in chunk.columns:
                sys.exit("[extract] colonne 'passage' absente — abort.")
            vals = chunk[all_genes].to_numpy(dtype=np.float64)
            sum_x += vals.sum(axis=0)
            sum_x2 += (vals ** 2).sum(axis=0)
            n_total += len(chunk)
            if (ci + 1) % 10 == 0:
                print(f"[extract]   ... {n_total} cellules lues (variance)")
        mean = sum_x / n_total
        var = (sum_x2 / n_total) - mean ** 2
        var_series = pd.Series(var, index=all_genes).sort_values(ascending=False)
        n_hvg = min(args.n_hvg, len(var_series))
        selected = var_series.head(n_hvg).index.tolist()
        sel_label = f"hvg{n_hvg}"
        print(f"[extract] mode hvg : {n_hvg} HVG (variance globale P4+P16)")

    # ----- Passe d'extraction : colonnes sélectionnées par condition -----
    keep = {index_name, "passage"} | set(selected)
    buffers = {"P4": [], "P16": []}
    reader2 = pd.read_csv(merged_path, index_col=0,
                          usecols=lambda c: c in keep,
                          chunksize=chunksize)
    for chunk in reader2:
        for cond in ("P4", "P16"):
            sub = chunk[chunk["passage"] == cond]
            if not sub.empty:
                buffers[cond].append(sub[selected])

    for cond in ("P4", "P16"):
        if not buffers[cond]:
            print(f"[extract] [warn] aucune cellule {cond} — skip")
            continue
        mat = pd.concat(buffers[cond], axis=0)
        out_path = out_dir / f"expr_matrix_{cond}.csv"
        mat.to_csv(out_path)
        print(f"[extract] {cond}: {mat.shape[0]} cellules × {mat.shape[1]} "
              f"gènes → {out_path}")

    # Traçabilité : liste des gènes sélectionnés (nom selon le mode)
    genes_file = out_dir / f"genes_{sel_label}.txt"
    genes_file.write_text("\n".join(selected))
    # Compat : hvg_genes.txt conservé comme alias en mode hvg
    if universe == "hvg":
        (out_dir / "hvg_genes.txt").write_text("\n".join(selected))
    print(f"[extract] gene list ({len(selected)}) → {genes_file}")
    print(f"[extract] OK. Étape suivante : print-grn-cmds")


# ---------------------------------------------------------------------------
# Étape 2 — print-grn-cmds (les commandes GRNBoost2 à lancer sur le cluster)
# ---------------------------------------------------------------------------
def cmd_print_grn_cmds(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    tf_list = "data/pyscenic/scenic_refs/allTFs_hg38.txt"
    print("# ============================================================")
    print("# GRNBoost2 P4 / P16 — réimplémentation sklearn (sans arboreto)")
    print("# ============================================================")
    print("# arboreto est cassé sur l'env GLiCID (dask ≥ 2025.1 a retiré")
    print("# l'API legacy dask.dataframe → NotImplementedError). On")
    print("# utilise la réimplémentation locale (sklearn pur, Moerman")
    print("# 2019, mêmes hyperparams SGBM). Seed=42 identique P4/P16.")
    print("# Préférer : bash scripts/run_diff_coexpr.sh (SLURM array).")
    print()
    for cond in ("P4", "P16"):
        print(f"python src/build_diff_coexpr.py grnboost2-local \\")
        print(f"    --expr '{out_dir}/expr_matrix_{cond}.csv' \\")
        print(f"    --tf-list '{tf_list}' \\")
        print(f"    --out '{out_dir}/adjacencies_{cond}.csv' \\")
        print(f"    --n-jobs 8 --seed 42")
        print()
    print("# Puis : merge-adjacencies (local)")


# ---------------------------------------------------------------------------
# Étape 2bis — grnboost2-local : GRNBoost2 réimplémenté en sklearn pur
# ---------------------------------------------------------------------------
# Pourquoi : arboreto est cassé sur l'env GLiCID (dask ≥ 2025.1 a retiré
# l'API legacy dask.dataframe que arboreto importe → NotImplementedError
# "The legacy implementation is no longer supported"). Plutôt que de
# pinner dask (fragile, risque de casser torch/pyg dans l'env `gnn`),
# on réimplémente l'algo GRNBoost2 (Moerman 2019) directement :
#   pour chaque gène cible g, régresser expr[g] sur expr[TFs] via
#   Stochastic Gradient Boosting (mêmes hyperparams que arboreto
#   SGBM_KWARGS) ; feature_importances_ → poids d'arête (TF, g, imp).
# Zéro dask, zéro arboreto. Parallélisé sur les gènes cibles (joblib).
# Référence : Moerman et al. 2019 Bioinformatics (GRNBoost2) ; Aibar
# 2017 Nat Methods (SCENIC). Hyperparams identiques à arboreto :
#   learning_rate=0.01, n_estimators=500, max_features=0.1,
#   subsample=0.9 + early stopping (n_iter_no_change=25).

# Hyperparamètres SGBM identiques à arboreto.core.SGBM_KWARGS
_SGBM_KWARGS = dict(
    learning_rate=0.01,
    n_estimators=500,
    max_features=0.1,
    subsample=0.9,
)
_EARLY_STOP_WINDOW = 25


def _fit_one_target(target, tf_names, X_tf, y, seed):
    """Régresse y (= expr du gène cible) sur X_tf (= expr des TFs).

    Exclut le gène cible de ses propres prédicteurs (un TF ne se
    régule pas lui-même dans GRNBoost2). Retourne une liste de
    (TF, target, importance) pour les importances > 0.
    """
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor

    # Masque : retirer le gène cible des features s'il est lui-même un TF
    keep = [i for i, tf in enumerate(tf_names) if tf != target]
    if not keep:
        return []
    X = X_tf[:, keep]
    feats = [tf_names[i] for i in keep]

    reg = GradientBoostingRegressor(
        random_state=seed,
        n_iter_no_change=_EARLY_STOP_WINDOW,
        validation_fraction=0.1,
        tol=1e-4,
        **_SGBM_KWARGS,
    )
    reg.fit(X, y)
    imp = reg.feature_importances_
    out = []
    for f, w in zip(feats, imp):
        if w > 0.0:
            out.append((f, target, float(w)))
    return out


def cmd_grnboost2_local(args: argparse.Namespace) -> None:
    """GRNBoost2 réimplémenté (sklearn pur, sans dask/arboreto)."""
    import numpy as np
    from joblib import Parallel, delayed

    expr_path = Path(args.expr)
    tf_path = Path(args.tf_list)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[grn-local] lecture {expr_path}")
    expr = pd.read_csv(expr_path, index_col=0)
    cols = list(expr.columns)
    col_set = set(cols)

    raw_tfs = [t.strip() for t in open(tf_path) if t.strip()]
    # Normalisation underscore↔dash (cf. scenic_from_r.py:121-130)
    tf_names = []
    for t in raw_tfs:
        if t in col_set:
            tf_names.append(t)
        elif t.replace("_", "-") in col_set:
            tf_names.append(t.replace("_", "-"))
    tf_names = sorted(set(tf_names))
    print(f"[grn-local] {expr.shape[0]} cellules × {expr.shape[1]} gènes ; "
          f"{len(tf_names)}/{len(raw_tfs)} TFs mappés")
    if len(tf_names) < 10:
        sys.exit(f"[grn-local] seulement {len(tf_names)} TFs mappés — "
                 f"vérifier la nomenclature ({cols[:5]} ...). Abort.")

    X_tf = expr[tf_names].to_numpy(dtype=np.float32)
    targets = cols  # tous les gènes sont des cibles potentielles
    n_jobs = args.n_jobs
    print(f"[grn-local] régression de {len(targets)} cibles "
          f"(GBM {_SGBM_KWARGS}, early-stop={_EARLY_STOP_WINDOW}) "
          f"sur {n_jobs} jobs…")

    results = Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(
        delayed(_fit_one_target)(
            tgt, tf_names, X_tf,
            expr[tgt].to_numpy(dtype=np.float32), args.seed,
        )
        for tgt in targets
    )

    rows = [r for sub in results for r in sub]
    adj = pd.DataFrame(rows, columns=["TF", "target", "importance"])
    adj = adj.sort_values("importance", ascending=False).reset_index(drop=True)
    adj.to_csv(out_path, index=False)
    print(f"[grn-local] {len(adj)} arêtes (TF,target,importance) → {out_path}")
    print(f"[grn-local] top 3 :\n{adj.head(3).to_string(index=False)}")


# ---------------------------------------------------------------------------
# Étape 3 — merge-adjacencies (option A)
# ---------------------------------------------------------------------------
def cmd_merge_adjacencies(args: argparse.Namespace) -> None:
    p4_path = Path(args.adj_p4)
    p16_path = Path(args.adj_p16)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for p in (p4_path, p16_path):
        if not p.exists():
            sys.exit(f"[merge] introuvable : {p} — lancer GRNBoost2 d'abord "
                     f"(print-grn-cmds).")

    df4 = pd.read_csv(p4_path)[["TF", "target", "importance"]]
    df16 = pd.read_csv(p16_path)[["TF", "target", "importance"]]
    print(f"[merge] P4  : {len(df4)} paires | P16 : {len(df16)} paires")

    # ── MERGE D'ABORD, ÉLAGAGE APRÈS (fix V4.2.1) ────────────────────────
    # BUG V4.2 (run 2026-05-19) : élaguer CHAQUE condition séparément puis
    # outer-merge explosait (P4-topK et P16-topK quasi-disjoints car les
    # importances du top-K sont proches → instables entre conditions →
    # 96% d'arêtes condition-spécifiques, coexpr=601k arêtes=2×PPI →
    # AUC VGAE 0.97→0.65). Fix : on fusionne les réseaux COMPLETS, on
    # calcule imp_max = max(imp_p4_norm, imp_p16_norm) PAR PAIRE, puis on
    # élague sur imp_max → exactement K arêtes/cible (couverture +
    # parcimonie + delta préservé), pas K×2 quasi-disjoints.
    mode = args.prune_mode
    K = args.per_target_k
    q = args.top_quantile

    # 1. Normalisation min-max PAR CONDITION sur le réseau COMPLET (avant
    #    merge — GRNBoost2 importance non calibrée entre runs).
    for d in (df4, df16):
        mx = d["importance"].max()
        d["imp_norm"] = d["importance"] / mx if mx > 0 else 0.0

    # 2. Outer-merge des réseaux COMPLETS sur (TF, target).
    merged = df4.merge(df16, on=["TF", "target"], how="outer",
                       suffixes=("_p4", "_p16"))
    for c in ("importance_p4", "importance_p16",
              "imp_norm_p4", "imp_norm_p16"):
        merged[c] = merged[c].fillna(0.0)
    merged = merged.rename(columns={"imp_norm_p4": "importance_p4_norm",
                                    "imp_norm_p16": "importance_p16_norm"})
    merged["delta"] = (merged["importance_p16_norm"]
                       - merged["importance_p4_norm"])
    # Force d'une arête différentielle = max d'importance inter-condition.
    merged["imp_max"] = merged[["importance_p4_norm",
                                "importance_p16_norm"]].max(axis=1)
    print(f"[merge] réseaux complets fusionnés : {len(merged)} paires "
          f"(P4={len(df4)}, P16={len(df16)})")

    # 2bis. Floor LAXISTE sur imp_max (optionnel, --min-imax-quantile).
    # But : éliminer les gènes SANS régulateur réel (régression-bruit,
    # imp_max tous faibles) que per-target-topk garderait quand même à
    # K arêtes de bruit. Floor = quantile(imp_max) — laxiste (0.5 =
    # médiane recommandée) vs q0.98/0.995 hub-dominé. Appliqué AVANT
    # le top-K : un gène dont TOUS les imp_max sont sous le floor →
    # 0 arête (droppé, souhaité) ; un vrai gène garde son top-K.
    fq = args.min_imax_quantile
    if fq and fq > 0.0:
        floor = merged["imp_max"].quantile(fq)
        n_before = len(merged)
        tgt_before = merged["target"].nunique()
        merged = merged[merged["imp_max"] >= floor].copy()
        print(f"[merge] floor laxiste imp_max ≥ q{fq} ({floor:.4g}) : "
              f"{n_before} → {len(merged)} paires ; cibles "
              f"{tgt_before} → {merged['target'].nunique()} "
              f"(gènes sans signal droppés)")

    # 3. Élagage sur imp_max (force inter-condition), APRÈS le merge.
    if mode == "global-quantile":
        thr = merged["imp_max"].quantile(q)
        merged = merged[merged["imp_max"] >= thr].copy()
        why = f"global-q{q} sur imp_max (seuil={thr:.4g})"
    elif mode == "per-target-topk":
        merged = (merged.sort_values("imp_max", ascending=False)
                        .groupby("target", sort=False).head(K).copy())
        why = f"per-target top-{K} sur imp_max"
    else:  # hybrid
        thr = merged["imp_max"].quantile(q)
        g = merged[merged["imp_max"] >= thr]
        pt = (merged.sort_values("imp_max", ascending=False)
                    .groupby("target", sort=False).head(K))
        merged = (pd.concat([g, pt])
                    .drop_duplicates(["TF", "target"]).copy())
        why = f"hybrid (global-q{q} ∪ per-target top-{K}) sur imp_max"
    print(f"[merge] élagage {why} : → {len(merged)} arêtes "
          f"({merged.target.nunique()} cibles, {merged.TF.nunique()} TFs)")

    def categorize(row) -> str:
        p4 = row["importance_p4"] > 0
        p16 = row["importance_p16"] > 0
        if p4 and p16:
            return "shared"
        if p16:
            return "p16_specific"
        return "p4_specific"

    merged["category"] = merged.apply(categorize, axis=1)
    # One-hot pour l'edge_attr (option A : edge_dim=6)
    merged["cat_shared"] = (merged["category"] == "shared").astype(float)
    merged["cat_p4"] = (merged["category"] == "p4_specific").astype(float)
    merged["cat_p16"] = (merged["category"] == "p16_specific").astype(float)

    cols = [
        "TF", "target",
        "importance_p4_norm", "importance_p16_norm", "delta",
        "cat_shared", "cat_p4", "cat_p16",
        "category", "importance_p4", "importance_p16",
    ]
    merged[cols].to_csv(out_path, sep="\t", index=False)

    print(f"[merge] {len(merged)} arêtes → {out_path}")
    print(f"[merge] catégories : "
          f"{merged['category'].value_counts().to_dict()}")
    print(f"[merge] delta : médiane={merged['delta'].median():.3f} "
          f"P16-biais={(merged['delta']>0).mean()*100:.0f}% des arêtes")
    print(f"[merge] colonnes edge_attr (option A, edge_dim=6) : "
          f"importance_p4_norm, importance_p16_norm, delta, "
          f"cat_shared, cat_p4, cat_p16")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract-matrices",
                        help="merged → expr_matrix_P4/P16 (jeu de gènes "
                             "commun ; cf. --gene-universe)")
    pe.add_argument("--merged",
                    default="data/gnn_data/merged_P4_P16_normalized.csv")
    pe.add_argument("--gene-universe", choices=["hvg", "graph", "all"],
                    default="hvg",
                    help="hvg = top-N variance (défaut, rétro-compat) ; "
                         "graph = ∩ univers VGAE (recommandé V4.2, "
                         "couverture coexpr 100%%, exige --graph-genes) ; "
                         "all = tous les gènes QC du merged (~15779).")
    pe.add_argument("--graph-genes", default=None,
                    help="Mode graph : fichier listant les gènes du "
                         "graphe VGAE (cross_seed_gene_ranking.tsv col "
                         "'target', ou gene_ranking_vgae.csv, ou 1 "
                         "gène/ligne).")
    pe.add_argument("--n-hvg", type=int, default=5000,
                    help="Mode hvg uniquement : nombre de HVG.")
    pe.add_argument("--out-dir", default="data/pyscenic/diff_coexpr")
    pe.set_defaults(func=cmd_extract_matrices)

    pg = sub.add_parser("print-grn-cmds",
                        help="Affiche les commandes GRNBoost2 (cluster)")
    pg.add_argument("--out-dir", default="data/pyscenic/diff_coexpr")
    pg.set_defaults(func=cmd_print_grn_cmds)

    pl = sub.add_parser(
        "grnboost2-local",
        help="GRNBoost2 réimplémenté sklearn pur (sans dask/arboreto). "
             "Robuste sur env où arboreto est cassé (dask ≥ 2025.1).")
    pl.add_argument("--expr", required=True,
                    help="expr_matrix_{P4,P16}.csv (cellules × gènes)")
    pl.add_argument("--tf-list",
                    default="data/pyscenic/scenic_refs/allTFs_hg38.txt")
    pl.add_argument("--out", required=True,
                    help="adjacencies_{P4,P16}.csv de sortie")
    pl.add_argument("--n-jobs", type=int, default=8,
                    help="joblib n_jobs (parallélisme sur les cibles)")
    pl.add_argument("--seed", type=int, default=42,
                    help="random_state GBM (identique P4/P16 → "
                         "delta non confondu)")
    pl.set_defaults(func=cmd_grnboost2_local)

    pm = sub.add_parser("merge-adjacencies",
                        help="adjacencies_P4/P16 → coexpr_diff.tsv (option A)")
    pm.add_argument("--adj-p4", required=True)
    pm.add_argument("--adj-p16", required=True)
    pm.add_argument("--prune-mode",
                    choices=["per-target-topk", "global-quantile", "hybrid"],
                    default="per-target-topk",
                    help="per-target-topk (DÉFAUT V4.2 : top-K régulateurs "
                         "par cible → 100%% couverture, hubs déconcentrés, "
                         "= élagage SCENIC) ; global-quantile (legacy, "
                         "dominé par hubs → affame ASNS/IL6/IL1B/DDIT3) ; "
                         "hybrid (union des deux).")
    pm.add_argument("--per-target-k", type=int, default=5,
                    help="K régulateurs gardés par cible, sur imp_max "
                         "(modes per-target-topk / hybrid). Défaut 5 : "
                         "coexpr ≈ 38%% du PPI en univers graphe (K=10 "
                         "≈ 76%%, risque de noyer la topologie — cf. "
                         "§14bis.6terdecies). Edge count déterministe = "
                         "n_cibles × K.")
    pm.add_argument("--top-quantile", type=float, default=0.98,
                    help="Quantile (modes global-quantile / hybrid).")
    pm.add_argument("--min-imax-quantile", type=float, default=0.0,
                    help="Floor LAXISTE optionnel : drop les paires dont "
                         "imp_max < quantile(imp_max, q) AVANT le top-K. "
                         "0.0 = off (défaut). 0.5 (médiane) recommandé : "
                         "élimine les gènes sans régulateur réel "
                         "(régression-bruit) que per-target-topk "
                         "garderait sinon à K arêtes de bruit. "
                         "Cf. §14bis.6terdecies.")
    pm.add_argument("--out",
                    default="data/pyscenic/diff_coexpr/coexpr_diff.tsv")
    pm.set_defaults(func=cmd_merge_adjacencies)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
