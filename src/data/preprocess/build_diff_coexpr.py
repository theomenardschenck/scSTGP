#!/usr/bin/env python3
"""build_diff_coexpr.py — Coexpression différentielle P4 vs P16 (V4.2/V4.3).

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

V4.3 — comparaison de méthodes d'inférence et d'élagage. Le script
fournit maintenant **4 méthodes GRN** (toutes émettent le même format
`(TF, target, importance)`) et **4 modes d'élagage** :

  GRN     : `grnboost2-local` (sklearn, défaut V4.2)
          : `grnboost2-diff` (arboreto canonique, cf. scenic_from_r.py)
          : `correlation`    (Pearson/Spearman TF×target)
          : `mutual-info`    (sklearn mutual_info_regression)

  élagage : `per-target-topk`   (DÉFAUT, = élagage SCENIC, K régul./cible)
          : `global-quantile`   (legacy, hub-dominé — baseline négative)
          : `mutual-rank`       (Obayashi 2018, COXPRESdb, sym. + débiaisé hub)
          : `z-score`           (par cible, μ + n·σ — adaptatif)

Convention de nommage (V4.3) :
    adjacencies_<COND>.<METHOD>.csv      # COND ∈ {P4,P16}, METHOD ∈ {sklearn,arboreto,corr,mi}
    coexpr_diff.<METHOD>.<PRUNE>.tsv     # PRUNE ∈ {topk,quantile,mr,zscore}

Compat. ascendante : sans `--method`, les sorties gardent leur nom
historique (`adjacencies_<COND>.csv`, `coexpr_diff.tsv`).

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
def cmd_prep_matrices(args: argparse.Namespace) -> None:
    """V6 — adaptateur dataset quelconque → expr_<groupe>.csv (échantillons ×
    gènes), format attendu par grnboost2-local. Lit une matrice (counts/FPKM,
    genes×samples OU samples×genes, sniff sép.) + un metadata (sample→groupe),
    transpose si besoin, et écrit une matrice par groupe.

    metadata : TSV/CSV, 1re col = échantillon (= colonnes/lignes de la matrice),
    2e col = groupe (ou un titre type 'HUVEC_C1_d5' → on garde tel quel ; le
    découpage en états fin reste au readout, ici on groupe par la valeur brute).
    """
    mp = Path(args.matrix)
    with open(mp) as fh:
        first = fh.readline()
    sep = "\t" if first.count("\t") >= first.count(",") else ","
    df = pd.read_csv(mp, sep=sep)
    md = pd.read_csv(args.metadata, sep="\t" if str(args.metadata).endswith(".tsv") else ",")
    skey, sval = md.columns[0], md.columns[args.group_col]
    grp = dict(zip(md[skey].astype(str), md[sval].astype(str)))
    samples_md = set(grp)

    # Colonne gène : --gene-col, sinon 1re colonne reconnue, sinon col[0].
    gene_cands = ["Tracking_id", "hgnc_symbol", "gene_symbol", "gene_name",
                  "symbol", "gene", "GeneName"]
    gcol = (args.gene_col or next((c for c in gene_cands if c in df.columns),
                                  df.columns[0]))
    # Oriente en samples × genes, piloté par les échantillons du metadata.
    cols_are_samples = len(samples_md & set(df.columns.astype(str)))
    if cols_are_samples >= 2:                      # genes × samples → transpose
        gi = df.set_index(gcol)
        gi = gi[~gi.index.astype(str).duplicated()]     # dédup gènes
        keep = [c for c in gi.columns.astype(str) if c in samples_md]
        df = gi[keep].apply(pd.to_numeric, errors="coerce").T   # samples × genes
    else:                                          # samples × genes déjà
        m = df.set_index(df.columns[0]).apply(pd.to_numeric, errors="coerce")
        df = m.loc[[i for i in m.index.astype(str) if i in samples_md]]
    print(f"[prep] {df.shape[0]} échantillons × {df.shape[1]} gènes "
          f"(gène='{gcol}', orient={args.orient})")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for g in sorted(set(grp.values())):
        samples = [s for s in df.index.astype(str) if grp.get(s) == g]
        if not samples:
            continue
        sub = df.loc[samples].dropna(axis=1, how="any")
        out = out_dir / f"expr_{g}.csv"
        sub.to_csv(out)
        written.append((g, len(samples), sub.shape[1], out))
        print(f"[prep] groupe {g}: {len(samples)} échantillons × {sub.shape[1]} gènes → {out}")
    if not written:
        sys.exit("[prep] aucun groupe apparié entre matrice et metadata "
                 "(vérifier que les noms d'échantillons correspondent).")
    # matrice POOLÉE (tous les échantillons appariés) — fallback non-différentiel
    # quand un groupe est trop petit pour GRNBoost2 (GBM crash à n<3).
    pooled = df.loc[[s for s in df.index.astype(str) if s in grp]].dropna(axis=1, how="any")
    pooled.to_csv(out_dir / "expr_all.csv")
    print(f"[prep] poolé : {pooled.shape[0]} échantillons × {pooled.shape[1]} gènes → {out_dir/'expr_all.csv'}")
    # Entrées HuMess (abundance_table genes×samples + samplesheet) — HuMess
    # construit un modèle métabolique par présence de gènes (robuste petit-n,
    # ≠ régression coexpr). cf. scripts/make_humess_config.py.
    if getattr(args, "emit_humess", False):
        ab = pooled.T  # genes × samples
        ab.index.name = ""
        ab.to_csv(out_dir / "abundance_table.tsv", sep="\t")
        with open(out_dir / "samplesheet.tsv", "w") as fh:
            for s in pooled.index.astype(str):
                fh.write(f"{s}\t{grp[s]}\n")
        print(f"[prep] HuMess : abundance_table.tsv ({ab.shape[0]}×{ab.shape[1]}) "
              f"+ samplesheet.tsv → {out_dir}")
    n_min = min(w[1] for w in written)
    if n_min < 20:
        print(f"[prep] ⚠️ min {n_min} échantillons/groupe : GRNBoost2/coexpr "
              f"peu fiable (p≫n) ; <3 = crash GBM → utiliser expr_all.csv (poolé). "
              f"HuMess (présence de gènes) reste viable.")


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

    # Garde-fou : cible constante ou trop peu d'échantillons → le GBM dégénère
    # ("Weights sum to zero"). On skip plutôt que de crasher tout le job.
    if len(y) < 3 or float(np.std(y)) < 1e-12:
        return []
    # Masque : retirer le gène cible des features s'il est lui-même un TF
    keep = [i for i, tf in enumerate(tf_names) if tf != target]
    if not keep:
        return []
    X = X_tf[:, keep]
    feats = [tf_names[i] for i in keep]

    # Early-stopping (split de validation 10%) nécessite assez d'échantillons ;
    # en-dessous de ~20 on le désactive (sinon 0 sample de validation).
    es = (dict(n_iter_no_change=_EARLY_STOP_WINDOW, validation_fraction=0.1, tol=1e-4)
          if len(y) >= 20 else {})
    reg = GradientBoostingRegressor(random_state=seed, **es, **_SGBM_KWARGS)
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
# Étape 2ter — correlation : Pearson/Spearman TF×target
# ---------------------------------------------------------------------------
# Baseline naïve pour la grille V4.3. Calcule la corrélation entre chaque
# TF et chaque cible (≠ TF lui-même), garde |r| comme `importance` et
# conserve le signe dans la colonne `sign` (consommable par un encodeur
# signé V5/V6 ; ignorée par le decoder InnerProduct V4.x).
# Référence : Eisen 1998 PNAS (clustering corr.) ; Stuart 2003 Science
# (coexpr corrélationnelle) ; Marbach 2012 Nat Methods (limites
# corrélation vs GRNBoost2 sur GRN benchmarks).
def cmd_correlation(args: argparse.Namespace) -> None:
    """Corrélation TF×target (Pearson ou Spearman), format (TF, target, importance, sign)."""
    expr_path = Path(args.expr)
    tf_path = Path(args.tf_list)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[corr] lecture {expr_path}")
    expr = pd.read_csv(expr_path, index_col=0)
    cols = list(expr.columns)
    col_set = set(cols)

    raw_tfs = [t.strip() for t in open(tf_path) if t.strip()]
    tf_names = []
    for t in raw_tfs:
        if t in col_set:
            tf_names.append(t)
        elif t.replace("_", "-") in col_set:
            tf_names.append(t.replace("_", "-"))
    tf_names = sorted(set(tf_names))
    print(f"[corr] {expr.shape[0]} cellules × {expr.shape[1]} gènes ; "
          f"{len(tf_names)}/{len(raw_tfs)} TFs mappés ; method={args.method}")
    if len(tf_names) < 10:
        sys.exit(f"[corr] seulement {len(tf_names)} TFs mappés — abort.")

    # Sortir la matrice rangée [cells × genes] en numpy. Pour Spearman :
    # remplacer par les rangs colonne par colonne (ranking par gène) puis
    # appliquer la même formule Pearson sur les rangs ⇒ Spearman.
    X = expr.to_numpy(dtype=np.float32)
    if args.method == "spearman":
        # rankdata par colonne (axis=0) — argsort.argsort = rangs entiers
        ranks = X.argsort(axis=0).argsort(axis=0).astype(np.float32)
        X = ranks

    # Centrer-réduire colonne par colonne (n cells = lignes).
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma == 0.0] = 1.0  # éviter NaN sur colonnes constantes
    Z = (X - mu) / sigma
    n_cells = Z.shape[0]

    name_to_idx = {g: i for i, g in enumerate(cols)}
    tf_idx = np.array([name_to_idx[t] for t in tf_names], dtype=np.int64)

    # Corrélation Z[:, tf] · Z[:, all] / n  →  matrice TF × allGenes.
    # Coût mémoire : len(tf) × len(cols) × 4 octets (e.g. 1500×15000 ≈ 90 Mo).
    print(f"[corr] produit matriciel TF×target ({len(tf_idx)}×{len(cols)})…")
    R = (Z[:, tf_idx].T @ Z) / n_cells  # shape (n_tf, n_genes)
    R = np.clip(R, -1.0, 1.0).astype(np.float32)

    # Masquer la diagonale TF↔TF (un TF ne se régule pas lui-même).
    for k, gi in enumerate(tf_idx):
        R[k, gi] = 0.0

    # Filtre optionnel sur |r|.
    thr = float(args.min_abs_r)
    print(f"[corr] filtre |r| ≥ {thr}")
    abs_R = np.abs(R)
    mask = abs_R >= thr
    tf_ix, tgt_ix = np.where(mask)
    importances = abs_R[tf_ix, tgt_ix]
    signs = np.sign(R[tf_ix, tgt_ix]).astype(np.int8)
    rows = list(zip(
        [tf_names[i] for i in tf_ix],
        [cols[j] for j in tgt_ix],
        importances.tolist(),
        signs.tolist(),
    ))
    adj = pd.DataFrame(rows, columns=["TF", "target", "importance", "sign"])
    adj = adj.sort_values("importance", ascending=False).reset_index(drop=True)
    adj.to_csv(out_path, index=False)
    print(f"[corr] {len(adj)} arêtes |r|≥{thr} → {out_path}")
    if len(adj) > 0:
        print(f"[corr] top 3 :\n{adj.head(3).to_string(index=False)}")


# ---------------------------------------------------------------------------
# Étape 2quater — mutual-info : sklearn mutual_info_regression
# ---------------------------------------------------------------------------
# Baseline non linéaire (capture dépendances non monotones). Coût ×N par
# rapport à la corrélation. Pour chaque cible, MI(TF, target) via
# sklearn.feature_selection.mutual_info_regression (k-NN Kraskov 2004).
# Pas de signe : MI ≥ 0.
# Référence : Margolin 2006 BMC Bioinf. (ARACNe, MI + DPI) ; Faith 2007
# PLoS Biol (CLR sur MI).
def cmd_mutual_info(args: argparse.Namespace) -> None:
    """Mutual information TF×target (sklearn mutual_info_regression)."""
    from joblib import Parallel, delayed
    from sklearn.feature_selection import mutual_info_regression

    expr_path = Path(args.expr)
    tf_path = Path(args.tf_list)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[mi] lecture {expr_path}")
    expr = pd.read_csv(expr_path, index_col=0)
    cols = list(expr.columns)
    col_set = set(cols)

    raw_tfs = [t.strip() for t in open(tf_path) if t.strip()]
    tf_names = []
    for t in raw_tfs:
        if t in col_set:
            tf_names.append(t)
        elif t.replace("_", "-") in col_set:
            tf_names.append(t.replace("_", "-"))
    tf_names = sorted(set(tf_names))
    print(f"[mi] {expr.shape[0]} cellules × {expr.shape[1]} gènes ; "
          f"{len(tf_names)}/{len(raw_tfs)} TFs mappés")
    if len(tf_names) < 10:
        sys.exit(f"[mi] seulement {len(tf_names)} TFs mappés — abort.")

    X_tf = expr[tf_names].to_numpy(dtype=np.float32)
    targets = cols
    n_jobs = args.n_jobs
    n_neighbors = args.n_neighbors

    def _mi_one(target):
        keep = [i for i, tf in enumerate(tf_names) if tf != target]
        if not keep:
            return []
        X = X_tf[:, keep]
        feats = [tf_names[i] for i in keep]
        y = expr[target].to_numpy(dtype=np.float32)
        mi = mutual_info_regression(
            X, y, n_neighbors=n_neighbors, random_state=args.seed,
        )
        return [(feats[k], target, float(mi[k])) for k in range(len(feats))
                if mi[k] > 0.0]

    print(f"[mi] régression MI sur {len(targets)} cibles "
          f"(k-NN n_neighbors={n_neighbors}, {n_jobs} jobs)…")
    results = Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(
        delayed(_mi_one)(tgt) for tgt in targets
    )
    rows = [r for sub in results for r in sub]
    adj = pd.DataFrame(rows, columns=["TF", "target", "importance"])
    adj = adj.sort_values("importance", ascending=False).reset_index(drop=True)
    adj.to_csv(out_path, index=False)
    print(f"[mi] {len(adj)} arêtes → {out_path}")
    if len(adj) > 0:
        print(f"[mi] top 3 :\n{adj.head(3).to_string(index=False)}")


# ---------------------------------------------------------------------------
# Étape 3 — merge-adjacencies (option A)
# ---------------------------------------------------------------------------
def cmd_merge_adjacencies(args: argparse.Namespace) -> None:
    p4_path = Path(args.adj_p4)
    p16_path = Path(args.adj_p16)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Garde-fou (V4.3) : ne pas écraser silencieusement.
    if out_path.exists() and not args.overwrite:
        sys.exit(f"[merge] sortie déjà présente : {out_path} — passer "
                 f"--overwrite pour remplacer (V4.3 : protège la "
                 f"grille méthode×prune contre les ré-écritures "
                 f"accidentelles).")

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
    elif mode == "mutual-rank":
        # Obayashi 2018 (COXPRESdb) : MR(TF,target) = sqrt(rank_TF×target ×
        # rank_target×TF) avec rang DENSE par origine. Symétrise et
        # corrige le biais hub (un hub n'apparaît top-K pour TOUTES les
        # cibles que si la réciproque est aussi vraie). Pour des données
        # TF→target asymétriques (cas GRNBoost2), on définit :
        #   r1 = rang de (TF, target) parmi les arêtes sortantes du TF
        #        (TF fixé, target variable) sur imp_max
        #   r2 = rang de (TF, target) parmi les arêtes entrantes du target
        #        (target fixé, TF variable) sur imp_max
        #   MR = sqrt(r1 * r2). On garde MR <= K (= top K les plus
        #   mutuellement co-classés).
        m = merged.copy()
        m["r1"] = (m.sort_values("imp_max", ascending=False)
                    .groupby("TF").cumcount() + 1)
        m["r2"] = (m.sort_values("imp_max", ascending=False)
                    .groupby("target").cumcount() + 1)
        m["mr"] = np.sqrt(m["r1"].to_numpy() * m["r2"].to_numpy())
        merged = m[m["mr"] <= K].copy()
        why = f"mutual-rank ≤ {K} (Obayashi 2018)"
    elif mode == "z-score":
        # Par cible : keep TF dont (imp - μ_target) / σ_target ≥ z_thresh.
        # Adaptatif à la dispersion locale ⇒ ne donne pas le même
        # nombre d'arêtes par cible (contrairement à top-K déterministe).
        z = float(args.z_thresh)
        mu = merged.groupby("target")["imp_max"].transform("mean")
        sd = merged.groupby("target")["imp_max"].transform("std").fillna(0.0)
        sd = sd.replace(0.0, 1.0)
        merged = merged[(merged["imp_max"] - mu) / sd >= z].copy()
        why = f"z-score per-target ≥ {z}σ"
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

    pp = sub.add_parser("prep-matrices",
                        help="V6 : matrice quelconque (+ metadata sample→groupe) "
                             "→ expr_<groupe>.csv (échantillons × gènes)")
    pp.add_argument("--matrix", required=True,
                    help="counts/FPKM, genes×samples ou samples×genes (sniff)")
    pp.add_argument("--metadata", required=True,
                    help="TSV/CSV : col1=échantillon, col-group=groupe/état")
    pp.add_argument("--group-col", type=int, default=1,
                    help="index (0-based) de la colonne groupe du metadata (déf. 1)")
    pp.add_argument("--gene-col", default=None,
                    help="colonne gène de la matrice (déf. auto : Tracking_id/"
                         "hgnc_symbol/… sinon 1re colonne)")
    pp.add_argument("--orient", choices=["auto", "genes-rows", "samples-rows"],
                    default="auto", help="orientation de la matrice (déf. auto)")
    pp.add_argument("--emit-humess", action="store_true",
                    help="émet aussi abundance_table.tsv + samplesheet.tsv (entrées HuMess)")
    pp.add_argument("--out-dir", default="data/pyscenic/diff_coexpr")
    pp.set_defaults(func=cmd_prep_matrices)

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

    # --- correlation (V4.3) ---
    pc = sub.add_parser(
        "correlation",
        help="Corrélation Pearson/Spearman TF×target — baseline V4.3.")
    pc.add_argument("--expr", required=True,
                    help="expr_matrix_{P4,P16}.csv (cellules × gènes)")
    pc.add_argument("--tf-list",
                    default="data/pyscenic/scenic_refs/allTFs_hg38.txt")
    pc.add_argument("--out", required=True,
                    help="adjacencies_{P4,P16}.corr.csv de sortie "
                         "(colonnes TF,target,importance=|r|,sign)")
    pc.add_argument("--method", choices=["pearson", "spearman"],
                    default="spearman",
                    help="Spearman (défaut, robuste rangs) ou Pearson.")
    pc.add_argument("--min-abs-r", type=float, default=0.1,
                    help="Filtre minimal sur |r| (défaut 0.1 : élimine "
                         "le bruit ; ajusté ensuite par merge --prune-mode).")
    pc.set_defaults(func=cmd_correlation)

    # --- mutual-info (V4.3) ---
    pmi = sub.add_parser(
        "mutual-info",
        help="Mutual information sklearn TF×target — baseline V4.3.")
    pmi.add_argument("--expr", required=True)
    pmi.add_argument("--tf-list",
                     default="data/pyscenic/scenic_refs/allTFs_hg38.txt")
    pmi.add_argument("--out", required=True,
                     help="adjacencies_{P4,P16}.mi.csv de sortie")
    pmi.add_argument("--n-jobs", type=int, default=8)
    pmi.add_argument("--n-neighbors", type=int, default=3,
                     help="k-NN MI Kraskov (sklearn défaut=3).")
    pmi.add_argument("--seed", type=int, default=42)
    pmi.set_defaults(func=cmd_mutual_info)

    pm = sub.add_parser("merge-adjacencies",
                        help="adjacencies_P4/P16 → coexpr_diff.tsv (option A)")
    pm.add_argument("--adj-p4", required=True)
    pm.add_argument("--adj-p16", required=True)
    pm.add_argument("--prune-mode",
                    choices=["per-target-topk", "global-quantile",
                             "hybrid", "mutual-rank", "z-score"],
                    default="per-target-topk",
                    help="per-target-topk (DÉFAUT V4.2 : top-K régulateurs "
                         "par cible → 100%% couverture, hubs déconcentrés, "
                         "= élagage SCENIC) ; global-quantile (legacy, "
                         "dominé par hubs → affame ASNS/IL6/IL1B/DDIT3) ; "
                         "hybrid (union des deux) ; mutual-rank "
                         "(Obayashi 2018 COXPRESdb, MR ≤ K sym. + débiaisé "
                         "hub) ; z-score (par cible, (imp-μ)/σ ≥ z_thresh).")
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
    pm.add_argument("--z-thresh", type=float, default=2.0,
                    help="Seuil σ pour --prune-mode z-score (défaut 2.0).")
    pm.add_argument("--overwrite", action="store_true",
                    help="V4.3 : autorise l'écrasement de --out si présent. "
                         "Par défaut, refuse pour protéger la grille "
                         "méthode×prune.")
    pm.add_argument("--out",
                    default="data/pyscenic/diff_coexpr/coexpr_diff.tsv")
    pm.set_defaults(func=cmd_merge_adjacencies)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
