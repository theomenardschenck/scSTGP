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

3. `merge-adjacencies` (LOCAL) : depuis `adjacencies_P4.csv` +
   `adjacencies_P16.csv`, produit `coexpr_diff.tsv` (option A).

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
        --top-quantile 0.98 \\
        --out data/pyscenic/diff_coexpr/coexpr_diff.tsv

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

    print(f"[extract] lecture {merged_path} (par chunks, ~976 MB) ...")
    # Lecture par chunks : variance incrémentale (Welford-like via somme +
    # somme des carrés) pour éviter de charger 976 MB d'un coup. On garde
    # aussi l'index des cellules P4/P16 pour la 2e passe.
    chunksize = 500
    n_total = 0
    sum_x = None
    sum_x2 = None
    gene_cols = None
    passage_per_idx: list[tuple[str, str]] = []  # (barcode, passage)

    reader = pd.read_csv(merged_path, index_col=0, chunksize=chunksize)
    for ci, chunk in enumerate(reader):
        if gene_cols is None:
            gene_cols = [c for c in chunk.columns if c not in META_COLS]
            sum_x = np.zeros(len(gene_cols), dtype=np.float64)
            sum_x2 = np.zeros(len(gene_cols), dtype=np.float64)
        if "passage" not in chunk.columns:
            sys.exit("[extract] colonne 'passage' absente du merged — abort.")
        vals = chunk[gene_cols].to_numpy(dtype=np.float64)
        sum_x += vals.sum(axis=0)
        sum_x2 += (vals ** 2).sum(axis=0)
        n_total += len(chunk)
        for bc, pas in zip(chunk.index.astype(str), chunk["passage"].astype(str)):
            passage_per_idx.append((bc, pas))
        if (ci + 1) % 5 == 0:
            print(f"[extract]   ... {n_total} cellules lues")

    # Variance globale (P4+P16) par gène : Var = E[x²] − E[x]²
    mean = sum_x / n_total
    var = (sum_x2 / n_total) - mean ** 2
    var_series = pd.Series(var, index=gene_cols).sort_values(ascending=False)
    n_hvg = min(args.n_hvg, len(var_series))
    hvg = var_series.head(n_hvg).index.tolist()
    print(f"[extract] {n_total} cellules × {len(gene_cols)} gènes ; "
          f"{n_hvg} HVG sélectionnés (variance globale)")

    # 2e passe : extraire les colonnes HVG par condition, par chunks.
    # Le nom de la colonne d'index (1ère colonne) est lu sur l'en-tête.
    with open(merged_path) as _fh:
        _header = _fh.readline().rstrip("\n").split(",")
    index_name = _header[0].strip().strip('"')
    hvg_set = set(hvg)
    keep = {index_name, "passage"} | hvg_set
    buffers = {"P4": [], "P16": []}
    reader2 = pd.read_csv(merged_path, index_col=0,
                          usecols=lambda c: c in keep,
                          chunksize=chunksize)
    for chunk in reader2:
        for cond in ("P4", "P16"):
            sub = chunk[chunk["passage"] == cond]
            if not sub.empty:
                buffers[cond].append(sub[hvg])

    for cond in ("P4", "P16"):
        if not buffers[cond]:
            print(f"[extract] [warn] aucune cellule {cond} — skip")
            continue
        mat = pd.concat(buffers[cond], axis=0)
        out_path = out_dir / f"expr_matrix_{cond}.csv"
        mat.to_csv(out_path)
        print(f"[extract] {cond}: {mat.shape[0]} cellules × {mat.shape[1]} "
              f"gènes → {out_path}")

    # Sauver la liste HVG pour traçabilité
    (out_dir / "hvg_genes.txt").write_text("\n".join(hvg))
    print(f"[extract] HVG list → {out_dir / 'hvg_genes.txt'}")
    print(f"[extract] OK. Étape suivante : print-grn-cmds")


# ---------------------------------------------------------------------------
# Étape 2 — print-grn-cmds (les commandes GRNBoost2 à lancer sur le cluster)
# ---------------------------------------------------------------------------
def cmd_print_grn_cmds(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    tf_list = "data/pyscenic/scenic_refs/allTFs_hg38.txt"
    print("# ============================================================")
    print("# GRNBoost2 sur P4 et P16 — À LANCER SUR LE FRONTAL (arboreto)")
    print("# ============================================================")
    print("# Réutilise la logique de scenic_from_r.py étape 1 mais en")
    print("# pointant sur les 2 matrices séparées. Seed=42 identique pour")
    print("# que le delta ne soit pas confondu par la stochasticité GRNBoost2.")
    print()
    for cond in ("P4", "P16"):
        print(f"python - <<'PYEOF'")
        print(f"import pandas as pd")
        print(f"from arboreto.algo import grnboost2")
        print(f"from distributed import Client, LocalCluster")
        print(f"expr = pd.read_csv('{out_dir}/expr_matrix_{cond}.csv', index_col=0)")
        print(f"tfs = [t.strip() for t in open('{tf_list}')]")
        print(f"tfs = [t for t in tfs if t in expr.columns]")
        print(f"cl = LocalCluster(n_workers=8, threads_per_worker=1)")
        print(f"adj = grnboost2(expression_data=expr, tf_names=tfs,")
        print(f"                client_or_address=Client(cl), seed=42, verbose=True)")
        print(f"adj.to_csv('{out_dir}/adjacencies_{cond}.csv', index=False)")
        print(f"print('done {cond}', len(adj))")
        print(f"PYEOF")
        print()
    print("# Puis : merge-adjacencies (local)")


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

    # Filtrage top-quantile PAR CONDITION (cohérent avec COEXPR_TOP_QUANTILE
    # de gnn_vgae.py — on ne garde que les arêtes fortes de chaque réseau).
    q = args.top_quantile
    t4 = df4["importance"].quantile(q)
    t16 = df16["importance"].quantile(q)
    df4f = df4[df4["importance"] >= t4].copy()
    df16f = df16[df16["importance"] >= t16].copy()
    print(f"[merge] après filtre q={q} : P4={len(df4f)} P16={len(df16f)}")

    merged = df4f.merge(
        df16f, on=["TF", "target"], how="outer",
        suffixes=("_p4", "_p16"),
    )
    merged["importance_p4"] = merged["importance_p4"].fillna(0.0)
    merged["importance_p16"] = merged["importance_p16"].fillna(0.0)

    # Normalisation min-max PAR CONDITION pour rendre les deux importances
    # comparables (GRNBoost2 importance n'est pas calibrée entre runs).
    for c in ("importance_p4", "importance_p16"):
        mx = merged[c].max()
        merged[c + "_norm"] = merged[c] / mx if mx > 0 else 0.0

    merged["delta"] = merged["importance_p16_norm"] - merged["importance_p4_norm"]

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
                        help="merged → expr_matrix_P4/P16 (HVG commun)")
    pe.add_argument("--merged",
                    default="data/gnn_data/merged_P4_P16_normalized.csv")
    pe.add_argument("--n-hvg", type=int, default=5000,
                    help="Nombre de HVG (variance sur toutes cellules)")
    pe.add_argument("--out-dir", default="data/pyscenic/diff_coexpr")
    pe.set_defaults(func=cmd_extract_matrices)

    pg = sub.add_parser("print-grn-cmds",
                        help="Affiche les commandes GRNBoost2 (cluster)")
    pg.add_argument("--out-dir", default="data/pyscenic/diff_coexpr")
    pg.set_defaults(func=cmd_print_grn_cmds)

    pm = sub.add_parser("merge-adjacencies",
                        help="adjacencies_P4/P16 → coexpr_diff.tsv (option A)")
    pm.add_argument("--adj-p4", required=True)
    pm.add_argument("--adj-p16", required=True)
    pm.add_argument("--top-quantile", type=float, default=0.98,
                    help="Quantile minimal d'importance par condition "
                         "(cohérent avec --coexpr-top-quantile de gnn_vgae)")
    pm.add_argument("--out",
                    default="data/pyscenic/diff_coexpr/coexpr_diff.tsv")
    pm.set_defaults(func=cmd_merge_adjacencies)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
