"""
perturb_supervised.py — Perturbation NATIVE du modèle supervisé (V-sup) par la tête.

CONTEXTE
--------
Le modèle supervisé (`_supervised.py`) = encodeur HeteroEncoder + tête de
classification (proba DEG par cluster). Son « décodeur » est la tête. Cette
perturbation KO/KD/OE mesure, pour chaque gène perturbé, le **Δ de la proba DEG
prédite par cluster** sur toute la population — le score driver NATIF du modèle
supervisé : « le KO de X réduit-il la sénescence-DEG prédite, et dans quels
clusters ? ». Complète la voie « format run VGAE » (perturbation Δμ le long de
l'axe sénescence via `perturb_top_genes.py`, encodeur seul).

Réutilise `apply_perturbation` de `gnn_perturbation.py` (sémantique V5.5 : KO =
feature→0 arêtes gardées ; KD = ×0.15 ; OE = ×facteur) — cohérence exacte avec
la perturbation du VGAE. Forward-only (pas de backward) → tractable en RAM
modeste, contrairement à l'importance par cluster.

USAGE
-----
    python src/perturbation/perturb_supervised.py --run-dir output/gnn_supervised/vsup_full \\
        [--modes knockout knockdown overexpress] \\
        [--all-genes | --top-n 200 | --genes HMGB1,HMGB2,ENO1] \\
        [--ko-factor 2.0] [--device cpu]

Sorties → <run-dir>/perturbation/ :
  * perturbation_supervised.tsv — par (gène, mode) : Δ net proba DEG par cluster
    + senescence_net (moyenne clusters P16) + Δ propre.
  * driver_supervised.tsv       — ranking driver agrégé (modes signe-alignés).

Author: Théo Ménard — CRCI2NA. Created 2026-06-30.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve()
# Robuste aux 2 layouts : local (src/perturbation/, src/gnn/) ET cluster à plat
# (tous les .py sous src/). On ajoute les candidats hébergeant _supervised.py /
# gnn_perturbation.py au sys.path.
for _cand in (_HERE.parent.parent / "gnn", _HERE.parent):
    if _cand.is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from _supervised import load_supervised_run          # noqa: E402
from gnn_perturbation import apply_perturbation      # noqa: E402


def _head_probs(model, head, data):
    """Forward encode + head → sigmoid probs (n_genes, n_labels). No grad."""
    x_dict = {"gene": data["gene"].x, "cell_group": data["cell_group"].x}
    eid, ead = {}, {}
    for et in data.edge_types:
        eid[et] = data[et].edge_index
        if "edge_attr" in data[et] and data[et].edge_attr is not None:
            ead[et] = data[et].edge_attr
    with torch.no_grad():
        _, mu, _ = model.encode(x_dict, eid, ead)
        return torch.sigmoid(head(mu)).cpu().numpy()


def perturb_supervised(run_dir, modes=("knockout", "knockdown", "overexpress"),
                       gene_subset=None, top_n=None, ko_factor=2.0,
                       device="cpu", verbose=True, aggregate=True):
    """Perturbe chaque gène cible et mesure le Δ des probas DEG par cluster.

    Écrit un fichier PAR mode. Si `aggregate`, agrège aussi les modes présents
    en `driver_supervised.tsv` (mettre False pour un job mono-mode ; l'agrégation
    est faite par un job `--finalize` séparé)."""
    run_dir = Path(run_dir)
    model, head, data, cfg = load_supervised_run(run_dir, device=device)
    gene_symbols = np.array(cfg["gene_symbols"])
    label_names = cfg["label_names"]
    sym_to_idx = {g: i for i, g in enumerate(gene_symbols)}
    cluster_labels = [n for n in label_names if n.startswith("cluster_")]
    cluster_cols = [label_names.index(n) for n in cluster_labels]

    base = _head_probs(model, head, data)   # (n_genes, n_labels)

    # ── Sélection des cibles ────────────────────────────────────────────────
    if gene_subset:
        targets = [g for g in gene_subset if g in sym_to_idx]
    elif top_n is not None:
        imp_path = run_dir / "cluster_importance.tsv"
        rank_path = run_dir / "gene_ranking.tsv"
        if imp_path.exists():
            r = pd.read_csv(imp_path, sep="\t")
            imp_cols = [c for c in r.columns if c.startswith("importance_")]
            r["_m"] = r[imp_cols].mean(axis=1)
            targets = r.sort_values("_m", ascending=False)["gene"].head(top_n).tolist()
        elif rank_path.exists():
            targets = pd.read_csv(rank_path, sep="\t")["gene"].head(top_n).tolist()
        else:
            targets = list(gene_symbols[:top_n])
    else:  # all-genes
        targets = list(gene_symbols)
    targets = [g for g in targets if g in sym_to_idx]
    if verbose:
        print(f"  [perturb-sup] {len(targets)} cibles × {len(modes)} modes "
              f"(base senescence clusters={cluster_labels})")

    rows = []
    for k, g in enumerate(targets):
        gi = sym_to_idx[g]
        tidx = torch.tensor([gi], device=device)
        for m in modes:
            pert = apply_perturbation(data, tidx, m, factor=ko_factor)
            pp = _head_probs(model, head, pert)
            delta = pp - base                      # (n_genes, n_labels)
            row = {"gene": g, "mode": m}
            for c, name in zip(cluster_cols, cluster_labels):
                row[f"net_{name}"] = float(delta[:, c].sum())
                row[f"self_{name}"] = float(delta[gi, c])
            # net global sur l'axe P4_vs_P16 (si présent)
            if "P4_vs_P16" in label_names:
                pc = label_names.index("P4_vs_P16")
                row["net_P4_vs_P16"] = float(delta[:, pc].sum())
            # senescence_net = Δ net moyen sur les clusters P16 (sénescents)
            row["senescence_net"] = float(np.mean(
                [delta[:, c].sum() for c in cluster_cols]))
            rows.append(row)
        if verbose and (k + 1) % 200 == 0:
            print(f"    {k+1}/{len(targets)} gènes perturbés")

    df = pd.DataFrame(rows)
    out_dir = run_dir / "perturbation"
    out_dir.mkdir(exist_ok=True)
    # Écriture PAR MODE (`perturbation_supervised_<mode>.tsv`) → permet de
    # découper les 3 modes en sous-jobs SLURM séparés (évite le time-limit).
    for m in modes:
        df[df["mode"] == m].to_csv(
            out_dir / f"perturbation_supervised_{m}.tsv", sep="\t", index=False)
    if verbose:
        print(f"  [perturb-sup] écrit {len(modes)} fichier(s) par mode → {out_dir}")
    if aggregate:
        return df, aggregate_driver(run_dir, verbose=verbose)
    return df, None


def aggregate_driver(run_dir, verbose=True):
    """Concatène les `perturbation_supervised_<mode>.tsv` présents → merge +
    ranking driver signe-aligné (−net KO/KD, +net OE). Job « finalize »
    indépendant des jobs par-mode (afterok)."""
    out_dir = Path(run_dir) / "perturbation"
    files = sorted(out_dir.glob("perturbation_supervised_*.tsv"))
    if not files:
        raise FileNotFoundError(
            f"aucun perturbation_supervised_<mode>.tsv dans {out_dir}")
    df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    df.to_csv(out_dir / "perturbation_supervised.tsv", sep="\t", index=False)

    def _aligned(sub):
        s = 0.0; n = 0
        for _, r in sub.iterrows():
            sign = -1.0 if r["mode"] in ("knockout", "knockdown") else 1.0
            s += sign * r["senescence_net"]; n += 1
        return s / max(n, 1)
    drv = (df.groupby("gene").apply(_aligned, include_groups=False)
           .rename("driver_supervised").reset_index())
    drv["abs_driver"] = drv["driver_supervised"].abs()
    drv = drv.sort_values("abs_driver", ascending=False)
    drv.to_csv(out_dir / "driver_supervised.tsv", sep="\t", index=False)
    if verbose:
        print(f"  [perturb-sup] agrégé {len(files)} mode(s) "
              f"({[f.stem.split('_')[-1] for f in files]}) → "
              f"perturbation_supervised.tsv, driver_supervised.tsv")
        print("  top-10 drivers (|effet sénescence| signe-aligné) :")
        for _, r in drv.head(10).iterrows():
            print(f"    {r['gene']:>12} : {r['driver_supervised']:+.4f}")
    return drv


def _cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="run supervisé sauvé (output/gnn_supervised/<tag>)")
    ap.add_argument("--modes", nargs="+",
                    default=["knockout", "knockdown", "overexpress"])
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--all-genes", action="store_true",
                     help="perturbe TOUS les gènes (lent).")
    grp.add_argument("--top-n", type=int, default=None,
                     help="perturbe le top-N par importance/ranking (défaut None).")
    grp.add_argument("--genes", default=None,
                     help="liste séparée par des virgules (ex. HMGB1,HMGB2,ENO1).")
    ap.add_argument("--ko-factor", type=float, default=2.0,
                    help="facteur d'overexpression (défaut 2.0).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="job mono-mode : écrit le fichier du mode SANS agréger "
                         "(l'agrégation est faite par --finalize).")
    ap.add_argument("--finalize", action="store_true",
                    help="n'exécute QUE l'agrégation des perturbation_supervised_"
                         "<mode>.tsv présents → driver_supervised.tsv (job afterok).")
    args = ap.parse_args()

    if args.finalize:
        aggregate_driver(args.run_dir)
        return
    gene_subset = args.genes.split(",") if args.genes else None
    top_n = None if (args.all_genes or gene_subset) else (args.top_n or 200)
    perturb_supervised(args.run_dir, modes=tuple(args.modes),
                       gene_subset=gene_subset, top_n=top_n,
                       ko_factor=args.ko_factor, device=args.device,
                       aggregate=not args.no_aggregate)


if __name__ == "__main__":
    _cli()
