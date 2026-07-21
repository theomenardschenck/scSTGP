#!/usr/bin/env python3
"""
perturb_decoy.py — Null-models décoy : le signal d'un driver vient-il de ses
VOISINS SPÉCIFIQUES (biologie) ou de son DEGRÉ / ses FEATURES (artefact) ?

Deux nulles (gnn_futur §7.12), encodeur GELÉ, post-hoc :
  N2 — rewire degré-préservant de TOUTES les arêtes gène-gène (partenaires +
       signes aléatoires) → si la projection réelle ≫ nulle = voisinage
       spécifique ; si elle tombe dans la nulle = artefact de degré.
  N4 — features du gène → moyenne (vrais voisins gardés) → si la projection
       s'effondre = signal porté par les features (circularité expression).

Verdict établi (V5.4.1 single-seed) : en OE le signal est marqueur-broadcast
(neighbor-invariant) ; en KO/KD le réel dépasse la nulle (p≈0 → info de
voisinage réelle). D'où la **colonne de confiance décoy** (mode confidence) :
    decoy_confidence = 1 − |proj_N2_moy| / |proj_réel|      (par gène, mode KO)
≈0 = diffusé/marqueur ; ↗1 = network-causal (type TP53).

⚠️ DEUX CORRECTIONS 2026-07-21 (LOG §25bis / §26) :
 1. **n≥50 obligatoire.** À `--n-rewires 5` les SD sont sous-estimées ~10×
    (HMGB2 0.3→4.4) et trois « passages » sur quatre disparaissent à n=50.
    Le défaut de `confidence` (3) est donc trop bas : les colonnes
    `decoy_confidence` produites avant cette date sont NON FIABLES.
 2. **Nulle sur le COSINUS ajoutée.** Le test ne portait que sur
    `proj_signed_diff`, la projection porteuse de MAGNITUDE — or LOG §25
    montre qu'elle est confondue au degré (ρ +0.45, signe qui s'inverse avec
    la densité du graphe). Le cosinus, qui est la statistique séparant
    réellement les cibles (SYNJ2 |cos| 0.85 vs TP53 0.05), n'avait AUCUNE
    nulle : il était calculé par `_project` puis jeté (`[0]`). Colonnes
    `n2_cos_mean/n2_p_cos` (per-gene) et `decoy_confidence_cos/decoy_p_cos`
    (confidence). Coût nul — même forward.
⚠️ N4 est non informatif sur un graphe SANS canal expression (op.all) :
   neutraliser les features y supprime l'objet même de la perturbation.

Usage
-----
    # quelques gènes, N2+N4, modes KO/KD/OE
    python src/validation/explain/perturb_decoy.py per-gene \\
        --run-dir output/gnn_vgae/V5.4.1/v5.4.baseline.s1 \\
        --genes H2AFZ HMGB1 HMGB2 TP53 --modes knockout knockdown --n-rewires 5

    # colonne de confiance sur le top-N d'un ranking (mode KO, degré total)
    python src/validation/explain/perturb_decoy.py confidence \\
        --run-dir output/gnn_vgae/V5.4.1/v5.4.baseline.s1 \\
        --ranking .../cross_seed_gene_ranking.tsv --top-n 40 --n-rewires 3 \\
        --out decoy_confidence_top.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# gnn_perturbation importable sous layout local (src/gnn) ou flat (src/)
_THIS = Path(__file__).resolve()
for _cand in (_THIS.parent.parent.parent / "gnn", _THIS.parent.parent):
    if (_cand / "gnn_perturbation.py").exists():
        sys.path.insert(0, str(_cand)); break
import torch  # noqa: E402
from gnn_perturbation import (load_run, apply_perturbation,  # noqa: E402
                              cell_group_shift_projected, compute_senescence_axes)

P16 = ("P16_cluster_1", "P16_cluster_2", "P16_cluster_3")
QUI = ("P4", "P16_cluster_0")
HP = dict(hidden=128, latent=64, n_layers=3, n_heads=4)


# --------------------------------------------------------------------------- #
# Primitives (encodeur gelé)
# --------------------------------------------------------------------------- #
def _encode_mu(model, d):
    xd = {"gene": d["gene"].x, "cell_group": d["cell_group"].x}
    ei = {k: d[k].edge_index for k in d.edge_types}
    ea = {k: d[k].edge_attr for k in d.edge_types
          if getattr(d[k], "edge_attr", None) is not None}
    with torch.no_grad():
        mu, _ = model.encode(xd, ei, ea)
    return mu.cpu().numpy()


def _base_axis(model, d, group_expr, gene_symbols):
    mu = _encode_mu(model, d)
    ag, ac, _at, _ = compute_senescence_axes(mu, group_expr, gene_symbols,
                                             p16_groups=P16, quiescent_groups=QUI)
    return mu, ag, ac


def _project(model, d, idx, mode, mu_base, ag, ac, group_expr, gene_symbols):
    """Projection signée (max |proj_signed_diff| sur les lignes axe-global)."""
    ti = torch.tensor([idx])
    pert = apply_perturbation(d, ti, mode, 3.0)            # factor ignoré KO/KD
    mu_pert = _encode_mu(model, pert)
    df = cell_group_shift_projected(mu_base, mu_pert, group_expr, gene_symbols,
                                    ti, ag, ac, target_degree=1)
    g = df[df.axis_type == "global"]
    row = g.loc[g["proj_signed_diff"].abs().idxmax()]
    return float(row["proj_signed_diff"]), float(row["proj_signed_cosine"])


def rewire(d, idx, rng):
    """N2 : rebranche TOUTES les arêtes gène-gène incidentes au gène `idx`
    sur des partenaires aléatoires (degré préservé par type) + signes random."""
    n_gene = d["gene"].x.shape[0]
    dd = d.clone()
    for et in dd.edge_types:
        if not (et[0] == "gene" and et[2] == "gene"):
            continue
        ei = dd[et].edge_index.clone()
        sm = ei[0] == idx; dm = ei[1] == idx
        for mask, repl_row in ((sm, 1), (dm, 0)):
            k = int(mask.sum())
            if k:
                r = torch.tensor(rng.integers(0, n_gene, k), dtype=ei.dtype)
                r[r == idx] = (r[r == idx] + 1) % n_gene
                ei[repl_row, mask] = r
        dd[et].edge_index = ei
        ea = getattr(dd[et], "edge_attr", None)
        if ea is not None and ea.ndim > 1 and ea.shape[1] >= 2:
            ea = ea.clone(); inc = sm | dm
            ea[inc, 1] = torch.tensor(rng.choice([-1.0, 1.0], int(inc.sum())),
                                      dtype=ea.dtype)
            dd[et].edge_attr = ea
    return dd


def neutralize(d, idx, x_mean):
    """N4 : remplace les features du gène par la moyenne (voisins gardés)."""
    dd = d.clone(); x = dd["gene"].x.clone(); x[idx] = x_mean; dd["gene"].x = x
    return dd


def total_degree(data):
    """Degré TOTAL (toutes arêtes gène-gène de reconstruction, pas que PPI)."""
    n = data["gene"].x.shape[0]; td = np.zeros(n)
    for et in data.edge_types:
        if et[0] == "gene" and et[2] == "gene":
            ei = data[et].edge_index
            np.add.at(td, ei[0].numpy(), 1); np.add.at(td, ei[1].numpy(), 1)
    return td


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_per_gene(args):
    data, model, gs, g2i, _b, gx = load_run(args.run_dir, **HP)
    x_mean = data["gene"].x.mean(dim=0)
    mu_base, ag, ac = _base_axis(model, data, gx, gs)
    print(f"{'gene':9s} {'mode':4s} | {'REAL':>8} {'cos':>6} | "
          f"{'N2 moy±sd':>13} {'p':>4} | {'N2cos±sd':>12} {'p_cos':>5} | "
          f"{'N4':>8}", flush=True)
    rows = []
    for G in args.genes:
        if G not in g2i:
            print(f"{G}: absent"); continue
        idx = g2i[G]
        rew = [_base_axis(model, rewire(data, idx, np.random.default_rng(s)), gx, gs)
               for s in range(args.n_rewires)]
        rew_d = [rewire(data, idx, np.random.default_rng(s)) for s in range(args.n_rewires)]
        dn = neutralize(data, idx, x_mean); mb4, ag4, ac4 = _base_axis(model, dn, gx, gs)
        for mode in args.modes:
            rd, rc = _project(model, data, idx, mode, mu_base, ag, ac, gx, gs)
            # V6.2 (2026-07-21) : the null is now kept on BOTH statistics.
            # It used to discard the cosine ([0] only) and test `proj_signed_diff`
            # alone -- i.e. the MAGNITUDE-carrying projection, which LOG §25 shows
            # is degree-confounded (rho +0.45, sign flipping with graph density).
            # The cosine is the statistic that actually separates the targets
            # (SYNJ2 |cos|=0.85 vs TP53 0.05), and it had NO null at all. Both
            # values are produced by the same forward pass, so this is free.
            _n = [_project(model, dr, idx, mode, mb, agr, acr, gx, gs)
                  for dr, (mb, agr, acr) in zip(rew_d, rew)]
            nd = np.array([v[0] for v in _n])
            nc = np.array([v[1] for v in _n])
            p = float((np.abs(nd) >= abs(rd)).mean())
            p_cos = float((np.abs(nc) >= abs(rc)).mean())
            n4d, n4c = _project(model, dn, idx, mode, mb4, ag4, ac4, gx, gs)
            print(f"{G:9s} {mode[:4]:4s} | {rd:8.1f} {rc:6.2f} | "
                  f"{nd.mean():8.1f}±{nd.std():4.1f} {p:4.2f} | "
                  f"{nc.mean():7.2f}±{nc.std():4.2f} {p_cos:5.2f} | "
                  f"{n4d:8.1f}", flush=True)
            rows.append(dict(gene=G, mode=mode, real=rd, cos=rc,
                             n2_mean=nd.mean(), n2_sd=nd.std(), n2_p=p,
                             n2_cos_mean=nc.mean(), n2_cos_sd=nc.std(),
                             n2_p_cos=p_cos, n4=n4d, n4_cos=n4c))
    if args.out:
        pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)
        print(f"[wrote] {args.out}")


def run_confidence(args):
    data, model, gs, g2i, _b, gx = load_run(args.run_dir, **HP)
    td = total_degree(data)
    mu_base, ag, ac = _base_axis(model, data, gx, gs)
    rk = pd.read_csv(args.ranking, sep="\t")
    key = "target" if "target" in rk.columns else rk.columns[0]
    rk = rk.sort_values("driver_score", ascending=False).head(args.top_n)
    rows = []
    for i, row in enumerate(rk.itertuples()):
        G = getattr(row, key)
        if G not in g2i:
            continue
        idx = g2i[G]
        rd, rc = _project(model, data, idx, args.mode, mu_base, ag, ac, gx, gs)
        nd, nc = [], []
        for s in range(args.n_rewires):
            dr = rewire(data, idx, np.random.default_rng(4000 + s))
            mb, agr, acr = _base_axis(model, dr, gx, gs)
            _v = _project(model, dr, idx, args.mode, mb, agr, acr, gx, gs)
            nd.append(_v[0]); nc.append(_v[1])
        nm = float(np.mean(nd))
        conf = (1.0 - abs(nm) / (abs(rd) + 1e-9)) if abs(rd) >= args.min_signal else np.nan
        # V6.2 : cosine-based confidence. Unlike `decoy_confidence` it needs no
        # `min_signal` gate -- the cosine is scale-free, so a weak-amplitude gene
        # (all our metabolic targets) still gets a defined value instead of NaN.
        ncm = float(np.mean(nc))
        conf_cos = 1.0 - abs(ncm) / (abs(rc) + 1e-9)
        p_cos = float((np.abs(np.asarray(nc)) >= abs(rc)).mean())
        rows.append(dict(rank=i + 1, gene=G,
                         driver_score=getattr(row, "driver_score", np.nan),
                         real_proj=round(rd, 1), null_proj=round(nm, 1),
                         decoy_confidence=round(conf, 3) if conf == conf else np.nan,
                         real_cos=round(rc, 3), null_cos=round(ncm, 3),
                         decoy_confidence_cos=round(conf_cos, 3),
                         decoy_p_cos=round(p_cos, 3),
                         total_degree=int(td[idx])))
        print(f"{i+1:3d} {G:9s} real={rd:7.1f} null={nm:7.1f} conf={conf:+.2f} | "
              f"cos={rc:+.2f} null_cos={ncm:+.2f} p_cos={p_cos:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)
    print(f"[wrote] {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("per-gene", help="N2+N4 sur une liste de gènes")
    g.add_argument("--run-dir", type=Path, required=True)
    g.add_argument("--genes", nargs="+", required=True)
    g.add_argument("--modes", nargs="+", default=["knockout", "knockdown", "overexpress"])
    g.add_argument("--n-rewires", type=int, default=5)
    g.add_argument("--out", type=Path, default=None)
    g.set_defaults(func=run_per_gene)

    c = sub.add_parser("confidence", help="colonne de confiance décoy sur top-N")
    c.add_argument("--run-dir", type=Path, required=True)
    c.add_argument("--ranking", type=Path, required=True)
    c.add_argument("--top-n", type=int, default=40)
    c.add_argument("--n-rewires", type=int, default=3)
    c.add_argument("--mode", default="knockout")
    c.add_argument("--min-signal", type=float, default=5.0,
                   help="|proj_réel| min pour calculer la confiance (sinon NaN).")
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(func=run_confidence)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
