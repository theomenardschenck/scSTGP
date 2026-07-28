#!/usr/bin/env python3
"""
purity_source_attribution.py — D'OÙ vient la `purity` (cos(dz_mean, axe)) qui
classe une cible ? Quelle SOURCE d'arête la porte, et est-elle spécifique à
l'axe sénescence ? (encodeur GELÉ, post-hoc sur Δz, aucun réentraînement)

Répond à l'objection : « ces cibles métaboliques sont portées par une seule
source qui ne porte même pas la perturbation contrefactuelle » (memoire_figures
T2e). Le rang du trio PI-métabolique (OCRL/SYNJ2/SMPD1) tient UNIQUEMENT sur la
purity — l'amplitude est morte (log-normalisée, saturée). On décompose donc ce
cosinus.

Deux sous-commandes, contribution alignée de chaque gène g au numérateur du
cosinus :  c[g] = w_diff[g]·(Δz[g]·u)   avec dz_mean = Σ_g c[g]·û / Σ w_diff.

  decompose — répartit |c[g]| par DISTANCE au nœud cible (1-hop / 2-hop / far)
      + concentration (n80, top-10 share) + edge_types du 1-hop.
      ⚠️ La distance seule est TROMPEUSE : tout gène 2-hop est dans le
      voisinage de la cible ; s'il est atteint via les arêtes du module, le
      module porte le signal (délocalisé d'un cran). D'où `mediate`.

  mediate — TEST DE MÉDIATION : recompute le cosinus sur des graphes ablatés
      LOCALEMENT (arêtes incidentes à la cible seulement, encodeur figé, axe
      figé), par classe d'arête. Si couper une classe effondre le cosinus, elle
      est le PONT porteur — même quand le signal ressort à 2-hop.
      + NULLE D'AXE ALÉATOIRE : projette dz_mean réel sur N axes unitaires
      aléatoires → p(|cos aléatoire| ≥ |cos réel|). Teste la spécificité à
      l'axe sénescence.

Résultats V5.4.1 `v5.4.baseline.s1` (single-seed, 2026-07-24, cf. T2e/T2c) :
  - Trio OCRL/SYNJ2/SMPD1 : cos≈−0.48 porté à ~84 % par le 2-hop MAIS médié
    par reactome_fi (couper rfi incident : cos→−0.11, amplitude −2.7→−0.2) ;
    couper la cocatalyse HuMess (data-dérivée) OU un nombre comparable
    d'arêtes génériques ne fait rien. Nulle d'axe p<0.001 ⇒ spécifique.
  - HMGB1 (chromatine) : miroir — porté par la COEXPRESSION (couper generic :
    0.90→0.17), insensible au module curé.
  - GCLC : exception, ne dépend d'aucun module (rfi Δ+0.00), GCLM à contre-sens.

⚠️ Portée / limites (à écrire) :
  (a) reactome_fi est NON SIGNÉ / symétrisé à 100 % ⇒ « le module porte le
      signal » ≠ effet causal ORIENTÉ ; le sens (activer/inhiber) n'est pas
      identifiable.
  (b) HuMess est en partie DATA-DÉRIVÉ (modèles métaboliques P4/P16). Ce test
      montre que ses ARÊTES ne portent pas la propagation, mais sa nécessité
      au RANG (T2a) transite par ses FEATURES de nœud à l'entraînement — voie
      de circularité NON fermée ici (⇒ run `no-humess-features` séparé).
  (c) single-seed : rejouer cross-seed (plancher de bruit T4, §27).
  (d) nulle d'axe = vs axes ALÉATOIRES, pas vs autres axes BIOLOGIQUES.

Usage
-----
    python src/validation/explain/purity_source_attribution.py mediate \\
        --run-dir output/gnn_vgae/V5.4.1/v5.4.baseline.s1 \\
        --ranking output/interpretation/V5.4.1/cross_seed/baseline/cross_seed_gene_ranking.tsv \\
        --targets OCRL SYNJ2 SMPD1 GCLC HMGB1 \\
        --out output/interpretation/V5.4.1/cross_seed/baseline/purity_mediation.tsv

    python src/validation/explain/purity_source_attribution.py decompose \\
        --run-dir output/gnn_vgae/V5.4.1/v5.4.baseline.s1 \\
        --targets OCRL SYNJ2 SMPD1 GCLC HMGB1 H2AFZ TP53 \\
        --out .../purity_decomposition.tsv

Axe : par défaut l'axe V4 (quiescent = P4+P16_cluster_0 → sénescent =
P16_cluster_1..3), comme le driver_score headline V5.4.1.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# import-safe : le module vit à côté de gnn_perturbation dans src/
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from gnn import gnn_perturbation as gp   # noqa: E402

# Classes d'arête pour le test de médiation.
MODULE_ETS = {"metabolic_cocatalysis", "reactome_fi", "reactome_fi_undirected"}
COCAT_ETS = {"metabolic_cocatalysis"}                       # HuMess, data-dérivée
RFI_ETS = {"reactome_fi", "reactome_fi_undirected"}         # DB curée statique
GENERIC_ETS = {"ppi", "same_pathway", "coexpression"}


# --------------------------------------------------------------------------- #
# Infrastructure partagée
# --------------------------------------------------------------------------- #
def _encode(model, d):
    """μ (gene) sur le graphe d, forward-only."""
    x_dict = {"gene": d["gene"].x, "cell_group": d["cell_group"].x}
    if "complex" in d.node_types:
        x_dict["complex"] = d["complex"].x
    ei = {k: d[k].edge_index for k in d.edge_types}
    ea = {k: d[k].edge_attr for k in d.edge_types
          if "edge_attr" in d[k] and d[k].edge_attr is not None}
    with torch.no_grad():
        mu, _ = model.encode(x_dict, ei, ea)
    return mu.cpu().numpy().astype(np.float32)


def _load(run_dir, quiescent, p16, device="cpu"):
    """run + axe sénescence FIGÉ (full graph) + poids w_diff par groupe."""
    data, model, gene_symbols, gene_to_idx, baseline, group_expr = gp.load_run(
        Path(run_dir), hidden=128, latent=64, n_layers=3, n_heads=4,
        device=device)
    mu_base = _encode(model, data)
    axis_g, _, _, _ = gp.compute_senescence_axes(
        mu_base, group_expr, gene_symbols, p4_group=quiescent[0],
        p16_groups=tuple(p16), quiescent_groups=list(quiescent))
    u = np.asarray(axis_g, dtype=np.float32)
    u /= (np.linalg.norm(u) + 1e-8)

    expr_by = group_expr.set_index("gene").reindex(gene_symbols).fillna(0.0)
    gnames = list(quiescent) + list(p16)
    expr = np.stack([expr_by[f"mean_{g}"].to_numpy().astype(np.float32)
                     for g in gnames], axis=1)
    expr_diff = np.clip(expr - expr.mean(axis=1, keepdims=True), 0.0, None)
    gg = [et for et in data.edge_types if et[0] == "gene" and et[2] == "gene"]
    return dict(data=data, model=model, gene_symbols=gene_symbols,
                gene_to_idx=gene_to_idx, baseline=baseline, u=u,
                expr_diff=expr_diff, gnames=gnames, gg=gg)


def _ko_dzmean(ctx, d, t):
    """KO du gène t sur le graphe d → (num, cos, dz_mean_unnorm, c_idx) sur le
    groupe dominant en |numérateur|. dz_mean = Σ_g w_diff·Δz / Σ w_diff."""
    u, expr_diff, gnames = ctx["u"], ctx["expr_diff"], ctx["gnames"]
    mb = _encode(ctx["model"], d)
    pert = gp.apply_perturbation(d, torch.tensor([t]), "knockout", 1.0)
    mp = _encode(ctx["model"], pert)
    dz = (mp - mb).astype(np.float32)
    proj = dz @ u
    best = None
    for c in range(len(gnames)):
        w = expr_diff[:, c].copy()
        w[t] = 0.0
        num = float((w * proj).sum())
        dv = (w[:, None] * dz).sum(axis=0)
        cv = float(np.dot(dv, u) / (np.linalg.norm(dv) + 1e-8))
        if best is None or abs(num) > abs(best[0]):
            best = (num, cv, dv, c, w, proj)
    return best


def _neighbors(ctx, t, hops):
    """union des voisins jusqu'à `hops` (arêtes gène-gène, non typées)."""
    gg = ctx["gg"]
    frontier = {t}
    seen = {t}
    for _ in range(hops):
        nxt = set()
        for et in gg:
            ei = ctx["data"][et].edge_index.cpu().numpy()
            nxt |= set(ei[1, np.isin(ei[0], list(frontier))])
            nxt |= set(ei[0, np.isin(ei[1], list(frontier))])
        nxt -= seen
        seen |= nxt
        frontier = nxt
    return seen - {t}


def _drop_incident(ctx, d, t, et_names):
    """clone d ; retire les arêtes des types et_names incidentes au nœud t."""
    dd = d.clone()
    for et in ctx["gg"]:
        if et[1] not in et_names:
            continue
        ei = dd[et].edge_index
        keep = (ei[0] != t) & (ei[1] != t)
        dd[et].edge_index = ei[:, keep]
        if "edge_attr" in dd[et] and dd[et].edge_attr is not None:
            dd[et].edge_attr = dd[et].edge_attr[keep]
    return dd


def _count_incident(ctx, t, et_names):
    return sum(int(((ctx["data"][et].edge_index[0] == t) |
                    (ctx["data"][et].edge_index[1] == t)).sum())
               for et in ctx["gg"] if et[1] in et_names)


# --------------------------------------------------------------------------- #
# Sous-commande : decompose
# --------------------------------------------------------------------------- #
def cmd_decompose(args):
    ctx = _load(args.run_dir, args.quiescent, args.p16, args.device)
    rank_map = _rank_map(args.ranking)
    rows = []
    for g in args.targets:
        if g not in ctx["gene_to_idx"]:
            print(f"  {g}: absent du graphe")
            continue
        t = ctx["gene_to_idx"][g]
        num, cos, _dv, _c, w, proj = _ko_dzmean(ctx, ctx["data"], t)
        contrib = w * proj
        order = np.argsort(-np.abs(contrib))
        abscum = np.cumsum(np.abs(contrib)[order])
        tot = abscum[-1] + 1e-12
        n80 = int(np.searchsorted(abscum, 0.80 * tot) + 1)
        top10 = float(abscum[min(9, len(abscum) - 1)] / tot)
        nb1 = _neighbors(ctx, t, 1)
        nb2 = _neighbors(ctx, t, 2)

        def sh(idxset):
            return (float(np.abs(contrib[list(idxset)]).sum()) / tot
                    if idxset else 0.0)
        # edge_types du 1-hop
        ets = {}
        for et in ctx["gg"]:
            ei = ctx["data"][et].edge_index.cpu().numpy()
            m = (ei[0] == t) | (ei[1] == t)
            n = len(set(ei[0, m]) | set(ei[1, m]) - {t})
            if n:
                ets[et[1]] = n
        rows.append(dict(
            gene=g, driver_rank=rank_map.get(g, -1),
            cos=round(cos, 3), amplitude=round(num, 3),
            n_1hop=len(nb1), n_2hop=len(nb2),
            n80=n80, top10_share=round(top10, 3),
            share_1hop=round(sh(nb1), 3), share_2hop=round(sh(nb2), 3),
            edge_types_1hop="|".join(f"{k}:{v}" for k, v in ets.items())))
        print(f"{g:6s} rang={rows[-1]['driver_rank']:>4} cos={cos:+.3f} "
              f"amp={num:+.2f} | 1-hop={sh(nb1):.2f} 2-hop={sh(nb2):.2f} "
              f"n80={n80}")
    _save(rows, args.out)


# --------------------------------------------------------------------------- #
# Sous-commande : mediate
# --------------------------------------------------------------------------- #
def cmd_mediate(args):
    ctx = _load(args.run_dir, args.quiescent, args.p16, args.device)
    rank_map = _rank_map(args.ranking)
    rng = np.random.default_rng(args.seed)
    lat = ctx["u"].shape[0]
    R = rng.standard_normal((args.n_random, lat)).astype(np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    present = {et[1] for et in ctx["gg"]}
    print("module présent :", "cocat" if COCAT_ETS & present else "-",
          "| rfi:", sorted(RFI_ETS & present))
    rows = []
    for g in args.targets:
        if g not in ctx["gene_to_idx"]:
            print(f"  {g}: absent du graphe")
            continue
        t = ctx["gene_to_idx"][g]
        num_f, cos_f, dzm, _c, _w, _p = _ko_dzmean(ctx, ctx["data"], t)
        _, cos_gen, *_ = _ko_dzmean(ctx, _drop_incident(ctx, ctx["data"], t, GENERIC_ETS), t)
        _, cos_coc, *_ = _ko_dzmean(ctx, _drop_incident(ctx, ctx["data"], t, COCAT_ETS), t)
        _, cos_rfi, *_ = _ko_dzmean(ctx, _drop_incident(ctx, ctx["data"], t, RFI_ETS), t)
        # nulle d'axe aléatoire sur dz_mean réel
        dzm_n = dzm / (np.linalg.norm(dzm) + 1e-8)
        cos_rand = R @ dzm_n
        p_rand = float((np.abs(cos_rand) >= abs(cos_f)).mean())
        rows.append(dict(
            gene=g, driver_rank=rank_map.get(g, -1),
            n_cocat=_count_incident(ctx, t, COCAT_ETS),
            n_rfi=_count_incident(ctx, t, RFI_ETS),
            n_generic=_count_incident(ctx, t, GENERIC_ETS),
            cos_full=round(cos_f, 3), amplitude=round(num_f, 2),
            cos_drop_generic=round(cos_gen, 3),
            cos_drop_cocat=round(cos_coc, 3),
            cos_drop_rfi=round(cos_rfi, 3),
            d_generic=round(cos_gen - cos_f, 3),
            d_cocat=round(cos_coc - cos_f, 3),
            d_rfi=round(cos_rfi - cos_f, 3),
            null_absmean=round(float(np.abs(cos_rand).mean()), 3),
            null_abs_p95=round(float(np.quantile(np.abs(cos_rand), 0.95)), 3),
            p_random_axis=round(p_rand, 4)))
        print(f"{g:6s} cos={cos_f:+.3f} | -gen={cos_gen:+.3f} "
              f"-cocat={cos_coc:+.3f} -rfi={cos_rfi:+.3f}(Δ{cos_rfi-cos_f:+.3f})"
              f" || nulle|cos|~{rows[-1]['null_absmean']:.3f} "
              f"p(rand≥réel)={p_rand:.4f}")
    _save(rows, args.out)


# --------------------------------------------------------------------------- #
def _rank_map(ranking_path):
    if not ranking_path:
        return {}
    df = pd.read_csv(ranking_path, sep="\t")
    col = "target" if "target" in df.columns else df.columns[0]
    df = df.sort_values("driver_score", ascending=False).reset_index(drop=True)
    return {g: i + 1 for i, g in enumerate(df[col].astype(str))}


def _save(rows, out):
    df = pd.DataFrame(rows)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, sep="\t", index=False)
        print("\nÉcrit :", out)
    print(df.to_string(index=False))


def _common(sub):
    sub.add_argument("--run-dir", required=True,
                     help="run VGAE (best_vgae.pt + hetero_graph_vgae.pt + ...)")
    sub.add_argument("--targets", nargs="+", required=True)
    sub.add_argument("--ranking", default=None,
                     help="cross_seed_gene_ranking.tsv (colonne driver_score) "
                          "pour annoter le rang ; optionnel")
    sub.add_argument("--out", default=None, help="TSV de sortie")
    sub.add_argument("--quiescent", nargs="+",
                     default=["P4", "P16_cluster_0"],
                     help="groupes du pôle quiescent (axe V4 par défaut)")
    sub.add_argument("--p16", nargs="+",
                     default=["P16_cluster_1", "P16_cluster_2", "P16_cluster_3"],
                     help="groupes du pôle sénescent")
    sub.add_argument("--device", default="cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = ap.add_subparsers(dest="cmd", required=True)
    d = subs.add_parser("decompose", help="contribution par distance (1/2-hop)")
    _common(d)
    d.set_defaults(func=cmd_decompose)
    m = subs.add_parser("mediate", help="ablation ciblée par source + nulle d'axe")
    _common(m)
    m.add_argument("--n-random", type=int, default=2000,
                   help="axes aléatoires pour la nulle d'axe")
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(func=cmd_mediate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
