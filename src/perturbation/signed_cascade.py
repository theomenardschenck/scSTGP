#!/usr/bin/env python3
"""
signed_cascade.py — Effector-anchored signed cascade (Solution C.1).

Post-hoc, multi-hop generalisation of the 1-hop signed fan-out readout
(`gnn_perturbation.precompute_signed_fanout_context`, Solution A). It runs
ENTIRELY on the frozen encoder + the signed regulatory graph — no re-training,
no per-mode perturbation forward pass.

Biological premise (gnn_futur.md §2.1 / §8.C)
---------------------------------------------
    A activates B ; B inhibits C(pro-senescence)  ⟹  A and B are anti-senescence
by composition of edge signs along the path = structural balance of signed
graphs (Harary 1953) applied to causality. Cf. CARNIVAL (Liu 2019, ILP signed
sub-network), VIPER/MARINa (Alvarez 2016, regulon sign coherence), CellOracle
(Kamimoto 2023, multi-hop signed propagation), and network propagation in
general (Cowen 2017, Nat Rev Genet).

Minimal labelling bias: only a handful of terminal EFFECTORS are labelled
(CDKN2A/p16, CDKN1A/p21, SERPINE1… pro ; LMNB1, MKI67… anti). Sign composition
*derives* the role of every upstream gene → generalises to another pathology
with a few known anchors.

Method (C.1 — iterated signed label spreading, degree-safe)
-----------------------------------------------------------
Signed directed graph W: an edge  S --s--> T  (s∈{-1,+1}) means "S regulates T
with sign s". We propagate the anchor role UPSTREAM (row = regulator = source):

    r_{k+1} = alpha · W_norm · r_k  +  (1-alpha) · Y

where W_norm[A,T] = sign(A→T)·|w_AT| / Σ_T |w_AT| (out-degree-normalised, so a
hub does not accumulate mass merely by degree), Y is the seed (anchor roles),
alpha the damping (effective depth ≈ 1/(1-alpha), exceeds the 2-hop GAT).

We track two NON-NEGATIVE flows (pro `p`, anti `n`) so cancellation — the
frustration signal (p53↔MDM2 cycles, Harary) — is measured, not assumed:

    p_{k+1} = alpha·(P·p_k + N·n_k) + (1-alpha)·Y_pro
    n_{k+1} = alpha·(P·n_k + N·p_k) + (1-alpha)·Y_anti

P/N = row-normalised positive/negative edge weights. A positive edge keeps the
channel, a negative edge swaps pro↔anti. One checks r = p−n satisfies the
single-channel recursion above (positive edge → +r, negative edge → −r).

Outputs (per gene, cascade_role/coherence/reach + score)
--------------------------------------------------------
    cascade_role_<src>      = sgn(p − n)                 multi-hop role
    cascade_coherence_<src> = |p − n| / (p + n) ∈ [0,1]  degree-free confidence
    cascade_reach_<src>     = p + n                      signed-propagation
                                                         proximity to the
                                                         effector core (≈ size
                                                         of the coherent
                                                         downstream cascade,
                                                         the low-degree upstream
                                                         controllability metric
                                                         of gnn_futur §2.3 /
                                                         Liu-Slotine-Barabási 2011)
    cascade_score_<src>     = cascade_role · coherence · reach   (non-ranking)

`<src>` ∈ {known, pred}. `known` = curated sign (CollecTRI/SIGNOR/Reactome) →
NON-circular, defensible. `pred` = bilinear-decoder learned sign (semi-circular,
tests whether the expression-trained latent internalised consistent signs).

The role is AXIS-FREE (anchor-derived); an optional axis only feeds the
`role_latent` agreement diagnostic.

Usage
-----
    # Curated-sign cascade on a trained run (defaults = HUVEC effector core)
    python src/perturbation/signed_cascade.py --run-dir output/gnn_vgae/.../s1

    # Both sign sources, custom anchors, deeper propagation
    python src/perturbation/signed_cascade.py --run-dir RUN \\
        --sign-source both --hops 6 --alpha 0.85 \\
        --effector-pro CDKN2A,CDKN1A,SERPINE1 --effector-anti LMNB1,MKI67

Reads  : <run>/hetero_graph_vgae.pt, gene_embeddings_vgae.csv, best_vgae.pt …
Writes : <run>/signed_cascade.tsv  (+ signed_cascade_summary.json)

Limits (assumed, gnn_futur §8.C)
--------------------------------
- Amplifier of the known: scores what is *reachable from the anchors* → great
  to complete a pathway, blind to a radically novel mechanism.
- Only regulators (genes with outgoing signed edges) get a role; pure sinks
  (incl. the effectors themselves) are NaN — by design, they regulate nothing.
- reactome_fi is often bidirectional/weakly oriented: it inflates reach. Use
  --no-reactome-fi to restrict to signaling + tf_curated (+ OmniPath) if needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# Make gnn_perturbation importable under both layouts (mirrors perturb_top_genes):
#   nested (repo)  : src/perturbation/this.py + src/gnn/gnn_perturbation.py
#   flat (cluster) : src/this.py             + src/gnn_perturbation.py
_THIS_DIR = Path(__file__).resolve().parent
_IMPORT_PATH = (_THIS_DIR if (_THIS_DIR / "gnn_perturbation.py").exists()
                else _THIS_DIR.parent / "gnn")
if str(_IMPORT_PATH) not in sys.path:
    sys.path.insert(0, str(_IMPORT_PATH))

# Default effector anchors — identical to perturb_top_genes.py --effector-pro/-anti
# so the cascade shares the effector core used by the effector-anchored axis.
DEFAULT_EFFECTOR_PRO = "CDKN2A,CDKN1A,CDKN2B,SERPINE1,GLB1"
DEFAULT_EFFECTOR_ANTI = "LMNB1,MKI67,CCNB1,CCNA2,PCNA,TOP2A"


# --------------------------------------------------------------------------- #
# Signed adjacency
# --------------------------------------------------------------------------- #
def build_signed_adjacency(data, model, mu_base, fanout_edge_types,
                           sign_col: int = 1, weight: str = "uniform"):
    """Collect the signed, forward-directed gene→gene edges into arrays.

    Mirrors the edge-collection of `precompute_signed_fanout_context` but keeps
    the flat (src, dst, sign_known, sign_pred, conf) arrays so several
    row-normalised propagation matrices can be built from them.

    Args:
        weight : "uniform" (|w|=1, sign only — most defensible) or "confidence"
                 (|w| = |edge score| = edge_attr[:, 0], if present).

    Returns:
        dict with src, dst (int arrays), sign_known, sign_pred (±1 float),
        conf (>0 float), n_genes, has_bilinear, edge_types (list[str]).
    """
    import torch

    present = [tuple(et) for et in fanout_edge_types
               if tuple(et) in set(data.edge_types)]
    src_all, dst_all, sk_all, conf_all = [], [], [], []
    for et in present:
        store = data[et]
        ei = store.edge_index.cpu().numpy()
        ea = getattr(store, "edge_attr", None)
        if ea is None or ea.ndim < 2 or ea.shape[1] <= sign_col:
            continue
        ea_np = ea.cpu().numpy()
        signs = ea_np[:, sign_col]
        if weight == "confidence" and ea_np.shape[1] >= 1:
            conf = np.abs(ea_np[:, 0]).astype(np.float32)
        else:
            conf = np.ones(ei.shape[1], dtype=np.float32)
        src_all.append(ei[0]); dst_all.append(ei[1])
        sk_all.append(signs); conf_all.append(conf)

    n_genes = int(data["gene"].x.shape[0])
    if not src_all:
        return {"src": np.array([], int), "dst": np.array([], int),
                "sign_known": np.array([], np.float32),
                "sign_pred": np.array([], np.float32),
                "conf": np.array([], np.float32), "n_genes": n_genes,
                "has_bilinear": False, "edge_types": []}

    src = np.concatenate(src_all)
    dst = np.concatenate(dst_all)
    sign_known = np.sign(np.concatenate(sk_all)).astype(np.float32)
    conf = np.concatenate(conf_all).astype(np.float32)
    # Drop self-loops (not a fan-out) and zero-confidence edges.
    keep = (src != dst) & (conf > 0)
    src, dst, sign_known, conf = src[keep], dst[keep], sign_known[keep], conf[keep]
    # Unsigned edges (sign==0, e.g. symmetrised reactome_fi) carry no directional
    # role information → drop them (they would only add cancelling mass).
    keep_s = sign_known != 0
    src, dst, sign_known, conf = src[keep_s], dst[keep_s], sign_known[keep_s], conf[keep_s]

    # Learned sign via the bilinear decoder (single pass on z = mu_base).
    bilinear = getattr(model, "bilinear_decoder", None)
    has_bilinear = bilinear is not None
    if has_bilinear and src.size:
        dev = next(bilinear.parameters()).device
        z = torch.as_tensor(np.asarray(mu_base, dtype=np.float32), device=dev)
        ei_t = torch.as_tensor(np.stack([src, dst]), dtype=torch.long, device=dev)
        with torch.no_grad():
            logit = bilinear.predict_sign_score(z, ei_t).cpu().numpy()
        sign_pred = np.sign(logit).astype(np.float32)
        zero = sign_pred == 0
        sign_pred[zero] = sign_known[zero]
    else:
        sign_pred = sign_known.copy()

    return {"src": src, "dst": dst, "sign_known": sign_known,
            "sign_pred": sign_pred, "conf": conf, "n_genes": n_genes,
            "has_bilinear": has_bilinear,
            "edge_types": ["/".join(et) for et in present]}


def _route_matrices(src, dst, sign, conf, n_genes, norm: str = "sym"):
    """Positive / negative routing matrices (row = regulator A, col = target T).

    The message from a downstream target T flows up to its regulator A, weighted
    by w[A,T]. The normalisation sets what "reach" rewards:

      * ``sym`` (DEFAULT): w = |w_AT| / sqrt(outdeg(A)·indeg(T)) — GCN/label-
        spreading operator (Zhou 2004). Bounded (spectral radius ≤ 1) yet mass
        ACCUMULATES with downstream breadth → a master regulator upstream of
        many effectors scores high, a single-out-edge leaf does not. This is
        the "size of the coherent downstream cascade" of gnn_futur §8.C.
      * ``row``: w = |w_AT| / outdeg(A) — row-stochastic AVERAGE. Degree-safe
        for the *sign* (coherence) but flattens breadth: a deg-1 leaf and a hub
        get comparable reach → NOT suitable for reach ranking. Kept for the
        degree-free coherence companion (gnn_futur §1.2) / ablation.
      * ``in``: w = |w_AT| / indeg(T) — credit a rarer controller of T more.

    outdeg(A)=Σ_T|w|, indeg(T)=Σ_S|w| over the signed graph. Isolated degrees
    fall back to 1 (no division by zero).
    """
    outdeg = np.zeros(n_genes, dtype=np.float64)
    indeg = np.zeros(n_genes, dtype=np.float64)
    np.add.at(outdeg, src, conf)
    np.add.at(indeg, dst, conf)
    outdeg[outdeg == 0] = 1.0
    indeg[indeg == 0] = 1.0
    if norm == "row":
        w = conf.astype(np.float64) / outdeg[src]
    elif norm == "in":
        w = conf.astype(np.float64) / indeg[dst]
    else:  # sym
        w = conf.astype(np.float64) / np.sqrt(outdeg[src] * indeg[dst])
    pos = sign > 0
    P = sparse.csr_matrix((w[pos], (src[pos], dst[pos])), shape=(n_genes, n_genes))
    N = sparse.csr_matrix((w[~pos], (src[~pos], dst[~pos])), shape=(n_genes, n_genes))
    return P, N


def propagate_signed_role(P, N, y_pro, y_anti, alpha: float, hops: int):
    """Iterated signed label spreading (dual non-negative flows).

    Returns (p, n) at the final hop. p−n = signed role magnitude, p+n = reach.
    """
    p = y_pro.astype(np.float64).copy()
    n = y_anti.astype(np.float64).copy()
    for _ in range(int(hops)):
        p_new = alpha * (P @ p + N @ n) + (1.0 - alpha) * y_pro
        n_new = alpha * (P @ n + N @ p) + (1.0 - alpha) * y_anti
        p, n = p_new, n_new
    return p, n


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def cascade_metrics(adj, anchors_pro_idx, anchors_anti_idx, gene_symbols,
                    alpha: float, hops: int, sign_sources=("known",),
                    role_latent_sign=None, norm: str = "sym"):
    """Build the per-gene cascade table for the requested sign source(s)."""
    n = adj["n_genes"]
    y_pro = np.zeros(n, dtype=np.float64)
    y_anti = np.zeros(n, dtype=np.float64)
    y_pro[anchors_pro_idx] = 1.0
    y_anti[anchors_anti_idx] = 1.0
    is_anchor = np.zeros(n, dtype=bool)
    is_anchor[anchors_pro_idx] = True
    is_anchor[anchors_anti_idx] = True
    # out-degree of the signed graph (regulator activity; degree-diagnostic).
    out_deg = np.zeros(n, dtype=np.int64)
    np.add.at(out_deg, adj["src"], 1)

    df = pd.DataFrame({"gene": gene_symbols})
    df["is_effector_anchor"] = is_anchor
    df["signed_out_degree"] = out_deg
    if role_latent_sign is not None:
        df["role_latent_sign"] = np.asarray(role_latent_sign, dtype=np.float32)

    for src_name in sign_sources:
        sign = adj["sign_known"] if src_name == "known" else adj["sign_pred"]
        P, N = _route_matrices(adj["src"], adj["dst"], sign, adj["conf"], n, norm=norm)
        p, q = propagate_signed_role(P, N, y_pro, y_anti, alpha, hops)
        reach = p + q
        role_raw = p - q
        with np.errstate(divide="ignore", invalid="ignore"):
            coherence = np.where(reach > 1e-12, np.abs(role_raw) / reach, 0.0)
        role = np.sign(role_raw).astype(np.float32)
        # Non-anchor regulators only carry a meaningful role.
        reachable = (reach > 1e-12) & (~is_anchor)
        role = np.where(reachable, role, np.nan)
        coherence = np.where(reachable, coherence, np.nan)
        reach_out = np.where(reachable, reach, np.nan)
        score = role * coherence * reach_out
        sfx = f"_{src_name}"
        df[f"cascade_role{sfx}"] = role
        df[f"cascade_coherence{sfx}"] = coherence.astype(np.float32)
        df[f"cascade_reach{sfx}"] = reach_out.astype(np.float32)
        df[f"cascade_score{sfx}"] = score.astype(np.float32)
        # Agreement with the resting-latent role (diagnostic, gnn_futur §8.A):
        # low agreement flags marker/driver conflicts to arbitrate cross-dataset.
        if role_latent_sign is not None:
            rl = np.asarray(role_latent_sign, dtype=np.float32)
            agree = np.where(reachable & (rl != 0), (role == np.sign(rl)), np.nan)
            df[f"cascade_latent_agree{sfx}"] = agree.astype(np.float32)
    return df


def _resolve_anchor_idx(genes_csv: str, gene_to_idx: dict, gene_symbols):
    """Parse a comma-list of symbols → present indices (+ report missing)."""
    want = [g.strip() for g in genes_csv.split(",") if g.strip()]
    idx, missing = [], []
    for g in want:
        if g in gene_to_idx:
            idx.append(gene_to_idx[g])
        else:
            missing.append(g)
    return np.array(idx, dtype=np.int64), want, missing


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Effector-anchored signed cascade (Solution C.1).")
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="Trained VGAE run directory (load_run-compatible).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output TSV (default: <run-dir>/signed_cascade.tsv).")
    ap.add_argument("--effector-pro", default=DEFAULT_EFFECTOR_PRO,
                    help="Pro-senescence effector anchors (comma-list).")
    ap.add_argument("--effector-anti", default=DEFAULT_EFFECTOR_ANTI,
                    help="Anti-senescence effector anchors (comma-list).")
    ap.add_argument("--alpha", type=float, default=0.85,
                    help="Damping (effective depth ~1/(1-alpha)). Default 0.85.")
    ap.add_argument("--hops", type=int, default=6,
                    help="Propagation iterations (> 2 exceeds the GAT). Default 6.")
    ap.add_argument("--sign-source", choices=["known", "pred", "both"],
                    default="known",
                    help="Curated sign (default, non-circular) / learned bilinear "
                         "/ both.")
    ap.add_argument("--weight", choices=["uniform", "confidence"],
                    default="uniform",
                    help="Edge weight: uniform (sign only) or |edge score|.")
    ap.add_argument("--norm", choices=["sym", "row", "in"], default="sym",
                    help="Propagation normalisation. sym (default) = breadth-"
                         "rewarding bounded label spreading (Zhou 2004) ; row = "
                         "degree-safe average (coherence companion) ; in = "
                         "rarer-controller credit.")
    ap.add_argument("--no-reactome-fi", action="store_true",
                    help="Exclude reactome_fi (weakly oriented) from the fan-out.")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | auto.")
    # load_run hyperparameters (mirror perturb_top_genes defaults; read from
    # vgae_metrics.json when present via load_run's own fallback path).
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    args = ap.parse_args()

    from gnn_perturbation import (  # noqa: E402  (heavy: torch + PyG)
        load_run, prepare_baseline, precompute_signed_fanout_context,
        FANOUT_EDGE_TYPES,
    )

    fanout_types = tuple(
        et for et in FANOUT_EDGE_TYPES
        if not (args.no_reactome_fi and et[1] == "reactome_fi"))

    print(f"[cascade] loading {args.run_dir} …")
    data, model, gene_symbols, gene_to_idx, baseline, group_expr = load_run(
        args.run_dir, args.hidden, args.latent, args.n_layers, args.n_heads,
        device=args.device)

    # Baseline forward pass → mu_base (needed for the bilinear sign_pred and for
    # the optional resting-latent role diagnostic).
    (spec, base_imp, base_rank, z_cg_base, mu_base,
     axis_global, axes_cluster, axes_transition) = prepare_baseline(
        model, data, baseline, gene_symbols, group_expr)

    role_latent_sign = None
    if axis_global is not None:
        ctx = precompute_signed_fanout_context(
            model, data, mu_base, axis_global, fanout_edge_types=fanout_types)
        role_latent_sign = ctx.get("role_latent_sign")

    adj = build_signed_adjacency(data, model, mu_base, fanout_types,
                                 weight=args.weight)
    print(f"[cascade] signed graph: {adj['src'].size} directed edges over "
          f"{adj['n_genes']} genes ; types={adj['edge_types']} ; "
          f"bilinear={adj['has_bilinear']}")

    pro_idx, pro_want, pro_miss = _resolve_anchor_idx(
        args.effector_pro, gene_to_idx, gene_symbols)
    anti_idx, anti_want, anti_miss = _resolve_anchor_idx(
        args.effector_anti, gene_to_idx, gene_symbols)
    print(f"[cascade] anchors pro {len(pro_idx)}/{len(pro_want)} "
          f"(missing {pro_miss or '-'}) ; anti {len(anti_idx)}/{len(anti_want)} "
          f"(missing {anti_miss or '-'})")
    if len(pro_idx) == 0 or len(anti_idx) == 0:
        raise SystemExit("[cascade] need ≥1 pro AND ≥1 anti anchor present in "
                         "the graph — aborting.")

    sources = (["known", "pred"] if args.sign_source == "both"
               else [args.sign_source])
    if "pred" in sources and not adj["has_bilinear"]:
        print("[cascade] no bilinear decoder in this run → 'pred' == 'known'.")

    df = cascade_metrics(adj, pro_idx, anti_idx, gene_symbols,
                         alpha=args.alpha, hops=args.hops, sign_sources=sources,
                         role_latent_sign=role_latent_sign, norm=args.norm)

    # Sort by |score| of the FIRST (headline) source for a readable head.
    head_src = sources[0]
    df = df.sort_values(f"cascade_score_{head_src}",
                        key=lambda s: s.abs(), ascending=False,
                        na_position="last").reset_index(drop=True)

    out = args.out or (args.run_dir / "signed_cascade.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)

    # Summary (frustration = mean 1-coherence over reached regulators).
    summary = {"run_dir": str(args.run_dir), "alpha": args.alpha,
               "hops": args.hops, "norm": args.norm, "sign_sources": sources,
               "weight": args.weight,
               "edge_types": adj["edge_types"], "n_signed_edges": int(adj["src"].size),
               "has_bilinear": adj["has_bilinear"],
               "anchors_pro": [g for g in pro_want if g in gene_to_idx],
               "anchors_anti": [g for g in anti_want if g in gene_to_idx],
               "missing_anchors": pro_miss + anti_miss}
    for s in sources:
        coh = df[f"cascade_coherence_{s}"].dropna()
        summary[f"n_scored_{s}"] = int(coh.size)
        summary[f"mean_coherence_{s}"] = float(coh.mean()) if coh.size else None
        summary[f"frustration_{s}"] = float((1 - coh).mean()) if coh.size else None
    (out.parent / (out.stem + "_summary.json")).write_text(json.dumps(summary, indent=2))

    print(f"[cascade] wrote {out}  ({len(df)} genes, "
          f"{summary.get(f'n_scored_{head_src}')} scored regulators)")
    show = [c for c in df.columns if c in ("gene", "is_effector_anchor",
            f"cascade_role_{head_src}", f"cascade_coherence_{head_src}",
            f"cascade_reach_{head_src}", f"cascade_score_{head_src}")]
    with pd.option_context("display.width", 140):
        print(df[show].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
