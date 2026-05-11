#!/usr/bin/env python3
"""
edge_attention.py — extraction et visualisation des poids d'attention GAT
appris par le VGAE entraîné.

Principe
--------
L'encodeur HeteroEncoder de `gnn_vgae.py` empile N couches de
HeteroConv ; chaque HeteroConv contient un GATConv indépendant par
type d'arête. Chaque GATConv apprend des coefficients d'attention
α_ij ∈ [0,1] par tête, qui pondèrent les messages de j vers i.

Ce script :
  1. Charge le graphe hétérogène `hetero_graph_vgae.pt` et les poids
     entraînés `vgae_weights.pt` d'un run donné.
  2. Effectue une forward pass en monkey-patchant chaque GATConv pour
     forcer `return_attention_weights=True` ; capture α (shape
     `[E, n_heads]`) sans modifier le modèle.
  3. Agrège α par tête (mean) et par couche (mean | max | last) selon
     l'option choisie — voir §3 ci-dessous.
  4. Exporte un TSV long-format edge_attention.tsv :
        layer, edge_type, src, dst, alpha_mean, alpha_max, edge_attr
     + figure réseau optionnelle centrée sur des gènes cibles.

Choix d'agrégation
------------------
- **Heads** : mean par défaut (`--head-agg max` possible).
  Référence : Veličković et al. 2018 §3.3 — la moyenne sur les têtes
  est l'agrégat standard, plus stable que max.
- **Layers** : `mean` (défaut), `max`, `last` (couche finale), `prod`
  (produit ≈ chemin de propagation). Aucun choix n'est "correct"
  académiquement — on expose plusieurs colonnes.
- **Avertissement** : l'attention GAT n'est pas une explication causale
  calibrée (cf. Ying et al. 2019, GNNExplainer §4 ; et discussion sur
  edge attention dans le rapport §19.8). Pour une explication
  défendable, utiliser GNNExplainer (cf. TODO Tier 2 Phase 5).

Usage
-----
    python src/validation/edge_attention.py extract \\
        --run-dir output/gnn_vgae/V4.0/v4-full.s1 \\
        --target-genes H2AFZ HMGB1 ASNS \\
        --top-k 30 \\
        --out-dir output/gnn_vgae/V4.0/v4-full.s1/attention

    python src/validation/edge_attention.py figure \\
        --attention-tsv .../attention/edge_attention.tsv \\
        --target-gene H2AFZ \\
        --top-k 20

Cf. §19.8 du rapport, et la TODO Tier 2 Phase 5 (GNNExplainer
préféré pour interprétabilité défendable).
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sys path bootstrap — permet d'importer HeteroEncoder/VGAE depuis gnn_vgae
# ---------------------------------------------------------------------------
def _bootstrap_paths():
    here = Path(__file__).resolve()
    project_root = here.parents[2]  # gnn_huvec/
    for p in [project_root / "src", project_root]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


_bootstrap_paths()
from gnn._vgae_model import HeteroEncoder, VGAE  # noqa: E402


# ---------------------------------------------------------------------------
# Monkey-patch : capture α sans modifier le modèle
# ---------------------------------------------------------------------------
@contextmanager
def capture_attention(model: VGAE):
    """
    Context manager qui monkey-patche chaque GATConv pour capturer α.

    HeteroConv appelle `gat(x, edge_index, edge_attr=...)` SANS
    `return_attention_weights` — on intercepte et on stocke α sur le
    module, sans changer le type de retour vu par HeteroConv.

    Yields
    ------
    attentions : dict
        Rempli pendant le bloc `with`. Clés : (layer_idx, edge_type).
        Valeurs : dict avec 'edge_index' (np.array shape [2, E_total],
        peut inclure self-loops ajoutées par GATConv) et 'alpha'
        (np.array shape [E_total, n_heads]).
    """
    attentions: dict[tuple[int, tuple], dict] = {}

    # Référence vers HeteroConv interne pour identifier (layer_idx, edge_type)
    encoder = model.encoder
    gat_to_id: dict[int, tuple[int, tuple]] = {}
    for layer_idx, hetero_conv in enumerate(encoder.convs):
        for et, gat in hetero_conv.convs.items():
            gat_to_id[id(gat)] = (layer_idx, et)

    # Sauvegarder les forwards originaux
    originals = {}
    for gat_id, (layer_idx, et) in gat_to_id.items():
        for hetero_conv in encoder.convs:
            for et2, gat in hetero_conv.convs.items():
                if id(gat) == gat_id:
                    originals[gat_id] = gat.forward
                    break

    def make_wrapper(gat, gat_id):
        original = gat.forward

        def wrapper(*args, **kwargs):
            kwargs["return_attention_weights"] = True
            out, (ei_out, alpha) = original(*args, **kwargs)
            key = gat_to_id[gat_id]
            attentions[key] = {
                "edge_index": ei_out.detach().cpu().numpy(),
                "alpha": alpha.detach().cpu().numpy(),
            }
            return out  # HeteroConv attend seulement out, pas le tuple

        return wrapper

    for hetero_conv in encoder.convs:
        for et, gat in hetero_conv.convs.items():
            gat.forward = make_wrapper(gat, id(gat))

    try:
        yield attentions
    finally:
        # Restaurer
        for hetero_conv in encoder.convs:
            for et, gat in hetero_conv.convs.items():
                if id(gat) in originals:
                    gat.forward = originals[id(gat)]


# ---------------------------------------------------------------------------
# Chargement run
# ---------------------------------------------------------------------------
def load_run(run_dir: Path, device: str = "cpu"):
    """Charge graph + weights + gene_symbols pour 1 run.

    Returns
    -------
    data : HeteroData
    model : VGAE en mode eval
    gene_symbols : list[str]
    """
    graph_path = run_dir / "hetero_graph_vgae.pt"
    weights_path = run_dir / "vgae_weights.pt"
    if not graph_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Missing hetero_graph_vgae.pt or vgae_weights.pt in {run_dir}"
        )

    data = torch.load(graph_path, weights_only=False, map_location=device)

    # Hyperparams inférés depuis le graphe (consistent avec gnn_vgae.py)
    gene_in = data["gene"].x.shape[1]
    cell_in = data["cell_group"].x.shape[1] if "cell_group" in data.node_types else 3
    n_genes = data["gene"].x.shape[0]

    state = torch.load(weights_path, weights_only=False, map_location=device)
    # Infer hyperparams from weights
    hidden = state["encoder.gene_proj.weight"].shape[0]
    latent = state["encoder.mu_head.weight"].shape[0]
    n_layers = sum(1 for k in state if k.startswith("encoder.norms.")
                   and k.endswith(".gene.weight"))
    # n_heads : déduit depuis une GATConv att_src ∈ (1, n_heads, head_dim)
    att_keys = [k for k in state if k.endswith(".att_src")]
    if not att_keys:
        raise RuntimeError("Pas de .att_src dans le state_dict — modèle non GAT ?")
    att_shape = state[att_keys[0]].shape  # (1, n_heads, head_dim)
    n_heads = int(att_shape[1])
    head_dim = int(att_shape[2])
    assert hidden == n_heads * head_dim, (
        f"hidden ({hidden}) != n_heads * head_dim ({n_heads}×{head_dim}). "
        f"Vérifie la cohérence du state_dict."
    )

    encoder = HeteroEncoder(
        gene_in=gene_in,
        cell_in=cell_in,
        hidden=hidden,
        latent=latent,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=0.0,
        available_edge_types=list(data.edge_types),
    )
    model = VGAE(encoder)
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)

    # Gene symbols
    emb_csv = run_dir / "gene_embeddings_vgae.csv"
    if emb_csv.exists():
        gene_symbols = pd.read_csv(emb_csv, index_col=0).index.tolist()
    else:
        gene_symbols = [f"gene_{i}" for i in range(n_genes)]

    return data, model, gene_symbols


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def aggregate_heads(alpha: np.ndarray, mode: str = "mean") -> np.ndarray:
    """alpha shape (E, n_heads) → (E,)"""
    if mode == "mean":
        return alpha.mean(axis=1)
    if mode == "max":
        return alpha.max(axis=1)
    raise ValueError(f"unknown head_agg: {mode}")


def attention_to_dataframe(
    attentions: dict,
    gene_symbols: list[str],
    head_agg: str = "mean",
) -> pd.DataFrame:
    """
    Concat tous les (layer, edge_type) en un DataFrame long.

    Colonnes : layer, edge_type, src_idx, dst_idx, src_gene, dst_gene,
               alpha_mean_heads, alpha_max_heads, n_heads, is_self_loop
    """
    rows = []
    n_genes = len(gene_symbols)
    for (layer_idx, et), payload in attentions.items():
        ei = payload["edge_index"]      # (2, E_total)
        alpha = payload["alpha"]        # (E_total, n_heads)
        if alpha.ndim == 1:
            alpha = alpha[:, None]
        n_heads = alpha.shape[1]
        a_mean = aggregate_heads(alpha, "mean")
        a_max = aggregate_heads(alpha, "max")
        src_type, rel, dst_type = et

        for k in range(ei.shape[1]):
            s, d = int(ei[0, k]), int(ei[1, k])
            # GATConv ajoute self-loops si add_self_loops=True ; chez nous False,
            # mais on garde un drapeau au cas où la version PyG les ajoute quand-même.
            is_self = (src_type == dst_type) and (s == d)
            src_g = gene_symbols[s] if src_type == "gene" and s < n_genes \
                else f"{src_type}_{s}"
            dst_g = gene_symbols[d] if dst_type == "gene" and d < n_genes \
                else f"{dst_type}_{d}"
            rows.append({
                "layer": layer_idx,
                "edge_type": f"{src_type}-{rel}-{dst_type}",
                "src_idx": s,
                "dst_idx": d,
                "src_node": src_g,
                "dst_node": dst_g,
                "alpha_mean_heads": float(a_mean[k]),
                "alpha_max_heads": float(a_max[k]),
                "n_heads": n_heads,
                "is_self_loop": is_self,
            })

    return pd.DataFrame(rows)


def aggregate_layers(df: pd.DataFrame, mode: str = "mean") -> pd.DataFrame:
    """
    Réduit la dimension `layer` pour chaque (src, dst, edge_type).

    mode : 'mean' | 'max' | 'last' | 'prod'
    """
    df = df[~df["is_self_loop"]].copy()
    group_cols = ["edge_type", "src_idx", "dst_idx", "src_node", "dst_node"]
    if mode == "mean":
        agg = df.groupby(group_cols, as_index=False)["alpha_mean_heads"].mean()
    elif mode == "max":
        agg = df.groupby(group_cols, as_index=False)["alpha_mean_heads"].max()
    elif mode == "last":
        last_layer = df["layer"].max()
        agg = df[df["layer"] == last_layer].drop(columns=["layer"])
        agg = agg.rename(columns={"alpha_mean_heads": "alpha_mean_heads"})
    elif mode == "prod":
        agg = (df.groupby(group_cols, as_index=False)
                 ["alpha_mean_heads"]
                 .apply(lambda s: float(np.prod(s.values))))
    else:
        raise ValueError(f"unknown layer_agg: {mode}")
    agg = agg.rename(columns={"alpha_mean_heads": f"alpha_{mode}_layers"})
    return agg.sort_values(f"alpha_{mode}_layers", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sub-graph extraction pour un gène cible
# ---------------------------------------------------------------------------
def attention_neighborhood(
    df_per_layer: pd.DataFrame,
    target_gene: str,
    top_k: int = 30,
    direction: str = "in",
) -> pd.DataFrame:
    """
    Sous-graphe d'attention autour de `target_gene` : top-k voisins
    par α moyen sur les couches.

    direction : 'in' = voisins qui envoient des messages vers la cible ;
                'out' = voisins que la cible influence ;
                'both' = les deux.
    """
    df = df_per_layer[~df_per_layer["is_self_loop"]].copy()
    if direction == "in":
        mask = df["dst_node"] == target_gene
    elif direction == "out":
        mask = df["src_node"] == target_gene
    else:
        mask = (df["src_node"] == target_gene) | (df["dst_node"] == target_gene)
    sub = df[mask].copy()
    if sub.empty:
        return sub

    # Agréger sur layers (mean) pour chaque arête unique
    group_cols = ["src_node", "dst_node", "edge_type"]
    sub_agg = (sub.groupby(group_cols, as_index=False)
                  .agg(alpha_mean=("alpha_mean_heads", "mean"),
                       alpha_max=("alpha_max_heads", "max"),
                       n_layers=("layer", "nunique")))
    sub_agg = sub_agg.sort_values("alpha_mean", ascending=False)
    return sub_agg.head(top_k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_attention_neighborhood(
    neighborhood: pd.DataFrame,
    target_gene: str,
    out_path: Path,
):
    """Bar chart horizontal : top-k voisins par α, coloré par edge_type."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if neighborhood.empty:
        print(f"[figure] no attention found for {target_gene}")
        return

    df = neighborhood.copy()
    df["neighbor"] = df.apply(
        lambda r: r["src_node"] if r["dst_node"] == target_gene else r["dst_node"],
        axis=1,
    )
    df = df.sort_values("alpha_mean", ascending=True)

    palette = sns.color_palette("tab10", n_colors=df["edge_type"].nunique())
    et_color = dict(zip(sorted(df["edge_type"].unique()), palette))
    colors = [et_color[e] for e in df["edge_type"]]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(df))))
    ax.barh(df["neighbor"], df["alpha_mean"], color=colors,
            edgecolor="black", linewidth=0.3)
    ax.set_xlabel("α moyen (heads, layers)")
    ax.set_title(f"Top-{len(df)} attention neighbors of {target_gene}\n"
                 f"(mean over heads × mean over layers)")
    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in et_color.values()]
    ax.legend(handles, et_color.keys(), loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] wrote {out_path}")


# ---------------------------------------------------------------------------
# Commandes
# ---------------------------------------------------------------------------
def cmd_extract(args):
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir \
        else run_dir / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] loading run {run_dir.name} ...")
    data, model, gene_symbols = load_run(run_dir, device=args.device)
    print(f"[extract] {data['gene'].x.shape[0]} genes, "
          f"{len(data.edge_types)} edge types, "
          f"{model.encoder.n_layers} GAT layers")

    # Préparer inputs
    x_dict = {k: data[k].x.to(args.device) for k in data.node_types}
    edge_index_dict = {et: data[et].edge_index.to(args.device)
                       for et in data.edge_types}
    edge_attr_dict = {}
    for et in data.edge_types:
        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
            edge_attr_dict[et] = data[et].edge_attr.to(args.device)

    print("[extract] forward + capture α ...")
    with torch.no_grad(), capture_attention(model) as captured:
        _ = model.encoder(x_dict, edge_index_dict,
                          edge_attr_dict or None)

    print(f"[extract] captured {len(captured)} (layer, edge_type) tensors")

    df = attention_to_dataframe(captured, gene_symbols, head_agg=args.head_agg)
    out_tsv = out_dir / "edge_attention.tsv"
    df.to_csv(out_tsv, sep="\t", index=False, float_format="%.5f")
    print(f"[extract] wrote {out_tsv}  ({len(df)} rows)")

    # Agrégation layers
    for mode in ("mean", "max", "last", "prod"):
        agg = aggregate_layers(df, mode=mode)
        out_agg = out_dir / f"edge_attention_layer_{mode}.tsv"
        agg.to_csv(out_agg, sep="\t", index=False, float_format="%.5f")
    print(f"[extract] wrote layer aggregations (mean/max/last/prod)")

    # Sous-graphes per-target + figures
    if args.target_genes:
        for gene in args.target_genes:
            neigh = attention_neighborhood(
                df, gene, top_k=args.top_k, direction=args.direction
            )
            if neigh.empty:
                print(f"[extract] target {gene} not found / no edges — skip")
                continue
            tsv = out_dir / f"neighborhood_{gene}.tsv"
            neigh.to_csv(tsv, sep="\t", index=False, float_format="%.5f")
            png = out_dir / f"neighborhood_{gene}.png"
            plot_attention_neighborhood(neigh, gene, png)

    # Manifest pour traçabilité
    manifest = {
        "run_dir": str(run_dir),
        "n_layers": int(model.encoder.n_layers),
        "edge_types": [str(et) for et in data.edge_types],
        "head_agg": args.head_agg,
        "target_genes": list(args.target_genes or []),
        "top_k": int(args.top_k),
        "direction": args.direction,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[extract] done — {out_dir}")


def cmd_figure(args):
    df = pd.read_csv(args.attention_tsv, sep="\t")
    out_path = Path(args.out_path) if args.out_path else \
        Path(args.attention_tsv).parent / f"neighborhood_{args.target_gene}.png"
    neigh = attention_neighborhood(
        df, args.target_gene, top_k=args.top_k, direction=args.direction
    )
    if neigh.empty:
        print(f"[figure] no attention for {args.target_gene}")
        return
    plot_attention_neighborhood(neigh, args.target_gene, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="Extract attention from a trained VGAE run.")
    pe.add_argument("--run-dir", required=True,
                    help="Run dir containing hetero_graph_vgae.pt + vgae_weights.pt")
    pe.add_argument("--out-dir", default=None,
                    help="Output dir (default: <run-dir>/attention)")
    pe.add_argument("--target-genes", nargs="*", default=[],
                    help="Genes to extract neighborhood for (TSV + figure each)")
    pe.add_argument("--top-k", type=int, default=30,
                    help="Top-K attention neighbors per target (default 30)")
    pe.add_argument("--direction", choices=["in", "out", "both"], default="in",
                    help="Edges going TO target / FROM target / both")
    pe.add_argument("--head-agg", choices=["mean", "max"], default="mean",
                    help="Aggregation across attention heads (default mean)")
    pe.add_argument("--device", default="cpu", help="cpu | cuda:0 | ...")
    pe.set_defaults(func=cmd_extract)

    pf = sub.add_parser("figure", help="Plot neighborhood figure from existing TSV.")
    pf.add_argument("--attention-tsv", required=True,
                    help="Path to edge_attention.tsv produced by extract")
    pf.add_argument("--target-gene", required=True)
    pf.add_argument("--top-k", type=int, default=30)
    pf.add_argument("--direction", choices=["in", "out", "both"], default="in")
    pf.add_argument("--out-path", default=None)
    pf.set_defaults(func=cmd_figure)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
