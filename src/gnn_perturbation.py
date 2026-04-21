#!/usr/bin/env python3
"""
gnn_perturbation.py — In silico perturbation of the trained VGAE.

Loads a trained VGAE run (output/gnn_vgae/V*_Run*/) and simulates
knockdown / knockout / over-expression of one gene, a set of genes, or
a full REACTOME pathway. Runs a forward pass through the FROZEN encoder
(no retraining), recomputes the composite importance score and reports:

  * delta_ranking.csv       — per-gene baseline vs perturbed rank & importance
  * delta_ora_top_up_reactome.tsv — REACTOME ORA on the top rising genes
  * summary.json            — top movers, stats, signature delta-pathways

Perturbation modes
------------------
  knockdown   : gene feature row -> 0
  overexpress : gene feature row *= FACTOR (default 2.0)
  knockout    : knockdown + remove every edge incident to the gene

Usage
-----
    # Single-gene knockdown on V3_Run2
    python src/gnn_perturbation.py \\
        --run-dir output/gnn_vgae/V3_Run2 \\
        --mode knockdown \\
        --genes PEX11B

    # Pathway-level knockout (gene symbols read from a file)
    python src/gnn_perturbation.py \\
        --run-dir output/gnn_vgae/V3_Run2 \\
        --mode knockout \\
        --gene-list docs/peroxisome_genes.txt \\
        --out-dir output/perturbation/peroxisome_KO

    # Over-expression with a custom factor
    python src/gnn_perturbation.py \\
        --run-dir output/gnn_vgae/V3_Run2 \\
        --mode overexpress \\
        --genes ADM2 --factor 3.0

Design notes
------------
- The 5th importance component (`specificity`) is computed from RNAseq
  expression (entropy across cell_groups), so it is invariant under gene
  feature perturbation. We therefore reuse the baseline `vgae_specificity`
  column instead of recomputing it — keeping the perturbed score directly
  comparable to the baseline published by `gnn_vgae.py`.
- The encoder hyperparameters (hidden/latent/n_layers/n_heads) default to
  the training-script defaults. Override via CLI if a run used different
  values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GATConv, HeteroConv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ora_consensus import (  # noqa: E402
    load_background,
    load_reactome_gmt,
    run_ora,
    write_tsv,
)

DEFAULT_HIDDEN = 128
DEFAULT_LATENT = 64
DEFAULT_LAYERS = 3
DEFAULT_HEADS = 4

# Ordre FIXE des 5 noeuds cell_group, tel que construit dans gnn_vgae.py:468.
# L'index dans ce tuple = l'index du noeud cell_group dans le HeteroData.
CELL_GROUPS = ("P4", "P16_cluster_0", "P16_cluster_1",
               "P16_cluster_2", "P16_cluster_3")


# --------------------------------------------------------------------------- #
# Model (duplicated from gnn_vgae.py to keep this tool self-contained).
# --------------------------------------------------------------------------- #
class HeteroEncoder(nn.Module):
    def __init__(self, gene_in, cell_in, hidden, latent, n_layers,
                 n_heads=4, dropout=0.2):
        super().__init__()
        self.n_layers = n_layers
        head_dim = hidden // n_heads
        self.gene_proj = nn.Linear(gene_in, hidden)
        self.cell_proj = nn.Linear(cell_in, hidden)

        edge_types_dims = [
            (("gene", "ppi", "gene"), 1),
            (("gene", "same_pathway", "gene"), None),
            (("gene", "regulates", "gene"), 1),
            (("gene", "regulated_by", "gene"), 1),
            (("cell_group", "expresses", "gene"), 7),
            (("gene", "expressed_in", "cell_group"), 7),
            (("gene", "coexpression", "gene"), 1),
            (("gene", "metabolic_cocatalysis", "gene"), 2),
        ]
        self.edge_dims = {et: dim for et, dim in edge_types_dims}
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            conv_dict = {}
            for et, ed in edge_types_dims:
                conv_kwargs = dict(heads=n_heads, concat=True,
                                   dropout=dropout, add_self_loops=False)
                if ed is not None:
                    conv_kwargs["edge_dim"] = ed
                conv_dict[et] = GATConv(hidden, head_dim, **conv_kwargs)
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(nn.ModuleDict({
                "gene": nn.BatchNorm1d(hidden),
                "cell_group": nn.BatchNorm1d(hidden),
            }))
        self.dropout = nn.Dropout(dropout)
        self.mu_head = nn.Linear(hidden, latent)
        self.logvar_head = nn.Linear(hidden, latent)
        # Stash du hidden cell_group après le dernier forward. Utilisé par
        # l'option A (shift per cell_group après perturbation) sans changer
        # la signature de forward() / VGAE.encode() / compute_importance().
        self.last_cell_group_h: torch.Tensor | None = None

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x_dict = {
            "gene": F.relu(self.gene_proj(x_dict["gene"])),
            "cell_group": F.relu(self.cell_proj(x_dict["cell_group"])),
        }
        for i in range(self.n_layers):
            x_prev = {k: v.clone() for k, v in x_dict.items()}
            active = {k: v for k, v in edge_index_dict.items() if v.numel() > 0}
            if edge_attr_dict is not None:
                active_attrs = {
                    k: edge_attr_dict[k]
                    for k in active
                    if k in edge_attr_dict and self.edge_dims.get(k) is not None
                }
                x_dict = self.convs[i](x_dict, active,
                                       edge_attr_dict=active_attrs)
            else:
                x_dict = self.convs[i](x_dict, active)
            for key in x_dict:
                x_dict[key] = self.norms[i][key](x_dict[key])
                x_dict[key] = F.relu(x_dict[key])
                x_dict[key] = self.dropout(x_dict[key])
                x_dict[key] = x_dict[key] + x_prev[key]
        self.last_cell_group_h = x_dict["cell_group"].detach().clone()
        return self.mu_head(x_dict["gene"]), self.logvar_head(x_dict["gene"])


class VGAE(nn.Module):
    def __init__(self, encoder, tau_init=2.0, tau_max=3.0):
        super().__init__()
        self.encoder = encoder
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(tau_init))))
        self.log_tau_max = np.log(tau_max)

    def encode(self, x_dict, edge_index_dict, edge_attr_dict=None):
        mu, logvar = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        return mu, logvar

    def decode(self, z, edge_index):
        src, dst = edge_index
        z_src = F.normalize(z[src], dim=1)
        z_dst = F.normalize(z[dst], dim=1)
        cos_sim = (z_src * z_dst).sum(dim=1)
        log_tau = torch.clamp(self.log_tau, max=self.log_tau_max)
        return torch.exp(log_tau) * cos_sim


# --------------------------------------------------------------------------- #
# Importance score (mirrors gnn_vgae.py §11).
# --------------------------------------------------------------------------- #
def compute_importance(model, data, baseline_specificity):
    model.eval()
    x_dict = {"gene": data["gene"].x, "cell_group": data["cell_group"].x}
    edge_index_dict = {k: data[k].edge_index for k in data.edge_types}
    edge_attr_dict = {
        k: data[k].edge_attr
        for k in data.edge_types
        if "edge_attr" in data[k] and data[k].edge_attr is not None
    }

    with torch.no_grad():
        mu, logvar = model.encode(x_dict, edge_index_dict, edge_attr_dict)

    mu_np = mu.cpu().numpy()
    n_nodes = mu_np.shape[0]

    # (1) latent norm
    emb_norm = np.linalg.norm(mu_np, axis=1)
    emb_norm_score = emb_norm / (emb_norm.max() + 1e-8)

    # (2) reconstruction fidelity averaged over all gene↔gene edge types
    gene_edges = []
    for et in data.edge_types:
        if et[0] == "gene" and et[2] == "gene":
            ei = data[et].edge_index
            if ei.numel() > 0:
                gene_edges.append(ei)
    if gene_edges:
        all_ei = torch.cat(gene_edges, dim=1)
        with torch.no_grad():
            sc = torch.sigmoid(model.decode(mu, all_ei)).cpu().numpy()
        src = all_ei[0].cpu().numpy()
        dst = all_ei[1].cpu().numpy()
        err = np.zeros(n_nodes, dtype=np.float32)
        cnt = np.zeros(n_nodes, dtype=np.float32)
        np.add.at(err, src, 1.0 - sc)
        np.add.at(err, dst, 1.0 - sc)
        np.add.at(cnt, src, 1.0)
        np.add.at(cnt, dst, 1.0)
        safe = np.where(cnt > 0, cnt, 1.0)
        recon_err = np.where(cnt > 0, err / safe, 0.0)
    else:
        recon_err = np.zeros(n_nodes, dtype=np.float32)
    recon_err_score = recon_err / (recon_err.max() + 1e-8)

    # (3) uncertainty from logvar
    sigma = torch.exp(0.5 * logvar).cpu().numpy()
    uncertainty = sigma.mean(axis=1)
    uncertainty_score = uncertainty / (uncertainty.max() + 1e-8)

    # (4) local density via cosine k-NN + variance regularisation
    k = min(20, n_nodes - 1)
    emb_normed = mu_np / (np.linalg.norm(mu_np, axis=1, keepdims=True) + 1e-8)
    knn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    knn.fit(emb_normed)
    dists, _ = knn.kneighbors(emb_normed)
    dists = dists[:, 1:]
    raw_density = 1.0 / (dists.mean(axis=1) + 1e-8)
    density = rankdata(raw_density) / len(raw_density)
    variance = rankdata(dists.var(axis=1)) / len(dists)
    density = density * variance
    density_score = (density / (density.max() + 1e-8)).astype(np.float32)

    # (5) specificity — invariant under gene feature perturbation
    importance = (
        emb_norm_score
        + density_score
        + (1.0 - recon_err_score)
        + (1.0 - uncertainty_score)
        + baseline_specificity
    ) / 5.0
    importance = importance / (importance.max() + 1e-8)

    return {
        "importance": importance.astype(np.float32),
        "emb_norm": emb_norm_score.astype(np.float32),
        "density": density_score.astype(np.float32),
        "recon_fidelity": (1.0 - recon_err_score).astype(np.float32),
        "certainty": (1.0 - uncertainty_score).astype(np.float32),
    }


# --------------------------------------------------------------------------- #
# Perturbation.
# --------------------------------------------------------------------------- #
def apply_perturbation(data, target_idx: torch.Tensor, mode: str, factor: float):
    pert = data.clone()
    x = pert["gene"].x.clone()
    if mode == "knockdown":
        x[target_idx] = 0.0
    elif mode == "overexpress":
        x[target_idx] = x[target_idx] * factor
    elif mode == "knockout":
        x[target_idx] = 0.0
    else:
        raise ValueError(f"unknown mode: {mode}")
    pert["gene"].x = x

    if mode == "knockout":
        for et in list(pert.edge_types):
            ei = pert[et].edge_index
            if ei is None or ei.numel() == 0:
                continue
            keep = torch.ones(ei.shape[1], dtype=torch.bool)
            if et[0] == "gene":
                keep &= ~torch.isin(ei[0], target_idx)
            if et[2] == "gene":
                keep &= ~torch.isin(ei[1], target_idx)
            pert[et].edge_index = ei[:, keep]
            if "edge_attr" in pert[et] and pert[et].edge_attr is not None:
                pert[et].edge_attr = pert[et].edge_attr[keep]
    return pert


# --------------------------------------------------------------------------- #
# Shift des cell_group après perturbation.
# --------------------------------------------------------------------------- #
def cell_group_shift(z_base: torch.Tensor,
                     z_pert: torch.Tensor,
                     group_names=CELL_GROUPS) -> pd.DataFrame:
    """Compare les embeddings cell_group avant vs après perturbation.

    Pour chaque noeud cell_group, on rapporte :
      * baseline_norm      : ‖z_base‖₂
      * perturbed_norm     : ‖z_pert‖₂
      * shift_L2           : ‖z_pert − z_base‖₂   (amplitude absolue du déplacement)
      * shift_relative     : shift_L2 / ‖z_base‖  (amplitude relative)
      * cosine_similarity  : cos(z_base, z_pert)  (direction conservée ?)

    Interprétation :
      - shift_L2 grand sur P4 → le gène perturbé est important pour P4.
      - shift_L2 grand sur P16_cluster_X → important pour ce sous-état sénescent.
      - Si le shift est concentré sur 1-2 groupes → gène spécifique de ces états.
      - Si réparti uniformément → gène housekeeping / hub global.
    """
    zb = z_base.cpu().numpy()
    zp = z_pert.cpu().numpy()
    assert zb.shape == zp.shape, (zb.shape, zp.shape)
    base_norm = np.linalg.norm(zb, axis=1)
    pert_norm = np.linalg.norm(zp, axis=1)
    shift_l2 = np.linalg.norm(zp - zb, axis=1)
    shift_rel = shift_l2 / (base_norm + 1e-8)
    cos = (zb * zp).sum(axis=1) / (base_norm * pert_norm + 1e-8)
    n = zb.shape[0]
    # Si le nombre de noeuds ne matche pas CELL_GROUPS, on retombe sur un
    # index numérique — évite un crash silencieux sur un run atypique.
    names = list(group_names) if n == len(group_names) else [f"group_{i}" for i in range(n)]
    return pd.DataFrame({
        "group": names,
        "baseline_norm": base_norm,
        "perturbed_norm": pert_norm,
        "shift_L2": shift_l2,
        "shift_relative": shift_rel,
        "cosine_similarity": cos,
    })


# --------------------------------------------------------------------------- #
# Run loader.
# --------------------------------------------------------------------------- #
def load_run(run_dir: Path, hidden, latent, n_layers, n_heads):
    data = torch.load(run_dir / "hetero_graph_vgae.pt", weights_only=False)
    baseline = pd.read_csv(run_dir / "gene_ranking_vgae.csv")
    emb_df = pd.read_csv(run_dir / "gene_embeddings_vgae.csv", index_col=0)
    gene_symbols = np.array(emb_df.index.astype(str))
    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}

    encoder = HeteroEncoder(
        gene_in=data["gene"].x.shape[1],
        cell_in=data["cell_group"].x.shape[1],
        hidden=hidden, latent=latent, n_layers=n_layers, n_heads=n_heads,
    )
    model = VGAE(encoder)
    state_path = run_dir / "best_vgae.pt"
    if not state_path.exists():
        state_path = run_dir / "vgae_weights.pt"
    model.load_state_dict(torch.load(state_path, weights_only=True))
    model.eval()
    return data, model, gene_symbols, gene_to_idx, baseline


# --------------------------------------------------------------------------- #
# Reusable API — called both by main() and by perturb_top_genes.py (--all-*).
# --------------------------------------------------------------------------- #
def prepare_baseline(model, data, baseline_df, gene_symbols):
    """Compute the baseline once; reused across many perturbations.

    Returns:
        spec       : (n_genes,) baseline vgae_specificity aligned on node idx.
        base_imp   : (n_genes,) composite baseline importance.
        base_rank  : (n_genes,) ordinal rank of base_imp (1 = best).
        z_cg_base  : (5, hidden) cell_group hidden states from the baseline pass.
    """
    spec_map = dict(zip(baseline_df["gene"].astype(str),
                        baseline_df["vgae_specificity"].astype(float)))
    spec = np.array([spec_map.get(g, 0.0) for g in gene_symbols],
                    dtype=np.float32)
    base = compute_importance(model, data, spec)
    base_imp = base["importance"]
    base_rank = rankdata(-base_imp, method="ordinal").astype(int)
    z_cg_base = model.encoder.last_cell_group_h.clone()
    return spec, base_imp, base_rank, z_cg_base


def run_perturbation_once(model, data, gene_symbols, gene_to_idx,
                          spec, base_imp, base_rank, z_cg_base,
                          targets, mode, factor, top_k, fdr,
                          reactome, background,
                          out_dir=None, write_full=True):
    """Run a single perturbation and return its summary dict.

    Args:
        targets     : list[str] gene symbols to perturb (pre-dedup).
        mode        : "knockdown" | "knockout" | "overexpress".
        factor      : multiplier for overexpress (ignored otherwise).
        out_dir     : if None, write nothing (in-memory only).
                      if set, always write summary.json.
        write_full  : if True AND out_dir set, also write delta_ranking.csv,
                      delta_ora_top_up_reactome.tsv, cell_group_shift.tsv.

    Returns:
        summary (dict) or None if no target is present in the graph.
    """
    n_genes = len(gene_symbols)
    hit = [g for g in targets if g in gene_to_idx]
    miss = [g for g in targets if g not in gene_to_idx]
    if not hit:
        return None
    target_idx = torch.tensor([gene_to_idx[g] for g in hit], dtype=torch.long)

    pert_data = apply_perturbation(data, target_idx, mode, factor)
    pert = compute_importance(model, pert_data, spec)
    pert_imp = pert["importance"]
    pert_rank = rankdata(-pert_imp, method="ordinal").astype(int)
    z_cg_pert = model.encoder.last_cell_group_h.clone()

    shift_df = cell_group_shift(z_cg_base, z_cg_pert)
    shift_map = dict(zip(shift_df["group"], shift_df["shift_relative"]))
    max_shift_row = shift_df.loc[shift_df["shift_relative"].idxmax()]

    delta_rank = base_rank - pert_rank
    delta_imp = pert_imp - base_imp

    ranking = pd.DataFrame({
        "gene": gene_symbols,
        "baseline_importance": base_imp,
        "perturbed_importance": pert_imp,
        "delta_importance": delta_imp,
        "baseline_rank": base_rank,
        "perturbed_rank": pert_rank,
        "delta_rank": delta_rank,
        "is_target": np.isin(np.arange(n_genes),
                             target_idx.cpu().numpy()).astype(int),
    }).sort_values("delta_rank", ascending=False).reset_index(drop=True)

    non_target = ranking[ranking["is_target"] == 0]
    top_up = non_target.head(top_k)
    top_down = non_target.sort_values("delta_rank").head(top_k)

    gl = set(top_up["gene"].astype(str))
    rows = run_ora(gl, background, reactome)
    sig = sum(1 for r in rows if r.p_adj < fdr)

    summary = {
        "mode": mode,
        "factor": factor if mode == "overexpress" else None,
        "n_targets_in_graph": len(hit),
        "targets_in_graph": hit,
        "targets_missing": miss,
        "top_k": top_k,
        "n_rising": int((ranking["delta_rank"] > 0).sum()),
        "n_falling": int((ranking["delta_rank"] < 0).sum()),
        "median_abs_delta_rank": float(np.median(np.abs(delta_rank))),
        "max_up_gene": top_up.iloc[0]["gene"] if len(top_up) else None,
        "max_up_delta_rank": (int(top_up.iloc[0]["delta_rank"])
                              if len(top_up) else None),
        "max_down_gene": top_down.iloc[0]["gene"] if len(top_down) else None,
        "max_down_delta_rank": (int(top_down.iloc[0]["delta_rank"])
                                if len(top_down) else None),
        "n_sig_delta_pathways": sig,
        "fdr_threshold": fdr,
        "top5_delta_pathways": [r.pathway for r in rows[:5]],
        "cell_group_shift_relative": {k: float(v) for k, v in shift_map.items()},
        "max_shift_group": str(max_shift_row["group"]),
        "max_shift_relative": float(max_shift_row["shift_relative"]),
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        if write_full:
            ranking.to_csv(out_dir / "delta_ranking.csv", index=False)
            shift_df.to_csv(out_dir / "cell_group_shift.tsv",
                            sep="\t", index=False)
            write_tsv(rows, out_dir / "delta_ora_top_up_reactome.tsv")
    return summary


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--mode", choices=["knockdown", "knockout", "overexpress"],
                    default="knockdown")
    ap.add_argument("--genes", nargs="+", default=None,
                    help="Gene symbol(s) to perturb.")
    ap.add_argument("--gene-list", type=Path, default=None,
                    help="File with one gene symbol per line (e.g. full pathway).")
    ap.add_argument("--factor", type=float, default=2.0,
                    help="Multiplier used in --mode overexpress (default 2.0).")
    ap.add_argument("--top-k", type=int, default=100,
                    help="Top-K rising genes used for delta-ORA (default 100).")
    ap.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--latent", type=int, default=DEFAULT_LATENT)
    ap.add_argument("--n-layers", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--n-heads", type=int, default=DEFAULT_HEADS)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: run-dir/perturbation/<tag>).")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--summary-only", action="store_true",
                    help="Skip writing delta_ranking.csv / cell_group_shift.tsv "
                         "/ delta_ora_*.tsv. Only summary.json is written.")
    args = ap.parse_args()

    targets: list[str] = []
    if args.genes:
        targets.extend(args.genes)
    if args.gene_list:
        targets.extend(
            g.strip() for g in args.gene_list.read_text().splitlines()
            if g.strip() and not g.strip().startswith("#")
        )
    targets = sorted(set(targets))
    if not targets:
        ap.error("must supply --genes and/or --gene-list")

    data, model, gene_symbols, gene_to_idx, baseline = load_run(
        args.run_dir, args.hidden, args.latent, args.n_layers, args.n_heads)
    print(f"Run loaded: {len(gene_symbols)} genes, "
          f"{len(data.edge_types)} edge types.")

    hit = [g for g in targets if g in gene_to_idx]
    miss = [g for g in targets if g not in gene_to_idx]
    if miss:
        print(f"[warn] {len(miss)} target(s) absent from the graph: "
              f"{miss[:8]}{'...' if len(miss) > 8 else ''}")
    if not hit:
        print("No target present in the graph. Aborting.")
        return

    tag = (f"{args.mode}_{'_'.join(hit[:3])}"
           f"{'_etc' if len(hit) > 3 else ''}")
    out_dir = args.out_dir or (args.run_dir / "perturbation" / tag)
    print(f"Mode={args.mode} | targets in graph={len(hit)} | "
          f"tag={tag} | out={out_dir}")

    spec, base_imp, base_rank, z_cg_base = prepare_baseline(
        model, data, baseline, gene_symbols)

    print("Loading REACTOME + background ...")
    reactome = load_reactome_gmt()
    background = load_background()

    summary = run_perturbation_once(
        model, data, gene_symbols, gene_to_idx,
        spec, base_imp, base_rank, z_cg_base,
        targets=hit, mode=args.mode, factor=args.factor,
        top_k=args.top_k, fdr=args.fdr,
        reactome=reactome, background=background,
        out_dir=out_dir, write_full=not args.summary_only)
    summary["run_dir"] = str(args.run_dir)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nWrote:")
    print(f"  {out_dir / 'summary.json'}")
    if not args.summary_only:
        print(f"  {out_dir / 'delta_ranking.csv'}")
        print(f"  {out_dir / 'delta_ora_top_up_reactome.tsv'}")
        print(f"  {out_dir / 'cell_group_shift.tsv'}")


if __name__ == "__main__":
    main()
