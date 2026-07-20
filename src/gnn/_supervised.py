"""
_supervised.py — Cluster-membership classification head for gnn_vgae.

SCOPE (narrowed 2026-07-20)
---------------------------
This module owns ONE thing: the multi-label head that predicts cell-state /
cluster membership {P4_vs_P16, cluster_0..3} from the per-gene latent μ, plus
its evaluation and per-cluster saliency. It does NOT train anything on its own.

There is a single training loop in the project — the VGAE loop in
`_train_body.py`. When `--supervised` is passed, the head defined here is
attached to μ and co-trained inside that loop:

    loss = recon + β·KL + λ_signed·signed_aux + λ_sup·weighted_bce(head(μ))

Reconstruction is never interrupted, so the run stays perturbation-ready
(standard Δμ along the senescence axis) while the head adds the per-cluster
DEG signal. `finalize_supervised` runs afterwards: forward-only evaluation,
saliency, outputs, checkpoint.

CIRCULARITY is a SEPARATE, ORTHOGONAL axis, controlled by `--de-features`,
which injects the DE statistics as node features **at graph-build time**
(`_graph_build_body.py`). That flag therefore invalidates the graph cache and
appears in the run tag ('de-feat'). This module knows nothing about it.

The former standalone end-to-end trainer (`train_supervised` / `run_supervised`
/ `save_supervised_run`) was a second, unused training loop; it now lives in
`archive/_supervised_standalone.py`.

Provides:
  - `SupervisedHead`     : μ_gene (latent) → n_labels logits.
  - `node_split`         : per-gene train/test split (node classification).
  - `weighted_bce`       : confidence-weighted multi-label BCE (used by the
                           joint loop AND by the evaluation here — one formula).
  - `cluster_importance` : per-cluster gradient×input saliency.
  - `finalize_supervised`: post-training eval + outputs + head checkpoint.
  - `load_supervised_run`: rebuild (model, head, data, config) from a run dir.

Kept import-safe and self-contained so it migrates unchanged when gnn_vgae is
split into lib/cli/workflow (pipeline_design.md).

Author: Théo Ménard — CRCI2NA. Created 2026-06-30. Narrowed 2026-07-20.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


class SupervisedHead(nn.Module):
    """MLP on per-gene latent μ → multi-label logits (one per DEG label)."""

    def __init__(self, latent_dim: int, n_labels: int, hidden: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_labels),
        )

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        return self.net(mu)


def node_split(n_genes: int, test_frac: float, seed: int):
    """Random per-gene train/test split (node classification)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_genes, generator=g)
    n_test = int(round(test_frac * n_genes))
    test_idx = perm[:n_test]
    train_mask = torch.ones(n_genes, dtype=torch.bool)
    train_mask[test_idx] = False
    test_mask = ~train_mask
    return train_mask, test_mask


def weighted_bce(logits, targets, conf, mask):
    """Confidence-weighted multi-label BCE over the masked genes.

    Per-gene weight = confidence (bootstrap×consensus); broadcast over labels.
    """
    bce = F.binary_cross_entropy_with_logits(
        logits[mask], targets[mask], reduction="none")          # (n_mask, L)
    w = conf[mask].unsqueeze(1)                                  # (n_mask, 1)
    return (bce * w).sum() / (w.sum() * logits.shape[1] + 1e-8)


def _eval_labels(logits, targets, mask, label_names):
    """Per-label AUROC / AP on the masked genes (skips degenerate labels)."""
    probs = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    y = targets[mask].detach().cpu().numpy()
    rows = []
    for j, name in enumerate(label_names):
        yj = y[:, j]
        if yj.sum() == 0 or yj.sum() == len(yj):
            rows.append({"label": name, "auroc": np.nan, "ap": np.nan,
                         "n_pos": int(yj.sum())})
            continue
        rows.append({
            "label": name,
            "auroc": float(roc_auc_score(yj, probs[:, j])),
            "ap": float(average_precision_score(yj, probs[:, j])),
            "n_pos": int(yj.sum()),
        })
    return pd.DataFrame(rows)


def cluster_importance(model, head, x_dict, edge_index_dict, edge_attr_dict,
                       cluster_label_idx):
    """Per-cluster gene importance via gradient×input saliency.

    For each cluster label c, importance[j] = |Σ_f x_j,f · ∂(mean_i logit_{i,c})
    /∂x_j,f|. Captures how much gene j's input features influence cluster-c
    predictions across the graph (message passing aggregates neighbours).
    Returns ndarray (n_genes, n_clusters), per-cluster max-normalised.
    """
    model.eval(); head.eval()
    base = x_dict["gene"].detach()
    n_genes = base.shape[0]
    sal = np.zeros((n_genes, len(cluster_label_idx)), dtype=np.float32)

    for k, c in enumerate(cluster_label_idx):
        x_gene = base.clone().requires_grad_(True)
        xd = dict(x_dict); xd["gene"] = x_gene
        _, mu, _ = model.encode(xd, edge_index_dict, edge_attr_dict)
        target = head(mu)[:, c].mean()
        grad = torch.autograd.grad(target, x_gene, retain_graph=False)[0]
        gi = (grad * x_gene).abs().sum(dim=1).detach().cpu().numpy()
        m = gi.max()
        sal[:, k] = gi / m if m > 0 else gi
    return sal


def finalize_supervised(model, head, x_dict, edge_index_dict, edge_attr_dict,
                        sup_labels, gene_symbols, out_dir, hyperparams,
                        train_mask, test_mask, data=None, top_n=30, verbose=True):
    """Post-entraînement (tête co-entraînée dans la boucle VGAE) : écrit
    predictions.tsv + cluster_importance.tsv + metrics.json + heatmap, et
    sauve la tête (supervised_model.pt + supervised_config.json) pour
    perturb_supervised. NE ré-entraîne PAS. Cf. gnn_vgae.py mode --supervised."""
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figure")
    os.makedirs(fig_dir, exist_ok=True)
    label_names = sup_labels.label_names
    device = x_dict["gene"].device
    labels_t = torch.tensor(sup_labels.labels, device=device)
    train_mask = train_mask.to(device); test_mask = test_mask.to(device)

    model.eval(); head.eval()
    with torch.no_grad():
        _, mu, _ = model.encode(x_dict, edge_index_dict, edge_attr_dict)
        logits = head(mu)
        probs = torch.sigmoid(logits).cpu().numpy()
    eval_test = _eval_labels(logits, labels_t, test_mask, label_names)
    eval_train = _eval_labels(logits, labels_t, train_mask, label_names)
    if verbose:
        print("  [V-sup] tête classif — AUROC/AP test par label :")
        for _, r in eval_test.iterrows():
            print(f"    {r['label']:>12}: AUROC={r['auroc']:.3f} AP={r['ap']:.3f} "
                  f"(n_pos={int(r['n_pos'])})")

    pred_df = pd.DataFrame({"gene": gene_symbols})
    for j, name in enumerate(label_names):
        pred_df[f"prob_{name}"] = probs[:, j]
        pred_df[f"label_{name}"] = sup_labels.labels[:, j]
    pred_df["test_set"] = test_mask.cpu().numpy()
    pred_df.to_csv(os.path.join(out_dir, "predictions.tsv"), sep="\t", index=False)

    cluster_idx = [j for j, n in enumerate(label_names) if n.startswith("cluster_")]
    cluster_names = [label_names[j] for j in cluster_idx]
    sal = cluster_importance(model, head, x_dict, edge_index_dict,
                             edge_attr_dict, cluster_idx)
    imp_df = pd.DataFrame(sal, columns=[f"importance_{n}" for n in cluster_names])
    imp_df.insert(0, "gene", gene_symbols)
    val_rho = {}
    for k, cname in enumerate(cluster_names):
        rho = spearmanr(sal[:, k], np.abs(sup_labels.cluster_lfc[:, k])).statistic
        val_rho[cname] = float(rho) if rho == rho else None
        imp_df[f"real_abs_log2fc_{cname}"] = np.abs(sup_labels.cluster_lfc[:, k])
    imp_df.to_csv(os.path.join(out_dir, "cluster_importance.tsv"),
                  sep="\t", index=False)
    _plot_importance_heatmap(sal, gene_symbols, cluster_names, top_n, fig_dir)
    if verbose:
        print(f"  [V-sup] importance↔|log2FC| Spearman/cluster : {val_rho}")

    with open(os.path.join(out_dir, "supervised_metrics.json"), "w") as fh:
        json.dump({"eval_test": eval_test.to_dict(orient="records"),
                   "eval_train": eval_train.to_dict(orient="records"),
                   "importance_vs_log2fc_spearman": val_rho,
                   "label_names": label_names}, fh, indent=2)

    # Sauvegarde tête pour perturb_supervised (le graphe augmenté + vgae_weights
    # sont écrits par le flux VGAE normal / le re-save gnn_vgae).
    torch.save({"model": model.state_dict(), "head": head.state_dict()},
               os.path.join(out_dir, "supervised_model.pt"))
    hp = dict(hyperparams); hp.setdefault("signed_message", False)
    hp.setdefault("signed_decoder", False)
    gin = data["gene"].x.shape[1] if data is not None else mu.shape[1]
    cin = data["cell_group"].x.shape[1] if data is not None else 3
    with open(os.path.join(out_dir, "supervised_config.json"), "w") as fh:
        json.dump({"hyperparams": hp, "label_names": list(label_names),
                   "gene_symbols": [str(g) for g in gene_symbols],
                   "gene_in": int(gin), "cell_in": int(cin),
                   "head_hidden": int(head.net[0].out_features)}, fh)
    if verbose:
        print(f"  [V-sup] tête + config sauvées → {out_dir} "
              f"(supervised_model.pt, supervised_config.json)")
    return {"eval_test": eval_test.to_dict(orient="records"),
            "importance_vs_log2fc_spearman": val_rho}


def load_supervised_run(run_dir, device="cpu"):
    """Reconstruit (model VGAE + head + data + config) depuis un run supervisé sauvé.

    Rebuild l'encodeur depuis la config + le graphe (edge_dim_overrides déduits
    des edge_attr réels, comme à l'entraînement), charge les state_dicts.
    Retourne (model, head, data, config).
    """
    import json as _json
    from _vgae_model import (  # duplicate import-safe
        BilinearSignedDecoder, HeteroEncoder, VGAE,
    )

    run_dir = str(run_dir)
    with open(os.path.join(run_dir, "supervised_config.json")) as fh:
        cfg = _json.load(fh)
    hp = cfg["hyperparams"]
    data = torch.load(os.path.join(run_dir, "hetero_graph_vgae.pt"),
                      weights_only=False)
    edge_dim_overrides = {
        et: int(data[et].edge_attr.shape[1])
        for et in data.edge_types
        if "edge_attr" in data[et] and data[et].edge_attr is not None
    }
    encoder = HeteroEncoder(
        gene_in=cfg["gene_in"], cell_in=cfg["cell_in"],
        hidden=hp["hidden"], latent=hp["latent"],
        n_layers=hp["n_layers"], n_heads=hp["n_heads"],
        available_edge_types=list(data.edge_types),
        edge_dim_overrides=edge_dim_overrides or None,
        signed_message=bool(hp.get("signed_message", False)),
        edge_type_weights=hp.get("edge_type_weights") or None)
    # V5 : a run trained with --signed-decoder carries bilinear_decoder.* keys in
    # its state_dict — without rebuilding it here, load_state_dict(strict=True)
    # below fails on every V5+ supervised run.
    _bilinear = None
    if hp.get("signed_decoder", False):
        _bilinear = BilinearSignedDecoder(
            hp["latent"], signed_dim=hp.get("signed_decoder_dim"))
    model = VGAE(encoder, bilinear_decoder=_bilinear)
    head = SupervisedHead(hp["latent"], n_labels=len(cfg["label_names"]),
                          hidden=int(cfg.get("head_hidden", 64)))
    ckpt = torch.load(os.path.join(run_dir, "supervised_model.pt"),
                      weights_only=True)
    model.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])
    model.to(device).eval(); head.to(device).eval()
    data = data.to(device)
    return model, head, data, cfg


def _plot_importance_heatmap(sal, gene_symbols, cluster_names, top_n, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    top_idx = np.argsort(sal.mean(axis=1))[::-1][:top_n]
    sub = sal[top_idx]
    fig, ax = plt.subplots(figsize=(1.2 * len(cluster_names) + 3, 0.3 * top_n + 2))
    sns.heatmap(sub, ax=ax, cmap="rocket_r",
                xticklabels=cluster_names,
                yticklabels=np.asarray(gene_symbols)[top_idx],
                cbar_kws={"label": "importance (norm.)"})
    ax.set_title(f"Per-cluster gene importance — top {top_n}")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "cluster_importance_heatmap.png"), dpi=150)
    plt.close(fig)
