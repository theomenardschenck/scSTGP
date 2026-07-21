"""
_supervised_standalone.py — ARCHIVED: the parallel end-to-end supervised trainer.

RETIRED 2026-07-20. This was the first V-sup design (2026-06-30): a *second*
training loop that optimised the encoder end-to-end on the DEG multi-labels
with NO reconstruction term, run separately from the VGAE loop. It had zero
Python callers — `gnn_vgae.py --supervised` has always gone through the JOINT
path instead (head co-trained inside the VGAE loop, loss = recon + beta*KL +
lambda_signed*signed_aux + lambda_sup*BCE, see `_train_body.py`), which keeps
the run perturbation-ready. Kept here only as a record of the discarded
"pure ceiling" variant: an encoder trained on labels alone, with no topology
reconstruction constraint.

NOT IMPORTED BY ANYTHING. Depends on symbols that now live in
`src/gnn/_supervised.py` (`SupervisedHead`, `node_split`, `weighted_bce`,
`_eval_labels`, `cluster_importance`, `_plot_importance_heatmap`).
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


def train_supervised(model, head, x_dict, edge_index_dict, edge_attr_dict,
                     labels_t, conf_t, train_mask, test_mask,
                     n_epochs=300, lr=5e-3, weight_decay=1e-4, patience=40,
                     log_every=20, verbose=True):
    """End-to-end training of encoder + head. Returns (best_state, history)."""
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    label_names = [f"L{j}" for j in range(labels_t.shape[1])]

    best_auc, best_epoch, best_state = -1.0, 0, None
    history = []
    for epoch in range(n_epochs):
        model.train(); head.train()
        opt.zero_grad()
        _, mu, _ = model.encode(x_dict, edge_index_dict, edge_attr_dict)
        logits = head(mu)
        loss = _weighted_bce(logits, labels_t, conf_t, train_mask)
        loss.backward()
        opt.step()

        model.eval(); head.eval()
        with torch.no_grad():
            _, mu_e, _ = model.encode(x_dict, edge_index_dict, edge_attr_dict)
            logits_e = head(mu_e)
            ev = _eval_labels(logits_e, labels_t, test_mask, label_names)
            mean_auc = float(np.nanmean(ev["auroc"].values))
        history.append({"epoch": epoch, "loss": float(loss.item()),
                        "test_mean_auroc": mean_auc})

        if mean_auc > best_auc:
            best_auc, best_epoch = mean_auc, epoch
            best_state = {
                "model": {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()},
                "head": {k: v.detach().cpu().clone()
                         for k, v in head.state_dict().items()},
            }
        if verbose and (epoch % log_every == 0 or epoch == n_epochs - 1):
            print(f"    epoch {epoch+1:3d}/{n_epochs} — loss={loss.item():.4f} "
                  f"test_mean_AUROC={mean_auc:.4f}")
        if epoch - best_epoch >= patience:
            if verbose:
                print(f"    early stop @ epoch {epoch+1} "
                      f"(best {best_epoch+1}, AUROC={best_auc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        head.load_state_dict(best_state["head"])
    return {"best_epoch": best_epoch, "best_mean_auroc": best_auc}, history


def run_supervised(model, head, x_dict, edge_index_dict, edge_attr_dict,
                   sup_labels, gene_symbols, out_dir,
                   n_epochs=300, lr=5e-3, patience=40, test_frac=0.2,
                   seed=42, top_n=30, verbose=True, data=None, hyperparams=None):
    """Orchestrate supervised training + per-cluster importance + outputs.

    `sup_labels` is a SupervisedLabels (build_supervised_labels). Writes
    predictions.tsv, cluster_importance.tsv, gene_ranking.tsv, metrics.json,
    figure/. Returns the metrics dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figure")
    os.makedirs(fig_dir, exist_ok=True)

    device = x_dict["gene"].device
    label_names = sup_labels.label_names
    labels_t = torch.tensor(sup_labels.labels, device=device)
    conf_t = torch.tensor(sup_labels.confidence, device=device)
    n_genes = labels_t.shape[0]

    train_mask, test_mask = _node_split(n_genes, test_frac, seed)
    train_mask, test_mask = train_mask.to(device), test_mask.to(device)
    if verbose:
        print(f"  [supervised] genes={n_genes} train={int(train_mask.sum())} "
              f"test={int(test_mask.sum())} labels={label_names}")

    metrics, history = train_supervised(
        model, head, x_dict, edge_index_dict, edge_attr_dict,
        labels_t, conf_t, train_mask, test_mask,
        n_epochs=n_epochs, lr=lr, patience=patience, verbose=verbose)

    # ── Final predictions + per-label evaluation ────────────────────────────
    model.eval(); head.eval()
    with torch.no_grad():
        _, mu, _ = model.encode(x_dict, edge_index_dict, edge_attr_dict)
        logits = head(mu)
        probs = torch.sigmoid(logits).cpu().numpy()
    eval_test = _eval_labels(logits, labels_t, test_mask, label_names)
    eval_train = _eval_labels(logits, labels_t, train_mask, label_names)
    if verbose:
        print("  [supervised] test per-label AUROC/AP:")
        for _, r in eval_test.iterrows():
            print(f"    {r['label']:>12}: AUROC={r['auroc']:.3f} "
                  f"AP={r['ap']:.3f} (n_pos={int(r['n_pos'])})")

    pred_df = pd.DataFrame({"gene": gene_symbols})
    for j, name in enumerate(label_names):
        pred_df[f"prob_{name}"] = probs[:, j]
        pred_df[f"label_{name}"] = sup_labels.labels[:, j]
    pred_df["test_set"] = test_mask.cpu().numpy()
    pred_df.to_csv(os.path.join(out_dir, "predictions.tsv"), sep="\t", index=False)

    # ── Per-cluster importance (saliency) ───────────────────────────────────
    cluster_idx = [j for j, n in enumerate(label_names) if n.startswith("cluster_")]
    cluster_names = [label_names[j] for j in cluster_idx]
    sal = cluster_importance(model, head, x_dict, edge_index_dict,
                             edge_attr_dict, cluster_idx)
    imp_df = pd.DataFrame(sal, columns=[f"importance_{n}" for n in cluster_names])
    imp_df.insert(0, "gene", gene_symbols)
    # Validation: does learned importance track the real per-cluster log2FC?
    val_rho = {}
    for k, cname in enumerate(cluster_names):
        rho = spearmanr(sal[:, k], np.abs(sup_labels.cluster_lfc[:, k])).statistic
        val_rho[cname] = float(rho) if rho == rho else None
        imp_df[f"real_abs_log2fc_{cname}"] = np.abs(sup_labels.cluster_lfc[:, k])
    imp_df.to_csv(os.path.join(out_dir, "cluster_importance.tsv"),
                  sep="\t", index=False)
    if verbose:
        print(f"  [supervised] importance↔|log2FC| Spearman per cluster: {val_rho}")

    # ── Global ranking (mean cluster importance) ────────────────────────────
    rank_df = pd.DataFrame({
        "gene": gene_symbols,
        "mean_cluster_importance": sal.mean(axis=1),
        "prob_P4_vs_P16": probs[:, label_names.index("P4_vs_P16")],
    }).sort_values("mean_cluster_importance", ascending=False)
    rank_df.to_csv(os.path.join(out_dir, "gene_ranking.tsv"), sep="\t", index=False)

    # ── Figures ─────────────────────────────────────────────────────────────
    _plot_importance_heatmap(sal, gene_symbols, cluster_names, top_n, fig_dir)
    _plot_history(history, eval_test, fig_dir)

    out_metrics = {
        "best_epoch": int(metrics["best_epoch"] + 1),
        "best_test_mean_auroc": metrics["best_mean_auroc"],
        "eval_test": eval_test.to_dict(orient="records"),
        "eval_train": eval_train.to_dict(orient="records"),
        "importance_vs_log2fc_spearman": val_rho,
        "n_genes": int(n_genes),
        "label_names": label_names,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(out_metrics, fh, indent=2)

    # ── Sauvegarde du modèle (perturbation) ─────────────────────────────────
    if data is not None:
        save_supervised_run(model, head, data, out_dir, hyperparams or {},
                            label_names, gene_symbols, verbose=verbose)
    elif verbose:
        print("  [supervised] data=None → modèle NON sauvé (perturbation "
              "indisponible ; passez data= pour activer).")

    if verbose:
        print(f"  [supervised] outputs → {out_dir}")
    return out_metrics


def save_supervised_run(model, head, data, out_dir, hyperparams,
                        label_names, gene_symbols, verbose=True):
    """Persist the supervised model in TWO formats (cf. docs/technical/gnn_supervised.md) :

    1. **Format run VGAE** (`hetero_graph_vgae.pt` + `vgae_weights.pt` +
       `vgae_metrics.json`) → perturbable par `gnn_perturbation.load_run` /
       `perturb_top_genes.py` le long de l'axe sénescence (Δμ, encodeur seul).
       `model` est un `VGAE(encoder)` : son state_dict est directement compatible.
    2. **Format supervisé** (`supervised_model.pt` = {model, head} + graphe +
       `supervised_config.json`) → perturbation NATIVE par la tête (Δ proba DEG
       par cluster) via `perturb_supervised`.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Graphe partagé par les deux voies (contient les features DE ajoutées).
    torch.save(data, os.path.join(out_dir, "hetero_graph_vgae.pt"))
    # Encodeur (dans VGAE) — compatible load_run.
    torch.save(model.state_dict(), os.path.join(out_dir, "vgae_weights.pt"))
    # Modèle supervisé complet (encodeur VGAE + tête).
    torch.save({"model": model.state_dict(), "head": head.state_dict()},
               os.path.join(out_dir, "supervised_model.pt"))

    hp = dict(hyperparams)
    hp.setdefault("signed_message", False)
    hp.setdefault("signed_decoder", False)
    # vgae_metrics.json pour load_run (flags signed) + hyperparams encoder.
    with open(os.path.join(out_dir, "vgae_metrics.json"), "w") as fh:
        json.dump({"hyperparams": hp, "supervised": True}, fh, indent=2)
    # Config supervisé pour load_supervised_run.
    with open(os.path.join(out_dir, "supervised_config.json"), "w") as fh:
        json.dump({"hyperparams": hp, "label_names": list(label_names),
                   "gene_symbols": [str(g) for g in gene_symbols],
                   "gene_in": int(data["gene"].x.shape[1]),
                   "cell_in": int(data["cell_group"].x.shape[1]),
                   "head_hidden": int(head.net[0].out_features)}, fh)
    if verbose:
        print(f"  [supervised] modèle sauvé → {out_dir} "
              f"(hetero_graph_vgae.pt, vgae_weights.pt, supervised_model.pt, "
              f"vgae_metrics.json, supervised_config.json)")


def _plot_history(history, eval_test, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(h["epoch"], h["loss"], color="#E74C3C")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
    axes[0].set_title("Training loss"); axes[0].grid(alpha=0.3)
    axes[1].plot(h["epoch"], h["test_mean_auroc"], color="#3498DB")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("test mean AUROC")
    axes[1].set_title("Test mean AUROC"); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "training_history.png"), dpi=150)
    plt.close(fig)
