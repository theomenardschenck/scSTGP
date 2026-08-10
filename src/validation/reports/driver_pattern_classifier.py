"""
driver_pattern_classifier.py — the SUPERVISED NULL for the GNN driver_score.

WHY (2026-07-30, user question 7). `head_to_head_baselines` asks whether a simple
UNSUPERVISED statistic (centrality) reproduces the targets. This is its supervised
twin: can a trivial model predict a KNOWN senescence driver (CellAge / GenAge / ...)
straight from a gene's RAW, PRE-GNN descriptors — node features + per-edge-type
degree — WITHOUT any message-passing?

    RF/GBM( node_features + degree_by_edge_type )  ->  P(known driver)

It is the NULL for the classification head (`_supervised.py`), which reads the
LEARNED latent mu. The gap
        AUROC(head on mu)  -  AUROC(this on raw descriptors)
is the value the GNN's representation adds for known-driver recall. If this NULL
already scores high, the signal was trivially in the degrees/features and the GNN
adds nothing (cf. the metabolic targets); if the head beats it, message-passing
composes the descriptors into something more predictive = the "GNN beyond DE" claim.

Interpretability: permutation importance (SHAP if installed) tells WHICH
feature/edge-type combination predicts a driver — "high coexpr degree + is_tf +
reactome_fi degree", etc.

Anti-circularity: labels are `role: validation` gene sets (never upstream of the
graph/score). Degree is a legitimate INPUT here (the whole point is to see whether
degree alone predicts drivers). Report the AUROC controlled for total degree too.

Usage:
    python src/validation/reports/driver_pattern_classifier.py \
        --graph output/gnn_vgae/V6.1.3/output_fi/rfi2.rich-dir/s1/hetero_graph_vgae.pt \
        --embeddings output/.../s1/gene_embeddings_vgae.csv \
        --ranking output/.../analysis/cross_seed_gene_ranking.tsv \
        --label-sets cellage,genage \
        --out output/.../analysis/driver_pattern_importance.tsv

Author: Théo Ménard — CRCI2NA. Created 2026-07-30.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data" / "loaders"))


def _node_descriptors(graph_path: str, emb_path: str) -> pd.DataFrame:
    """Per-gene raw descriptors: node features + total degree per gene-gene edge type."""
    d = torch.load(graph_path, weights_only=False)
    g = d["gene"]
    n = g.num_nodes
    names = list(pd.read_csv(emb_path).iloc[:, 0])
    if len(names) != n:
        raise SystemExit(f"embeddings ({len(names)}) != n_gene ({n})")
    feat_names = list(getattr(g, "feature_names", [f"f{i}" for i in range(g.x.shape[1])]))
    X = pd.DataFrame(g.x.numpy(), columns=feat_names)
    # degree per gene-gene edge type (both endpoints)
    for et in d.edge_types:
        s, r, t = et
        if s == "gene" and t == "gene":
            ei = d[et].edge_index
            deg = np.bincount(ei[0].numpy(), minlength=n) + np.bincount(ei[1].numpy(), minlength=n)
            X[f"deg_{r}"] = deg
    X["deg_total"] = X[[c for c in X.columns if c.startswith("deg_")]].sum(axis=1)
    X.insert(0, "gene", names)
    return X


def _labels(genes, label_sets: list[str]):
    from gene_sets import load_registry, annotate  # noqa: E402
    sets = load_registry()
    sets = [s for s in sets if s.name in label_sets]
    if not sets:
        raise SystemExit(f"none of {label_sets} in registry")
    ann = annotate(sets, genes)              # in_<name> columns
    cols = [f"in_{s.name}" for s in sets if f"in_{s.name}" in ann.columns]
    y = (ann[cols].sum(axis=1) > 0).astype(int).values
    return y, cols


def _auroc_cv(X, y, seed=0, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(X, y):
        clf = GradientBoostingClassifier(random_state=seed)
        clf.fit(X[tr], y[tr])
        scores.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(scores)), float(np.std(scores))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--ranking", help="cross_seed_gene_ranking.tsv — for the GNN-score reference AUROC")
    p.add_argument("--label-sets", default="cellage,genage")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    X = _node_descriptors(a.graph, a.embeddings)
    y, label_cols = _labels(X["gene"].tolist(), [s.strip() for s in a.label_sets.split(",")])
    feat_cols = [c for c in X.columns if c != "gene"]
    Xv = X[feat_cols].fillna(0).values.astype(float)
    print(f"[data] {len(y)} genes, {int(y.sum())} positives ({100*y.mean():.1f}%), "
          f"{len(feat_cols)} descriptors, labels={label_cols}")

    auroc, sd = _auroc_cv(Xv, y, seed=a.seed)
    print(f"[NULL raw-descriptor GBM]  AUROC = {auroc:.3f} ± {sd:.3f}")

    # degree-only NULL (does total degree alone predict a driver?)
    deg_auroc, _ = _auroc_cv(X[["deg_total"]].values, y, seed=a.seed)
    print(f"[degree-only NULL]         AUROC = {deg_auroc:.3f}")

    # LATENT arm: GBM on the LEARNED embedding mu (post-message-passing). The gap
    # AUROC(mu) - AUROC(raw) = what the GNN's representation ADDS for known-driver
    # recall. mu is already loaded as the embeddings CSV (same gene order as X).
    mu = pd.read_csv(a.embeddings, index_col=0)
    if len(mu) == len(X):
        mu_auroc, mu_sd = _auroc_cv(mu.values.astype(float), y, seed=a.seed)
        print(f"[LATENT mu GBM]            AUROC = {mu_auroc:.3f} ± {mu_sd:.3f}   "
              f"=> message-passing value (mu - raw) = {mu_auroc-auroc:+.3f}")

    # GNN-score reference (the thing the NULL must beat to justify the GNN)
    if a.ranking and Path(a.ranking).exists():
        rk = pd.read_csv(a.ranking, sep="\t")
        m = dict(zip(X["gene"], y))
        rk = rk[rk["target"].isin(m)]
        yy = rk["target"].map(m).values
        gnn_auroc = roc_auc_score(yy, rk["driver_score"].values)
        print(f"[GNN driver_score ref]     AUROC = {gnn_auroc:.3f}   "
              f"=> GNN gap vs raw-NULL = {gnn_auroc-auroc:+.3f}")

    # permutation importance on a full-fit model
    clf = GradientBoostingClassifier(random_state=a.seed).fit(Xv, y)
    imp = permutation_importance(clf, Xv, y, n_repeats=10, random_state=a.seed, scoring="roc_auc")
    out = (pd.DataFrame({"feature": feat_cols,
                         "perm_importance": imp.importances_mean,
                         "perm_sd": imp.importances_std})
           .sort_values("perm_importance", ascending=False))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, sep="\t", index=False)
    print("\n[top predictive descriptors]")
    print(out.head(12).to_string(index=False))
    print(f"\n[wrote] {a.out}")


if __name__ == "__main__":
    main()
