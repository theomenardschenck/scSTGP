#!/usr/bin/env python3
"""
visualize_global.py — Defense-ready cross-run visualisations.

Produces global figures summarising the VGAE HUVEC senescence pipeline
across multiple runs and versions (V2 without HuMess vs V3 with HuMess,
three seeds each). Each subcommand generates one figure family.

Subcommands
-----------
    umap            UMAP of gene embeddings colored by REACTOME pathway
    network         Top-N gene network (PPI + metabolic cocatalysis)
    v2_vs_v3        V2 vs V3 comparison panels (rank correlation, overlap, etc.)
    consensus       Cross-seed V3 consensus (confidence A/B/C, stability)
    perturbation    Perturbation overview (before/after, key movers)
    all             Run every subcommand with sensible defaults

Usage
-----
    # Everything at once
    python src/visualize_global.py all

    # Just the UMAP, with a specific run and pathway list
    python src/visualize_global.py umap \\
        --run-dir output/gnn_vgae/V3_Run3 \\
        --pathways REACTOME_PEROXISOMAL_LIPID_METABOLISM \\
                   REACTOME_ESCRT_DEPENDENT_MVB_BIOGENESIS \\
                   REACTOME_POST_TRANSLATIONAL_MODIFICATION_GPI_ANCHOR_BIOSYNTHESIS

    # V2 vs V3 barplots over all available runs
    python src/visualize_global.py v2_vs_v3

Outputs
-------
    output/gnn_vgae/global_figures/<figure_name>.png
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

def _find_project_root(start: Path, fallback_levels: int = 1) -> Path:
    """Trouve le projet en cherchant data/databases/ vers le haut.
    Robuste à src/validation/foo.py (local) et src/foo.py (cluster flat).
    Override env GNN_PROJECT_ROOT possible."""
    import os
    env = os.environ.get("GNN_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    s = start.resolve()
    for p in [s] + list(s.parents):
        if (p / "data" / "databases").is_dir():
            return p
    return s.parents[fallback_levels]


ROOT = _find_project_root(Path(__file__))
RUNS_ROOT = ROOT / "output/gnn_vgae"
GMT_PATH = ROOT / "data/databases/c2.cp.reactome.symbols.gmt"
OUT_ROOT_DEFAULT = RUNS_ROOT / "global_figures"

# Default "interesting" pathways (peroxisome / ESCRT / GPI signature + a few
# senescence standards). Used by the UMAP when no pathway list is provided.
DEFAULT_PATHWAYS = [
    "REACTOME_PEROXISOMAL_LIPID_METABOLISM",
    "REACTOME_ESCRT_DEPENDENT_MVB_BIOGENESIS",
    "REACTOME_POST_TRANSLATIONAL_MODIFICATION_SYNTHESIS_OF_GPI_ANCHORED_PROTEINS",
    "REACTOME_CILIUM_ASSEMBLY",
    "REACTOME_CELL_CYCLE_MITOTIC",
    "REACTOME_TNF_SIGNALING",
]

MODE_ORDER = ("knockdown", "knockout", "overexpress")
MODE_COLORS = {"knockdown": "#4C72B0", "knockout": "#C44E52", "overexpress": "#55A868"}

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 10})


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def list_runs(prefix: str | None = None) -> list[Path]:
    """Return all run directories under output/gnn_vgae, optionally filtered."""
    if not RUNS_ROOT.exists():
        return []
    out = []
    for d in sorted(RUNS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if prefix and not d.name.startswith(prefix):
            continue
        if not (d / "gene_ranking_vgae.csv").exists():
            continue
        out.append(d)
    return out


def load_ranking(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "gene_ranking_vgae.csv")


def load_embeddings(run_dir: Path) -> pd.DataFrame:
    """genes × 64-D latent; first column is the gene symbol index."""
    emb = pd.read_csv(run_dir / "gene_embeddings_vgae.csv", index_col=0)
    return emb


def load_reactome() -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    with open(GMT_PATH) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name, _url, *genes = parts
            sets[name] = {g.strip() for g in genes if g.strip()}
    return sets


def ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# --------------------------------------------------------------------------- #
# 1. UMAP embeddings colored by pathway
# --------------------------------------------------------------------------- #
def fig_umap(run_dir: Path,
             pathways: list[str],
             out_dir: Path,
             top_n_label: int = 15,
             n_neighbors: int = 30,
             min_dist: float = 0.2,
             seed: int = 42) -> Path:
    import umap

    print(f"[umap] Loading embeddings from {run_dir.name} ...")
    emb = load_embeddings(run_dir)
    ranking = load_ranking(run_dir).set_index("gene")

    # Align ranking with embeddings
    common = [g for g in emb.index if g in ranking.index]
    emb = emb.loc[common]
    ranking = ranking.loc[common]

    print(f"[umap] {len(emb)} genes × {emb.shape[1]} dims — running UMAP ...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        metric="cosine", random_state=seed)
    xy = reducer.fit_transform(emb.values)

    reactome = load_reactome()
    pw_present = [p for p in pathways if p in reactome]
    if not pw_present:
        print(f"[umap] WARN — none of the requested pathways found in GMT; "
              f"falling back to DEFAULT_PATHWAYS filtered to existing.")
        pw_present = [p for p in DEFAULT_PATHWAYS if p in reactome]

    # Colour palette
    palette = sns.color_palette("tab10", n_colors=len(pw_present))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    # Panel A : colored by importance (heatmap background)
    ax = axes[0]
    order = ranking["vgae_importance"].argsort().values
    sc = ax.scatter(xy[order, 0], xy[order, 1],
                    c=ranking["vgae_importance"].iloc[order].values,
                    s=5, cmap="viridis", alpha=0.7, linewidths=0)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("vgae_importance")
    # Annotate top-K
    top_genes = ranking.nlargest(top_n_label, "vgae_importance").index
    for g in top_genes:
        i = list(emb.index).index(g)
        ax.annotate(g, (xy[i, 0], xy[i, 1]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points",
                    color="black")
    ax.set_title(f"A — UMAP of VGAE latent space ({run_dir.name})\n"
                 f"colored by importance, top-{top_n_label} labelled")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    # Panel B : colored by pathway membership
    ax = axes[1]
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c="#d8d8d8", alpha=0.4, linewidths=0,
               label="other")
    gene_to_i = {g: i for i, g in enumerate(emb.index)}
    legend_counts = []
    for pw, col in zip(pw_present, palette):
        idxs = [gene_to_i[g] for g in reactome[pw] if g in gene_to_i]
        if not idxs:
            continue
        label = pw.replace("REACTOME_", "").replace("_", " ").lower()
        if len(label) > 45:
            label = label[:42] + "..."
        ax.scatter(xy[idxs, 0], xy[idxs, 1], s=18, color=col,
                   alpha=0.85, linewidths=0.3, edgecolor="white",
                   label=f"{label} (n={len(idxs)})")
        legend_counts.append((pw, len(idxs)))

    ax.set_title("B — UMAP colored by REACTOME pathway membership")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=7, frameon=True, markerscale=1.2)

    fig.suptitle(f"VGAE latent space — {run_dir.name}", fontsize=13, y=1.02)
    fig.tight_layout()

    path = out_dir / f"umap_{run_dir.name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[umap] Wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# 2. Top-N gene network (PPI + metabolic cocatalysis)
# --------------------------------------------------------------------------- #
def fig_network(run_dir: Path,
                out_dir: Path,
                top_n: int = 100,
                seed: int = 42) -> Path:
    import networkx as nx
    import torch

    graph_path = run_dir / "hetero_graph_vgae.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"missing {graph_path} — need hetero_graph for network figure")

    print(f"[network] Loading hetero graph from {graph_path} ...")
    data = torch.load(graph_path, weights_only=False)
    ranking = load_ranking(run_dir)

    # Recover gene ordering: embeddings file index is the graph node order
    emb = load_embeddings(run_dir)
    gene_symbols = list(emb.index)
    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}

    # Pick top-N by importance
    top = (ranking.sort_values("vgae_importance", ascending=False)
                  .head(top_n)["gene"].astype(str).tolist())
    top_set = set(top)
    top_idx = {gene_to_idx[g] for g in top if g in gene_to_idx}

    # Collect edges restricted to top-N
    def edges_for(rel: tuple[str, str, str]) -> list[tuple[int, int]]:
        if rel not in data.edge_types:
            return []
        ei = data[rel].edge_index.cpu().numpy()
        out = []
        for s, d in zip(ei[0], ei[1]):
            if int(s) in top_idx and int(d) in top_idx and int(s) != int(d):
                pair = (int(min(s, d)), int(max(s, d)))
                out.append(pair)
        return list(set(out))

    # All gene-gene relations we display. Order = priority when an edge
    # exists under several types: the earlier type wins the visible style,
    # and the rest are stored in the "extra" set for the tooltip/title.
    # (regulates / regulated_by are merged since they're symmetric for drawing.)
    edge_kinds = [
        ("ppi",          ("gene", "ppi", "gene")),
        ("cocat",        ("gene", "metabolic_cocatalysis", "gene")),
        ("pathway",      ("gene", "same_pathway", "gene")),
        ("regulates",    ("gene", "regulates", "gene")),
        ("regulates",    ("gene", "regulated_by", "gene")),
        ("coexpression", ("gene", "coexpression", "gene")),
    ]
    # Merge multi-type edges into a single graph. Each edge keeps the set of
    # relations it belongs to.
    edge_sets: dict[str, list[tuple[int, int]]] = {}
    for kind, rel in edge_kinds:
        pairs = edges_for(rel)
        edge_sets.setdefault(kind, []).extend(pairs)

    print(f"[network] top-{top_n} genes — edges by kind: "
          + ", ".join(f"{k}={len(set(v))}" for k, v in edge_sets.items()))

    # Build graph
    G = nx.Graph()
    for g in top:
        if g in gene_to_idx:
            G.add_node(gene_to_idx[g], symbol=g)
    for kind, pairs in edge_sets.items():
        for s, d in set(pairs):
            if G.has_edge(s, d):
                G[s][d]["kinds"].add(kind)
            else:
                G.add_edge(s, d, kinds={kind})

    # Importance for sizing / coloring
    imp = (ranking.set_index("gene")["vgae_importance"]
                  .reindex(top).fillna(0.0).to_dict())

    # Layout — spring on the full graph, then an inverse-spring correction:
    # spring pulls connected nodes *close* and pushes disconnected nodes
    # *apart*. For readability we invert that bias:
    #   • within each connected component → expand (spread nodes from their
    #     cluster centroid),
    #   • between components → shrink (pull cluster centroids toward the
    #     figure centre).
    # The structural information (who is connected to whom) is preserved;
    # only the visual spacing is adjusted.
    connected = [n for n in G.nodes if G.degree(n) > 0]
    isolated = [n for n in G.nodes if G.degree(n) == 0]
    pos = nx.spring_layout(G, k=0.5, iterations=300, seed=seed, weight=None)

    # Uniform fit into a [-1, 1] viewport (preserves aspect + component
    # separation). Do NOT scale x and y independently.
    xs_all = np.array([p[0] for p in pos.values()])
    ys_all = np.array([p[1] for p in pos.values()])
    max_abs = max(np.abs(xs_all).max(), np.abs(ys_all).max()) or 1.0
    pos = {n: (p[0] / max_abs, p[1] / max_abs) for n, p in pos.items()}

    # Inverse-spring rescaling.
    EXPAND = 2.2   # factor applied to each node's offset from its centroid
    SHRINK = 0.45  # factor applied to each centroid's position
    for comp in nx.connected_components(G):
        if len(comp) < 2:
            continue
        cx = float(np.mean([pos[n][0] for n in comp]))
        cy = float(np.mean([pos[n][1] for n in comp]))
        new_cx, new_cy = cx * SHRINK, cy * SHRINK
        for n in comp:
            dx, dy = pos[n][0] - cx, pos[n][1] - cy
            pos[n] = (new_cx + EXPAND * dx, new_cy + EXPAND * dy)
    # Isolated nodes: also pulled toward the centre so they don't hog the border
    for n in isolated:
        pos[n] = (pos[n][0] * SHRINK, pos[n][1] * SHRINK)

    # Re-fit after the rescaling so everything stays in view.
    xs_all = np.array([p[0] for p in pos.values()])
    ys_all = np.array([p[1] for p in pos.values()])
    max_abs = max(np.abs(xs_all).max(), np.abs(ys_all).max()) or 1.0
    pos = {n: (p[0] / max_abs, p[1] / max_abs) for n, p in pos.items()}

    fig, ax = plt.subplots(figsize=(15, 12))

    # Per-kind style; drawing order (later = on top). Multi-type edges are
    # drawn once per kind they belong to, so overlapping edges show as
    # layered colours — dense connections appear brighter.
    kind_styles = {
        "coexpression": {"color": "#555555", "alpha": 0.8, "width": 1.2,
                         "label": "co-expression"},
        "pathway":      {"color": "#2ca02c", "alpha": 0.5, "width": 1.0,
                         "label": "same REACTOME pathway"},
        "regulates":    {"color": "#ff7f0e", "alpha": 0.95, "width": 1.8,
                         "label": "SCENIC regulates"},
        "ppi":          {"color": "#1f77b4", "alpha": 0.85, "width": 1.3,
                         "label": "PPI"},
        "cocat":        {"color": "#d62728", "alpha": 1.0, "width": 2.4,
                         "label": "metabolic cocatalysis"},
    }
    # Draw in the defined order (faint relations first, so strong
    # relations are drawn on top).
    drawn_labels: set[str] = set()
    kind_counts: dict[str, int] = {k: 0 for k in kind_styles}
    for kind, style in kind_styles.items():
        for u, v, d in G.edges(data=True):
            if kind not in d["kinds"]:
                continue
            kind_counts[kind] += 1
            lbl = style["label"] if style["label"] not in drawn_labels else None
            drawn_labels.add(style["label"])
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=style["color"], alpha=style["alpha"],
                    linewidth=style["width"], label=lbl, zorder=1,
                    solid_capstyle="round")

    # Nodes — smaller to let edges breathe; scaled by importance
    xs = [pos[n][0] for n in G.nodes]
    ys = [pos[n][1] for n in G.nodes]
    syms = [G.nodes[n]["symbol"] for n in G.nodes]
    sizes = [25 + 220 * imp.get(s, 0.0) for s in syms]
    colors = [imp.get(s, 0.0) for s in syms]
    sc = ax.scatter(xs, ys, s=sizes, c=colors, cmap="viridis",
                    edgecolor="black", linewidths=0.5, zorder=3)
    cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("vgae_importance")

    # Labels — only top-30 to avoid clutter, with a small offset
    label_top = set(top[:30])
    for n in G.nodes:
        sym = G.nodes[n]["symbol"]
        if sym in label_top:
            ax.annotate(sym, pos[n], fontsize=7, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points",
                        color="black", zorder=4,
                        path_effects=None)

    title_counts = "  ·  ".join(
        f"{kind}={kind_counts[kind]}"
        for kind in ["ppi", "cocat", "pathway", "regulates", "coexpression"]
    )
    ax.set_title(
        f"Top-{top_n} genes by VGAE importance — full gene-gene edge set\n"
        f"{run_dir.name}  ·  {G.number_of_edges()} unique edges  ·  "
        f"{title_counts}\n"
        f"{len(connected)} connected / {len(isolated)} isolated")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.tight_layout()
    path = out_dir / f"network_top{top_n}_{run_dir.name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[network] Wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# 3. V2 vs V3 comparison
# --------------------------------------------------------------------------- #
def fig_v2_vs_v3(out_dir: Path, top_n: int = 100) -> Path:
    v2_runs = list_runs("V2_")
    v3_runs = list_runs("V3_")
    if not v2_runs or not v3_runs:
        raise RuntimeError("Need at least one V2_* and one V3_* run.")

    print(f"[v2_vs_v3] V2 runs: {[r.name for r in v2_runs]}")
    print(f"[v2_vs_v3] V3 runs: {[r.name for r in v3_runs]}")

    # Load all rankings, aligned on gene
    def load_series(run: Path) -> pd.Series:
        df = load_ranking(run)
        return df.set_index("gene")["vgae_importance"].rename(run.name)

    v2 = pd.concat([load_series(r) for r in v2_runs], axis=1)
    v3 = pd.concat([load_series(r) for r in v3_runs], axis=1)
    all_imp = pd.concat([v2, v3], axis=1).dropna()

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    # Panel A : mean importance V2 vs V3 scatter
    ax = fig.add_subplot(gs[0, 0])
    v2_mean = v2.mean(axis=1).reindex(all_imp.index)
    v3_mean = v3.mean(axis=1).reindex(all_imp.index)
    rho, _ = spearmanr(v2_mean, v3_mean)
    ax.scatter(v2_mean, v3_mean, s=5, alpha=0.35, c="#4C72B0", linewidths=0)
    min_val = min(v2_mean.min(), v3_mean.min())
    max_val = max(v2_mean.max(), v3_mean.max())
    lim = [min_val * 0.98, max_val * 1.02]
    ax.plot(lim, lim, "--", c="grey", lw=1)
    # Annotate top movers (largest rank change)
    delta = v3_mean.rank() - v2_mean.rank()
    risers = delta.nlargest(5).index
    fallers = delta.nsmallest(5).index
    for g in list(risers) + list(fallers):
        ax.annotate(g, (v2_mean[g], v3_mean[g]), fontsize=6.5,
                    xytext=(2, 2), textcoords="offset points")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("V2 mean importance (no HuMess)")
    ax.set_ylabel("V3 mean importance (with HuMess)")
    ax.set_title(f"A — V2 vs V3 mean importance\nSpearman ρ = {rho:.3f}")

    # Panel B : Top-N overlap (V2 top-N vs V3 top-N)
    ax = fig.add_subplot(gs[0, 1])
    v2_top = set(v2_mean.nlargest(top_n).index)
    v3_top = set(v3_mean.nlargest(top_n).index)
    inter = v2_top & v3_top
    only_v2 = v2_top - v3_top
    only_v3 = v3_top - v2_top
    counts = [len(only_v2), len(inter), len(only_v3)]
    ax.bar(["V2 only", "V2 ∩ V3", "V3 only"], counts,
           color=["#4C72B0", "#8172B3", "#C44E52"], edgecolor="black")
    for i, c in enumerate(counts):
        ax.text(i, c + 0.5, str(c), ha="center", fontsize=10)
    ax.set_ylabel(f"# genes in top-{top_n}")
    ax.set_title(f"B — Top-{top_n} overlap")

    # Panel C : V2 vs V3 — which top-N genes moved in/out
    ax = fig.add_subplot(gs[0, 2])
    # Show top-10 newcomers + top-10 dropouts by rank shift
    v2_rank = v2_mean.rank(ascending=False)
    v3_rank = v3_mean.rank(ascending=False)
    rank_shift = (v2_rank - v3_rank)  # positive = climbed in V3
    newcomers = rank_shift.loc[list(only_v3)].nlargest(10)
    dropouts = rank_shift.loc[list(only_v2)].nsmallest(10)
    bars_df = pd.concat([newcomers, dropouts]).sort_values()
    colors = ["#C44E52" if v < 0 else "#55A868" for v in bars_df.values]
    ax.barh(bars_df.index, bars_df.values, color=colors, edgecolor="black")
    ax.axvline(0, c="black", lw=0.8)
    ax.set_xlabel("Rank shift (V2_rank − V3_rank), + = climbed")
    ax.set_title(f"C — Genes in/out of top-{top_n} by V3")
    ax.tick_params(axis="y", labelsize=7)

    # Panel D : distribution of importance scores (V2 vs V3)
    ax = fig.add_subplot(gs[1, 0])
    df_long = pd.DataFrame({
        "importance": pd.concat([v2.stack(), v3.stack()]).values,
        "version": (["V2"] * v2.stack().shape[0]
                    + ["V3"] * v3.stack().shape[0]),
    })
    sns.violinplot(data=df_long, x="version", y="importance", hue="version",
                   palette={"V2": "#4C72B0", "V3": "#C44E52"},
                   inner="quartile", ax=ax, cut=0, legend=False)
    ax.set_title("D — Distribution of vgae_importance")
    ax.set_xlabel("")

    # Panel E : DB overlap — how many top-N genes are in gold-standards
    ax = fig.add_subplot(gs[1, 1])
    dbs = ["in_genage", "in_cellage", "in_msigdb_aging",
           "in_ageanno", "in_aging_local"]
    rows = []
    for run in v2_runs + v3_runs:
        r = load_ranking(run)
        tops = r.nlargest(top_n, "vgae_importance")
        for db in dbs:
            rows.append({"run": run.name,
                         "version": "V2" if run.name.startswith("V2") else "V3",
                         "db": db.replace("in_", ""),
                         "count": int(tops[db].sum())})
    db_df = pd.DataFrame(rows)
    sns.barplot(data=db_df, x="db", y="count", hue="version",
                palette={"V2": "#4C72B0", "V3": "#C44E52"},
                errorbar="sd", ax=ax)
    ax.set_title(f"E — Gold-standard overlap in top-{top_n}\n(mean ± sd across seeds)")
    ax.set_ylabel(f"# genes in top-{top_n}")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)

    # Panel F : component contribution — which importance component changed most?
    ax = fig.add_subplot(gs[1, 2])
    comps = ["vgae_emb_norm", "vgae_density", "vgae_recon_fidelity",
             "vgae_certainty", "vgae_specificity"]

    def comp_means(runs: list[Path]) -> pd.Series:
        vals = []
        for run in runs:
            r = load_ranking(run).set_index("gene")
            vals.append(r[comps].mean())
        return pd.concat(vals, axis=1).mean(axis=1)

    v2_c = comp_means(v2_runs)
    v3_c = comp_means(v3_runs)
    x = np.arange(len(comps))
    w = 0.38
    ax.bar(x - w/2, v2_c.values, w, color="#4C72B0", label="V2", edgecolor="black")
    ax.bar(x + w/2, v3_c.values, w, color="#C44E52", label="V3", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("vgae_", "") for c in comps], rotation=25)
    ax.set_ylabel("mean component value")
    ax.set_title("F — Importance components: V2 vs V3")
    ax.legend()

    fig.suptitle(f"V2 (no HuMess) vs V3 (with HuMess) — {len(v2_runs)} vs {len(v3_runs)} seeds",
                 fontsize=14, y=1.0)
    path = out_dir / "v2_vs_v3_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[v2_vs_v3] Wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# 4. Cross-run V3 consensus (A/B/C confidence)
# --------------------------------------------------------------------------- #
def fig_consensus(out_dir: Path,
                  version: str = "V3",
                  top_n: int = 100) -> Path:
    runs = list_runs(f"{version}_")
    if len(runs) < 2:
        raise RuntimeError(f"Need at least 2 {version}_* runs for consensus "
                           f"(found {len(runs)})")

    print(f"[consensus] {version} runs: {[r.name for r in runs]}")

    # Build top-N set per run
    top_sets = {}
    rank_frames = {}
    for run in runs:
        r = load_ranking(run)
        top_sets[run.name] = set(r.nlargest(top_n, "vgae_importance")["gene"])
        rank_frames[run.name] = r.set_index("gene")["vgae_importance"]

    # Confidence classification
    all_top = set().union(*top_sets.values())
    counts = {g: sum(g in s for s in top_sets.values()) for g in all_top}
    n_runs = len(runs)
    conf = pd.Series(counts).rename("n_runs").to_frame()
    conf["confidence"] = conf["n_runs"].map({
        n_runs: "A (all)",
        **{k: "B (majority)" for k in range(n_runs // 2 + 1, n_runs)},
        **{k: "C (minority)" for k in range(1, n_runs // 2 + 1)},
    })

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.35)

    # Panel A : confidence bar
    ax = fig.add_subplot(gs[0, 0])
    counts_by_k = conf["n_runs"].value_counts().sort_index()
    colors = sns.color_palette("RdYlGn_r", n_colors=n_runs)[::-1]
    bars = ax.bar(counts_by_k.index.astype(str), counts_by_k.values,
                  color=[colors[k-1] for k in counts_by_k.index],
                  edgecolor="black")
    for b, v in zip(bars, counts_by_k.values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, str(v),
                ha="center", fontsize=9)
    ax.set_xlabel(f"# seeds with gene in top-{top_n}")
    ax.set_ylabel("# genes")
    ax.set_title(f"A — Top-{top_n} confidence tiers "
                 f"({len(conf)} unique genes)")

    # Panel B : pairwise Jaccard
    ax = fig.add_subplot(gs[0, 1])
    names = list(top_sets.keys())
    mat = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            inter = len(top_sets[a] & top_sets[b])
            union = len(top_sets[a] | top_sets[b])
            mat[i, j] = inter / union if union else 0
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=names, yticklabels=names, ax=ax,
                vmin=0, vmax=1, cbar_kws={"label": "Jaccard"})
    ax.set_title(f"B — Top-{top_n} pairwise Jaccard")

    # Panel C : Spearman rank correlation between runs (all genes)
    ax = fig.add_subplot(gs[0, 2])
    rank_df = pd.concat(rank_frames.values(), axis=1, keys=rank_frames.keys()).dropna()
    spear = rank_df.corr(method="spearman")
    sns.heatmap(spear, annot=True, fmt=".3f", cmap="viridis",
                xticklabels=spear.columns, yticklabels=spear.index, ax=ax,
                vmin=0.7, vmax=1.0, cbar_kws={"label": "Spearman ρ"})
    ax.set_title("C — Full-ranking Spearman correlation")

    # Panel D : A-tier genes (all seeds) — top 25 by mean importance
    ax = fig.add_subplot(gs[1, 0])
    a_tier = conf[conf["n_runs"] == n_runs].index.tolist()
    if a_tier:
        means = rank_df.loc[[g for g in a_tier if g in rank_df.index]].mean(axis=1)
        top_a = means.nlargest(25)
        ax.barh(top_a.index[::-1], top_a.values[::-1],
                color="#2ca02c", edgecolor="black")
        ax.set_xlim(left=min(top_a.values) * 0.98)
        ax.set_xlabel("mean importance across seeds")
        ax.set_title(f"D — Tier A (all {n_runs} seeds) — top 25")
        ax.tick_params(axis="y", labelsize=7)
    else:
        ax.text(0.5, 0.5, "No tier-A genes", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("D — Tier A")

    # Panel E : rank stability for tier-A genes (min/max/mean across seeds)
    ax = fig.add_subplot(gs[1, 1])
    if a_tier:
        top_a_genes = top_a.index[:25].tolist()
        ranks = rank_df.rank(ascending=False)
        rk = ranks.loc[top_a_genes]
        rk_stats = pd.DataFrame({
            "min": rk.min(axis=1),
            "max": rk.max(axis=1),
            "mean": rk.mean(axis=1),
        }).sort_values("mean")
        y = np.arange(len(rk_stats))
        ax.hlines(y, rk_stats["min"], rk_stats["max"],
                  color="grey", alpha=0.6, lw=3)
        ax.scatter(rk_stats["mean"], y, color="#c44e52", s=40,
                   edgecolor="black", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(rk_stats.index, fontsize=7)
        ax.set_xlabel("rank (1 = most important)")
        ax.set_title("E — Tier-A rank range across seeds")
        ax.invert_xaxis()  # rank 1 on the right, best at the top
    else:
        ax.text(0.5, 0.5, "No tier-A genes", ha="center", va="center",
                transform=ax.transAxes)

    # Panel F : DB overlap per tier
    ax = fig.add_subplot(gs[1, 2])
    ref = load_ranking(runs[0]).set_index("gene")
    db_cols = ["in_genage", "in_cellage", "in_msigdb_aging",
               "in_ageanno", "in_aging_local"]
    rows = []
    for tier, genes in [("A (all)", conf[conf["n_runs"] == n_runs].index),
                        ("B (majority)", conf[(conf["n_runs"] > n_runs//2) &
                                              (conf["n_runs"] < n_runs)].index),
                        ("C (minority)", conf[conf["n_runs"] <= n_runs//2].index)]:
        g_in = [g for g in genes if g in ref.index]
        if not g_in:
            continue
        sub = ref.loc[g_in]
        n_any = int((sub[db_cols].sum(axis=1) > 0).sum())
        rows.append({"tier": tier, "n_genes": len(g_in),
                     "n_in_db": n_any,
                     "pct": 100 * n_any / max(len(g_in), 1)})
    tier_df = pd.DataFrame(rows)
    if not tier_df.empty:
        bars = ax.bar(tier_df["tier"], tier_df["pct"],
                      color=["#2ca02c", "#ff9933", "#d62728"],
                      edgecolor="black")
        for b, row in zip(bars, tier_df.itertuples()):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                    f"{row.n_in_db}/{row.n_genes}", ha="center", fontsize=8)
        ax.set_ylabel("% genes present in ≥1 aging DB")
        ax.set_title("F — DB validation by confidence tier")
        ax.set_ylim(0, 100)

    fig.suptitle(f"{version} cross-seed consensus — {len(runs)} seeds",
                 fontsize=14, y=1.0)
    path = out_dir / f"consensus_{version}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[consensus] Wrote {path}")

    # Side product : tier table
    tier_tsv = out_dir / f"consensus_{version}_tiers.tsv"
    (conf.sort_values("n_runs", ascending=False)
         .to_csv(tier_tsv, sep="\t"))
    print(f"[consensus] Wrote {tier_tsv}")

    return path


# --------------------------------------------------------------------------- #
# 5. Perturbation before/after summary
# --------------------------------------------------------------------------- #
def fig_perturbation(out_dir: Path, top_k: int = 15) -> Path:
    runs = [r for r in list_runs() if (r / "perturbation/manifest.csv").exists()]
    if not runs:
        raise RuntimeError("No runs with perturbation data found.")
    print(f"[perturbation] Using runs: {[r.name for r in runs]}")

    # Aggregate comparison tables across all runs
    frames = []
    for run in runs:
        ct = run / "perturbation/report/comparison_table_filtered.tsv"
        if not ct.exists():
            continue
        df = pd.read_csv(ct, sep="\t")
        df["run"] = run.name
        frames.append(df)
    if not frames:
        raise RuntimeError("No comparison_table_filtered.tsv found in any run.")
    all_ct = pd.concat(frames, ignore_index=True)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # Panel A : risers vs fallers per mode (boxplot across runs)
    ax = fig.add_subplot(gs[0, 0])
    m = all_ct.melt(id_vars=["mode", "run", "tag"],
                    value_vars=["n_rising", "n_falling"],
                    var_name="direction", value_name="count")
    sns.boxplot(data=m, x="mode", y="count", hue="direction",
                order=MODE_ORDER,
                palette={"n_rising": "#55A868", "n_falling": "#C44E52"},
                ax=ax)
    ax.set_title("A — Risers vs fallers per perturbation\n(all targets, all runs)")
    ax.set_ylabel("# genes")
    ax.legend(title="")

    # Panel B : median |Δrank| per mode
    ax = fig.add_subplot(gs[0, 1])
    sns.violinplot(data=all_ct, x="mode", y="median_abs_delta_rank",
                   order=MODE_ORDER, hue="mode",
                   palette=MODE_COLORS, ax=ax, cut=0, inner="quartile",
                   legend=False)
    ax.set_yscale("log")
    ax.set_title("B — Median |Δrank| per perturbation\n(log scale)")
    ax.set_ylabel("median |Δrank|")

    # Panel C : # significant pathways per mode
    ax = fig.add_subplot(gs[0, 2])
    sns.boxplot(data=all_ct, x="mode", y="n_sig_delta_pathways",
                order=MODE_ORDER, hue="mode", palette=MODE_COLORS,
                ax=ax, legend=False)
    ax.set_title("C — # significant ORA pathways\n(padj < 0.05, top-100 risers)")
    ax.set_ylabel("# pathways")

    # Panel D : Top-K perturbations with largest |max_up_delta_rank|
    ax = fig.add_subplot(gs[1, 0])
    # Take best per-run per-mode, then overall top
    top_up = (all_ct.sort_values("max_up_delta_rank", ascending=False)
                    .head(top_k))
    labels = [f"{r.tag} ({r.run.replace('Run','R')})"
              for r in top_up.itertuples()]
    colors = [MODE_COLORS[m] for m in top_up["mode"]]
    ax.barh(labels[::-1], top_up["max_up_delta_rank"].values[::-1],
            color=colors[::-1], edgecolor="black")
    ax.set_xlim(left=min(top_up["max_up_delta_rank"].values) * 0.98)
    ax.set_xlabel("max_up_delta_rank")
    ax.set_title(f"D — Top-{top_k} strongest risers across all perturbations")
    ax.tick_params(axis="y", labelsize=7)

    # Panel E : Reactome frequency among top1 pathways
    ax = fig.add_subplot(gs[1, 1])
    pw_counts = (all_ct["top1_pathway"].dropna()
                        .str.replace("REACTOME_", "")
                        .value_counts()
                        .head(top_k))
    if len(pw_counts):
        ax.barh(pw_counts.index[::-1], pw_counts.values[::-1],
                color="#4C72B0", edgecolor="black")
        ax.set_xlabel("# perturbations where this is top-1")
        ax.set_title(f"E — Most frequent top-1 ORA pathway (top-{top_k})")
        ax.tick_params(axis="y", labelsize=7)

    # Panel F : before/after — a single exemplar perturbation (biggest mover)
    ax = fig.add_subplot(gs[1, 2])
    exemplar = top_up.iloc[0]
    ex_run = [r for r in runs if r.name == exemplar["run"]][0]
    # Look up the out_dir via manifest.csv (tag column matches the sub-folder name).
    manifest = pd.read_csv(ex_run / "perturbation/manifest.csv")
    # manifest["out_dir"] is like "perturbation/knockdown_ATF3" — basename == tag.
    manifest["basename"] = manifest["out_dir"].apply(lambda s: Path(s).name)
    row = manifest[manifest["basename"] == exemplar["tag"]]
    dpath = None
    if not row.empty:
        dpath = ex_run / row.iloc[0]["out_dir"] / "delta_ranking.csv"
    else:
        # Fallback: infer from tag (format = <mode>_<target>)
        fallback = ex_run / "perturbation" / exemplar["tag"] / "delta_ranking.csv"
        if fallback.exists():
            dpath = fallback

    if dpath is not None and dpath.exists():
        d = pd.read_csv(dpath)
        ax.scatter(d["baseline_rank"], d["perturbed_rank"],
                   s=3, alpha=0.35, c="#aaaaaa", linewidths=0)
        movers = d.reindex(d["delta_rank"].abs().nlargest(10).index)
        max_abs = abs(movers["delta_rank"]).max() if len(movers) else 1
        sc = ax.scatter(movers["baseline_rank"], movers["perturbed_rank"],
                        c=movers["delta_rank"], cmap="coolwarm",
                        s=30, edgecolor="black", linewidths=0.5,
                        vmin=-max_abs, vmax=max_abs)
        for _, m_ in movers.iterrows():
            ax.annotate(m_["gene"], (m_["baseline_rank"], m_["perturbed_rank"]),
                        fontsize=6.5, xytext=(3, 3),
                        textcoords="offset points")
        min_val = min(d["baseline_rank"].min(), d["perturbed_rank"].min())
        max_val = max(d["baseline_rank"].max(), d["perturbed_rank"].max())
        lim = [min_val * 0.98, max_val * 1.02]
        ax.plot(lim, lim, "--", c="grey", lw=0.8)
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="Δrank")
        ax.set_xlabel("baseline rank")
        ax.set_ylabel("perturbed rank")
        ax.set_title(f"F — Before vs after: {exemplar['tag']}\n"
                     f"({ex_run.name}, {exemplar['mode']})")
    else:
        ax.text(0.5, 0.5, f"No delta_ranking for {exemplar['tag']}",
                ha="center", transform=ax.transAxes)

    fig.suptitle(f"Perturbation overview — {len(runs)} runs, "
                 f"{len(all_ct)} perturbations",
                 fontsize=14, y=1.0)
    path = out_dir / "perturbation_overview.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[perturbation] Wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT_DEFAULT,
                    help=f"Output directory (default: {OUT_ROOT_DEFAULT}).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("umap", help="UMAP of VGAE embeddings coloured by pathway")
    s.add_argument("--run-dir", type=Path, default=RUNS_ROOT / "V3_Run3")
    s.add_argument("--pathways", nargs="+", default=DEFAULT_PATHWAYS)
    s.add_argument("--top-n-label", type=int, default=15)
    s.add_argument("--n-neighbors", type=int, default=30)
    s.add_argument("--min-dist", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=42)

    s = sub.add_parser("network", help="Top-N gene network (PPI + cocatalysis)")
    s.add_argument("--run-dir", type=Path, default=RUNS_ROOT / "V3_Run3")
    s.add_argument("--top-n", type=int, default=100)
    s.add_argument("--seed", type=int, default=42)

    s = sub.add_parser("v2_vs_v3", help="V2 vs V3 comparison panels")
    s.add_argument("--top-n", type=int, default=100)

    s = sub.add_parser("consensus", help="Cross-seed consensus (A/B/C)")
    s.add_argument("--version", default="V3")
    s.add_argument("--top-n", type=int, default=100)

    s = sub.add_parser("perturbation", help="Perturbation before/after overview")
    s.add_argument("--top-k", type=int, default=15)

    sub.add_parser("all", help="Run every figure with defaults")

    args = ap.parse_args()
    out_dir = ensure_out_dir(args.out_dir)

    if args.cmd == "umap":
        fig_umap(args.run_dir, args.pathways, out_dir,
                 args.top_n_label, args.n_neighbors, args.min_dist, args.seed)
    elif args.cmd == "network":
        fig_network(args.run_dir, out_dir, args.top_n, args.seed)
    elif args.cmd == "v2_vs_v3":
        fig_v2_vs_v3(out_dir, args.top_n)
    elif args.cmd == "consensus":
        fig_consensus(out_dir, args.version, args.top_n)
    elif args.cmd == "perturbation":
        fig_perturbation(out_dir, args.top_k)
    elif args.cmd == "all":
        ref_run = RUNS_ROOT / "V3_Run3"
        if ref_run.exists():
            fig_umap(ref_run, DEFAULT_PATHWAYS, out_dir)
            fig_network(ref_run, out_dir, top_n=100)
        fig_v2_vs_v3(out_dir, top_n=100)
        fig_consensus(out_dir, version="V3", top_n=100)
        fig_perturbation(out_dir, top_k=15)
        print(f"\n[all] Done — figures in {out_dir}")


if __name__ == "__main__":
    main()
