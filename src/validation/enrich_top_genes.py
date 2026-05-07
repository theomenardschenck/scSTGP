#!/usr/bin/env python3
"""
enrich_top_genes.py — Annotate the top-N VGAE genes with public-DB info.

Queries MyGene.info (which aggregates NCBI, UniProt, Ensembl) to retrieve
a human-readable name and a NCBI/RefSeq summary for each top gene, and
produces:
    • a TSV with rank, importance, cluster, gene symbol, full name, brief
      description (first sentence), aging-DB flags;
    • a figure: horizontal bars of importance with the brief description
      written next to each bar, coloured by cluster.

Why MyGene.info and not GeneCards?
----------------------------------
GeneCards does not expose a public API and its terms of use prohibit
scraping. MyGene.info is a free REST service (https://mygene.info) that
is explicitly designed for programmatic access and aggregates data from
NCBI/Entrez (where "summary" comes from), UniProt, Ensembl, GO, and
others.

Usage
-----
    python src/enrich_top_genes.py --run-dir output/gnn_vgae/V3_Run3 --top-n 25

    # Skip network calls (reuse cache)
    python src/enrich_top_genes.py --run-dir ... --top-n 25 --no-fetch

Outputs
-------
    <run-dir>/figure/top{N}_enriched.tsv
    <run-dir>/figure/top{N}_enriched.png
    data/cache/mygene_cache.tsv  (persistent cross-run cache)
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
CACHE_PATH = ROOT / "data/cache/mygene_cache.tsv"

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 10})


# --------------------------------------------------------------------------- #
# MyGene.info helpers
# --------------------------------------------------------------------------- #
def load_cache() -> dict[str, dict]:
    """Return {symbol -> {name, summary}} from disk cache."""
    if not CACHE_PATH.exists():
        return {}
    df = pd.read_csv(CACHE_PATH, sep="\t").fillna("")
    return {row["symbol"]: {"name": row["name"], "summary": row["summary"]}
            for _, row in df.iterrows()}


def save_cache(cache: dict[str, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"symbol": s, "name": v.get("name", ""),
             "summary": v.get("summary", "")}
            for s, v in sorted(cache.items())]
    pd.DataFrame(rows).to_csv(CACHE_PATH, sep="\t", index=False)


def fetch_mygene(symbols: list[str], cache: dict[str, dict],
                 species: str = "human",
                 batch_sleep: float = 0.2) -> dict[str, dict]:
    """Fill `cache` with MyGene.info data for any missing symbols."""
    import mygene

    missing = [s for s in symbols if s not in cache]
    if not missing:
        print(f"[enrich] all {len(symbols)} symbols already cached.")
        return cache

    mg = mygene.MyGeneInfo()
    print(f"[enrich] querying MyGene.info for {len(missing)} new symbols ...")
    # Query in a single batch — MyGene handles up to 1000 ids per call
    results = mg.querymany(missing, scopes="symbol", species=species,
                           fields="symbol,name,summary", size=1,
                           returnall=False)
    for r in results:
        sym = r.get("query") or r.get("symbol")
        if not sym or r.get("notfound"):
            cache[sym] = {"name": "", "summary": ""}
            continue
        cache[sym] = {
            "name": r.get("name", "") or "",
            "summary": r.get("summary", "") or "",
        }
    # Any symbols that came back empty get an empty record so we don't
    # re-query them on the next run.
    for s in missing:
        cache.setdefault(s, {"name": "", "summary": ""})
    time.sleep(batch_sleep)
    save_cache(cache)
    print(f"[enrich] cache written to {CACHE_PATH}")
    return cache


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def first_sentence(text: str, max_chars: int = 160) -> str:
    """Return the first sentence of `text`, truncated to `max_chars`."""
    if not text:
        return ""
    # Strip [provided by ...] trailing provider annotation.
    text = re.sub(r"\s*\[provided by[^\]]*\]", "", text).strip()
    # Split on the first sentence boundary (. ! ?) followed by space/quote.
    m = re.search(r"(.+?[.!?])(\s|$)", text)
    sent = m.group(1) if m else text
    if len(sent) > max_chars:
        sent = sent[:max_chars - 1].rstrip() + "…"
    return sent


# --------------------------------------------------------------------------- #
# TSV
# --------------------------------------------------------------------------- #
DB_COLS = ["in_genage", "in_cellage", "in_msigdb_aging",
           "in_ageanno", "in_aging_local"]


def build_tsv(run_dir: Path, top_n: int,
              cache: dict[str, dict]) -> pd.DataFrame:
    rk = pd.read_csv(run_dir / "gene_ranking_vgae.csv")
    top = (rk.sort_values("vgae_importance", ascending=False)
             .head(top_n).copy().reset_index(drop=True))
    top.insert(0, "rank", top.index + 1)

    names = [cache.get(g, {}).get("name", "") for g in top["gene"]]
    summaries = [cache.get(g, {}).get("summary", "") for g in top["gene"]]
    top["name"] = names
    top["brief"] = [first_sentence(s) for s in summaries]

    # Carry over DB flags + cluster + importance components
    cols = ["rank", "gene", "vgae_importance", "cluster", "n_databases",
            *DB_COLS, "name", "brief"]
    cols = [c for c in cols if c in top.columns]
    return top[cols]


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def plot_enriched(df: pd.DataFrame, out_path: Path, top_n: int) -> Path:
    df = df.copy().sort_values("vgae_importance", ascending=True).reset_index(drop=True)
    n = len(df)
    # Generous vertical room (≈0.75" per row) so the annotation on the
    # right has space for up to 3 wrapped lines without overlap.
    fig_h = max(6, 0.75 * n + 1.5)
    fig, ax = plt.subplots(figsize=(18, fig_h))

    # Colour by cluster
    if "cluster" in df.columns:
        clusters = sorted(df["cluster"].dropna().unique())
        palette = dict(zip(clusters,
                           sns.color_palette("tab10", n_colors=len(clusters))))
        colors = [palette.get(c, "#888888") for c in df["cluster"]]
    else:
        colors = ["#4C72B0"] * n

    y = list(range(n))
    bar_h = 0.7  # taller bars for readability; rows are 1 unit apart
    ax.barh(y, df["vgae_importance"], color=colors,
            edgecolor="black", linewidth=0.5, height=bar_h)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['gene']}  ·  rk{int(r['rank'])}"
                        for _, r in df.iterrows()], fontsize=9)
    # Row boundaries (subtle grey lines) aid visual separation
    for i in range(n):
        ax.axhline(i + 0.5, color="#e0e0e0", linewidth=0.4, zorder=0)

    # Right-side annotation: wrapped brief description.
    x_max = df["vgae_importance"].max()
    WRAP_W = 70  # characters per line
    for i, row in df.iterrows():
        dbs = int(row.get("n_databases", 0)) if "n_databases" in df.columns else 0
        brief = (row.get("brief") or "").strip()
        full_name = (row.get("name") or "").strip()
        # Compose the annotation.
        parts = []
        if full_name:
            parts.append(textwrap.fill(full_name, width=WRAP_W))
        if brief and brief.lower() != full_name.lower():
            parts.append(textwrap.fill(brief, width=WRAP_W))
        if dbs:
            parts.append(f"[in {dbs} aging DB{'s' if dbs > 1 else ''}]")
        txt = "\n".join(parts) if parts else "(no description available)"
        ax.text(x_max * 1.02, i, txt, va="center", ha="left",
                fontsize=8, color="black")

    # Extra x-space for the annotations (reserve ~60% of the axis width).
    ax.set_xlim(0, x_max * 2.8)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("vgae_importance")
    ax.set_title(f"Top-{top_n} VGAE genes — brief annotation (MyGene.info / NCBI)")
    # Hide x-ticks beyond the data range (the right part is just text)
    ax.set_xticks([0, x_max * 0.25, x_max * 0.5, x_max * 0.75, x_max])

    # Cluster legend
    if "cluster" in df.columns:
        handles = [plt.Rectangle((0, 0), 1, 1,
                                 facecolor=palette.get(c, "#888888"),
                                 edgecolor="black")
                   for c in clusters]
        ax.legend(handles, [f"cluster {c}" for c in clusters],
                  loc="lower right", fontsize=7, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path,
                    default=ROOT / "output/gnn_vgae/V3_Run3")
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: <run-dir>/figure/).")
    ap.add_argument("--no-fetch", action="store_true",
                    help="Do not query MyGene.info; use cache only.")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    out_dir: Path = args.out_dir or (run_dir / "figure")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load top-N symbols
    rk = pd.read_csv(run_dir / "gene_ranking_vgae.csv")
    top_syms = (rk.sort_values("vgae_importance", ascending=False)
                  .head(args.top_n)["gene"].astype(str).tolist())

    # 2) Fetch / load cache
    cache = load_cache()
    if not args.no_fetch:
        cache = fetch_mygene(top_syms, cache)

    # 3) Build TSV
    df = build_tsv(run_dir, args.top_n, cache)
    tsv_path = out_dir / f"top{args.top_n}_enriched.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"[enrich] Wrote {tsv_path}  ({len(df)} rows)")

    # 4) Figure
    fig_path = out_dir / f"top{args.top_n}_enriched.png"
    plot_enriched(df, fig_path, args.top_n)
    print(f"[enrich] Wrote {fig_path}")


if __name__ == "__main__":
    main()
