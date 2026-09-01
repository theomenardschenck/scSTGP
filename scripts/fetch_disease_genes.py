#!/usr/bin/env python3
"""Fetch a gene-disease reference set for ANY phenotype, from DISEASES (Jensen lab).

WHY THIS EXISTS
---------------
The validation sets shipped with this project are senescence-specific (GenAge,
CellAge, SenMayo...). Applying the tool to another transition means bringing
your own reference, and the registry (`data/gene_sets/registry.yaml`) is built
for exactly that. What was missing was a way to GET such a set for an arbitrary
disease without an account: DisGeNET now sits behind a login, Open Targets ships
gigabytes of JSON. DISEASES (diseases.jensenlab.org, Pletscher-Frankild 2015)
is a plain TSV over HTTP, keyed by Disease Ontology term.

    knowledge   curated database records (UniProtKB, MedlinePlus...) — small, clean
    experiments GWAS-derived associations
    textmining  literature co-occurrence — large, noisy
    integrated  all channels merged into one score

USAGE
-----
    # find the term first
    python scripts/fetch_disease_genes.py --search lupus

    # curated set (goes into registry.yaml as role: validation)
    python scripts/fetch_disease_genes.py --doid DOID:9074 --name sle

    # broader set, score-filtered
    python scripts/fetch_disease_genes.py --doid DOID:9074 --name sle \
        --channel integrated --min-score 2.0

Downloads land in data/databases/jensen_diseases/ and are reused when present
(offline-first, like the other cache scripts). The per-disease slice is written
to data/gene_sets/ and the registry entry to paste is printed at the end.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "databases" / "jensen_diseases"
OUT = ROOT / "data" / "gene_sets"
BASE = "https://download.jensenlab.org"

CHANNELS = {
    # channel -> (file, columns, score column)
    "knowledge": ("human_disease_knowledge_filtered.tsv",
                  ["ensp", "gene_symbol", "doid", "disease", "source",
                   "evidence", "confidence"], "confidence"),
    "experiments": ("human_disease_experiments_filtered.tsv",
                    ["ensp", "gene_symbol", "doid", "disease", "source",
                     "evidence", "confidence"], "confidence"),
    "textmining": ("human_disease_textmining_filtered.tsv",
                   ["ensp", "gene_symbol", "doid", "disease", "zscore",
                    "confidence", "url"], "confidence"),
    "integrated": ("human_disease_integrated_full.tsv",
                   ["ensp", "gene_symbol", "doid", "disease", "confidence"],
                   "confidence"),
}


def ensure(channel: str) -> Path:
    fname, _, _ = CHANNELS[channel]
    path = CACHE / fname
    if path.exists() and path.stat().st_size > 0:
        print(f"[disease] cache : {path} ({path.stat().st_size/1e6:.1f} Mo)")
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{fname}"
    print(f"[disease] téléchargement {url}")
    urllib.request.urlretrieve(url, path)
    print(f"[disease] écrit {path} ({path.stat().st_size/1e6:.1f} Mo)")
    return path


def load(channel: str) -> pd.DataFrame:
    path = ensure(channel)
    _, cols, _ = CHANNELS[channel]
    df = pd.read_csv(path, sep="\t", header=None, names=cols,
                     dtype=str, on_bad_lines="skip")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", help="list Disease Ontology terms matching this text")
    ap.add_argument("--doid", help="Disease Ontology id, e.g. DOID:9074")
    ap.add_argument("--name", help="short slug for the output file and registry entry")
    ap.add_argument("--channel", default="knowledge", choices=list(CHANNELS))
    ap.add_argument("--min-score", type=float, default=None,
                    help="keep associations at or above this confidence")
    ap.add_argument("--top", type=int, default=None, help="keep the N best-scoring genes")
    ap.add_argument("--append-registry", action="store_true",
                    help="append the entry to data/gene_sets/registry.yaml instead of "
                         "only printing it (skips an entry of the same name)")
    args = ap.parse_args()

    if args.search:
        df = load("knowledge")
        hits = (df[df.disease.str.contains(args.search, case=False, na=False)]
                .groupby(["doid", "disease"]).size()
                .reset_index(name="n_genes").sort_values("n_genes", ascending=False))
        if hits.empty:
            print(f"[disease] aucun terme ne contient {args.search!r} "
                  "(essayez --channel integrated pour un vocabulaire plus large)")
        else:
            print(hits.head(25).to_string(index=False))
        return 0

    if not (args.doid and args.name):
        ap.error("--doid et --name sont requis (ou utilisez --search)")

    df = load(args.channel)
    sub = df[df.doid == args.doid].copy()
    if sub.empty:
        print(f"[disease] {args.doid} absent du canal {args.channel}", file=sys.stderr)
        return 1
    disease = sub.disease.iloc[0]
    if args.min_score is not None:
        sub = sub[sub.confidence >= args.min_score]
    sub = (sub.sort_values("confidence", ascending=False)
              .drop_duplicates("gene_symbol"))
    if args.top:
        sub = sub.head(args.top)

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{args.name}_{args.channel}.tsv"
    sub[["gene_symbol", "confidence", "disease"]].to_csv(dest, sep="\t", index=False)
    print(f"[disease] {disease} ({args.doid}) — canal {args.channel} : "
          f"{len(sub)} gènes → {dest}")
    print(sub.gene_symbol.head(12).tolist())
    entry = (f"\n# {disease} ({args.doid}) — DISEASES/{args.channel}"
             f"{f', score >= {args.min_score}' if args.min_score else ''}"
             f" — {len(sub)} gènes, ajouté par scripts/fetch_disease_genes.py\n"
             f"- name: {args.name}_{args.channel}\n"
             f"  path: data/gene_sets/{dest.name}\n"
             f"  symbol_col: gene_symbol\n"
             f"  format: tsv\n"
             f"  role: validation      # post-hoc uniquement — jamais dans le graphe\n")
    registry = ROOT / "data" / "gene_sets" / "registry.yaml"
    if args.append_registry:
        current = registry.read_text(encoding="utf-8") if registry.exists() else ""
        if f"name: {args.name}_{args.channel}" in current:
            print(f"[disease] {args.name}_{args.channel} déjà dans {registry} — inchangé")
        else:
            registry.write_text(current.rstrip("\n") + "\n" + entry, encoding="utf-8")
            print(f"[disease] entrée ajoutée à {registry}")
    else:
        print("\n  À coller dans data/gene_sets/registry.yaml "
              "(ou relancer avec --append-registry) :")
        print(entry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
