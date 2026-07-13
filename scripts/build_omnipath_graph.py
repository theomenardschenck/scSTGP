#!/usr/bin/env python3
"""
Build the full standalone OmniPath knowledge graph and cache it as TSVs.

Run ONCE on an internet-facing machine (local or Nautilus frontend) to
prefetch + assemble, then compute nodes can load it offline. The graph
is NOT wired into the VGAE pipeline — it is a standalone artifact the
project can load on demand via `omnipath_graph.load_omnipath_graph()`.

Examples:
    # core protein layer only (default), download what's missing
    python scripts/build_omnipath_graph.py --download

    # everything: protein + miRNA + drug + complex layers
    python scripts/build_omnipath_graph.py --layers all --download

    # rebuild from existing raw cache, no network
    python scripts/build_omnipath_graph.py --layers all

Outputs (under --cache-dir):
    graph_raw/<resource>.tsv.gz   per-resource normalized cache
    graph/edges.tsv.gz            assembled edge table
    graph/nodes.tsv.gz            node table + annotations
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "gnn"))

import omnipath_graph as og  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default="data/omnipath",
                   help="cache root (default: data/omnipath)")
    p.add_argument("--layers", default="core",
                   help="comma-separated subset of {core,mirna,drug,complex} "
                        "or 'all' (default: core). 'core' is always included.")
    p.add_argument("--organism", default="human",
                   choices=["human", "mouse", "rat"])
    p.add_argument("--download", action="store_true",
                   help="allow web fetch when a raw cache is absent "
                        "(omit on offline compute nodes)")
    p.add_argument("--no-annotations", action="store_true",
                   help="skip intercell roles + DGIdb druggability")
    p.add_argument("--force", action="store_true",
                   help="re-download every resource even if cached")
    args = p.parse_args()

    layers = [s.strip() for s in args.layers.split(",") if s.strip()]
    os.makedirs(args.cache_dir, exist_ok=True)

    og.build_omnipath_graph(
        cache_dir=args.cache_dir,
        layers=layers,
        organism=args.organism,
        download_if_missing=args.download,
        with_node_annotations=not args.no_annotations,
        force=args.force,
    )


if __name__ == "__main__":
    main()
