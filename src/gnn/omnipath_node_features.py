"""
omnipath_node_features.py — OmniPath-derived node features for the 'gene'
node type (Module 1 of the OmniPath integration, gnn_futur §10).
=========================================================================
Adds *features* to the existing homogeneous 'gene' node — **no new node
type**. Three groups, all optional / individually excludable:

  1. Intercellular-communication role (OmniPath `intercell`):
       op_transmitter, op_receiver, op_secreted, op_plasma_membrane
     (op_secreted is the SASP-relevant flag.)
  2. Druggability (OmniPath annotations / DGIdb):
       op_druggable (0/1), op_drug_records (log1p-normalized count)
  3. Molecular class (HGNC `locus_group`/`locus_type`, via hgnc_alias):
       is_protein_coding, is_lncrna, is_mirna
     (small-molecule does not apply to a gene node → intentionally absent.)

Data sources are the OmniPath graph artifact + HGNC biotype cache, both
prebuilt on an internet-facing machine (same offline pattern as
`cache_omnipath.py` / `hgnc_alias.py`):
  - <cache_dir>/graph/nodes.tsv.gz   (from build_omnipath_graph.py)
  - <cache_dir>/hgnc_biotype_map.tsv.gz (from hgnc_alias.build_biotype_map)

Everything is offline-safe: any missing source yields all-zero features
for that group (never crashes), so the feature block degrades gracefully
and stays fully reversible via the toggling flag.

Symbols are matched in HGNC **approved** space when an `alias_map` is
provided (our gene_to_idx keys are legacy symbols, OmniPath/HGNC use
current ones — cf. hgnc_alias.py).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# Canonical order of the OmniPath node-feature columns (appended AFTER the
# scRNA feature block, so existing feature ordering is untouched when OFF).
INTERCELL_FEATURES = [
    "op_transmitter", "op_receiver", "op_secreted", "op_plasma_membrane",
]
DRUGGABILITY_FEATURES = ["op_druggable", "op_drug_records"]
BIOTYPE_FEATURES = ["is_protein_coding", "is_lncrna", "is_mirna"]
ALL_FEATURES: List[str] = (INTERCELL_FEATURES + DRUGGABILITY_FEATURES
                           + BIOTYPE_FEATURES)

# Column mapping in the OmniPath nodes.tsv.gz (bool) → our feature name.
_INTERCELL_SRC = {
    "op_transmitter": "is_transmitter",
    "op_receiver": "is_receiver",
    "op_secreted": "is_secreted",
    "op_plasma_membrane": "is_plasma_membrane",
}


def _canonicalize(gene_symbols: Iterable[str],
                  alias_map: Optional[dict]) -> List[str]:
    amap = alias_map or {}
    return [amap.get(g, g) for g in gene_symbols]


def build_node_feature_arrays(
    gene_symbols: Iterable[str],
    cache_dir: str = "data/omnipath",
    alias_map: Optional[dict] = None,
    download_if_missing: bool = False,
) -> "OrderedDict[str, np.ndarray]":
    """Return `{feature_name: float32 array (n_genes,)}` aligned to
    `gene_symbols` (node-index order).

    Missing sources → the corresponding group is all-zeros. Never raises.
    """
    gene_symbols = list(gene_symbols)
    n = len(gene_symbols)
    canon = _canonicalize(gene_symbols, alias_map)
    out: "OrderedDict[str, np.ndarray]" = OrderedDict()

    # --- groups 1+2 : OmniPath node annotations (graph/nodes.tsv.gz) --------
    nodes_path = os.path.join(cache_dir, "graph", "nodes.tsv.gz")
    ann: Optional[pd.DataFrame] = None
    if os.path.exists(nodes_path):
        try:
            nd = pd.read_csv(nodes_path, sep="\t")
            ann = nd.drop_duplicates("symbol").set_index("symbol").reindex(canon)
        except Exception as e:  # pragma: no cover
            print(f"  [op-nodefeat] nodes.tsv.gz illisible ({e}) → intercell/"
                  f"druggability = 0")
    else:
        print(f"  [op-nodefeat] {nodes_path} absent → intercell/druggability "
              f"= 0 (build_omnipath_graph d'abord)")

    for feat, src in _INTERCELL_SRC.items():
        if ann is not None and src in ann.columns:
            out[feat] = (ann[src].fillna(False).to_numpy().astype(np.float32))
        else:
            out[feat] = np.zeros(n, dtype=np.float32)

    if ann is not None and "is_druggable" in ann.columns:
        out["op_druggable"] = (ann["is_druggable"].fillna(False)
                               .to_numpy().astype(np.float32))
    else:
        out["op_druggable"] = np.zeros(n, dtype=np.float32)

    if ann is not None and "n_drug_records" in ann.columns:
        raw = ann["n_drug_records"].fillna(0).to_numpy().astype(np.float32)
        logv = np.log1p(raw)
        vmax = float(logv.max())
        out["op_drug_records"] = (logv / vmax if vmax > 0
                                  else logv).astype(np.float32)
    else:
        out["op_drug_records"] = np.zeros(n, dtype=np.float32)

    # --- group 3 : HGNC biotype -------------------------------------------
    bio_flags = _load_biotype(canon, cache_dir, download_if_missing)
    for feat in BIOTYPE_FEATURES:
        out[feat] = bio_flags.get(feat, np.zeros(n, dtype=np.float32))

    return out


def _load_biotype(canon_symbols: List[str], cache_dir: str,
                  download_if_missing: bool) -> Dict[str, np.ndarray]:
    n = len(canon_symbols)
    try:
        import hgnc_alias
        bdf = hgnc_alias.build_biotype_map(
            cache_dir, download_if_missing=download_if_missing)
    except Exception as e:  # pragma: no cover
        print(f"  [op-nodefeat] biotype indisponible ({e}) → classes = 0")
        return {}
    if bdf is None or bdf.empty:
        return {}
    bdf = bdf.drop_duplicates("symbol").set_index("symbol").reindex(canon_symbols)
    res: Dict[str, np.ndarray] = {}
    for feat in BIOTYPE_FEATURES:
        if feat in bdf.columns:
            res[feat] = bdf[feat].fillna(0).to_numpy().astype(np.float32)
        else:
            res[feat] = np.zeros(n, dtype=np.float32)
    return res


def coverage_summary(arrays: "OrderedDict[str, np.ndarray]") -> str:
    """One-line non-zero coverage per feature (for build logs)."""
    parts = [f"{k}={int((v != 0).sum())}" for k, v in arrays.items()]
    return "  [op-nodefeat] non-zero: " + ", ".join(parts)
