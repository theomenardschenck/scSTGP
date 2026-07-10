"""
omnipath_graph.py — Standalone full-OmniPath knowledge-graph builder.
=====================================================================
Builds a *complete* heterogeneous graph from ALL OmniPath resources and
caches it as flat edge/node TSVs. **Not wired into the VGAE pipeline** —
it is a self-contained artifact that the project can load on demand
(either as a NetworkX graph for exploration, or projected onto a
`gene_to_idx` to feed the encoder later).

Design choices (decided 2026-06-29):
  - **Heterogeneous nodes, miRNA / drug / complex layers opt-in.** The
    `core` layer (protein↔protein) is always built; `mirna`, `drug` and
    `complex` layers are added only when requested via `layers=`.
  - **Edge-table storage.** Two gzipped TSVs:
        <cache_dir>/graph/edges.tsv.gz   (one row per directed edge)
        <cache_dir>/graph/nodes.tsv.gz   (one row per node + annotations)
    Plus a per-resource raw cache <cache_dir>/graph_raw/<edge_type>.tsv.gz
    so rebuilds are offline-friendly (same frontend-prefetch pattern as
    `cache_omnipath.py`).

Resource coverage (vs the 3 currently used by `omnipath_integration.py`):

  INTERACTIONS (op.interactions.*) — uniform 17-col schema:
    signaling          OmniPath          core  protein→protein   (used)
    collectri_tf       CollecTRI         core  protein→protein   (used)
    tf_target          TFtarget          core  protein→protein   NEW
    transcriptional    Transcriptional   core  protein→protein   NEW
    kinase_substrate   KinaseExtra       core  protein→protein   NEW
    ligand_receptor    LigRecExtra       core  protein→protein   NEW
    pathway            PathwayExtra      core  protein–protein    NEW (undirected)
    tf_mirna           TFmiRNA           mirna protein→mirna      NEW
    mirna_target       PostTranslational mirna mirna→protein      NEW
    small_molecule     SmallMolecule     drug  drug→protein       NEW

  OTHER ENDPOINTS (op.requests.*):
    enzyme_substrate   Enzsub+SignedPTMs core  protein→protein   NEW (PTM + sign)
    complex_membership Complexes         cmplx complex–protein    NEW
    intercell roles    Intercell         (node annotations)       NEW
    druggability       Annotations/DGIdb (node annotations)       NEW

Every edge carries `[score, sign]` exactly like `omnipath_integration`:
  - score = curation_effort (raw; normalized to [0,1] at projection time)
  - sign  ∈ {−1, 0, +1}  (inhibition / unsigned / activation)

The shared OmniPath plumbing (lazy import, HTTP retry, sign derivation)
is reused from `omnipath_integration` to avoid divergence.

References:
- Türei D. et al., Nat Commun 2021 — OmniPath integrated knowledge.
- Türei D. et al., Mol Syst Biol 2021 — OmniPath intercell / annotations.
- Lo Surdo P. et al., NAR 2023 — SIGNOR 3.0 (signed causal).
- Müller-Dott S. et al., Genome Biol 2023 — CollecTRI.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse shared helpers from the existing integration module (lazy import,
# retry, sign derivation, score normalization, logging silencing).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import omnipath_integration as opi  # noqa: E402


# --------------------------------------------------------------------------- #
# Schema + resource registry
# --------------------------------------------------------------------------- #
EDGE_COLS = [
    "source_symbol", "target_symbol", "edge_type", "score", "sign",
    "directed", "n_resources", "source_entity", "target_entity",
]
NODE_COLS = [
    "symbol", "entity_type",
    "is_transmitter", "is_receiver", "is_secreted", "is_plasma_membrane",
    "is_druggable", "n_drug_records",
]

# Node-entity priority when a symbol appears under several entity types
# (a protein that is also listed as a complex component stays "protein").
_ENTITY_PRIORITY = {"protein": 0, "mirna": 1, "small_molecule": 2, "complex": 3}

# Interaction endpoints with a uniform schema → handled by one normalizer.
# edge_type: (endpoint_attr, layer, directed, source_entity, target_entity)
INTERACTION_RESOURCES = {
    "signaling":        ("OmniPath",        "core",  True,  "protein", "protein"),
    "collectri_tf":     ("CollecTRI",       "core",  True,  "protein", "protein"),
    "tf_target":        ("TFtarget",        "core",  True,  "protein", "protein"),
    "transcriptional":  ("Transcriptional", "core",  True,  "protein", "protein"),
    "kinase_substrate": ("KinaseExtra",     "core",  True,  "protein", "protein"),
    "ligand_receptor":  ("LigRecExtra",     "core",  True,  "protein", "protein"),
    "pathway":          ("PathwayExtra",    "core",  False, "protein", "protein"),
    "tf_mirna":         ("TFmiRNA",         "mirna", True,  "protein", "mirna"),
    "mirna_target":     ("PostTranslational", "mirna", True, "mirna", "protein"),
    "small_molecule":   ("SmallMolecule",   "drug",  True,  "small_molecule", "protein"),
}

# Which non-interaction resources belong to which layer.
_LAYER_OF_EXTRA = {
    "enzyme_substrate":   "core",
    "complex_membership": "complex",
}

VALID_LAYERS = ("core", "mirna", "drug", "complex")


# --------------------------------------------------------------------------- #
# Low-level cache I/O (graph-specific schema, distinct from opi._CACHE_COLS)
# --------------------------------------------------------------------------- #
def _graph_dir(cache_dir: str) -> str:
    return os.path.join(cache_dir, "graph")


def _raw_dir(cache_dir: str) -> str:
    return os.path.join(cache_dir, "graph_raw")


def _read_tsv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, sep="\t", compression="infer")
    except Exception as e:  # pragma: no cover — corrupted cache
        warnings.warn(f"omnipath_graph: unreadable cache {path}: {e}",
                      RuntimeWarning)
        return None


def _write_tsv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False, compression="gzip")


def _resource_or_cache(cache_dir: str, name: str, fetcher,
                       download_if_missing: bool, force: bool,
                       label: str) -> Optional[pd.DataFrame]:
    """Read a per-resource raw cache; fetch + write it if allowed/forced."""
    path = os.path.join(_raw_dir(cache_dir), name)
    if not force:
        cached = _read_tsv(path)
        if cached is not None:
            print(f"  [graph] {label}: raw cache ({len(cached)} rows)")
            return cached
    if not download_if_missing:
        print(f"  [graph] {label}: cache absent, download disabled → skip")
        return None
    print(f"  [graph] {label}: downloading…")
    df = fetcher()
    if df is None or df.empty:
        print(f"  [graph] {label}: no data")
        return None
    _write_tsv(df, path)
    print(f"  [graph] {label}: {len(df)} rows → {path}")
    return df


# --------------------------------------------------------------------------- #
# Fetchers — normalize each resource to the EDGE_COLS / NODE annotation schema
# --------------------------------------------------------------------------- #
def _op():
    """Return the lazily-imported omnipath module, or None if unavailable."""
    return opi.op if opi._lazy_import_omnipath() else None


def _fetch_interaction(edge_type: str, organism: str) -> Optional[pd.DataFrame]:
    """Fetch + normalize one interaction endpoint to EDGE_COLS."""
    op = _op()
    if op is None:
        return None
    endpoint, _layer, directed, src_ent, dst_ent = INTERACTION_RESOURCES[edge_type]
    cls = getattr(op.interactions, endpoint, None)
    if cls is None:
        warnings.warn(f"omnipath_graph: endpoint {endpoint} absent from API",
                      RuntimeWarning)
        return None
    df = opi._retry_fetch(
        edge_type,
        lambda: cls.get(organism=organism, genesymbols=True),
    )
    if df is None or df.empty:
        return None

    sign = opi._stim_inhib_to_sign(
        df.get("consensus_stimulation", df.get("is_stimulation", 0)),
        df.get("consensus_inhibition",  df.get("is_inhibition",  0)),
    )
    score = (df["curation_effort"].astype(float)
             if "curation_effort" in df.columns
             else pd.Series(np.ones(len(df), dtype=float)))
    n_res = (df["n_resources"].astype(int) if "n_resources" in df.columns
             else (df["n_sources"].astype(int) if "n_sources" in df.columns
                   else pd.Series(np.ones(len(df), dtype=int))))
    out = pd.DataFrame({
        "source_symbol": df["source_genesymbol"].astype(str),
        "target_symbol": df["target_genesymbol"].astype(str),
        "edge_type": edge_type,
        "score": score.values,
        "sign": sign,
        "directed": int(directed),
        "n_resources": n_res.values,
        "source_entity": src_ent,
        "target_entity": dst_ent,
    })
    return out


def _fetch_enzyme_substrate(organism: str) -> Optional[pd.DataFrame]:
    """Enzyme→substrate PTM edges with sign (Enzsub joined with SignedPTMs).

    Enzsub carries gene symbols + modification; SignedPTMs carries the
    activation/inhibition flags but only UniProt ids. We merge on the
    UniProt enzyme/substrate + residue + modification to attach the sign.
    """
    op = _op()
    if op is None:
        return None
    # NB: requests.* endpoints reject `organism=`; they filter via the
    # plural `organisms=` or not at all. Keep the calls minimal and let
    # the default (human) apply — pass genesymbols only.
    enz = opi._retry_fetch(
        "enzsub",
        lambda: op.requests.Enzsub.get(genesymbols=True),
    )
    if enz is None or enz.empty:
        return None
    signed = opi._retry_fetch(
        "signed_ptms",
        lambda: op.requests.SignedPTMs.get(),
    )

    sign = np.zeros(len(enz), dtype=np.float32)
    if signed is not None and not signed.empty:
        key_cols = ["enzyme", "substrate", "residue_type",
                    "residue_offset", "modification"]
        have = [c for c in key_cols if c in enz.columns and c in signed.columns]
        if have:
            s = signed.copy()
            s["_sign"] = opi._stim_inhib_to_sign(
                s.get("is_stimulation", 0), s.get("is_inhibition", 0))
            merged = enz.merge(s[have + ["_sign"]], on=have, how="left")
            sign = merged["_sign"].fillna(0).astype(np.float32).to_numpy()

    n_res = (enz["n_resources"].astype(int) if "n_resources" in enz.columns
             else pd.Series(np.ones(len(enz), dtype=int)))
    score = (enz["curation_effort"].astype(float)
             if "curation_effort" in enz.columns
             else pd.Series(np.ones(len(enz), dtype=float)))
    out = pd.DataFrame({
        "source_symbol": enz["enzyme_genesymbol"].astype(str),
        "target_symbol": enz["substrate_genesymbol"].astype(str),
        "edge_type": "enzyme_substrate",
        "score": score.values,
        "sign": sign,
        "directed": 1,
        "n_resources": n_res.values,
        "source_entity": "protein",
        "target_entity": "protein",
    })
    return out


def _fetch_complex_membership(organism: str) -> Optional[pd.DataFrame]:
    """Complex→component bipartite edges (one row per component)."""
    op = _op()
    if op is None:
        return None
    df = opi._retry_fetch("complexes", lambda: op.requests.Complexes.get())
    if df is None or df.empty:
        return None
    comp_col = ("components_genesymbols" if "components_genesymbols" in df.columns
                else "components")
    rows = []
    for name, comps in zip(df["name"].astype(str), df[comp_col].astype(str)):
        members = [c for c in comps.replace("-", "_").split("_") if c]
        for m in members:
            rows.append((f"COMPLEX:{name}", m))
    if not rows:
        return None
    src, dst = zip(*rows)
    out = pd.DataFrame({
        "source_symbol": src,
        "target_symbol": dst,
        "edge_type": "complex_membership",
        "score": 1.0,
        "sign": 0.0,
        "directed": 0,
        "n_resources": 1,
        "source_entity": "complex",
        "target_entity": "protein",
    })
    return out


def _fetch_intercell_roles(organism: str) -> Optional[pd.DataFrame]:
    """Per-protein intercellular-communication roles (node annotation)."""
    op = _op()
    if op is None:
        return None
    df = opi._retry_fetch("intercell", lambda: op.requests.Intercell.get())
    if df is None or df.empty or "genesymbol" not in df.columns:
        return None

    def _any(col):
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].astype(str).str.lower().isin(["true", "1", "1.0"])

    tmp = pd.DataFrame({
        "symbol": df["genesymbol"].astype(str),
        "is_transmitter": _any("transmitter"),
        "is_receiver": _any("receiver"),
        "is_secreted": _any("secreted"),
        "is_plasma_membrane": (_any("plasma_membrane_transmembrane")
                               | _any("plasma_membrane_peripheral")),
    })
    agg = tmp.groupby("symbol").max().reset_index()
    return agg


def _fetch_druggability(organism: str) -> Optional[pd.DataFrame]:
    """DGIdb drug-gene records per protein (node annotation)."""
    op = _op()
    if op is None:
        return None
    df = opi._retry_fetch(
        "druggability",
        lambda: op.requests.Annotations.get(resources="DGIdb"),
    )
    if df is None or df.empty or "genesymbol" not in df.columns:
        return None
    counts = (df.groupby("genesymbol").size()
                .rename("n_drug_records").reset_index()
                .rename(columns={"genesymbol": "symbol"}))
    counts["is_druggable"] = True
    return counts


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def _resolve_layers(layers: Iterable[str]) -> set[str]:
    req = set(layers)
    if "all" in req:
        return set(VALID_LAYERS)
    unknown = req - set(VALID_LAYERS)
    if unknown:
        warnings.warn(f"omnipath_graph: unknown layers ignored {unknown}",
                      RuntimeWarning)
    req &= set(VALID_LAYERS)
    req.add("core")  # core is mandatory
    return req


def build_omnipath_graph(
    cache_dir: str = "data/omnipath",
    layers: Iterable[str] = ("core",),
    organism: str = "human",
    download_if_missing: bool = False,
    with_node_annotations: bool = True,
    force: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the full OmniPath graph → (nodes_df, edges_df), and write
    `<cache_dir>/graph/{edges,nodes}.tsv.gz`.

    Args:
        cache_dir: root cache dir (raw resources under graph_raw/, the
            assembled graph under graph/).
        layers: subset of {core, mirna, drug, complex} or {"all"}.
            `core` is always included.
        organism: passed to OmniPath ("human"/"mouse"/"rat").
        download_if_missing: allow web fetch when a raw cache is absent
            (set True only on the internet-facing frontend).
        with_node_annotations: attach intercell roles + DGIdb druggability.
        force: re-download every resource even if cached.

    Returns:
        (nodes_df, edges_df) — also persisted to disk.
    """
    active = _resolve_layers(layers)
    if not download_if_missing:
        opi.silence_omnipath_logging()
    print(f"[omnipath_graph] layers={sorted(active)} organism={organism} "
          f"download={download_if_missing}")

    edge_frames = []

    # 1) interaction endpoints (filtered by layer)
    for edge_type, (endpoint, layer, *_rest) in INTERACTION_RESOURCES.items():
        if layer not in active:
            continue
        df = _resource_or_cache(
            cache_dir, f"{edge_type}.tsv.gz",
            lambda et=edge_type: _fetch_interaction(et, organism),
            download_if_missing, force, label=edge_type)
        if df is not None and not df.empty:
            edge_frames.append(df)

    # 2) enzyme-substrate PTM (core)
    if "core" in active:
        df = _resource_or_cache(
            cache_dir, "enzyme_substrate.tsv.gz",
            lambda: _fetch_enzyme_substrate(organism),
            download_if_missing, force, label="enzyme_substrate")
        if df is not None and not df.empty:
            edge_frames.append(df)

    # 3) complex membership (complex layer)
    if "complex" in active:
        df = _resource_or_cache(
            cache_dir, "complex_membership.tsv.gz",
            lambda: _fetch_complex_membership(organism),
            download_if_missing, force, label="complex_membership")
        if df is not None and not df.empty:
            edge_frames.append(df)

    if not edge_frames:
        warnings.warn("omnipath_graph: no edges built (all resources empty?)",
                      RuntimeWarning)
        edges = pd.DataFrame(columns=EDGE_COLS)
    else:
        edges = pd.concat(edge_frames, ignore_index=True)
        edges = edges[(edges["source_symbol"] != edges["target_symbol"])
                      & edges["source_symbol"].astype(bool)
                      & edges["target_symbol"].astype(bool)]
        edges = edges[EDGE_COLS].reset_index(drop=True)

    # 4) node table = union of edge endpoints (+ entity by priority)
    nodes = _build_node_table(edges)

    # 5) node annotations (intercell roles + druggability)
    if with_node_annotations:
        nodes = _attach_annotations(nodes, cache_dir, organism,
                                    download_if_missing, force)

    # 6) persist
    _write_tsv(edges, os.path.join(_graph_dir(cache_dir), "edges.tsv.gz"))
    _write_tsv(nodes, os.path.join(_graph_dir(cache_dir), "nodes.tsv.gz"))
    print(summary(nodes, edges))
    return nodes, edges


def _build_node_table(edges: pd.DataFrame) -> pd.DataFrame:
    """Collapse edge endpoints into a node table with one entity type each."""
    if edges.empty:
        return pd.DataFrame(columns=NODE_COLS)
    a = edges[["source_symbol", "source_entity"]].rename(
        columns={"source_symbol": "symbol", "source_entity": "entity_type"})
    b = edges[["target_symbol", "target_entity"]].rename(
        columns={"target_symbol": "symbol", "target_entity": "entity_type"})
    alln = pd.concat([a, b], ignore_index=True).drop_duplicates()
    alln["_prio"] = alln["entity_type"].map(_ENTITY_PRIORITY).fillna(9)
    alln = (alln.sort_values("_prio")
                .drop_duplicates("symbol", keep="first")
                .drop(columns="_prio"))
    for c in ("is_transmitter", "is_receiver", "is_secreted",
              "is_plasma_membrane", "is_druggable"):
        alln[c] = False
    alln["n_drug_records"] = 0
    return alln[NODE_COLS].reset_index(drop=True)


def _attach_annotations(nodes: pd.DataFrame, cache_dir: str, organism: str,
                        download_if_missing: bool, force: bool) -> pd.DataFrame:
    roles = _resource_or_cache(
        cache_dir, "intercell_roles.tsv.gz",
        lambda: _fetch_intercell_roles(organism),
        download_if_missing, force, label="intercell_roles")
    drug = _resource_or_cache(
        cache_dir, "druggability.tsv.gz",
        lambda: _fetch_druggability(organism),
        download_if_missing, force, label="druggability")

    if roles is not None and not roles.empty:
        nodes = nodes.drop(columns=[c for c in roles.columns if c != "symbol"
                                    and c in nodes.columns])
        nodes = nodes.merge(roles, on="symbol", how="left")
    if drug is not None and not drug.empty:
        nodes = nodes.drop(columns=[c for c in drug.columns if c != "symbol"
                                    and c in nodes.columns])
        nodes = nodes.merge(drug, on="symbol", how="left")

    for c in ("is_transmitter", "is_receiver", "is_secreted",
              "is_plasma_membrane", "is_druggable"):
        if c in nodes.columns:
            nodes[c] = nodes[c].fillna(False).astype(bool)
    if "n_drug_records" in nodes.columns:
        nodes["n_drug_records"] = nodes["n_drug_records"].fillna(0).astype(int)
    return nodes[NODE_COLS].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Load / consume
# --------------------------------------------------------------------------- #
def load_omnipath_graph(cache_dir: str = "data/omnipath",
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the persisted (nodes_df, edges_df). Raises if not built yet."""
    gdir = _graph_dir(cache_dir)
    nodes = _read_tsv(os.path.join(gdir, "nodes.tsv.gz"))
    edges = _read_tsv(os.path.join(gdir, "edges.tsv.gz"))
    if nodes is None or edges is None:
        raise FileNotFoundError(
            f"omnipath_graph: build first — missing graph/ TSVs under {cache_dir}")
    return nodes, edges


def to_networkx(nodes: pd.DataFrame, edges: pd.DataFrame,
                multigraph: bool = True):
    """Materialize a NetworkX (Multi)DiGraph for exploration.

    Node attributes = annotation columns; edge attributes = edge_type,
    score, sign, directed, n_resources.
    """
    import networkx as nx
    G = nx.MultiDiGraph() if multigraph else nx.DiGraph()
    for r in nodes.itertuples(index=False):
        d = r._asdict()
        sym = d.pop("symbol")
        G.add_node(sym, **d)
    for r in edges.itertuples(index=False):
        G.add_edge(r.source_symbol, r.target_symbol,
                   key=r.edge_type if multigraph else None,
                   edge_type=r.edge_type, score=float(r.score),
                   sign=float(r.sign), directed=int(r.directed),
                   n_resources=int(r.n_resources))
    return G


def project_to_gene_indices(edges: pd.DataFrame, gene_to_idx: dict,
                            edge_types: Optional[Iterable[str]] = None,
                            alias_map: Optional[dict] = None) -> dict:
    """Project edges onto graph indices, grouped by edge_type.

    Reuses `omnipath_integration._project_to_graph` (strict filter on
    `gene_to_idx`, dedup, sign by majority vote). Returns
    `{edge_type: (src_idx, dst_idx, attr=[score,sign], metadata_df)}`.
    Only protein↔protein edge types are meaningful here (miRNA / drug /
    complex symbols won't be in a gene `gene_to_idx`).

    HGNC alias handling: pass an `alias_map` (from `hgnc_alias.build_alias_map`)
    to match in **approved-symbol space**. Both the edge endpoints and the
    `gene_to_idx` keys are canonicalized to their approved symbol before the
    join, then original graph indices are returned. Without it, matching is a
    raw exact-symbol join (legacy behaviour) and drops symbol-drift genes
    such as H2AFZ↔H2AZ1. See `hgnc_alias.py`.
    """
    types = (list(edge_types) if edge_types is not None
             else sorted(edges["edge_type"].unique()))

    # Build the (possibly canonical) index the join runs against.
    if alias_map:
        import hgnc_alias  # local, sub-project module
        canon_g2i: dict = {}
        for key, idx in gene_to_idx.items():
            canon = alias_map.get(key, key)
            canon_g2i.setdefault(canon, idx)  # first key wins on collision
        join_index = canon_g2i
    else:
        join_index = gene_to_idx

    out = {}
    for et in types:
        sub = edges.loc[edges["edge_type"] == et,
                        ["source_symbol", "target_symbol", "score", "sign"]].copy()
        if sub.empty:
            continue
        if alias_map:
            sub["source_symbol"] = hgnc_alias.canonicalize_series(
                sub["source_symbol"], alias_map)
            sub["target_symbol"] = hgnc_alias.canonicalize_series(
                sub["target_symbol"], alias_map)
        out[et] = opi._project_to_graph(sub, join_index.keys(), join_index)
    return out


def summary(nodes: pd.DataFrame, edges: pd.DataFrame) -> str:
    """Human-readable build summary."""
    lines = ["[omnipath_graph] summary",
             f"  nodes: {len(nodes)}"]
    if not nodes.empty:
        for et, n in nodes["entity_type"].value_counts().items():
            lines.append(f"    {et:>14s}: {n}")
        if "is_druggable" in nodes.columns:
            lines.append(f"    druggable     : {int(nodes['is_druggable'].sum())}")
    lines.append(f"  edges: {len(edges)}")
    if not edges.empty:
        for et, n in edges["edge_type"].value_counts().items():
            n_signed = int((edges.loc[edges['edge_type'] == et, 'sign'] != 0).sum())
            lines.append(f"    {et:>18s}: {n:>7d}  (signed {n_signed})")
    return "\n".join(lines)
