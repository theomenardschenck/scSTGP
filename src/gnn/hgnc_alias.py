"""
hgnc_alias.py — HGNC gene-symbol alias normalization (offline-first).
====================================================================
Maps any gene symbol variant (previous symbol, alias symbol) to its
current HGNC-**approved** symbol, so that graphs built from data using
legacy symbols (our scRNA HUVEC graph: H2AFZ, AARS, ATP5A1, HIST1H1B…)
can be matched against resources using current symbols (OmniPath:
H2AZ1, AARS1, ATP5F1A, H1-5…).

Motivation: a raw symbol-match between our 11 133-gene graph and the
OmniPath protein set loses ~813 "real" genes purely to symbol drift —
including top drivers (H2AFZ = driver #2). Normalizing both sides to the
approved symbol recovers them.

Source of truth: the HGNC "complete set" TSV (genenames.org / EBI), which
lists for each approved symbol its `alias_symbol` and `prev_symbol`
(pipe-separated). We keep only the 4 columns we need and cache a compact
2-column map `<cache_dir>/hgnc_alias_map.tsv.gz`.

Same offline pattern as `cache_omnipath.py`: prefetch once on an
internet-facing machine, then compute nodes read the compact cache.

Conflict policy (deliberately conservative — never create a wrong edge):
  - An approved symbol always maps to itself (self-map wins).
  - A variant that is ALSO an approved symbol of another gene is left
    untouched (its own approved identity wins; not remapped).
  - A variant that points to ≥2 distinct approved symbols is dropped
    (ambiguous → identity fallback, no guess).

References:
- Seal R.L. et al., "Genenames.org: the HGNC resources in 2023",
  Nucleic Acids Res 2023 — HGNC nomenclature + complete set.
- Braschi B. et al., "Consensus gene nomenclature", 2022.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Iterable, Optional

import pandas as pd

# HGNC complete set (public mirror on Google Cloud Storage). ~17 MB.
HGNC_COMPLETE_SET_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/"
    "hgnc_complete_set.txt"
)
_NEEDED_COLS = ["symbol", "alias_symbol", "prev_symbol", "status"]
_BIOTYPE_COLS = ["symbol", "locus_group", "locus_type", "status"]
_MAP_CACHE_NAME = "hgnc_alias_map.tsv.gz"
_BIOTYPE_CACHE_NAME = "hgnc_biotype_map.tsv.gz"
_MAP_COLS = ["variant", "approved", "kind"]
_BIOTYPE_OUT_COLS = ["symbol", "is_protein_coding", "is_lncrna", "is_mirna"]


# --------------------------------------------------------------------------- #
# Fetch + build
# --------------------------------------------------------------------------- #
def _fetch_hgnc_table(url: str = HGNC_COMPLETE_SET_URL,
                      usecols=None) -> Optional[pd.DataFrame]:
    """Download the HGNC complete set (selected columns only)."""
    try:
        df = pd.read_csv(url, sep="\t", usecols=usecols or _NEEDED_COLS,
                         dtype=str, low_memory=False)
    except Exception as e:
        warnings.warn(f"hgnc_alias: HGNC download failed ({type(e).__name__}: "
                      f"{e})", RuntimeWarning)
        return None
    return df


def _split_pipe(val) -> list[str]:
    if not isinstance(val, str) or not val:
        return []
    return [v.strip() for v in val.split("|") if v.strip()]


def _build_map_from_table(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the HGNC table into a compact variant→approved map DataFrame."""
    df = df[df["status"].astype(str).str.lower() == "approved"].copy()
    approved = set(df["symbol"].astype(str))

    rows: list[tuple[str, str, str]] = []
    # 1) approved self-maps (always win)
    for s in approved:
        rows.append((s, s, "approved"))

    # 2) alias / prev variants — collect candidates, drop ambiguous ones
    candidates: Dict[str, set] = {}
    for sym, aliases, prevs in zip(df["symbol"], df["alias_symbol"],
                                   df["prev_symbol"]):
        for v in _split_pipe(aliases) + _split_pipe(prevs):
            if v in approved:
                continue  # an approved symbol keeps its own identity
            candidates.setdefault(v, set()).add(str(sym))

    for v, targets in candidates.items():
        if len(targets) == 1:
            rows.append((v, next(iter(targets)), "alias"))
        # else: ambiguous → skip (identity fallback at normalize time)

    out = pd.DataFrame(rows, columns=_MAP_COLS)
    out = out.drop_duplicates("variant", keep="first").reset_index(drop=True)
    return out


def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, _MAP_CACHE_NAME)


def build_alias_map(
    cache_dir: str = "data/omnipath",
    download_if_missing: bool = False,
    force: bool = False,
) -> Dict[str, str]:
    """Return a `{symbol_variant: approved_symbol}` dict.

    Reads the compact cache if present; otherwise (and if allowed) fetches
    the HGNC complete set, builds the map, and writes the cache. Returns an
    empty dict when unavailable → callers then fall back to identity, which
    is safe (no normalization, same as before).

    Args:
        cache_dir: where the compact `hgnc_alias_map.tsv.gz` lives.
        download_if_missing: allow the HGNC web fetch when cache is absent.
        force: rebuild even if the cache exists.
    """
    path = _cache_path(cache_dir)
    if not force and os.path.exists(path):
        try:
            m = pd.read_csv(path, sep="\t", compression="infer", dtype=str)
            if set(_MAP_COLS).issubset(m.columns):
                print(f"  [hgnc] alias map cache ({len(m)} variants)")
                return dict(zip(m["variant"], m["approved"]))
        except Exception as e:
            warnings.warn(f"hgnc_alias: unreadable cache {path}: {e}",
                          RuntimeWarning)

    if not download_if_missing:
        print(f"  [hgnc] alias map absent, download disabled → identity map")
        return {}

    print("  [hgnc] downloading HGNC complete set…")
    df = _fetch_hgnc_table()
    if df is None or df.empty:
        return {}
    m = _build_map_from_table(df)
    os.makedirs(cache_dir, exist_ok=True)
    m.to_csv(path, sep="\t", index=False, compression="gzip")
    n_alias = int((m["kind"] == "alias").sum())
    print(f"  [hgnc] alias map built: {len(m)} variants "
          f"({n_alias} alias/prev → approved) → {path}")
    return dict(zip(m["variant"], m["approved"]))


# --------------------------------------------------------------------------- #
# Apply
# --------------------------------------------------------------------------- #
def normalize(symbols: Iterable[str], alias_map: Dict[str, str],
              upper_fallback: bool = True) -> Dict[str, str]:
    """Map each input symbol to its approved form (identity if unknown).

    `upper_fallback`: if an exact match fails, retry on the uppercased
    symbol (HGNC symbols are upper-case; guards against minor casing).
    """
    if not alias_map:
        return {s: s for s in symbols}
    upper_index = None
    if upper_fallback:
        # Lazily build an uppercase view of the map for case-robust lookup.
        upper_index = {k.upper(): v for k, v in alias_map.items()}
    out = {}
    for s in symbols:
        if s in alias_map:
            out[s] = alias_map[s]
        elif upper_index is not None and s.upper() in upper_index:
            out[s] = upper_index[s.upper()]
        else:
            out[s] = s
    return out


def canonicalize_series(series: pd.Series,
                        alias_map: Dict[str, str]) -> pd.Series:
    """Vectorized approved-symbol mapping of a pandas Series (identity if
    unknown)."""
    if not alias_map:
        return series.astype(str)
    return series.astype(str).map(lambda s: alias_map.get(s, s))


def coverage_report(symbols: Iterable[str],
                    alias_map: Dict[str, str]) -> dict:
    """Small diagnostic: how many symbols are approved / remapped / unknown."""
    syms = list(symbols)
    norm = normalize(syms, alias_map)
    approved_set = set(alias_map.values())
    n_remapped = sum(1 for s in syms if norm[s] != s)
    n_already = sum(1 for s in syms if s in approved_set)
    return {
        "n_input": len(syms),
        "n_already_approved": n_already,
        "n_remapped": n_remapped,
        "n_map_variants": len(alias_map),
    }


# --------------------------------------------------------------------------- #
# Molecular-class (biotype) map — protein-coding / lncRNA / miRNA
# --------------------------------------------------------------------------- #
def build_biotype_map(
    cache_dir: str = "data/omnipath",
    download_if_missing: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame [symbol, is_protein_coding, is_lncrna, is_mirna]
    from the HGNC `locus_group` / `locus_type`.

    Same offline-first cache pattern as `build_alias_map`. Returns an empty
    DataFrame when unavailable → callers fall back to all-zero flags (safe).
    Keyed by the HGNC **approved** symbol (canonicalize your symbols with the
    alias map before joining).
    """
    path = os.path.join(cache_dir, _BIOTYPE_CACHE_NAME)
    if not force and os.path.exists(path):
        try:
            m = pd.read_csv(path, sep="\t", compression="infer")
            if set(_BIOTYPE_OUT_COLS).issubset(m.columns):
                print(f"  [hgnc] biotype map cache ({len(m)} symbols)")
                return m
        except Exception as e:
            warnings.warn(f"hgnc_alias: unreadable biotype cache {path}: {e}",
                          RuntimeWarning)

    if not download_if_missing:
        print("  [hgnc] biotype map absent, download disabled → empty (zeros)")
        return pd.DataFrame(columns=_BIOTYPE_OUT_COLS)

    print("  [hgnc] downloading HGNC complete set (biotype)…")
    df = _fetch_hgnc_table(usecols=_BIOTYPE_COLS)
    if df is None or df.empty:
        return pd.DataFrame(columns=_BIOTYPE_OUT_COLS)
    df = df[df["status"].astype(str).str.lower() == "approved"].copy()
    lg = df["locus_group"].astype(str).str.lower()
    lt = df["locus_type"].astype(str).str.lower()
    out = pd.DataFrame({
        "symbol": df["symbol"].astype(str),
        "is_protein_coding": (lg == "protein-coding gene").astype(int),
        "is_lncrna": (lt == "rna, long non-coding").astype(int),
        "is_mirna": (lt == "rna, micro").astype(int),
    }).drop_duplicates("symbol").reset_index(drop=True)
    os.makedirs(cache_dir, exist_ok=True)
    out.to_csv(path, sep="\t", index=False, compression="gzip")
    print(f"  [hgnc] biotype map built: {len(out)} symbols "
          f"(protein-coding {int(out.is_protein_coding.sum())}, "
          f"lncRNA {int(out.is_lncrna.sum())}, "
          f"miRNA {int(out.is_mirna.sum())}) → {path}")
    return out
