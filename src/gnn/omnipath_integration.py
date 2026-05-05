"""
OmniPath integration — V4 (signed/directed edges + curated TFs).
================================================================
Rewrite complete pour omnipath-py >= 1.0 (l'ancienne API
`OmnipathInteractions.get(...)` n'existe plus).

3 sources de données prises en charge, chacune cachée sur disque
sous forme TSV pour pouvoir tourner sur les compute nodes du cluster
Nautilus (qui n'ont pas accès Internet) :

    1. SIGNALING DIRIGÉ (kinase-substrat + causal signaling) :
       — agrégat OmniPath `omnipath` filtré sur consensus_direction=True.
       — sign = is_stimulation − is_inhibition ∈ {−1, 0, +1}.
       — score = curation_effort normalisé.
       — cache : data/omnipath/signaling_omnipath.tsv.gz

    2. TF→target curé (CollecTRI, fallback DoRothEA) :
       — CollecTRI = méta-ressource 2023 (Müller-Dott et al., Genome Biol)
         qui agrège DoRothEA + ChEA3 + ENCODE + 11 autres → ~1186 TFs vs
         ~50 SCENIC, chaque interaction porte un mode-of-regulation (mor).
       — sign = mor ∈ {−1, +1}.
       — cache : data/omnipath/tf_collectri.tsv.gz (ou tf_dorothea.tsv.gz)

    3. PPI causal SIGNÉ (SIGNOR 3.0 via OmniPath) :
       — sources='SIGNOR', interaction signée + dirigée curée
         (Lo Surdo et al., Nucleic Acids Res 2023).
       — sign = is_stimulation − is_inhibition.
       — cache : data/omnipath/signed_ppi_signor.tsv.gz

Toutes les fonctions retournent (edge_src, edge_dst, edge_attr,
metadata_df). edge_attr a la forme (n_edges, 2) : [score, sign] avec
score ∈ [0, 1] et sign ∈ {−1, 0, +1}. Si la source est indisponible,
elles renvoient des arrays vides — le caller doit gérer ce cas.

Usage typique côté pipeline :

    from omnipath_integration import (
        load_signaling_directed, load_collectri_tf_target,
        load_signed_ppi_signor,
    )
    src, dst, attr, _ = load_signaling_directed(
        cache_dir="data/omnipath",
        available_genes=available_set,
        gene_to_idx=gene_to_idx,
        download_if_missing=False,   # True uniquement sur le frontal
    )

Pré-télécharger une fois sur le frontal (qui a Internet) avec
`scripts/cache_omnipath.py`, puis les jobs SLURM lisent juste les TSV.

Références :
- Türei D. et al., "Integrated intra- and intercellular signaling
  knowledge for multicellular omics analysis", Nat Commun 2021.
- Müller-Dott S. et al., "Expanding the coverage of regulons from
  high-confidence prior knowledge for accurate estimation of
  transcription factor activities" — CollecTRI, Genome Biol 2023.
- Lo Surdo P. et al., "SIGNOR 3.0, the SIGnaling Network Open
  Resource 3.0", Nucleic Acids Res 2023.
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import omnipath as op
    OMNIPATH_AVAILABLE = True
except ImportError:
    OMNIPATH_AVAILABLE = False


# Schéma commun aux 3 caches (5 colonnes minimales)
_CACHE_COLS = ["source_symbol", "target_symbol", "score", "sign", "n_resources"]


# --------------------------------------------------------------------------- #
# Helpers — cache lecture/écriture, normalisation
# --------------------------------------------------------------------------- #
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _cache_path(cache_dir: str, name: str) -> str:
    return os.path.join(cache_dir, name)


def _read_cache(path: str) -> Optional[pd.DataFrame]:
    """Retourne le DataFrame caché ou None si absent/corrompu."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep="\t", compression="infer")
    except Exception as e:
        warnings.warn(f"OmniPath cache illisible ({path}): {e}", RuntimeWarning)
        return None
    if not all(c in df.columns for c in _CACHE_COLS):
        warnings.warn(f"OmniPath cache schéma invalide ({path})", RuntimeWarning)
        return None
    return df


def _write_cache(df: pd.DataFrame, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    df[_CACHE_COLS].to_csv(path, sep="\t", index=False, compression="gzip")


def _normalize_score(values: pd.Series) -> np.ndarray:
    """Min-max sur (0, 1] avec garde contre série vide / constante."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    vmax = float(np.nanmax(arr))
    if vmax <= 0 or not np.isfinite(vmax):
        return np.ones_like(arr, dtype=np.float32)
    return np.clip(arr / vmax, 0.0, 1.0)


def _stim_inhib_to_sign(is_stim: pd.Series, is_inhib: pd.Series) -> np.ndarray:
    """sign = stim − inhib  ∈ {−1, 0, +1}.

    OmniPath retourne 0/1 (parfois NaN si effet inconnu) — on coerce en 0
    pour les NaN. Si stim ET inhib sont à 1 (rare, conflits de curation),
    on laisse 0 (ambigu).
    """
    s = pd.to_numeric(is_stim, errors="coerce").fillna(0).astype(np.int8)
    i = pd.to_numeric(is_inhib, errors="coerce").fillna(0).astype(np.int8)
    sign = (s - i).astype(np.int8)
    sign = np.where((s == 1) & (i == 1), 0, sign)
    return sign.astype(np.float32)


def _project_to_graph(
    df: pd.DataFrame,
    available_genes: Iterable[str],
    gene_to_idx: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Projette les paires symbol→symbol sur les indices du graphe.

    Filtre :
      - gènes présents dans `available_genes` ET dans `gene_to_idx`
      - paires (src == dst) éliminées (self-loop pas pertinent ici)

    Retourne : edge_src, edge_dst, edge_attr=[score, sign], metadata_df
    Toutes vides si aucun lien ne survit au filtrage.
    """
    avail = set(available_genes)
    keep = (
        df["source_symbol"].isin(avail)
        & df["target_symbol"].isin(avail)
        & (df["source_symbol"] != df["target_symbol"])
    )
    sub = df.loc[keep].copy()
    if sub.empty:
        return (np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.zeros((0, 2), dtype=np.float32),
                sub)

    src_idx = sub["source_symbol"].map(gene_to_idx).to_numpy(dtype=np.int64)
    dst_idx = sub["target_symbol"].map(gene_to_idx).to_numpy(dtype=np.int64)
    score = _normalize_score(sub["score"])
    sign = sub["sign"].astype(np.float32).to_numpy()
    attr = np.stack([score, sign], axis=1).astype(np.float32)

    # Dédup : si un même (src, dst) existe plusieurs fois (resources
    # multiples), on agrège — score = max, sign = vote majoritaire (signe
    # du sum).
    pair_key = src_idx.astype(np.int64) * (max(int(dst_idx.max()) + 1, 1)) + dst_idx
    unique_keys, inv = np.unique(pair_key, return_inverse=True)
    if unique_keys.size != src_idx.size:
        agg_src = np.empty(unique_keys.size, dtype=np.int64)
        agg_dst = np.empty(unique_keys.size, dtype=np.int64)
        agg_attr = np.zeros((unique_keys.size, 2), dtype=np.float32)
        sign_sums = np.zeros(unique_keys.size, dtype=np.float32)
        for k, group_idx in enumerate(np.split(np.argsort(inv),
                                               np.cumsum(np.bincount(inv))[:-1])):
            agg_src[k] = src_idx[group_idx[0]]
            agg_dst[k] = dst_idx[group_idx[0]]
            agg_attr[k, 0] = score[group_idx].max()
            sign_sums[k] = sign[group_idx].sum()
        agg_attr[:, 1] = np.sign(sign_sums)  # ∈ {−1, 0, +1}
        return agg_src, agg_dst, agg_attr, sub

    return src_idx, dst_idx, attr, sub


# --------------------------------------------------------------------------- #
# Fetchers — chacun frappe l'API web, normalise au schéma _CACHE_COLS,
# et écrit le TSV. Renvoie None si omnipath indisponible ou échec.
# --------------------------------------------------------------------------- #
def _fetch_signaling_omnipath(organism: int = 9606) -> Optional[pd.DataFrame]:
    """Signaling dirigé OmniPath (kinase-substrat + causal)."""
    if not OMNIPATH_AVAILABLE:
        return None
    try:
        df = op.interactions.OmniPath.get(
            organism=organism,
            directed=True,
            genesymbols=True,
        )
    except Exception as e:
        warnings.warn(f"OmniPath fetch signaling KO : {e}", RuntimeWarning)
        return None
    if df is None or df.empty:
        return None

    # On garde uniquement les liens consensus-dirigés
    if "consensus_direction" in df.columns:
        df = df[df["consensus_direction"] == True]  # noqa: E712

    score_col = (
        "curation_effort" if "curation_effort" in df.columns
        else "n_references" if "n_references" in df.columns
        else None
    )
    if score_col is None:
        df = df.assign(_score=1.0)
        score_col = "_score"

    sign = _stim_inhib_to_sign(
        df.get("consensus_stimulation", df.get("is_stimulation", 0)),
        df.get("consensus_inhibition", df.get("is_inhibition", 0)),
    )
    n_res = (
        df["n_resources"].astype(int)
        if "n_resources" in df.columns
        else (df["n_references"].astype(int) if "n_references" in df.columns
              else pd.Series(np.ones(len(df), dtype=int)))
    )
    out = pd.DataFrame({
        "source_symbol": df["source_genesymbol"].astype(str),
        "target_symbol": df["target_genesymbol"].astype(str),
        "score": df[score_col].astype(float),
        "sign": sign,
        "n_resources": n_res.values,
    })
    return out


def _fetch_collectri(organism: int = 9606) -> Optional[pd.DataFrame]:
    """CollecTRI TF→target (méta-ressource curée 2023, ~1186 TFs)."""
    if not OMNIPATH_AVAILABLE:
        return None
    df = None
    # Endpoint dédié si dispo dans la version installée
    if hasattr(op.interactions, "CollecTRI"):
        try:
            df = op.interactions.CollecTRI.get(
                organism=organism,
                genesymbols=True,
            )
        except Exception as e:
            warnings.warn(f"CollecTRI fetch direct KO : {e}", RuntimeWarning)
    # Fallback : via AllInteractions filtré sur la ressource CollecTRI
    if df is None or df.empty:
        try:
            all_int = op.interactions.AllInteractions.get(
                organism=organism,
                genesymbols=True,
                resources=["CollecTRI"],
            )
            df = all_int
        except Exception as e:
            warnings.warn(f"CollecTRI fallback AllInteractions KO : {e}",
                          RuntimeWarning)
            return None
    if df is None or df.empty:
        return None

    sign = _stim_inhib_to_sign(
        df.get("is_stimulation", df.get("consensus_stimulation", 0)),
        df.get("is_inhibition",  df.get("consensus_inhibition",  0)),
    )
    n_res = (df["n_resources"].astype(int) if "n_resources" in df.columns
             else pd.Series(np.ones(len(df), dtype=int)))
    score_col = ("curation_effort" if "curation_effort" in df.columns
                 else "n_references" if "n_references" in df.columns else None)
    score = (df[score_col].astype(float)
             if score_col is not None else pd.Series(np.ones(len(df))))
    out = pd.DataFrame({
        "source_symbol": df["source_genesymbol"].astype(str),
        "target_symbol": df["target_genesymbol"].astype(str),
        "score": score.values,
        "sign": sign,
        "n_resources": n_res.values,
    })
    return out


def _fetch_dorothea(
    organism: int = 9606,
    levels: Iterable[str] = ("A", "B", "C"),
) -> Optional[pd.DataFrame]:
    """DoRothEA TF→target (legacy fallback si CollecTRI absent)."""
    if not OMNIPATH_AVAILABLE:
        return None
    try:
        df = op.interactions.Dorothea.get(
            organism=organism,
            dorothea_levels=list(levels),
            genesymbols=True,
        )
    except Exception as e:
        warnings.warn(f"DoRothEA fetch KO : {e}", RuntimeWarning)
        return None
    if df is None or df.empty:
        return None

    sign = _stim_inhib_to_sign(
        df.get("is_stimulation", 0),
        df.get("is_inhibition", 0),
    )
    # Confiance DoRothEA : level A=1.0, B=0.75, C=0.5, D=0.25, E=0
    level_map = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25, "E": 0.0}
    score = (df["dorothea_level"].astype(str).str[0].map(level_map).fillna(0.5)
             if "dorothea_level" in df.columns else pd.Series(np.ones(len(df))))
    n_res = (df["n_resources"].astype(int) if "n_resources" in df.columns
             else pd.Series(np.ones(len(df), dtype=int)))
    out = pd.DataFrame({
        "source_symbol": df["source_genesymbol"].astype(str),
        "target_symbol": df["target_genesymbol"].astype(str),
        "score": score.values,
        "sign": sign,
        "n_resources": n_res.values,
    })
    return out


def _fetch_signor_signed_ppi(organism: int = 9606) -> Optional[pd.DataFrame]:
    """PPI signé/dirigé curé SIGNOR 3.0."""
    if not OMNIPATH_AVAILABLE:
        return None
    try:
        df = op.interactions.AllInteractions.get(
            organism=organism,
            genesymbols=True,
            resources=["SIGNOR"],
            directed=True,
        )
    except Exception as e:
        warnings.warn(f"SIGNOR fetch KO : {e}", RuntimeWarning)
        return None
    if df is None or df.empty:
        return None

    sign = _stim_inhib_to_sign(
        df.get("consensus_stimulation", df.get("is_stimulation", 0)),
        df.get("consensus_inhibition",  df.get("is_inhibition",  0)),
    )
    score_col = ("curation_effort" if "curation_effort" in df.columns
                 else "n_references" if "n_references" in df.columns else None)
    score = (df[score_col].astype(float)
             if score_col is not None else pd.Series(np.ones(len(df))))
    n_res = (df["n_resources"].astype(int) if "n_resources" in df.columns
             else pd.Series(np.ones(len(df), dtype=int)))
    out = pd.DataFrame({
        "source_symbol": df["source_genesymbol"].astype(str),
        "target_symbol": df["target_genesymbol"].astype(str),
        "score": score.values,
        "sign": sign,
        "n_resources": n_res.values,
    })
    return out


# --------------------------------------------------------------------------- #
# API publique — appelée par gnn_vgae.py (et scripts/cache_omnipath.py)
# --------------------------------------------------------------------------- #
def _load_or_fetch(
    cache_dir: str,
    cache_name: str,
    fetcher,
    download_if_missing: bool,
    label: str,
) -> Optional[pd.DataFrame]:
    """Lit le cache ; si absent et autorisé, télécharge + écrit."""
    path = _cache_path(cache_dir, cache_name)
    cached = _read_cache(path)
    if cached is not None:
        print(f"  [omnipath] {label} : cache TSV ({len(cached)} liens)")
        return cached
    if not download_if_missing:
        print(f"  [omnipath] {label} : cache absent, download désactivé "
              f"→ skip ({path})")
        return None
    print(f"  [omnipath] {label} : téléchargement en cours…")
    fetched = fetcher()
    if fetched is None or fetched.empty:
        print(f"  [omnipath] {label} : aucune donnée récupérée")
        return None
    _write_cache(fetched, path)
    print(f"  [omnipath] {label} : {len(fetched)} liens téléchargés "
          f"→ cache écrit ({path})")
    return fetched


def load_signaling_directed(
    cache_dir: str,
    available_genes: Iterable[str],
    gene_to_idx: dict,
    download_if_missing: bool = False,
    organism: int = 9606,
):
    """Signaling dirigé OmniPath (kinase-substrat + causal).

    Returns:
        edge_src, edge_dst : (n,) np.int64 (indices du graphe)
        edge_attr          : (n, 2) np.float32 — colonnes [score, sign]
        metadata_df        : DataFrame des liens projetés
    """
    df = _load_or_fetch(
        cache_dir, "signaling_omnipath.tsv.gz",
        lambda: _fetch_signaling_omnipath(organism=organism),
        download_if_missing, "signaling",
    )
    if df is None:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                np.zeros((0, 2), dtype=np.float32), pd.DataFrame())
    return _project_to_graph(df, available_genes, gene_to_idx)


def load_collectri_tf_target(
    cache_dir: str,
    available_genes: Iterable[str],
    gene_to_idx: dict,
    download_if_missing: bool = False,
    organism: int = 9606,
    fallback_dorothea: bool = True,
):
    """TF→target curé : CollecTRI, fallback DoRothEA.

    Returns: idem load_signaling_directed.
    """
    df = _load_or_fetch(
        cache_dir, "tf_collectri.tsv.gz",
        lambda: _fetch_collectri(organism=organism),
        download_if_missing, "TF/CollecTRI",
    )
    if df is None and fallback_dorothea:
        df = _load_or_fetch(
            cache_dir, "tf_dorothea.tsv.gz",
            lambda: _fetch_dorothea(organism=organism),
            download_if_missing, "TF/DoRothEA(fallback)",
        )
    if df is None:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                np.zeros((0, 2), dtype=np.float32), pd.DataFrame())
    return _project_to_graph(df, available_genes, gene_to_idx)


def load_signed_ppi_signor(
    cache_dir: str,
    available_genes: Iterable[str],
    gene_to_idx: dict,
    download_if_missing: bool = False,
    organism: int = 9606,
):
    """PPI signé/dirigé curé SIGNOR 3.0.

    Returns: idem load_signaling_directed.
    """
    df = _load_or_fetch(
        cache_dir, "signed_ppi_signor.tsv.gz",
        lambda: _fetch_signor_signed_ppi(organism=organism),
        download_if_missing, "SIGNOR",
    )
    if df is None:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                np.zeros((0, 2), dtype=np.float32), pd.DataFrame())
    return _project_to_graph(df, available_genes, gene_to_idx)


# --------------------------------------------------------------------------- #
# Fusion utilitaire — combine signaling + SIGNOR en un seul edge_type
# (les deux portent la sémantique "lien causal signé dirigé").
# --------------------------------------------------------------------------- #
def merge_signed_directed(
    *triples,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatène plusieurs (src, dst, attr=[score,sign]) en un seul.

    Si la même paire (src, dst) apparaît plusieurs fois, on garde le score
    max et le signe par vote majoritaire (somme des signes). Utile pour
    fusionner kinase-substrat OmniPath + SIGNOR causal.
    """
    triples = [t for t in triples if len(t[0]) > 0]
    if not triples:
        return (np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.zeros((0, 2), dtype=np.float32))
    src = np.concatenate([t[0] for t in triples]).astype(np.int64)
    dst = np.concatenate([t[1] for t in triples]).astype(np.int64)
    attr = np.concatenate([t[2] for t in triples], axis=0).astype(np.float32)

    if src.size == 0:
        return src, dst, attr

    key = src * (int(dst.max()) + 1) + dst
    uniq, inv = np.unique(key, return_inverse=True)
    if uniq.size == src.size:
        return src, dst, attr

    out_src = np.empty(uniq.size, dtype=np.int64)
    out_dst = np.empty(uniq.size, dtype=np.int64)
    out_attr = np.zeros((uniq.size, 2), dtype=np.float32)
    sign_sum = np.zeros(uniq.size, dtype=np.float32)
    for k, group in enumerate(np.split(np.argsort(inv),
                                       np.cumsum(np.bincount(inv))[:-1])):
        out_src[k] = src[group[0]]
        out_dst[k] = dst[group[0]]
        out_attr[k, 0] = attr[group, 0].max()
        sign_sum[k] = attr[group, 1].sum()
    out_attr[:, 1] = np.sign(sign_sum)
    return out_src, out_dst, out_attr
