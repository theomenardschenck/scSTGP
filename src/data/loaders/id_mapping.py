#!/usr/bin/env python3
"""id_mapping.py — résolution offline-first UniProt accession → HGNC symbol.

V6 charge des données protéomiques indexées par accession UniProt
(`P00325`, `Q9Y2J2`, …). Pour les joindre au graphe (indexé HGNC),
on a besoin d'un mapping.

Stratégie en 3 couches, dans l'ordre :

1. **Cache local** : un TSV `data/cache/uniprot_to_hgnc.tsv`
   (2 colonnes : `uniprot, hgnc`). Construit incrémentalement par les
   appels successifs.
2. **OmniPath identifiers** : si le client `omnipath` est installé,
   on peut récupérer le mapping via `omnipath.interactions.AllInteractions`
   (Türei et al. 2021 *Mol Syst Biol*). Pour V6 on évite la dépendance
   à un fetch online — on consomme le cache existant
   (`data/omnipath/tf_collectri.tsv.gz`, `signed_ppi_signor.tsv.gz`).
3. **UniProt REST API** (fallback, désactivé par défaut) — `requests`
   GET sur `rest.uniprot.org/idmapping`. Activable via `online=True`.

Pas de dépendance dure : si `omnipath` n'est pas dispo, on saute la
couche 2 ; si `requests` n'est pas dispo, on saute la couche 3.

Référence : Türei et al. 2021 OmniPath identifier translation ;
UniProt Consortium 2025 *Nucleic Acids Res* (REST API).
"""
from __future__ import annotations

import gzip
import os
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd


CACHE_FILENAME = "uniprot_to_hgnc.tsv"
DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_OMNIPATH_DIR = Path("data/omnipath")


# ---------------------------------------------------------------------------
# Cache local
# ---------------------------------------------------------------------------
def _load_cache(cache_dir: Path) -> dict[str, str]:
    f = cache_dir / CACHE_FILENAME
    if not f.exists():
        return {}
    try:
        df = pd.read_csv(f, sep="\t", dtype=str)
    except Exception as e:
        warnings.warn(f"Cache UniProt illisible ({f}) : {e}. Recommencement à zéro.")
        return {}
    if not {"uniprot", "hgnc"}.issubset(df.columns):
        warnings.warn(f"Cache UniProt mal formé ({f}) : colonnes attendues uniprot,hgnc.")
        return {}
    return dict(zip(df["uniprot"].astype(str), df["hgnc"].astype(str)))


def _save_cache(mapping: dict[str, str], cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"uniprot": list(mapping.keys()),
                       "hgnc": list(mapping.values())})
    df = df.sort_values("uniprot")
    df.to_csv(cache_dir / CACHE_FILENAME, sep="\t", index=False)


# ---------------------------------------------------------------------------
# Couche 2 : OmniPath caches existants (offline)
# ---------------------------------------------------------------------------
def _scan_omnipath_caches(omnipath_dir: Path) -> dict[str, str]:
    """Extrait toutes les paires uniprot↔hgnc visibles dans les caches.

    Les fichiers OmniPath stockent simultanément les colonnes
    `source`/`target` (HGNC) ET les colonnes `*_uniprot` quand
    disponibles. On scanne sans présupposer une colonne précise.
    """
    if not omnipath_dir.exists():
        return {}
    mapping: dict[str, str] = {}
    for f in sorted(omnipath_dir.glob("*.tsv.gz")):
        try:
            df = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)
        except Exception as e:
            warnings.warn(f"Lecture {f} : {e}")
            continue
        # Colonnes plausibles HGNC
        hgnc_cols = [c for c in df.columns
                     if c.lower() in ("source_genesymbol", "target_genesymbol",
                                      "source", "target", "genesymbol",
                                      "hgnc_symbol")]
        # Colonnes plausibles UniProt
        uni_cols = [c for c in df.columns
                    if "uniprot" in c.lower() or c.lower() in ("source_id", "target_id")]
        # Aligne paires (uni_col, hgnc_col) par préfixe source_/target_
        pairs = []
        for u in uni_cols:
            prefix = u.lower().replace("uniprot", "").replace("_id", "").rstrip("_")
            for h in hgnc_cols:
                if h.lower().startswith(prefix) or prefix == "":
                    pairs.append((u, h))
                    break
        for u, h in pairs:
            sub = df[[u, h]].dropna().astype(str)
            mapping.update(dict(zip(sub[u], sub[h])))
    return mapping


# ---------------------------------------------------------------------------
# Couche 3 : UniProt REST API (online)
# ---------------------------------------------------------------------------
def _fetch_uniprot_online(
    accessions: Iterable[str],
    batch_size: int = 50,
    timeout: int = 30,
) -> dict[str, str]:
    """Résout en ligne via rest.uniprot.org/uniprotkb/search.

    Batch par `batch_size` (URL limite ~8 KB ⇒ ~50 accessions sûres).
    Best-effort : si `requests` indispo ou échec réseau, retourne {}.

    UniProt-style query : `accession:P12345 OR accession:Q67890 ...`.
    Le format TSV renvoie `accession\tgene_primary` ; on prend le 1er
    symbole avant le `;` (gene_primary peut être multi pour les
    isoformes).
    """
    try:
        import requests
    except ImportError:
        warnings.warn("requests indispo : skip fallback UniProt REST.")
        return {}

    acc = sorted(set(accessions))
    if not acc:
        return {}

    out: dict[str, str] = {}
    n_batches = (len(acc) + batch_size - 1) // batch_size
    for i in range(n_batches):
        batch = acc[i * batch_size : (i + 1) * batch_size]
        joined = " OR ".join(f"accession:{a}" for a in batch)
        try:
            r = requests.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={
                    "query": joined,
                    "fields": "accession,gene_primary",
                    "format": "tsv",
                    "size": batch_size,
                },
                timeout=timeout,
            )
            r.raise_for_status()
        except Exception as e:
            warnings.warn(
                f"UniProt REST batch {i+1}/{n_batches} failed: {e}. "
                f"Poursuite avec les autres batches."
            )
            continue
        for line in r.text.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1]:
                out[parts[0]] = parts[1].split(";")[0].strip()
    return out


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------
def map_uniprot_to_hgnc(
    accessions: Iterable[str],
    *,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    omnipath_dir: Path | str = DEFAULT_OMNIPATH_DIR,
    online: bool = False,
    write_cache: bool = True,
) -> dict[str, str]:
    """Résout un set d'accessions UniProt en symboles HGNC.

    Parameters
    ----------
    accessions : iterable
        Liste/tuple/set d'accessions (ex. "P00325").
    cache_dir : path
        Cache TSV cumulatif (lu d'abord, mis à jour à la fin).
    omnipath_dir : path
        Dossier contenant les caches OmniPath gz (scannés une fois).
    online : bool
        Si True, fallback HTTP UniProt REST pour les non-résolus.
        Désactivé par défaut (compatible cluster offline).
    write_cache : bool
        Persistance du cache mis à jour.

    Returns
    -------
    dict
        {accession: hgnc_symbol}. Les accessions non résolues sont
        absentes du dict — au caller de gérer (ex. drop ou marquer).
    """
    cache_dir = Path(cache_dir)
    omnipath_dir = Path(omnipath_dir)
    acc_set = {a for a in (str(x).strip() for x in accessions) if a}

    mapping = _load_cache(cache_dir)
    missing = acc_set - set(mapping)

    if missing:
        omni = _scan_omnipath_caches(omnipath_dir)
        new_hits = {a: omni[a] for a in missing if a in omni}
        if new_hits:
            mapping.update(new_hits)
            missing -= set(new_hits)

    if missing and online:
        fetched = _fetch_uniprot_online(missing)
        mapping.update(fetched)
        missing -= set(fetched)

    if write_cache:
        # On ne sauvegarde QUE les entrées correspondant à `acc_set`,
        # plus le cache préexistant (pour ne pas tout perdre si on
        # restreint).
        merged = _load_cache(cache_dir)
        merged.update({a: mapping[a] for a in acc_set if a in mapping})
        _save_cache(merged, cache_dir)

    if missing:
        n = len(missing)
        sample = sorted(missing)[:5]
        warnings.warn(
            f"{n} accession(s) UniProt non résolue(s) (échantillon : {sample}). "
            f"Activez online=True pour fetch REST ou enrichissez le cache."
        )

    return {a: mapping[a] for a in acc_set if a in mapping}
