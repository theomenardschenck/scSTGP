#!/usr/bin/env python3
"""proteomics.py — chargement V6 de tables protéomiques différentielles.

Conçu pour les exports type FAM111B :
`data/protemique/FAM111B/proteomique_serie_B.csv`
Colonnes brutes (format français) :
    Protein Set;bbinomial PValue;log Ratio (mutant non traité/wt non traité)
    P00325;1,90E-07;-4,462455093
    Q99715;1,57E-06;0,967856364
    ...

Le loader :
1. Auto-détecte sep `;` et décimal `,` (cf. `de_schema.sniff_*`).
2. Mappe l'accession UniProt → symbole HGNC via
   `id_mapping.map_uniprot_to_hgnc` (offline-first : cache local +
   scan OmniPath, optionnel REST UniProt).
3. Normalise au schéma V6 (`de_schema.REQUIRED_COLUMNS`).

Caveat (cf. §14bis.8 du rapport) : ces données sont *cross-système*
(FAM111B = fibroblastes, ≠ HUVEC). Convention : on les charge mais
on les caveate via `condition_label` explicite et `source="proteomics"` ;
le scoring V6 multi-tier marquera les drivers comme « système-indépendants »
uniquement si concordants avec un axe matched (scrna_pseudobulk ou
bulk_rna HUVEC).

Convention de signe : log_fc > 0 ⇔ up dans la *première* condition
nommée par `condition_label = "<A>_vs_<B>"`. Pour FAM111B serie B,
la colonne brute est `log Ratio (mutant/wt)` ⇒ logfc>0 ⇔ up dans mutant
⇒ utiliser `condition_label="mutant_vs_wt"`.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .de_schema import (
    detect_column,
    empty_de_table,
    normalize_de_frame,
    sniff_decimal,
    sniff_delimiter,
    sniff_encoding,
)
from .id_mapping import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OMNIPATH_DIR,
    map_uniprot_to_hgnc,
)


__all__ = ["load_proteomics_de"]


def load_proteomics_de(
    path: str | Path,
    *,
    condition_label: str,
    sep: str = "auto",
    decimal: str = "auto",
    id_col: str | None = None,
    logfc_col: str | None = None,
    pvalue_col: str | None = None,
    padj_col: str | None = None,
    map_to_hgnc: bool = True,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    omnipath_dir: Path | str = DEFAULT_OMNIPATH_DIR,
    online_mapping: bool = False,
    drop_unmapped: bool = False,
    flip_sign: bool = False,
) -> pd.DataFrame:
    """Charge une table protéomique différentielle et la normalise au schéma V6.

    Parameters
    ----------
    path : str | Path
        Fichier CSV/TSV avec accessions UniProt + log ratio + pvalue.
    condition_label : str
        Format `<A>_vs_<B>` ; ex. `"mutant_vs_wt"` pour FAM111B serie B.
    sep, decimal : "auto" | str
        Détectés par défaut (cf. sniff_delimiter / sniff_decimal).
    id_col, logfc_col, pvalue_col, padj_col : str | None
        Override de l'auto-détection.
    map_to_hgnc : bool
        Si True (défaut), résout les UniProt en symboles HGNC. Sinon
        garde `gene_symbol` vide et place l'accession dans `gene_id_native`.
    cache_dir, omnipath_dir : path
        Cf. `id_mapping.map_uniprot_to_hgnc`.
    online_mapping : bool
        Active le fallback REST UniProt (off par défaut, cluster-safe).
    drop_unmapped : bool
        Si True, drop les lignes sans HGNC résolu (sinon `gene_symbol`
        reste `NA` mais la ligne est conservée — utile pour audit).
    flip_sign : bool
        Inverse le `log_fc` (utile si la convention amont est `<B>/<A>`).

    Returns
    -------
    pd.DataFrame
        Au schéma `REQUIRED_COLUMNS` de `de_schema`, `source="proteomics"`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if sep == "auto":
        sep = sniff_delimiter(path)
    if decimal == "auto":
        decimal = sniff_decimal(path, sep)
    encoding = sniff_encoding(path)

    df = pd.read_csv(path, sep=sep, decimal=decimal, encoding=encoding)
    if df.empty:
        warnings.warn(f"{path} : table protéomique vide.")
        return empty_de_table()

    eid = id_col or detect_column(df, "uniprot")
    lfc = logfc_col or detect_column(df, "log_fc")
    pv  = pvalue_col or detect_column(df, "pvalue")
    pa  = padj_col or detect_column(df, "padj")

    if eid is None:
        raise ValueError(
            f"{path} : aucune colonne UniProt reconnue parmi {list(df.columns)}. "
            f"Passez `id_col=` explicite."
        )
    if lfc is None:
        raise ValueError(
            f"{path} : aucune colonne logFC/log ratio reconnue parmi "
            f"{list(df.columns)}. Passez `logfc_col=` explicite."
        )

    out = pd.DataFrame(index=df.index)
    out["gene_id_native"] = df[eid].astype("string").str.strip()
    out["log_fc"] = pd.to_numeric(df[lfc], errors="coerce")
    out["pvalue"] = pd.to_numeric(df[pv], errors="coerce") if pv else pd.NA
    out["padj"]   = pd.to_numeric(df[pa], errors="coerce") if pa else pd.NA
    out["stat"]   = pd.NA

    if flip_sign:
        out["log_fc"] = -out["log_fc"]

    n_in = len(out)
    out = out[out["log_fc"].notna() & out["gene_id_native"].notna()]
    if (dropped := n_in - len(out)):
        warnings.warn(f"{path} : {dropped} ligne(s) sans log_fc ou ID — drop.")

    # --- mapping UniProt → HGNC -------------------------------------------
    if map_to_hgnc:
        mapping = map_uniprot_to_hgnc(
            out["gene_id_native"].tolist(),
            cache_dir=cache_dir,
            omnipath_dir=omnipath_dir,
            online=online_mapping,
        )
        out["gene_symbol"] = out["gene_id_native"].map(mapping).astype("string")
        n_mapped = out["gene_symbol"].notna().sum()
        warnings.warn(
            f"{path} : {n_mapped}/{len(out)} UniProt → HGNC résolus "
            f"({100*n_mapped/max(len(out),1):.1f}%)."
        )
        if drop_unmapped:
            out = out[out["gene_symbol"].notna()]
    else:
        out["gene_symbol"] = pd.NA

    # Dédup HGNC (isoformes UniProt many-to-one) : |log_fc| max gagne.
    if out["gene_symbol"].notna().any():
        keep_mask = out["gene_symbol"].notna()
        deduped = (out[keep_mask]
                   .assign(_abs=out.loc[keep_mask, "log_fc"].abs())
                   .sort_values("_abs", ascending=False)
                   .drop_duplicates("gene_symbol", keep="first")
                   .drop(columns="_abs"))
        out = pd.concat([deduped, out[~keep_mask]], ignore_index=True)

    return normalize_de_frame(
        out,
        condition_label=condition_label,
        source="proteomics",
        drop_na_logfc=True,
    )
