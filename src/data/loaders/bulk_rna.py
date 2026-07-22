#!/usr/bin/env python3
"""bulk_rna.py — chargement V6 de tables DE bulk RNA-seq pré-calculées.

Préparation TIER 1f / TIER 4d (cf. TODO + §14bis.8 du rapport) : V6
adopte des DE déjà calculées en amont (DESeq2/limma/edgeR), pas de
matrice d'expression brute. Le contrat d'entrée est donc la *table*,
quel que soit le format source.

Deux formats sont attendus, plus une voie générique :

1. **GSE98440 HUVEC pro/sen (system-matched)** —
   `data/RNAseq/GSE98440_diff_expr_analysis_afterNorm_HUVEC_2reps.txt`
   Format DESeq2-like : `ensembl_gene_id, baseMean, log2FoldChange,
   lfcSE, stat, pvalue, padj, hgnc_symbol, description`. Séparateur
   tab, décimal point.

2. **FAM111B bulk patient/contrôle (cross-système, ENSG + GeneName)** —
   `data/RNAseq/FAM111B/Bulk_RNAseq_FAM111B.csv`. Format français :
   séparateur `;`, décimal `,`. Colonne logFC : `logFC (Nonmyo... / ...Control)`.
   Pas de `padj` exporté.

3. **Voie générique** — `load_bulk_rna_de(path, condition_label,
   sep="auto", decimal="auto", ...)` détecte les délimiteurs et les
   colonnes par heuristique (cf. de_schema.detect_column).

Sortie : `pd.DataFrame` au schéma `de_schema.REQUIRED_COLUMNS`.

Exemple :
    from data.loaders.bulk_rna import load_bulk_rna_de
    df = load_bulk_rna_de(
        "data/RNAseq/FAM111B/Bulk_RNAseq_FAM111B.csv",
        condition_label="patient_vs_control",
    )
    # df : gene_symbol, gene_id_native (ENSG), log_fc, pvalue, padj=NaN,
    #      stat, condition_label="patient_vs_control", source="bulk_rna"
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


__all__ = ["load_bulk_rna_de"]


def load_bulk_rna_de(
    path: str | Path,
    *,
    condition_label: str,
    sep: str = "auto",
    decimal: str = "auto",
    symbol_col: str | None = None,
    id_col: str | None = None,
    logfc_col: str | None = None,
    pvalue_col: str | None = None,
    padj_col: str | None = None,
    stat_col: str | None = None,
    flip_sign: bool = False,
) -> pd.DataFrame:
    """Charge une table DE bulk RNA-seq et la normalise au schéma V6.

    Parameters
    ----------
    path : str | Path
        Fichier TSV/CSV pré-DE (DESeq2/limma/edgeR exporté).
    condition_label : str
        Format `<A>_vs_<B>`. Convention : `log_fc > 0` ⇔ up dans A.
    sep, decimal : "auto" | str
        Détectés si "auto". Sinon forcés.
    symbol_col, id_col, logfc_col, pvalue_col, padj_col, stat_col : str | None
        Override de l'auto-détection (cf. de_schema._COLUMN_PATTERNS).
    flip_sign : bool
        Si l'export d'amont a inversé la convention `<A>/<B>`, mettre
        `True` pour multiplier `log_fc` par -1. (Utile pour aligner
        GSE98440 « pro vs sen » → « sen vs pro » selon le besoin.)

    Returns
    -------
    pd.DataFrame
        Au schéma `REQUIRED_COLUMNS` de `de_schema`, trié |log_fc| desc.
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
        warnings.warn(f"{path} : table DE vide.")
        return empty_de_table()

    # --- détection / override colonnes -------------------------------------
    sym = symbol_col or detect_column(df, "symbol")
    eid = id_col or detect_column(df, "ensembl") or detect_column(df, "uniprot")
    lfc = logfc_col or detect_column(df, "log_fc")
    pv  = pvalue_col or detect_column(df, "pvalue")
    pa  = padj_col or detect_column(df, "padj")
    st  = stat_col or detect_column(df, "stat")

    if lfc is None:
        raise ValueError(
            f"{path} : aucune colonne logFC reconnue parmi {list(df.columns)}. "
            f"Passez `logfc_col=` explicite."
        )
    if sym is None and eid is None:
        raise ValueError(
            f"{path} : ni symbole HGNC ni ID brut reconnu parmi "
            f"{list(df.columns)}. Passez `symbol_col=` ou `id_col=`."
        )

    out = pd.DataFrame(index=df.index)
    out["gene_symbol"] = (df[sym].astype("string").str.strip()
                          if sym is not None else pd.NA)
    out["gene_id_native"] = (df[eid].astype("string").str.strip()
                             if eid is not None else pd.NA)
    out["log_fc"] = pd.to_numeric(df[lfc], errors="coerce")
    out["pvalue"] = pd.to_numeric(df[pv], errors="coerce") if pv else pd.NA
    out["padj"]   = pd.to_numeric(df[pa], errors="coerce") if pa else pd.NA
    out["stat"]   = pd.to_numeric(df[st], errors="coerce") if st else pd.NA

    if flip_sign:
        out["log_fc"] = -out["log_fc"]

    n_in = len(out)
    out = out[out["log_fc"].notna()]
    if (dropped := n_in - len(out)):
        warnings.warn(f"{path} : {dropped} ligne(s) sans log_fc — drop.")

    # Dédup : si plusieurs lignes pour le même symbole, garder |log_fc| max.
    if out["gene_symbol"].notna().any():
        out = (out.assign(_abs=out["log_fc"].abs())
                  .sort_values("_abs", ascending=False)
                  .drop_duplicates("gene_symbol", keep="first")
                  .drop(columns="_abs"))

    return normalize_de_frame(
        out,
        condition_label=condition_label,
        source="bulk_rna",
        drop_na_logfc=True,
    )
