#!/usr/bin/env python3
"""de_schema.py — schéma canonique des tables DE chargées pour V6.

Toutes les sources « readout » (scRNA pseudobulk, bulk RNA, protéomique)
produisent des `pd.DataFrame` au même schéma, consommé ensuite par les
axes V6 (cf. §14bis.8 du rapport et §2.4 de pipeline_design : Plan 2 =
contraste A-vs-B au readout).

Ce schéma reste compatible — par sa colonne `source` et son `condition_label`
libre — avec l'usage P4/P16 historique (« P16_vs_P4 ») comme avec des
contrastes externes (« mutant_vs_wt », « senescent_vs_proliferative »…).

Le loader ne *réinterprète* rien : il rebaptise les colonnes, vérifie
le signe et propage les NaN pour les champs absents (pvalue/padj/stat
selon la source).

Fournit aussi (2026-06-30) la sélection d'ancres pour l'axe DE-ancré, unifiée
sc/bulk et robuste à la méthode de DE : `read_de_auto` (sniff) + détection de
colonnes étendue (MAST `avg_log2FC`/`p_val`/`p_val_adj`, `gene` nu) +
`select_de_anchors` (modes percentile/threshold/topn, rang stat→repli log_fc,
plancher/plafond n_min/n_max). Cf. design_log §20, results §16.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constantes de schéma
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: tuple[str, ...] = (
    "gene_symbol",      # HGNC officiel ; "" si non-résolu (warning loader)
    "gene_id_native",   # ENSG / UniProt / autre — identifiant brut source
    "log_fc",           # signed log2(A/B) ; NaN si absent
    "pvalue",           # p-value brute ; NaN si absent
    "padj",             # BH-FDR ; NaN si absent (laissé au caller)
    "stat",             # statistique test ; NaN si absent
    "condition_label",  # ex. "P16_vs_P4", "mutant_vs_wt"
    "source",           # "scrna_pseudobulk" | "bulk_rna" | "proteomics"
)

SOURCES = ("scrna_pseudobulk", "bulk_rna", "proteomics")

# Convention : <A>_vs_<B>  ⇒  log_fc > 0 ⇔ up dans A vs B
_CONDITION_RE = re.compile(r"^[A-Za-z0-9.+\-]+_vs_[A-Za-z0-9.+\-]+$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def validate_condition_label(label: str) -> tuple[str, str]:
    """Vérifie le format `<A>_vs_<B>` et retourne (A, B).

    Le signe attendu du `log_fc` est positif pour les gènes up dans A.
    """
    if not isinstance(label, str) or not _CONDITION_RE.match(label):
        raise ValueError(
            f"condition_label invalide : {label!r}. "
            f"Format attendu : `<cond_A>_vs_<cond_B>` (ex. 'P16_vs_P4', "
            f"'mutant_vs_wt')."
        )
    a, b = label.split("_vs_", 1)
    return a, b


def empty_de_table() -> pd.DataFrame:
    """Retourne un DataFrame vide au schéma V6."""
    return pd.DataFrame({col: pd.Series(dtype=_DTYPES[col]) for col in REQUIRED_COLUMNS})


_DTYPES = {
    "gene_symbol": "string",
    "gene_id_native": "string",
    "log_fc": "float64",
    "pvalue": "float64",
    "padj": "float64",
    "stat": "float64",
    "condition_label": "string",
    "source": "string",
}


def normalize_de_frame(
    df: pd.DataFrame,
    *,
    condition_label: str,
    source: str,
    drop_na_symbol: bool = False,
    drop_na_logfc: bool = True,
) -> pd.DataFrame:
    """Force un DataFrame brut au schéma V6.

    - Vérifie la présence des colonnes requises (NaN pour les manquantes).
    - Vérifie le format `condition_label` et la valeur de `source`.
    - Convertit les dtypes.
    - Optionnellement drop les lignes sans symbole ou sans logFC.
    - Trie par |log_fc| descendant pour stabilité.
    """
    if source not in SOURCES:
        raise ValueError(f"source={source!r} invalide. Choix : {SOURCES}.")
    validate_condition_label(condition_label)

    out = df.copy()
    out["condition_label"] = condition_label
    out["source"] = source

    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[list(REQUIRED_COLUMNS)]
    out = out.astype(_DTYPES, errors="ignore")

    if drop_na_logfc:
        out = out[out["log_fc"].notna()]
    if drop_na_symbol:
        out = out[out["gene_symbol"].fillna("").astype(str).str.len() > 0]

    out = out.sort_values(
        "log_fc", key=lambda s: s.abs(), ascending=False, na_position="last"
    ).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Heuristiques de détection (séparateur / décimal / colonnes)
# ---------------------------------------------------------------------------
def sniff_encoding(path) -> str:
    """Heuristique encoding utf-8 vs latin-1 (CSV français Windows).

    Lit jusqu'à 64 KB et tente utf-8 ; si décodage échoue, retombe sur
    latin-1 (jamais en erreur, surjection sur tous les octets).
    """
    with open(path, "rb") as fh:
        blob = fh.read(65536)
    try:
        blob.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def sniff_delimiter(path, candidates: tuple[str, ...] = ("\t", ";", ",")) -> str:
    """Détecte le séparateur par fréquence sur la 1ère ligne non vide."""
    enc = sniff_encoding(path)
    with open(path, "r", encoding=enc, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                counts = {sep: line.count(sep) for sep in candidates}
                best = max(counts, key=counts.get)
                if counts[best] == 0:
                    raise ValueError(f"{path} : aucun séparateur reconnu dans {candidates}.")
                return best
    raise ValueError(f"{path} : fichier vide")


def sniff_decimal(path, sep: str, n_lines: int = 50) -> str:
    """Détecte si les nombres utilisent '.' ou ',' comme décimal.

    Lit n_lines de données après l'en-tête, regarde si les champs
    « tokens entre séparateurs » contiennent davantage de ',' ou '.'
    précédés/suivis de chiffres.
    """
    enc = sniff_encoding(path)
    with open(path, "r", encoding=enc, errors="replace") as fh:
        next(fh, None)  # skip header
        sample = []
        for _ in range(n_lines):
            line = fh.readline()
            if not line:
                break
            sample.append(line)
    blob = "".join(sample)
    n_dot = len(re.findall(r"\d\.\d", blob))
    n_comma = len(re.findall(r"\d,\d", blob))
    # ',' comme décimal n'est plausible que si pas conflit avec sep=','
    if sep == "," and n_comma > 0 and n_dot == 0:
        return ","  # exotique, mais on signale
    return "," if n_comma > n_dot else "."


# ---------------------------------------------------------------------------
# Détection des colonnes utiles (regex par sémantique)
# ---------------------------------------------------------------------------
_COLUMN_PATTERNS = {
    "log_fc":   [r"^log2foldchange$", r"^log[\s_]?fc.*", r"^log[\s_]?ratio.*",
                 r"^logfc.*", r"^avg[\s_]?log2?[\s_]?fc$", r".*log2fc$"],
    "pvalue":   [r"^p[\s_]?value$", r"^pval$", r"^p[\s_.]?val$",
                 r"^bbinomial.*pvalue$", r"^binomial.*pvalue$"],
    "padj":     [r"^padj$", r"^adj.?p.?val.*", r"^p[\s_.]?val[\s_.]?adj$",
                 r"^fdr$", r"^qvalue$"],
    "stat":     [r"^stat$", r"^t[\s_]?stat.*", r"^z[\s_]?score$"],
    "symbol":   [r"^gene[\s_]?name$", r"^hgnc[\s_]?symbol$",
                 r"^gene[\s_]?symbol$", r"^symbol$", r"^gene$"],
    "ensembl":  [r"^gene[\s_]?id$", r"^ensembl[\s_]?gene[\s_]?id$",
                 r"^ensg.*"],
    "uniprot":  [r"^protein[\s_]?set$", r"^uniprot.*", r"^accession$",
                 r"^primary[\s_]?accession$"],
}


def detect_column(df: pd.DataFrame, kind: str) -> str | None:
    """Retourne la colonne de `df` correspondant à `kind` ou None.

    Match case-insensitive sur les patterns _COLUMN_PATTERNS[kind].
    Première occurrence gagne (ordre des patterns = priorité).
    """
    patterns = [re.compile(p, re.IGNORECASE) for p in _COLUMN_PATTERNS[kind]]
    for col in df.columns:
        norm = str(col).strip()
        for p in patterns:
            if p.match(norm):
                return col
    return None


# ---------------------------------------------------------------------------
# Raw read + DE-anchor selection (DE-anchored axis / DE role)
# ---------------------------------------------------------------------------
def read_de_auto(path) -> pd.DataFrame:
    """Read a DE table with sniffed encoding/delimiter/decimal (raw columns)."""
    enc = sniff_encoding(path)
    sep = sniff_delimiter(path)
    dec = sniff_decimal(path, sep)
    return pd.read_csv(path, sep=sep, encoding=enc, decimal=dec, engine="python")


@dataclass
class AnchorResult:
    """Up/down anchor gene lists for a DE-anchored axis + provenance.

    Convention: `up` = genes with log_fc > 0 (pro-senescence under <A>_vs_<B>
    when A is the senescent pole); `down` = log_fc < 0 (anti-senescence).
    """
    up: list[str]
    down: list[str]
    mode: str
    rank_used: str
    notes: list[str]

    def __len__(self) -> int:
        return len(self.up) + len(self.down)


def _select_pole(df, *, pole, gene_col, lfc_col, rank_col, padj_col, mode,
                 n_min, n_max, top_n, lfc_thresh, padj_thresh, pct, notes):
    """Pick anchors for one pole (sign of log_fc), ranked by |rank_col|."""
    sub = df[np.sign(df[lfc_col]) == pole]
    order = sub.reindex(sub[rank_col].abs().sort_values(ascending=False).index)
    label = "up" if pole > 0 else "down"
    if mode == "topn":
        picked = order.head(top_n)
    elif mode == "threshold":
        mem = order[order[lfc_col].abs() > lfc_thresh]
        if padj_col is not None:
            mem = mem[mem[padj_col] < padj_thresh]
        if len(mem) < n_min:
            notes.append(f"{label}: {len(mem)} < n_min({n_min}) au seuil "
                         f"→ repli top-{n_min} par {rank_col}")
            picked = order.head(n_min)
        else:
            picked = mem.head(n_max)
    elif mode == "percentile":
        k = int(np.ceil(len(order) * pct / 100.0))
        k = max(n_min, min(n_max, k))
        picked = order.head(k)
    else:
        raise ValueError(f"mode inconnu : {mode!r} (percentile|threshold|topn)")
    return picked[gene_col].astype(str).tolist()


def select_de_anchors(de, *, mode="percentile", rank="stat",
                      n_min=150, n_max=500, top_n=200,
                      lfc_thresh=0.5, padj_thresh=0.05, pct=10.0) -> AnchorResult:
    """Select up/down DE anchors for a DE-anchored axis — method-robust & unified.

    Works on sc (MAST/Wilcoxon) and bulk (DESeq2/limma) alike: columns are
    auto-detected via `detect_column` on the raw table (or a path), so the same
    logic serves every DE method. Three selection modes:

      - ``percentile`` (default) : top ``pct``% per pole by |rank| — relative to
        the method, so the absolute logFC scale (Seurat mean-of-logs vs DESeq2
        GLM coefficient) does not matter.
      - ``threshold`` : ``|log_fc| > lfc_thresh`` & ``padj < padj_thresh`` per
        pole (MAST-style fluid cutoff). Falls back to top-``n_min`` by rank if
        too few genes pass (keeps the axis stable).
      - ``topn`` : top ``top_n`` per pole by |rank| (legacy fixed count).

    Every mode clamps the per-pole count to ``[n_min, n_max]`` — the axis needs
    >= ~150 anchors to be stable (cf. §14bis.8.1, S1b). Ranking uses ``stat``
    (Wald = effect/SE, robust to low-count logFC inflation) when present, else
    falls back to ``log_fc`` with a note.

    Args:
        de : path to a DE table OR a DataFrame (raw columns; auto-detected).
    Returns:
        AnchorResult(up, down, mode, rank_used, notes).
    """
    notes: list[str] = []
    df = de if isinstance(de, pd.DataFrame) else read_de_auto(de)
    df = df.copy()

    gene_col = detect_column(df, "symbol") or detect_column(df, "ensembl")
    lfc_col = detect_column(df, "log_fc")
    if gene_col is None or lfc_col is None:
        raise ValueError("colonnes gène/logFC introuvables "
                         f"(colonnes={list(df.columns)})")
    padj_col = detect_column(df, "padj")
    stat_col = detect_column(df, "stat")

    df[lfc_col] = pd.to_numeric(df[lfc_col], errors="coerce")
    df[gene_col] = df[gene_col].astype(str)
    df = df[df[lfc_col].notna() & (df[gene_col].str.len() > 0)
            & (df[gene_col].str.lower() != "nan")]

    # Ranking column: prefer the requested `stat` when usable, else log_fc.
    if rank == "stat" and stat_col is not None:
        df[stat_col] = pd.to_numeric(df[stat_col], errors="coerce")
        if df[stat_col].notna().any():
            df[stat_col] = df[stat_col].fillna(0.0)
            rank_col = stat_col
        else:
            notes.append("rank=stat demandé mais colonne stat vide → repli log_fc")
            rank_col = lfc_col
    else:
        if rank == "stat":
            notes.append("rank=stat demandé mais colonne stat absente → repli log_fc")
        rank_col = lfc_col

    if padj_col is not None:
        df[padj_col] = pd.to_numeric(df[padj_col], errors="coerce")
    elif mode == "threshold":
        notes.append("padj absent → seuil sur |log_fc| seul")

    # Dedup genes, keeping the strongest by |rank|.
    df = df.reindex(df[rank_col].abs().sort_values(ascending=False).index)
    df = df.drop_duplicates(gene_col, keep="first")

    kw = dict(gene_col=gene_col, lfc_col=lfc_col, rank_col=rank_col,
              padj_col=padj_col, mode=mode, n_min=n_min, n_max=n_max,
              top_n=top_n, lfc_thresh=lfc_thresh, padj_thresh=padj_thresh,
              pct=pct, notes=notes)
    up = _select_pole(df, pole=+1, **kw)
    down = _select_pole(df, pole=-1, **kw)
    return AnchorResult(up=up, down=down, mode=mode,
                        rank_used=("stat" if rank_col == stat_col else "log_fc"),
                        notes=notes)
