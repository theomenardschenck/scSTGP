"""
Schéma TSV unifié pour la comparaison de méthodes de priorisation de gènes.

Toute méthode (VGAE perturbation, GNN_Lite supervisé, MLP, DeepWalk, stat,
scTenifoldKnk, CellOracle, DREAMwalk, decoupler-py ULM, …) doit exporter
ses résultats dans un TSV conforme pour être agrégée par
`cross_method_report.py` et comparée aux autres.

Le schéma vise 3 propriétés :
  1. **Sortabilité** : score (float, higher = more important) + rank.
  2. **Comparabilité** : `score_norm` ∈ [0, 1] min-max normalisé dans le
     groupe (method, version, seed) pour Spearman / precision@K.
  3. **Identifiabilité** : (method, version, seed, mode) doit être unique
     par gène. Les méthodes déterministes laissent seed=NaN ; les
     méthodes externes laissent version=NaN.

Workflow attendu :
  runners/<method>.py  →  output/method_comparison/<method>_<tag>.tsv
  cross_method_report.py joint tous les TSV via concat + pivot, calcule
    Spearman / precision@K / venn / matrice méthode×méthode.

Cf. §18 et §19 du rapport vgae_report.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Colonnes
# ─────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS: tuple[str, ...] = (
    "gene_symbol",   # str, HGNC symbol (canonique). Doit matcher gene_symbols
                     # de gnn_vgae.py (10 504 gènes V3.6 par défaut).
    "method",        # str, clé de METHOD_REGISTRY.
    "score",         # float, sortable (higher = more important).
)

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "rank",          # int ≥ 1, rank by score descending dans (method,
                     # version, seed, mode). Recalculé si manquant.
    "version",       # str, ex. "V3.6", "V4". NaN pour non-VGAE.
    "seed",          # int, NaN pour méthodes déterministes.
    "mode",          # str ∈ MODE_VOCABULARY.
    "score_signed",  # float, effet signé (perturbation only) ; NaN sinon.
    "score_norm",    # float ∈ [0, 1], min-max dans (method, version, seed,
                     # mode). Recalculé si absent.
    "is_significant",# bool, seuil method-specific (FDR / robustness / …).
    "metadata_json", # str, dump JSON d'extras (cosine, n_aging_dbs, …).
)

ALL_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


# ─────────────────────────────────────────────────────────────────────
# Vocabulaire mode
# ─────────────────────────────────────────────────────────────────────

MODE_VOCABULARY: tuple[str, ...] = (
    "KO",            # Knock-out (perturbation)
    "KD",            # Knock-down (perturbation, ×0.5)
    "OE",            # Over-expression (perturbation, ×2)
    "centrality",    # Score topologique sans perturbation
    "DE",            # Différentiel d'expression
    "TF_activity",   # Activité TF inférée (decoupler-py)
    "supervised",    # Score d'un classifieur supervisé (GNN_Lite, MLP)
    "consensus",     # Agrégat multi-mode (driver_score)
)


# ─────────────────────────────────────────────────────────────────────
# Registre des méthodes
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MethodSpec:
    """Métadonnées d'une méthode pour grouping et figures."""
    name: str
    family: str            # group pour les figures (color/legend)
    graph_based: bool
    uses_perturbation: bool
    is_internal: bool      # True si exporté par notre pipeline
    description: str


METHOD_REGISTRY: dict[str, MethodSpec] = {
    # ── Méthodes internes (notre pipeline) ──────────────────────────
    "vgae_perturb_driver": MethodSpec(
        name="vgae_perturb_driver",
        family="graph_dynamic",
        graph_based=True, uses_perturbation=True, is_internal=True,
        description="driver_score V3.6 — composite multi-source post-perturb.",
    ),
    "vgae_perturb_discovery": MethodSpec(
        name="vgae_perturb_discovery",
        family="graph_dynamic",
        graph_based=True, uses_perturbation=True, is_internal=True,
        description="discovery_score V3.6 — bonus non-DE (graph-only).",
    ),
    "vgae_perturb_validation": MethodSpec(
        name="vgae_perturb_validation",
        family="graph_dynamic",
        graph_based=True, uses_perturbation=True, is_internal=True,
        description="validation_score V3.6 — driver + DE-sig + aging DBs.",
    ),
    "vgae_importance": MethodSpec(
        name="vgae_importance",
        family="graph_static",
        graph_based=True, uses_perturbation=False, is_internal=True,
        description="vgae_importance — centralité latente baseline (§8.6).",
    ),
    "gnn_lite_supervised": MethodSpec(
        name="gnn_lite_supervised",
        family="graph_supervised",
        graph_based=True, uses_perturbation=False, is_internal=True,
        description="HeteroGNN multi-label classifier (gnn_classification.py).",
    ),
    "mlp_baseline": MethodSpec(
        name="mlp_baseline",
        family="feature_only",
        graph_based=False, uses_perturbation=False, is_internal=True,
        description="MLP same features no graph (gnn_vgae.py §12).",
    ),
    "deepwalk": MethodSpec(
        name="deepwalk",
        family="graph_static",
        graph_based=True, uses_perturbation=False, is_internal=True,
        description="DeepWalk random walks (gnn_vgae.py §13bis).",
    ),
    "stat_de": MethodSpec(
        name="stat_de",
        family="expression",
        graph_based=False, uses_perturbation=False, is_internal=True,
        description="|ΔExpr P16-P4| ranking (gnn_vgae.py §13).",
    ),
    # ── Méthodes externes (à intégrer via runners) ──────────────────
    "sctenifoldknk": MethodSpec(
        name="sctenifoldknk",
        family="expression_manifold",
        graph_based=False, uses_perturbation=True, is_internal=False,
        description="Manifold-aligned virtual KO (Osorio 2022).",
    ),
    "celloracle": MethodSpec(
        name="celloracle",
        family="graph_dynamic",
        graph_based=True, uses_perturbation=True, is_internal=False,
        description="GRN-propagated TF perturbation (Kamimoto 2023).",
    ),
    "dreamwalk": MethodSpec(
        name="dreamwalk",
        family="graph_static",
        graph_based=True, uses_perturbation=False, is_internal=False,
        description="Biased random walk on heterogeneous graph (Bang 2023).",
    ),
    "decoupler_ulm": MethodSpec(
        name="decoupler_ulm",
        family="tf_activity",
        graph_based=False, uses_perturbation=False, is_internal=False,
        description="ULM TF activity from DE (Badia-i-Mompel 2022).",
    ),
}


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────

class SchemaError(ValueError):
    """Schéma TSV non conforme."""


def validate(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Vérifie qu'un DataFrame respecte le schéma. Renvoie le df normalisé.

    - Vérifie présence des colonnes requises.
    - Vérifie unicité de (method, version, seed, mode, gene_symbol).
    - Vérifie que `method` appartient au registre.
    - Recalcule `rank` si manquant.
    - Recalcule `score_norm` si manquant (min-max par groupe).
    - Si strict=False, log les warnings au lieu de raise.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise SchemaError(f"Colonnes requises manquantes : {sorted(missing)}")

    bad_methods = set(df["method"].unique()) - set(METHOD_REGISTRY)
    if bad_methods:
        msg = f"Méthodes inconnues du registre : {sorted(bad_methods)}"
        if strict:
            raise SchemaError(msg)

    out = df.copy()
    for col in OPTIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    group_cols = ["method", "version", "seed", "mode"]
    if "rank" not in df.columns or df["rank"].isna().any():
        out["rank"] = (
            out.groupby(group_cols, dropna=False)["score"]
               .rank(method="min", ascending=False)
               .astype("Int64")
        )

    if "score_norm" not in df.columns or df["score_norm"].isna().any():
        def _minmax(s: pd.Series) -> pd.Series:
            lo, hi = s.min(), s.max()
            if hi == lo:
                return pd.Series(0.5, index=s.index)
            return (s - lo) / (hi - lo)
        out["score_norm"] = (
            out.groupby(group_cols, dropna=False)["score"]
               .transform(_minmax)
        )

    dup = out.duplicated(subset=group_cols + ["gene_symbol"])
    if dup.any():
        msg = f"{dup.sum()} duplicats sur (method, version, seed, mode, gene_symbol)"
        if strict:
            raise SchemaError(msg)

    return out[list(ALL_COLUMNS)]


def write_tsv(df: pd.DataFrame, path, *, validate_first: bool = True) -> None:
    """Écrit un TSV conforme. Valide d'abord par défaut."""
    if validate_first:
        df = validate(df, strict=True)
    df.to_csv(path, sep="\t", index=False, na_rep="")


def read_tsv(path) -> pd.DataFrame:
    """Charge un TSV et valide."""
    df = pd.read_csv(path, sep="\t")
    return validate(df, strict=True)


def concat_methods(paths: Iterable) -> pd.DataFrame:
    """Concatène plusieurs TSV méthode en un long-format unique."""
    frames = [read_tsv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)
