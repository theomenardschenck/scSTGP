"""Pluggable objectives for the automatic hyper-parameter search.

An objective answers one question: *given a finished pipeline run, how good was
it?* — as a single float to MAXIMISE, plus diagnostics.

WHY THIS IS A REGISTRY AND NOT ONE FUNCTION
-------------------------------------------
There is no consensus target here, and the project's own history says so:

  * reconstruction AUC does not decide drivers — the `mr5` config reached
    AUC 0.95 while placing 0 of 18 Tier-1 genes in the top-100, whereas `topk`
    reached only 0.87-0.89 with 7-8 of them;
  * recall of known drivers is the closest thing to the biological goal, but it
    is circular by construction and hostage to the reference list;
  * cross-seed stability targets the failure mode that actually blocks every
    ablation — see the noise-floor warning below.

So the search plumbing is fixed and the target is chosen per study.

THE NOISE FLOOR — READ BEFORE USING ANY OF THIS
----------------------------------------------
Two runs of an IDENTICAL config, same seed, gave Spearman rho 0.556-0.687 on
the gene ranking, with a median shift of 942 places. That is LARGER than most
treatment effects this pipeline is asked to measure.

Consequences, enforced by `search.py`:

  * a single-seed trial optimises noise. `--seeds 3` is the floor.
  * before trusting any study, run `search.py calibrate`: it repeats ONE config
    N times and reports the spread of the objective. A study whose best-to-worst
    range sits inside that spread has found nothing.
  * `compute.deterministic: true` shrinks the floor (threads=1, deterministic
    kernels, fixed PYTHONHASHSEED) at a real cost in speed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Résultat
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ObjectiveResult:
    """Score to maximise, plus everything needed to audit it afterwards."""

    value: float
    diagnostics: dict = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.value)


class ObjectiveError(RuntimeError):
    """Raised when a trial produced no usable output (crashed, timed out…)."""


# ─────────────────────────────────────────────────────────────────────────────
# Registre
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, callable] = {}


def register(name):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn

    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"objectif inconnu : {name!r} (disponibles : {available()})")
    return _REGISTRY[name]


# ─────────────────────────────────────────────────────────────────────────────
# Aides de lecture
# ─────────────────────────────────────────────────────────────────────────────


def _read_ranking(analysis_dir: Path) -> pd.DataFrame:
    path = Path(analysis_dir) / "cross_seed_gene_ranking.tsv"
    if not path.exists():
        raise ObjectiveError(f"ranking absent : {path}")
    df = pd.read_csv(path, sep="\t")
    if "driver_score" not in df.columns:
        raise ObjectiveError(f"colonne driver_score absente de {path}")
    return df


def _read_seed_metrics(config_dir: Path) -> list[dict]:
    metrics = []
    for p in sorted(Path(config_dir).glob("s*/vgae_metrics.json")):
        try:
            metrics.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    if not metrics:
        raise ObjectiveError(f"aucun vgae_metrics.json sous {config_dir}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Objectifs
# ─────────────────────────────────────────────────────────────────────────────


@register("cross_seed_stability")
def cross_seed_stability(config_dir: Path, analysis_dir: Path, top_n: int = 200, **_):
    """Cross-seed agreement of the ranking — the recommended default.

    Uses `direction_stability`, the per-gene fraction of seeds agreeing on the
    SIGN of the effect, averaged over the top-`top_n` drivers. It is produced by
    `perturb_report` in the cross-seed step, so it costs nothing extra.

    Known limit — it measures agreement on DIRECTION, not on RANK. A config
    where every seed agrees on the sign but shuffles the order scores well here.
    Rank agreement is what `pipeline_qc repro` measures, and it needs two runs
    of the same config, which is what `calibrate` produces. Read both.
    """
    df = _read_ranking(analysis_dir)
    col = "mean_stability" if "mean_stability" in df.columns else "direction_stability"
    if col not in df.columns:
        raise ObjectiveError(
            "ni mean_stability ni direction_stability dans le ranking : "
            "la config a-t-elle tourné avec >= 2 seeds ?"
        )
    top = df.nlargest(top_n, "driver_score")
    value = float(top[col].mean())
    return ObjectiveResult(
        value=value,
        diagnostics={
            "column": col,
            "top_n": top_n,
            "n_genes_total": int(len(df)),
            "stability_median": float(top[col].median()),
            "driver_score_max": float(df["driver_score"].max()),
        },
    )


@register("recon_auc")
def recon_auc(config_dir: Path, analysis_dir: Path, **_):
    """Mean link-prediction AUC across seeds — cheap, and NOT the biology.

    Kept because it is the standard VGAE target and a useful sanity signal (a
    collapsed encoder shows up here first). Do not select a final config on it:
    this project has a documented case of AUC and driver recall pointing in
    opposite directions.
    """
    metrics = _read_seed_metrics(config_dir)
    aucs = [m["best_auc"] for m in metrics if "best_auc" in m]
    if not aucs:
        raise ObjectiveError("aucun best_auc dans les vgae_metrics.json")
    deltas = [m.get("delta_auc_vgae_minus_mlp") for m in metrics]
    deltas = [d for d in deltas if d is not None]
    return ObjectiveResult(
        value=float(np.mean(aucs)),
        diagnostics={
            "n_seeds": len(aucs),
            "auc_std": float(np.std(aucs)),
            "auc_min": float(np.min(aucs)),
            "delta_vs_mlp_mean": float(np.mean(deltas)) if deltas else None,
        },
    )


@register("known_driver_recall")
def known_driver_recall(
    config_dir: Path,
    analysis_dir: Path,
    reference_genes: list[str] | None = None,
    reference_file: str | None = None,
    top_n: int = 100,
    **_,
):
    """Fraction of a reference gene list recovered in the top-`top_n`.

    Closest to the biological goal, and the most dangerous: the reference list
    is a choice, and optimising against it bakes that choice into the model. If
    the list overlaps the evidence already in the graph, this rewards
    circularity rather than discovery. Use it as a CHECK, and prefer a list
    held out from the graph's own sources.
    """
    if reference_genes is None:
        if not reference_file:
            raise ObjectiveError(
                "known_driver_recall exige reference_genes ou reference_file"
            )
        ref_path = Path(reference_file)
        if not ref_path.exists():
            raise ObjectiveError(f"liste de référence absente : {ref_path}")
        reference_genes = [
            ln.strip().split("\t")[0]
            for ln in ref_path.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]

    df = _read_ranking(analysis_dir)
    ref = {g.upper() for g in reference_genes}
    present = ref & {str(g).upper() for g in df["target"]}
    if not present:
        raise ObjectiveError(
            "aucun gène de référence présent dans le graphe — liste ou "
            "nomenclature incompatible (alias HGNC ?)"
        )
    top = {str(g).upper() for g in df.nlargest(top_n, "driver_score")["target"]}
    hits = present & top
    return ObjectiveResult(
        value=len(hits) / len(present),
        diagnostics={
            "n_reference": len(ref),
            "n_in_graph": len(present),
            "n_recovered": len(hits),
            "recovered": sorted(hits),
            "top_n": top_n,
        },
    )
