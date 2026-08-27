"""Unit tests for the DE schema layer (`stateshift.data.loaders.de_schema`).

Why this module and not another: `select_de_anchors` defines the poles of the
DE-anchored axis, and the axis is what the whole driver score is projected
onto. A silent change in anchor selection moves every ranking without failing
anything downstream. These tests pin the contract that the method-invariance
result rests on (cos 0.99 between sc-MAST and bulk-Wald axes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stateshift.data.loaders.de_schema import (
    detect_column,
    select_de_anchors,
    validate_condition_label,
)


def make_de(n=2000, seed=0, with_stat=True, with_padj=True):
    """Synthetic DE table in Seurat/MAST column naming."""
    rng = np.random.default_rng(seed)
    lfc = rng.normal(0, 1.5, n)
    df = pd.DataFrame(
        {
            "gene": [f"G{i:05d}" for i in range(n)],
            "avg_log2FC": lfc,
            "p_val": rng.uniform(0, 1, n),
        }
    )
    if with_padj:
        df["p_val_adj"] = df["p_val"].clip(upper=0.99)
    if with_stat:
        # Wald-like statistic, deliberately NOT collinear with logFC so that a
        # rank fallback to log_fc would produce a different selection.
        df["stat"] = lfc * rng.uniform(0.5, 2.0, n)
    return df


# ── schéma ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "label,expected",
    [("sen_vs_pro", ("sen", "pro")), ("P16_vs_P4", ("P16", "P4"))],
)
def test_validate_condition_label_ok(label, expected):
    assert validate_condition_label(label) == expected


@pytest.mark.parametrize("bad", ["sen-vs-pro", "senescent", "a_vs_b_vs_c", ""])
def test_validate_condition_label_rejects(bad):
    with pytest.raises(Exception):
        validate_condition_label(bad)


def test_detect_column_handles_seurat_naming():
    """Seurat/MAST names (`gene`, `avg_log2FC`, `p_val_adj`) must resolve, since
    that is the naming of the HUVEC reference DE table."""
    df = make_de(50)
    assert detect_column(df, "symbol") == "gene"
    assert detect_column(df, "log_fc") == "avg_log2FC"
    assert detect_column(df, "padj") == "p_val_adj"
    assert detect_column(df, "stat") == "stat"


def test_detect_column_returns_none_when_absent():
    assert detect_column(make_de(10), "uniprot") is None


# ── sélection d'ancres ───────────────────────────────────────────────

def test_anchors_are_split_by_sign():
    res = select_de_anchors(make_de(), mode="topn", top_n=100)
    assert not set(res.up) & set(res.down), "un gène ne peut pas ancrer les deux pôles"
    assert len(res.up) == len(res.down) == 100


def test_percentile_mode_clamps_to_n_min_n_max():
    """The axis needs >= ~150 anchors per pole to be stable (S1b)."""
    res = select_de_anchors(make_de(n=4000), mode="percentile", pct=1.0, n_min=150, n_max=500)
    assert len(res.up) >= 150 and len(res.down) >= 150
    res_wide = select_de_anchors(make_de(n=40000), mode="percentile", pct=50.0, n_max=500)
    assert len(res_wide.up) <= 500 and len(res_wide.down) <= 500


def test_rank_falls_back_to_logfc_and_says_so():
    """Absent `stat`, ranking must degrade to log_fc *and* leave a trace —
    silent degradation would hide a change of axis definition."""
    res = select_de_anchors(make_de(with_stat=False), mode="topn", top_n=50)
    assert res.rank_used != "stat"
    assert res.notes, "le repli de rang doit être consigné dans notes"


def test_stat_and_logfc_ranking_select_different_anchors():
    """Guards the fixture itself: if the two rankings agreed, the fallback test
    above would prove nothing."""
    by_stat = select_de_anchors(make_de(), mode="topn", top_n=100, rank="stat")
    by_lfc = select_de_anchors(make_de(with_stat=False), mode="topn", top_n=100)
    assert set(by_stat.up) != set(by_lfc.up)


def test_threshold_mode_falls_back_when_too_few_pass():
    """MAST-style cutoffs can starve a pole; the axis must not collapse."""
    df = make_de(n=1000)
    df["avg_log2FC"] = df["avg_log2FC"] * 0.01  # nothing clears |lfc| > 0.5
    res = select_de_anchors(df, mode="threshold", lfc_thresh=0.5, n_min=150)
    assert len(res.up) >= 150 and len(res.down) >= 150
    assert any("repli" in n for n in res.notes)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        select_de_anchors(make_de(100), mode="whatever")


@pytest.mark.xfail(
    reason="Défaut connu (relevé 2026-07-27) : le clamp [n_min, n_max] porte sur "
           "le nombre DEMANDÉ, pas sur le nombre LIVRÉ. Sur une table plus petite "
           "que n_min, head(k) renvoie silencieusement moins d'ancres que le "
           "plancher de stabilité, sans note. À corriger dans de_schema, pas ici.",
    strict=True,
)
def test_small_table_still_honours_n_min_floor():
    res = select_de_anchors(make_de(n=120), mode="percentile", n_min=150)
    assert len(res.up) >= 150 or any("n_min" in n for n in res.notes)
