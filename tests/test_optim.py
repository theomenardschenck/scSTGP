"""Tests for the automatic hyper-parameter search skeleton.

They cover the plumbing and the guard-rails, not the search quality: a real
study needs a cluster and hours. What must not break silently is (a) config
derivation, (b) the refusal to search on too few seeds, (c) the objective
registry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "optim"))

import objectives as obj  # noqa: E402
import search  # noqa: E402


# ── registre d'objectifs ─────────────────────────────────────────────

def test_registry_exposes_the_three_objectives():
    assert set(obj.available()) == {
        "cross_seed_stability",
        "recon_auc",
        "known_driver_recall",
    }


def test_unknown_objective_is_rejected():
    with pytest.raises(KeyError):
        obj.get("maximise_le_hasard")


def test_missing_ranking_raises_objective_error(tmp_path):
    """A crashed trial must surface as ObjectiveError (→ pruned), never as a
    silent zero that the study would treat as a real measurement."""
    with pytest.raises(obj.ObjectiveError):
        obj.cross_seed_stability(config_dir=tmp_path, analysis_dir=tmp_path)


def test_cross_seed_stability_reads_the_ranking(tmp_path):
    # driver_score décroissant : les 5 PREMIÈRES lignes sont les meilleurs
    # drivers, et ce sont elles qui doivent peser dans le score.
    pd.DataFrame(
        {
            "target": [f"G{i}" for i in range(10)],
            "driver_score": list(range(10, 0, -1)),
            "mean_stability": [0.9] * 5 + [0.1] * 5,
        }
    ).to_csv(tmp_path / "cross_seed_gene_ranking.tsv", sep="\t", index=False)
    res = obj.cross_seed_stability(config_dir=tmp_path, analysis_dir=tmp_path, top_n=5)
    assert float(res) == pytest.approx(0.9)
    assert res.diagnostics["top_n"] == 5

    # et le top_n compte réellement : élargir jusqu'aux drivers faibles dilue
    whole = obj.cross_seed_stability(config_dir=tmp_path, analysis_dir=tmp_path, top_n=10)
    assert float(whole) == pytest.approx(0.5)


def test_known_driver_recall_flags_nomenclature_mismatch(tmp_path):
    """Zero overlap means a broken gene-name mapping, not a score of 0."""
    pd.DataFrame({"target": ["AAA", "BBB"], "driver_score": [1.0, 0.5]}).to_csv(
        tmp_path / "cross_seed_gene_ranking.tsv", sep="\t", index=False
    )
    with pytest.raises(obj.ObjectiveError):
        obj.known_driver_recall(
            config_dir=tmp_path, analysis_dir=tmp_path, reference_genes=["ZZZ"]
        )


# ── espace de recherche et dérivation de config ──────────────────────

def test_params_become_cli_flags():
    flags = search.params_to_flags({"kl_beta_max": 0.001, "latent_dim": 64})
    assert "--kl-beta-max 0.001" in flags
    assert "--latent-dim 64" in flags


def test_trial_config_isolates_run_tag_and_forces_qc(tmp_path, repo_root):
    cfg = search.write_trial_config(
        base_config=repo_root / "workflow" / "config" / "config.smoke.yaml",
        run_tag="study/trial_007",
        flags="--kl-beta-max 0.001",
        seeds=3,
        out_dir=tmp_path,
    )
    got = yaml.safe_load(cfg.read_text())
    # run_tag propre à l'essai : deux essais ne doivent jamais écrire au même endroit
    assert got["models"]["vgae"]["run_tag"] == "study/trial_007"
    assert got["models"]["vgae"]["seeds"] == [1, 2, 3]
    # les drapeaux de l'essai s'AJOUTENT, ils n'écrasent pas ceux de la base
    assert "--kl-beta-max 0.001" in got["models"]["vgae"]["extra_flags"]
    assert "--n-epochs" in got["models"]["vgae"]["extra_flags"]
    # le QC est forcé : sans lui la comparaison entre essais n'est pas lisible
    assert got["validation"]["qc"]["enabled"] is True


# ── garde-fous ───────────────────────────────────────────────────────

def test_search_refuses_fewer_than_three_seeds(monkeypatch, capsys, repo_root):
    """The measured noise floor (rho 0.556-0.687 between identical runs) makes
    a 1-seed search meaningless. It must refuse rather than produce a plausible
    but worthless study."""
    parser = search.build_parser()
    args = parser.parse_args(
        ["search", "--seeds", "1", "--base-config",
         str(repo_root / "workflow" / "config" / "config.smoke.yaml")]
    )
    pytest.importorskip("optuna")
    with pytest.raises(SystemExit) as exc:
        search.cmd_search(args)
    assert "seeds" in str(exc.value).lower() or "bruit" in str(exc.value).lower()


def test_default_seeds_is_three():
    args = search.build_parser().parse_args(["search"])
    assert args.seeds == 3
