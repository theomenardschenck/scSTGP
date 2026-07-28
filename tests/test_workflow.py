"""Workflow-level tests: the Snakemake DAG must stay wired.

A dry-run is cheap and catches the failure mode that actually bites here — a
rule whose script path, flag or config key drifted, so the DAG silently stops
producing a deliverable. It does NOT prove the science runs; see
``config.smoke.yaml`` and the `slow` marker for that.

Requires snakemake on PATH (conda env `gnn`); skipped otherwise so that a
package-only checkout still gets a green suite.
"""

from __future__ import annotations

import subprocess

import pytest
import yaml

CONFIGS = ["config.yaml", "config.smoke.yaml"]

# Rules that must appear in a smoke dry-run. These are the deliverables: the
# ranking, the "is the GNN better than a trivial baseline" check, the
# functional read-out, and the summary.
EXPECTED_SMOKE_RULES = {
    "build_graph",
    "train_vgae",
    "perturb",
    "aggregate_cross_seed",
    "driver_baselines",
    "interpret_embedding",
    "ora_top_drivers",
    "report",
}


def dry_run(snakemake_exe, repo_root, configfile):
    return subprocess.run(
        [
            snakemake_exe,
            "-n",
            "--quiet",
            "-s",
            "workflow/Snakefile",
            "--configfile",
            f"workflow/config/{configfile}",
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=600,
    )


@pytest.mark.parametrize("configfile", CONFIGS)
def test_dag_resolves(snakemake_exe, repo_root, configfile):
    proc = dry_run(snakemake_exe, repo_root, configfile)
    assert proc.returncode == 0, (
        f"dry-run KO sur {configfile}\n--- stderr (fin) ---\n{proc.stderr[-2500:]}"
    )


def test_smoke_dag_covers_every_deliverable(snakemake_exe, repo_root):
    proc = dry_run(snakemake_exe, repo_root, "config.smoke.yaml")
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = proc.stdout + proc.stderr
    missing = {r for r in EXPECTED_SMOKE_RULES if r not in out}
    assert not missing, (
        f"règles absentes du DAG smoke : {sorted(missing)}. Une étape de "
        f"validation a été débranchée du `rule all`."
    )


@pytest.mark.parametrize("configfile", CONFIGS)
def test_config_declares_required_sections(repo_root, configfile):
    cfg = yaml.safe_load((repo_root / "workflow" / "config" / configfile).read_text())
    for section in ("paths", "compute", "models", "perturbation", "validation"):
        assert section in cfg, f"{configfile} : section '{section}' manquante"
    for key in ("data_root", "out_base", "humess_dir", "scenic_dir"):
        assert key in cfg["paths"], f"{configfile} : paths.{key} manquant"


def test_wired_validations_enter_the_dag(snakemake_exe, repo_root, tmp_path):
    """The five validation modules wired on 2026-07-27 must still resolve.

    They were dead code before — present in src/, called from nowhere. A rule
    whose script path or flags drift would silently drop them back out of the
    DAG, which is exactly how they went unnoticed the first time.
    """
    cfg = yaml.safe_load((repo_root / "workflow/config/config.smoke.yaml").read_text())
    cfg["validation"].update(
        {
            "qc": {"enabled": True, "blocking": False, "repro_a": "", "repro_b": ""},
            "purity_source": {"enabled": True, "targets": "OCRL SYNJ2", "n_random": 5},
            "head_to_head": {"enabled": True, "targets": ""},
            "readout_specificity": {"enabled": True},
            "ora_de_baseline": {"enabled": True},
        }
    )
    tmp_cfg = repo_root / "workflow" / "config" / "_pytest_valall.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    try:
        proc = dry_run(snakemake_exe, repo_root, tmp_cfg.name)
    finally:
        tmp_cfg.unlink(missing_ok=True)

    assert proc.returncode == 0, proc.stderr[-2500:]
    out = proc.stdout + proc.stderr
    expected = {
        "pipeline_qc",
        "purity_source_attribution",
        "head_to_head_baselines",
        "readout_specificity",
        "ora_de_baseline",
    }
    missing = {r for r in expected if r not in out}
    assert not missing, f"validations débranchées du DAG : {sorted(missing)}"


def test_purity_without_targets_is_rejected(snakemake_exe, repo_root):
    """`purity_source.enabled: true` with no targets must fail loudly at parse
    time rather than produce an empty table."""
    cfg = yaml.safe_load((repo_root / "workflow/config/config.smoke.yaml").read_text())
    cfg["validation"]["purity_source"] = {"enabled": True, "targets": ""}
    tmp_cfg = repo_root / "workflow" / "config" / "_pytest_notargets.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    try:
        proc = dry_run(snakemake_exe, repo_root, tmp_cfg.name)
    finally:
        tmp_cfg.unlink(missing_ok=True)
    assert proc.returncode != 0
    assert "targets" in (proc.stdout + proc.stderr)


def test_smoke_config_is_actually_reduced(repo_root):
    """Guards against config.smoke.yaml drifting into a full-size run — a smoke
    test that takes hours stops being run at all."""
    cfg = yaml.safe_load((repo_root / "workflow" / "config" / "config.smoke.yaml").read_text())
    assert len(cfg["models"]["vgae"]["seeds"]) == 1
    assert len(cfg["perturbation"]["modes"]) == 1
    assert cfg["validation"]["decoy"]["enabled"] is False
    assert "--n-epochs" in cfg["models"]["vgae"]["extra_flags"]
