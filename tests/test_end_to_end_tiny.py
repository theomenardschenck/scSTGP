"""End-to-end run of the whole pipeline on a generated tiny dataset.

This is the only test that EXECUTES the science path: graph build, VGAE
training, in-silico perturbation, cross-seed aggregation, QC, ORA, report. Every
other test checks structure. Marked `slow` — a cold run is a couple of minutes —
so it is opt-in locally and a separate job in CI:

    pytest tests/test_end_to_end_tiny.py -m slow

It found four real defects the day it was written (2026-07-28), each of which
had been silently waiting for a non-HUVEC dataset:

  * `perturb_top_genes` ignored the checkpoint's persisted `latent_dim`, so any
    run trained with a non-default value was impossible to perturb — including
    every hyper-parameter trial that tunes it;
  * `gnn_perturbation.CELL_GROUPS` was frozen to the five HUVEC groups while the
    graph builder already read `GNN_CELL_GROUPS`: the two halves of the pipeline
    had drifted, so any other dataset built a correct graph then died at readout;
  * the Snakefile injected the dataset description only under `build.enabled`,
    coupling "describe my data" to "rebuild my features";
  * `--no-baselines` crashed a downstream figure, taking the whole aggregation
    rule down with it — including the ranking, which was never written.

None of them was reachable without actually running the thing.
"""

from __future__ import annotations

import shutil
import subprocess

import pandas as pd
import pytest

pytestmark = pytest.mark.slow

CONFIG = "workflow/config/config.tiny.yaml"


@pytest.fixture(scope="module")
def tiny_run(repo_root, python_exe, tmp_path_factory):
    """Generate the fixture, run the pipeline, yield the analysis directory."""
    if shutil.which("snakemake") is None:
        pytest.skip("snakemake absent du PATH")

    data_dir = repo_root / "data_tiny"
    out_dir = repo_root / "output" / "tiny"
    shutil.rmtree(out_dir, ignore_errors=True)

    gen = subprocess.run(
        [python_exe, "tests/fixtures/make_tiny_dataset.py", "--out", str(data_dir)],
        capture_output=True, text=True, cwd=repo_root, timeout=300,
    )
    assert gen.returncode == 0, gen.stderr[-2000:]

    # Pin the interpreter the rules will use. Without this the Snakefile falls
    # back on `compute.python: "python"`, i.e. whatever the ambient PATH points
    # at — routinely another project's venv, with no torch. The suite must test
    # the pipeline, not the developer's shell.
    env = {
        "GNN_ALLOW_DOWNLOADS": "0",
        "MPLBACKEND": "Agg",
        "STATESHIFT_PYTHON": python_exe,
        "STATESHIFT_PYTHON_TORCH": python_exe,
    }
    import os
    full_env = {**os.environ, **env}
    proc = subprocess.run(
        ["bash", "workflow/run.sh", "--backend", "local", "--cores", "4",
         "--configfile", CONFIG],
        capture_output=True, text=True, cwd=repo_root, timeout=2400, env=full_env,
    )
    assert proc.returncode == 0, (
        "le pipeline tiny a échoué\n--- fin de sortie ---\n"
        + (proc.stdout + proc.stderr)[-4000:]
    )
    return out_dir / "tiny" / "analysis"


def test_pipeline_completes(tiny_run):
    assert tiny_run.is_dir()


def test_ranking_has_the_expected_shape(tiny_run):
    """The deliverable is a per-gene table carrying the scoring columns."""
    df = pd.read_csv(tiny_run / "cross_seed_gene_ranking.tsv", sep="\t")
    assert len(df) > 0
    for col in ("target", "driver_score", "mean_stability", "evidence_tier"):
        assert col in df.columns, f"colonne {col} absente du ranking"
    assert df["driver_score"].notna().all()
    # trié par driver_score décroissant
    assert df["driver_score"].is_monotonic_decreasing


def test_perturbed_genes_come_back_on_top(tiny_run):
    """Weak sanity check, and deliberately labelled as such.

    The fixture injects an effect on a known handful of genes AND restricts the
    perturbation to them, so recovering them proves only that signal survives
    the machinery end to end. It is not evidence that the method works.
    """
    df = pd.read_csv(tiny_run / "cross_seed_gene_ranking.tsv", sep="\t")
    top = set(df.nlargest(4, "driver_score")["target"])
    injected = {"SERPINE1", "CDKN1A", "IL6", "CDKN2A", "TOP2A", "MKI67", "HMGB2"}
    assert top & injected, f"aucun gène perturbé dans le top-4 : {top}"


def test_validation_outputs_exist(tiny_run):
    """The wired validations must produce their files, not just resolve in the DAG."""
    for rel in (
        "interpret/driver_baselines.tsv",
        "interpret/communities.tsv",
        "interpret/ora/top10_reactome.tsv",
        "interpret/ora/top10_aging.tsv",
        "qc/qc_report.md",
        "report/SUMMARY.md",
    ):
        assert (tiny_run / rel).exists(), f"livrable manquant : {rel}"


def test_qc_runs_its_five_checks(tiny_run):
    report = (tiny_run / "qc" / "qc_report.md").read_text()
    for check in ("repro", "duplicates", "overlap", "degree", "axis"):
        assert check in report, f"contrôle QC '{check}' absent du rapport"


def test_qc_duplicates_check_actually_inspects_the_edges(tiny_run):
    """`duplicates` must report a per-edge-type table, not just a verdict.

    Deliberately NOT asserting a ×2 factor. A first run appeared to reproduce
    the known "PPI stored twice" defect, but that came from a STALE graph cache
    — on a clean build the fixture shows dup_factor 1.0. The multiplicity is a
    property of the real STRING file plus the builder, not something this
    fixture can be relied on to reproduce. What IS stable, and worth pinning,
    is that the check runs and inspects every edge type.
    """
    report = (tiny_run / "qc" / "qc_report.md").read_text()
    assert "duplicates" in report
    for column in ("edge_type", "n_edges", "unique_ordered", "dup_factor"):
        assert column in report, f"colonne '{column}' absente du tableau duplicates"
    assert "ppi" in report and "same_pathway" in report
