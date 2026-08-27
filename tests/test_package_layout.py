"""Package-mode guarantees.

These tests exist because the layout was broken in two distinct ways before
2026-07-27, and both failures were silent:

  1. ``__init__.py`` was listed in ``.gitignore`` — any package file would have
     been silently untracked, reproducing the ``/data/`` incident that left six
     source files off the cluster.
  2. ``bulk_rna.py`` and ``proteomics.py`` use relative imports
     (``from .de_schema import …``) with no ``__init__.py`` anywhere, so they
     could not be imported at all — by any means — despite being documented as
     tested.

If either regresses, the suite fails here rather than three weeks later on a
cluster wave.
"""

from __future__ import annotations

import importlib
import subprocess

import pytest

SUBPACKAGES = [
    "stateshift",
    "stateshift.gnn",
    "stateshift.data",
    "stateshift.data.extract",
    "stateshift.data.loaders",
    "stateshift.data.preprocess",
    "stateshift.perturbation",
    "stateshift.validation",
    "stateshift.validation.cluster",
    "stateshift.validation.explain",
    "stateshift.validation.ora",
    "stateshift.validation.probe",
    "stateshift.validation.qc",
    "stateshift.validation.reports",
    "stateshift.validation.schema",
    "stateshift.validation.viz",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    assert importlib.import_module(name) is not None


def test_version_exposed():
    import stateshift

    assert stateshift.__version__.count(".") == 2


@pytest.mark.parametrize("module", ["bulk_rna", "proteomics"])
def test_v6_loaders_are_importable(module):
    """Regression: relative imports used to make these two unimportable."""
    mod = importlib.import_module(f"stateshift.data.loaders.{module}")
    entry = [n for n in dir(mod) if n.startswith("load_")]
    assert entry, f"{module} expose aucune fonction load_*"


def test_workflow_ships_with_the_package():
    """The pipeline must travel with the library.

    Regression guard for the packaging gap: a wheel containing only ``src/``
    installs a library, not a tool — ``stateshift run`` would have nothing to
    execute, and the failure would only appear on a user's machine.
    """
    import stateshift

    assert stateshift.snakefile().is_file()
    configs = list(stateshift.config_dir().glob("*.yaml"))
    assert configs, "aucune config embarquée"
    assert (stateshift.profile_dir() / "submit.sh").is_file()


def test_every_script_the_snakefile_calls_is_installed():
    """Each ``{SRC}/…`` in the Snakefile must exist in the installed package.

    A dry-run does NOT catch a missing script — Snakemake never inspects the
    shell command it would run — so a packaging gap surfaces only mid-DAG, after
    the graph build. Deriving the list from the Snakefile rather than hard-coding
    it means a newly wired rule is covered the day it lands.

    This is the check that would have caught ``validation/figures/`` shipping
    without an ``__init__.py`` and without an entry in ``packages``.

    Scope, deliberately: it reads the Snakefile ONLY, not the optional
    ``rules/*.local.smk`` includes. Those name scripts that are local by
    construction and absent from the wheel, so asserting on them would fail by
    design. The trade-off is that a local rule is not covered here — which is
    exactly why a rule belongs in the Snakefile as soon as its script ships.
    """
    import re

    import stateshift

    rels = sorted(set(re.findall(r"\{SRC\}/([A-Za-z0-9_/]+\.py)",
                                 stateshift.snakefile().read_text(encoding="utf-8"))))
    assert len(rels) >= 10, f"seulement {len(rels)} scripts détectés — regex à revoir"

    missing = [r for r in rels if not (stateshift.package_dir() / r).is_file()]
    assert not missing, f"scripts appelés par le Snakefile mais absents : {missing}"


def test_script_paths_resolve_without_a_clone():
    """``script()`` answers absolute paths, whatever the working directory."""
    import stateshift

    path = stateshift.script("gnn/gnn_vgae.py")
    assert path.is_absolute() and path.is_file()

    # The historical `src/…` spelling used across the SLURM scripts must keep
    # resolving, otherwise migrating them becomes a flag day.
    assert stateshift.script("src/gnn/gnn_vgae.py") == path


@pytest.mark.parametrize("argv", [["--version"], ["--help"], ["path", "snakefile"]])
def test_cli_answers(python_exe, argv, tmp_path):
    """The console entry point works from a neutral cwd — the whole point."""
    proc = subprocess.run(
        [python_exe, "-m", "stateshift.cli", *argv],
        capture_output=True, text=True, cwd=tmp_path, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-1000:]
    assert proc.stdout.strip()


def test_no_generic_root_packages(python_exe, tmp_path):
    """`gnn`, `validation`, `perturbation` must NOT be installed at root level.

    An auto-discovered layout (``packages.find`` + ``package-dir``) installs
    them as top-level names — a global ``import data`` that would shadow
    unrelated code. Run from a neutral cwd so the repo's own ``data/``
    directory cannot answer as an implicit namespace package.
    """
    for name in ("gnn", "validation", "perturbation"):
        proc = subprocess.run(
            [python_exe, "-c", f"import {name}"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert proc.returncode != 0, (
            f"`import {name}` réussit au niveau racine — collision de noms "
            f"réintroduite dans pyproject.toml"
        )
