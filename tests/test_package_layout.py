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
    "stgp",
    "stgp.gnn",
    "stgp.data",
    "stgp.data.extract",
    "stgp.data.loaders",
    "stgp.data.preprocess",
    "stgp.perturbation",
    "stgp.validation",
    "stgp.validation.cluster",
    "stgp.validation.explain",
    "stgp.validation.ora",
    "stgp.validation.probe",
    "stgp.validation.qc",
    "stgp.validation.reports",
    "stgp.validation.schema",
    "stgp.validation.viz",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    assert importlib.import_module(name) is not None


def test_version_exposed():
    import stgp

    assert stgp.__version__.count(".") == 2


@pytest.mark.parametrize("module", ["bulk_rna", "proteomics"])
def test_v6_loaders_are_importable(module):
    """Regression: relative imports used to make these two unimportable."""
    mod = importlib.import_module(f"stgp.data.loaders.{module}")
    entry = [n for n in dir(mod) if n.startswith("load_")]
    assert entry, f"{module} expose aucune fonction load_*"


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
