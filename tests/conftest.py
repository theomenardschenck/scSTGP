"""Shared fixtures for the stateshift test-suite.

Design note — why so much subprocess: several modules of this code base do real
work at import time (``gnn_vgae.py`` parses the CLI at module level;
``gnn_classification.py`` reads CSVs at line 134). Importing them in-process
would hang or crash pytest for reasons that have nothing to do with the test.
Anything script-shaped is therefore exercised through a subprocess, and only
genuinely import-safe modules are imported directly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def python_exe() -> str:
    """Interpreter used for subprocess checks.

    Prefers the project's local venv when present (it holds torch + PyG), so a
    bare `pytest` invocation still exercises the real stack.
    """
    local = REPO_ROOT / ".venv-local" / "bin" / "python"
    return str(local) if local.exists() else sys.executable


@pytest.fixture(scope="session")
def snakemake_exe() -> str:
    exe = shutil.which("snakemake")
    if exe is None:
        pytest.skip("snakemake absent du PATH (env conda 'gnn' non activé)")
    return exe


def run_script(python_exe: str, script: Path, *args: str, timeout: int = 180):
    """Run a repo script from the repo root and return the CompletedProcess."""
    env = dict(os.environ)
    # Never let a test reach the network: several modules lazily download
    # OmniPath / Reactome caches, which would make the suite slow and flaky.
    env["GNN_ALLOW_DOWNLOADS"] = "0"
    env.setdefault("MPLBACKEND", "Agg")
    return subprocess.run(
        [python_exe, str(script), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
        env=env,
    )


# ``cli.py`` is the console entry point: it is invoked as ``stateshift`` or
# ``python -m stateshift.cli``, never as a loose file, and it uses package-
# relative imports accordingly. The script-mode contract below does not apply to
# it — its equivalent guarantee is ``test_package_layout.py::test_cli_answers``,
# which exercises it the way users actually reach it.
PACKAGE_MODE_ONLY = {"cli.py"}


def cli_scripts() -> list[Path]:
    """Every src/ module that looks like a command-line entry point.

    Discovered rather than hard-coded, so a newly added tool is covered by the
    CLI contract test the day it lands.
    """
    found = []
    for path in sorted(REPO_ROOT.glob("src/**/*.py")):
        if path.name == "__init__.py" or "coexpr_benchmark" in path.parts:
            continue
        if path.name in PACKAGE_MODE_ONLY and path.parent == REPO_ROOT / "src":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "argparse" in text and "__main__" in text:
            found.append(path)
    return found
