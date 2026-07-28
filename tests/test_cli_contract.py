"""CLI contract: every entry point must still answer `--help`.

This is the refactor safety-net. The historical net was ``tests/golden/``, a
bit-exact comparison against a frozen reference — but that reference depends on
``output/gnn_vgae/_graph_cache_scrna.pkl``, which no longer exists, so the
golden check cannot run. Until a cache is regenerated, this suite is what
stands between an import-level mistake and a wasted cluster wave.

`--help` is a deliberately shallow contract, but it is not a trivial one here:
it exercises module import, the flat ``sys.path`` bootstrap, the argparse
construction in ``_config.py``, and every module-level side effect. All nine
import failures found on 2026-07-27 would have been caught by it.
"""

from __future__ import annotations

import pytest

from conftest import cli_scripts, run_script

SCRIPTS = cli_scripts()


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_cli_help(python_exe, repo_root, script):
    proc = run_script(python_exe, script, "--help", timeout=300)
    rel = script.relative_to(repo_root)
    assert proc.returncode == 0, (
        f"{rel} --help sort en {proc.returncode}\n"
        f"--- stderr (fin) ---\n{proc.stderr[-1500:]}"
    )
    assert "usage" in (proc.stdout + proc.stderr).lower(), (
        f"{rel} --help ne produit pas de bloc usage"
    )


def test_discovery_is_not_empty():
    """Guards the discovery heuristic itself: an empty list would make the
    whole parametrised suite vacuously green."""
    assert len(SCRIPTS) >= 20, f"seulement {len(SCRIPTS)} CLI découvertes"
