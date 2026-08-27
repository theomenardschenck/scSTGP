"""Where do the scripts and the workflow actually live?

The package supports two installation layouts, and every path in the pipeline
has to resolve correctly under both:

    clone       <repo>/src/gnn/gnn_vgae.py        <repo>/workflow/Snakefile
    installed   <site-packages>/stateshift/gnn/…  <site-packages>/stateshift/workflow/…

Historically the Snakefile and the twenty SLURM scripts addressed their targets
as ``src/gnn/gnn_vgae.py`` — relative to the *current working directory*. That
works only when the CWD is a clone root, which is why the tool could not be
used from anywhere else, pip install or not. This module is the single place
that answers the question, so no caller has to guess.

The rule is deliberately dumb, because a clever rule fails silently: the
workflow directory sits either *inside* the package (installed) or *beside* it
(clone). We look for the Snakefile, we do not infer from ``__file__`` shapes or
from the presence of a ``.git``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "package_dir",
    "workflow_dir",
    "snakefile",
    "config_dir",
    "profile_dir",
    "repo_root",
    "script",
    "describe",
]

# Directory holding this file: ``src/`` in a clone, ``…/stateshift/`` installed.
_PKG = Path(__file__).resolve().parent

# Environment escape hatch. Needed on clusters where the workflow is staged on a
# shared filesystem separate from the Python environment (a real pattern on
# GLiCID: env in $HOME, pipeline on LAB-DATA).
_ENV_WORKFLOW = "STATESHIFT_WORKFLOW"


def package_dir() -> Path:
    """Root of the importable package — also the root of every script path."""
    return _PKG


def workflow_dir() -> Path:
    """Directory containing ``Snakefile``, ``config/`` and ``profiles/``.

    Raises rather than returning a wrong path: a missing workflow means the
    wheel was built without its package data, and a silent fallback here would
    surface much later as an unreadable Snakemake error.
    """
    override = os.environ.get(_ENV_WORKFLOW)
    if override:
        candidate = Path(override).expanduser().resolve()
        if not (candidate / "Snakefile").is_file():
            raise FileNotFoundError(
                f"{_ENV_WORKFLOW}={override} ne contient pas de Snakefile."
            )
        return candidate

    # Installed layout first: in an editable install both the package and the
    # clone are visible, and the packaged copy is the one that matches the
    # installed code.
    for candidate in (_PKG / "workflow", _PKG.parent / "workflow"):
        if (candidate / "Snakefile").is_file():
            return candidate

    raise FileNotFoundError(
        "Workflow introuvable (ni dans le paquet, ni à côté). "
        "Installation incomplète : réinstalle avec `pip install stateshift`, "
        f"ou désigne le répertoire avec {_ENV_WORKFLOW}=/chemin/vers/workflow."
    )


def snakefile() -> Path:
    return workflow_dir() / "Snakefile"


def config_dir() -> Path:
    return workflow_dir() / "config"


def profile_dir() -> Path:
    return workflow_dir() / "profiles" / "slurm"


def repo_root() -> Path | None:
    """The clone root, or ``None`` when running from an installed wheel.

    Only the things that genuinely need a clone should call this — the tests,
    the golden comparator, the ablation shell scripts. The pipeline itself must
    not, or it stops working for pip users.
    """
    parent = _PKG.parent
    if (parent / "workflow" / "Snakefile").is_file() and (parent / "pyproject.toml").is_file():
        return parent
    return None


def script(relative: str) -> Path:
    """Absolute path of a pipeline script, e.g. ``script("gnn/gnn_vgae.py")``.

    Accepts a leading ``src/`` so the historical spelling used across the
    Snakefile and the SLURM scripts keeps working verbatim.
    """
    rel = relative[4:] if relative.startswith("src/") else relative
    path = _PKG / rel
    if not path.is_file():
        raise FileNotFoundError(
            f"Script introuvable : {path}\n"
            f"Le paquet est installé dans {_PKG}. Si ce fichier existe dans ton "
            f"clone mais pas ici, la roue a été construite sans lui — vérifie la "
            f"liste `packages` de pyproject.toml."
        )
    return path


def describe() -> dict[str, str]:
    """Flat summary for ``stateshift doctor`` and for bug reports."""
    try:
        workflow = str(workflow_dir())
    except FileNotFoundError as exc:
        workflow = f"INTROUVABLE ({exc.__class__.__name__})"
    root = repo_root()
    return {
        "package": str(_PKG),
        "workflow": workflow,
        "mode": "clone" if root else "installé",
        "repo_root": str(root) if root else "—",
    }
