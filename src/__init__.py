"""stateshift — State Transition Gene Prediction.

Prioritise genes that DRIVE a transition between two cell states, using a
heterogeneous VGAE over curated + data-derived networks, followed by in-silico
perturbation (KO / KD / OE) read out along a state axis.

The reference application is endothelial replicative senescence (HUVEC P4 vs
P16), but the method is state-agnostic: a transition is defined by the pair of
cell groups contrasted at readout, not by the encoder. See ``docs/`` (local)
and ``README.md`` for the generalisation boundary.

Layout — this package maps onto ``src/`` (see ``pyproject.toml``):

    stateshift.gnn           model, graph build, training, scoring
    stateshift.data          loaders (bulk / proteomics / DE schema), preprocessing
    stateshift.perturbation  in-silico KO/KD/OE and axis re-projection
    stateshift.validation    everything that CHALLENGES the result: baselines,
                             decoys, source attribution, ORA, QC, figures
    stateshift.optim         automatic hyper-parameter search (Optuna)
    stateshift.workflow      the Snakemake pipeline, shipped as package data

Two import modes coexist by design; ``pyproject.toml`` documents why:

    script mode   ``python src/gnn/gnn_vgae.py …``   (Snakefile, cluster)
    package mode  ``import stateshift.data.loaders.bulk_rna``  (tests, new code)

Path helpers are re-exported here so that neither mode has to know which layout
it is running under — see ``_layout`` for the reasoning.
"""

from ._layout import (
    config_dir,
    describe,
    package_dir,
    profile_dir,
    repo_root,
    script,
    snakefile,
    workflow_dir,
)

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "config_dir",
    "describe",
    "package_dir",
    "profile_dir",
    "repo_root",
    "script",
    "snakefile",
    "workflow_dir",
]
