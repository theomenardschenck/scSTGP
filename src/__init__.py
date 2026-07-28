"""(sc)STGP — State Transition Gene Prediction.

Prioritise genes that DRIVE a transition between two cell states, using a
heterogeneous VGAE over curated + data-derived networks, followed by in-silico
perturbation (KO / KD / OE) read out along a state axis.

The reference application is endothelial replicative senescence (HUVEC P4 vs
P16), but the method is state-agnostic: a transition is defined by the pair of
cell groups contrasted at readout, not by the encoder. See ``docs/`` (local)
and ``README.md`` for the generalisation boundary.

Layout — this package maps onto ``src/`` (see ``pyproject.toml``):

    stgp.gnn           model, graph build, training, scoring
    stgp.data          loaders (bulk / proteomics / DE schema), preprocessing
    stgp.perturbation  in-silico KO/KD/OE and axis re-projection
    stgp.validation    everything that CHALLENGES the result: baselines,
                       decoys, source attribution, ORA, QC, figures
    stgp.optim         automatic hyper-parameter search (Optuna)

Two import modes coexist by design; ``pyproject.toml`` documents why:

    script mode   ``python src/gnn/gnn_vgae.py …``   (Snakefile, cluster)
    package mode  ``import stgp.data.loaders.bulk_rna``  (tests, new code)
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
