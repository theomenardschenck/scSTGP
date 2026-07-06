"""
_score.py — scoring + export du VGAE (§11-16), import-safe.

Extrait du monolithe gnn_vgae.py (split Tier 2.5). `score_and_write(ctx)` extrait les
embeddings, calcule le score d'importance (5 composantes) + baselines, annote via les
BDD de sénescence, assemble et ÉCRIT le ranking + les sorties. Le corps (_score_body.py)
est compilé puis exécuté dans un dict-namespace pré-rempli par ctx (sémantique
module-level du monolithe).

    from _score import score_and_write
    score_and_write(dict(globals()))   # depuis gnn_vgae, après train

Interface transitoire (ctx = globals de l'appelant) ; (model, embeddings, bundle, cfg,
paths) → results_df propre viendra une fois la couche config extraite.
"""
import os

_BODY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_score_body.py")
with open(_BODY_PATH, encoding="utf-8") as _fh:
    _SCORE_CODE = compile(_fh.read(), _BODY_PATH, "exec")


def score_and_write(ctx):
    """Exécute §11-16 dans un namespace pré-rempli par ctx (écrit les sorties).
    Retourne le DataFrame `results` (ranking) s'il existe, sinon None."""
    ns = {k: v for k, v in ctx.items() if not k.startswith("__")}
    exec(_SCORE_CODE, ns)
    return ns.get("results")
