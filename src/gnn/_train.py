"""
_train.py — entraînement du VGAE (§8-10), import-safe.

Extrait du monolithe gnn_vgae.py (split Tier 2.5). `train_vgae(ctx)` instancie le
modèle, prépare les arêtes, entraîne (reconstruction + KL + loss signée + tête
supervisée jointe optionnelle), recharge le meilleur checkpoint et finalise. Comme
`build_graph`, le corps (_train_body.py) est COMPILÉ une fois puis exécuté dans un
dict-namespace pré-rempli par `ctx` — sémantique module-level exacte du monolithe.

    from _train import train_vgae
    globals().update(train_vgae(dict(globals())))   # depuis gnn_vgae, après build

Interface transitoire (ctx = globals de l'appelant) ; un couple (bundle, cfg, paths)
→ (model, embeddings, metrics) propre le remplacera une fois la couche config extraite.
"""
import os

_BODY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_train_body.py")
with open(_BODY_PATH, encoding="utf-8") as _fh:
    _TRAIN_CODE = compile(_fh.read(), _BODY_PATH, "exec")


def train_vgae(ctx):
    """Exécute §8-10 dans un namespace pré-rempli par ctx. Retourne le namespace
    résultant (nouvelles liaisons : model, embeddings, tête sup, métriques…) pour
    que gnn_vgae le fusionne dans ses globals avant le scoring."""
    ns = {k: v for k, v in ctx.items() if not k.startswith("__")}
    exec(_TRAIN_CODE, ns)
    return {k: v for k, v in ns.items() if not k.startswith("__")}
