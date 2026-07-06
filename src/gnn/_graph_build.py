"""
_graph_build.py — construction du graphe hétérogène VGAE (§1-7), import-safe.

Extrait du monolithe gnn_vgae.py (split Tier 2.5). `build_graph(ctx)` exécute les
sections 1-7 (chargement scRNA/PPI/SCENIC/HuMess/OmniPath/Reactome, features nœud,
assemblage HeteroData) et retourne le BUNDLE = variables de `_CACHE_VARS` — le même
contrat que le cache --reuse-graph (déjà éprouvé : le reuse ne restaure QUE ces
variables et l'aval fonctionne).

Implémentation : le corps §1-7 (_graph_build_body.py) est COMPILÉ une fois puis
exécuté dans un dict-namespace pré-rempli par `ctx`. Exécuter dans un dict reproduit
EXACTEMENT la sémantique module-level du monolithe (assignations/lectures sur le même
dict), sans les pièges de portée d'une fonction Python.

Interface (transitoire) : `ctx` = espace de noms de l'appelant (config + chemins +
constantes + helpers + imports). Un couple `cfg`/`paths` propre le remplacera une
fois la couche config entièrement extraite (le corps reste inchangé).

    from _graph_build import build_graph
    bundle = build_graph(dict(globals()))   # depuis gnn_vgae, après config+paths
    globals().update(bundle)                # expose §1-7 pour train/score
"""
import os

_BODY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_graph_build_body.py")
with open(_BODY_PATH, encoding="utf-8") as _fh:
    _BUILD_CODE = compile(_fh.read(), _BODY_PATH, "exec")


def build_graph(ctx):
    """Exécute §1-7 dans un namespace pré-rempli par ctx. Retourne le bundle _CACHE_VARS."""
    ns = {k: v for k, v in ctx.items() if not k.startswith("__")}
    exec(_BUILD_CODE, ns)          # sémantique module-level (assign/read sur ns)
    return {k: ns[k] for k in ns["_CACHE_VARS"] if k in ns}
