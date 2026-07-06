"""
_paths.py — résolution des chemins du pipeline VGAE (import-safe, sans effet de bord).

Extrait du monolithe gnn_vgae.py (split Tier 2.5). Regroupe toute la logique de
résolution des chemins d'entrée/sortie, pilotée par variables d'environnement
(clone portable / orchestration Snakemake), avec les MÊMES défauts historiques
GLiCID que le monolithe — comportement iso.

Correctif inclus : `find_repo_root()` remplace l'ancien `dirname³(__file__)` qui
CASSAIT sur le déploiement À PLAT du cluster (src/gnn_vgae.py, 2 niveaux) en
remontant un cran trop haut (/scratch/nautilus/users, non writable →
PermissionError). On remonte désormais jusqu'au dossier contenant `data/`
(racine dépôt/clone), avec repli sur l'ancien comportement.

Usage :
    from _paths import resolve_paths, ensure_dirs
    paths = resolve_paths(cli_args, run_tag)   # namespace de chemins
    ensure_dirs(paths)                          # crée PPI/DB/OUT/FIG
"""
import os
from types import SimpleNamespace


def find_repo_root(start_file):
    """Racine du dépôt/clone, robuste au layout (imbriqué src/gnn/ OU à plat src/).

    Le module vit à côté de gnn_vgae.py, dans `src/gnn/` (layout local imbriqué)
    OU directement dans `src/` (déploiement À PLAT du cluster). On identifie le
    dossier `src` puis on remonte d'un cran → racine du dépôt.

    NB : on NE cherche PAS un dossier contenant `data/` — `src/data/` existe
    (loaders), ce qui ferait s'arrêter à `.../src` par erreur. Repli défensif =
    ancien `dirname³`. Toujours surchargeable par GNN_BASE_DIR / GNN_DATA_DIR.
    """
    d = os.path.dirname(os.path.abspath(start_file))   # .../src/gnn (imbriqué) OU .../src (à plat)
    base = os.path.basename(d)
    if base == "gnn":
        return os.path.dirname(os.path.dirname(d))     # dirname(.../src) = racine
    if base == "src":
        return os.path.dirname(d)                      # dirname(src) = racine
    # Repli : 3 niveaux au-dessus (comportement legacy si layout inattendu).
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(start_file))))


def resolve_paths(cli_args, run_tag, start_file=None, env=None):
    """Résout tous les chemins du pipeline. Défauts = comportement GLiCID historique.

    Args:
        cli_args : Namespace argparse (lit `omnipath_cache_dir`).
        run_tag  : tag du run → OUT_DIR = OUT_DIR_BASE / run_tag.
        start_file : fichier de référence pour la racine du dépôt (défaut : ce module,
            qui vit dans src/gnn/ → même racine que gnn_vgae.py).
        env : mapping d'environnement (défaut os.environ) — testable.
    Returns:
        SimpleNamespace avec tous les chemins (attributs = anciens globals).
    """
    env = os.environ if env is None else env
    start_file = start_file if start_file is not None else __file__

    repo_root = find_repo_root(start_file)
    lab_dir = env.get("GNN_LAB_DIR", "/LAB-DATA/GLiCID/users/USER@univ-nantes.fr/")
    base_dir = env.get("GNN_BASE_DIR", repo_root)
    data_dir = env.get("GNN_DATA_DIR", os.path.join(base_dir, "data"))
    scenic_dir = env.get("GNN_SCENIC_DIR", os.path.join(base_dir, "output", "pyscenic"))
    out_dir_base = env.get("GNN_OUT_DIR_BASE", os.path.join(repo_root, "output", "gnn_vgae"))
    out_dir = os.path.join(out_dir_base, run_tag)

    omnipath_cache_dir = (cli_args.omnipath_cache_dir
                          if getattr(cli_args, "omnipath_cache_dir", None) is not None
                          else os.path.join(data_dir, "omnipath"))
    humess_dir = env.get("GNN_HUMESS_DIR", os.path.join(lab_dir, "humess", "output_huvec"))
    humess_conditions = [c.strip() for c in
                         env.get("GNN_HUMESS_CONDITIONS", "P4,P16").split(",")
                         if c.strip()]
    expr_matrix = env.get("GNN_EXPR_MATRIX", "merged_P4_P16_normalized.csv")
    group_meta = env.get("GNN_GROUP_META", "")

    return SimpleNamespace(
        REPO_ROOT=repo_root,
        LAB_DIR=lab_dir,
        BASE_DIR=base_dir,
        DATA_DIR=data_dir,
        SCENIC_DIR=scenic_dir,
        OUT_DIR_BASE=out_dir_base,
        OUT_DIR=out_dir,
        GNN_DATA_DIR=os.path.join(data_dir, "gnn_data"),
        PPI_DIR=os.path.join(data_dir, "PPI"),
        DB_DIR=os.path.join(data_dir, "databases"),
        FIG_DIR=os.path.join(out_dir, "figure"),
        OMNIPATH_CACHE_DIR=omnipath_cache_dir,
        HUMESS_DIR=humess_dir,
        HUMESS_CONDITIONS=humess_conditions,
        EXPR_MATRIX=expr_matrix,
        GROUP_META=group_meta,
    )


def ensure_dirs(paths):
    """Crée les dossiers de sortie (side effect explicite, ex-l.665-667 du monolithe)."""
    for d in (paths.PPI_DIR, paths.DB_DIR, paths.OUT_DIR, paths.FIG_DIR):
        os.makedirs(d, exist_ok=True)
