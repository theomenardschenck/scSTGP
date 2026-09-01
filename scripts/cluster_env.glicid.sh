# =============================================================================
# cluster_env.glicid.sh — à SOURCER sur le frontal GLiCID avant tout lancement.
#
#   source scripts/cluster_env.glicid.sh
#   bash workflow/run.sh --backend cluster --configfile <cfg>
#
# Pourquoi ce fichier existe : une session SSH non interactive n'active aucun
# environnement, et la libstdc++ système des nœuds GLiCID est trop vieille pour
# le numpy/torch de l'env conda (il manque GLIBCXX_3.4.29). Sans les trois
# exports ci-dessous, `import numpy` échoue AVANT même la soumission — et le
# wrapper workflow/profiles/slurm/submit.sh ne peut pas propager le correctif
# aux jobs, puisqu'il le lit dans CONDA_PREFIX.
#
# Adapter MAMBA_ENV à votre site ; le reste est générique.
# =============================================================================
MAMBA_ENV="${MAMBA_ENV:-/micromamba/$USER/envs/gnn}"

if [ ! -x "$MAMBA_ENV/bin/python" ]; then
  echo "[cluster_env] environnement introuvable : $MAMBA_ENV" >&2
  echo "[cluster_env] définissez MAMBA_ENV=<chemin de l'env> puis re-sourcez." >&2
  return 1 2>/dev/null || exit 1
fi

export CONDA_PREFIX="$MAMBA_ENV"
export PATH="$MAMBA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$MAMBA_ENV/lib:${LD_LIBRARY_PATH:-}"

echo "[cluster_env] env      : $MAMBA_ENV"
echo "[cluster_env] python   : $(command -v python)"
echo "[cluster_env] snakemake: $(command -v snakemake || echo ABSENT)"
