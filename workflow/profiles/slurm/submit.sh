#!/usr/bin/env bash
# ===========================================================================
# submit.sh — wrapper de soumission SLURM pour le profil Snakemake 7.
# Appelé par `cluster:` (workflow/profiles/slurm/config.yaml). Snakemake ajoute
# le jobscript en DERNIER argument. On construit le sbatch et on n'ajoute
# --gres QUE s'il est non vide (les règles CPU passent slurm_gres='').
#
# Args (dans l'ordre passé par le profil) :
#   $1 rule  $2 threads  $3 mem_mb  $4 runtime(min)  $5 partition  $6 qos
#   $7 gres (peut être vide)   $8 jobscript (ajouté par Snakemake)
# Doit imprimer l'ID de job (sbatch --parsable) pour le suivi de dépendances.
# ===========================================================================
set -euo pipefail
RULE="$1"; THREADS="$2"; MEM="$3"; RUNTIME="$4"; PART="$5"; QOS="$6"; GRES="$7"; JOB="$8"

mkdir -p logs/slurm
args=(--parsable
      --partition="$PART"
      --qos="$QOS"
      --cpus-per-task="$THREADS"
      --mem="${MEM}"
      --time="$RUNTIME"
      --job-name="smk-$RULE"
      --output="logs/slurm/${RULE}_%j.log")
[[ -n "$GRES" ]] && args+=(--gres="$GRES")

# Fix GLIBCXX (compute nodes GLiCID) : leur libstdc++ système (/lib64) est trop
# vieux pour numpy/torch de l'env conda (manque GLIBCXX_3.4.29). On pose le lib de
# l'env en tête de LD_LIBRARY_PATH DANS NOTRE PROPRE env, puis --export=ALL le
# propage au job — mécanisme identique au srun qui a fonctionné (plus fiable que
# la syntaxe --export=ALL,VAR= dont le parsing SLURM est capricieux).
[[ -n "${CONDA_PREFIX:-}" ]] && \
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
args+=(--export=ALL)

exec sbatch "${args[@]}" "$JOB"
