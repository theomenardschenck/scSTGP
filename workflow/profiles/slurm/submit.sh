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

exec sbatch "${args[@]}" "$JOB"
