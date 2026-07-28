#!/usr/bin/env bash
# =============================================================================
# run_optuna.sh — recherche automatique d'hyperparamètres sur le cluster.
#
# Optuna pilote Snakemake : le CONTRÔLEUR est un job SLURM long, et c'est LUI
# qui soumet les jobs de chaque essai (via `workflow/run.sh --backend cluster`).
#
# Pourquoi un job contrôleur et pas un simple lancement sur le frontal : une
# recherche dure des heures et Snakemake exige un processus vivant pendant tout
# le DAG. Sur le frontal, une session coupée fige la recherche (les jobs déjà
# soumis finissent, mais plus rien n'avance). Dans un job SLURM, elle survit.
#
#   bash scripts/run_optuna.sh calibrate --repeats 3
#   bash scripts/run_optuna.sh search    --n-trials 20
#
# ⚠️ CALIBRER D'ABORD. Deux runs d'une config identique diffèrent de rho
#    0.556-0.687 ; sans plancher de bruit mesuré, aucun résultat de recherche
#    n'est interprétable. `search.py report` refuse de conclure sans lui.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CMD="${1:-search}"; shift || true

STUDY="${OPTUNA_STUDY:-stgp-search}"
BASE_CONFIG="${OPTUNA_BASE_CONFIG:-workflow/config/config.yaml}"
OBJECTIVE="${OPTUNA_OBJECTIVE:-cross_seed_stability}"
SEEDS="${OPTUNA_SEEDS:-3}"
PARTITION="${OPTUNA_PARTITION:-standard}"
QOS="${OPTUNA_QOS:-short}"
RUNTIME="${OPTUNA_RUNTIME:-1440}"          # minutes ; le contrôleur vit longtemps
PY="${OPTUNA_PY:-python3}"

LOG_DIR="logs/optuna"; mkdir -p "$LOG_DIR"
SBATCH_FILE="$LOG_DIR/sbatch_optuna_${CMD}.sh"

cat > "$SBATCH_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=stgp-optuna-${CMD}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --time=${RUNTIME}
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=${LOG_DIR}/optuna_${CMD}_%j.log
set -euo pipefail
cd "\$SLURM_SUBMIT_DIR"

# Le contrôleur ne calcule rien : il soumet. 2 CPU suffisent.
${PY} src/optim/search.py ${CMD} \\
    --base-config "${BASE_CONFIG}" \\
    --objective "${OBJECTIVE}" \\
    --study-name "${STUDY}" \\
    --seeds ${SEEDS} \\
    --backend cluster \\
    $*
EOF

echo "[optuna] script  : $SBATCH_FILE"
echo "[optuna] étude   : $STUDY   objectif : $OBJECTIVE   seeds/essai : $SEEDS"
if [[ "${OPTUNA_DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY-RUN] sbatch $SBATCH_FILE"; exit 0
fi
JOB_ID=$(sbatch --parsable "$SBATCH_FILE")
echo "[optuna] job contrôleur soumis : $JOB_ID"
echo "[optuna] suivi : tail -f ${LOG_DIR}/optuna_${CMD}_${JOB_ID}.log"
