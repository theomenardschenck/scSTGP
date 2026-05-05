#!/usr/bin/env bash
# =============================================================================
# run_perturbation_grid.sh — Job array SLURM pour les perturbations V3.6
# =============================================================================
# Découvre tous les run_dirs entraînés (auto-discovery via glob) et soumet
# UN seul job array sbatch — `--array=0-N%MAX_PARALLEL`. Le cluster gère le
# parallélisme. Tu peux fermer la session après soumission.
#
# Usage :
#   bash scripts/run_perturbation_grid.sh
#       → tous les *.s* sous $OUT_DIR_BASE, mode knockout, --all-genes
#
#   bash scripts/run_perturbation_grid.sh --modes "knockout knockdown overexpress"
#   bash scripts/run_perturbation_grid.sh --runs "V3.3_Run1 V3.3_Run2"
#   bash scripts/run_perturbation_grid.sh --pattern '<base>/no-humess.s*'
#   bash scripts/run_perturbation_grid.sh --also-pathways
#   bash scripts/run_perturbation_grid.sh --dry-run
#   bash scripts/run_perturbation_grid.sh --max-parallel 5
#   bash scripts/run_perturbation_grid.sh --time 1-00:00:00
#
# Doc GLiCID : https://doc.glicid.fr/GLiCID-PUBLIC/quickstart_advanced_user.html
# =============================================================================

set -euo pipefail

# --- Paramètres par défaut --------------------------------------------------
OUT_DIR_BASE_DEFAULT="/scratch/nautilus/users/USER@univ-nantes.fr/gnn_vgae"
OUT_DIR_BASE="${OUT_DIR_BASE:-$OUT_DIR_BASE_DEFAULT}"
DEFAULT_PATTERN="${OUT_DIR_BASE}/*.s*"

DEFAULT_MODES=(knockout)
MAX_PARALLEL=10
TIME_LIMIT="0-12:00:00"          # 12h par perturbation
DRY_RUN=0
ALSO_PATHWAYS=0

# --- Parsing CLI ------------------------------------------------------------
PATTERN=""
EXPLICIT_RUNS=()
MODES=("${DEFAULT_MODES[@]}")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --max-parallel)   MAX_PARALLEL="$2"; shift 2 ;;
        --time)           TIME_LIMIT="$2"; shift 2 ;;
        --pattern)        PATTERN="$2"; shift 2 ;;
        --runs)           IFS=' ' read -r -a EXPLICIT_RUNS <<< "$2"; shift 2 ;;
        --modes)          IFS=' ' read -r -a MODES <<< "$2"; shift 2 ;;
        --also-pathways)  ALSO_PATHWAYS=1; shift ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Argument inconnu : $1"; exit 1 ;;
    esac
done

# --- Découverte des run_dirs -----------------------------------------------
RUN_DIRS=()
if [[ ${#EXPLICIT_RUNS[@]} -gt 0 ]]; then
    RUN_DIRS=("${EXPLICIT_RUNS[@]}")
else
    # shellcheck disable=SC2206
    RUN_DIRS=( ${PATTERN:-$DEFAULT_PATTERN} )
fi

# Filtrage : ne garder que les dossiers avec hetero_graph_vgae.pt
VALID_RUNS=()
for d in "${RUN_DIRS[@]}"; do
    if [[ -d "$d" && -f "$d/hetero_graph_vgae.pt" ]]; then
        VALID_RUNS+=("$d")
    else
        echo "[skip] $d (pas de hetero_graph_vgae.pt)"
    fi
done

if [[ ${#VALID_RUNS[@]} -eq 0 ]]; then
    echo "[grid] aucun run valide trouvé. Pattern : ${PATTERN:-$DEFAULT_PATTERN}"
    exit 1
fi

# --- Préparation des logs + génération du configs.tsv ----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs/perturb_${TS}"
mkdir -p "$LOG_DIR"
CONFIGS_FILE="$LOG_DIR/configs.tsv"

# Une ligne par tâche : run_dir<TAB>mode<TAB>target_flag
> "$CONFIGS_FILE"
for run_dir in "${VALID_RUNS[@]}"; do
    for mode in "${MODES[@]}"; do
        printf '%s\t%s\t%s\n' "$run_dir" "$mode" "--all-genes" >> "$CONFIGS_FILE"
        if [[ $ALSO_PATHWAYS -eq 1 ]]; then
            printf '%s\t%s\t%s\n' "$run_dir" "$mode" "--all-pathways" >> "$CONFIGS_FILE"
        fi
    done
done

N_TASKS=$(wc -l < "$CONFIGS_FILE")
ARRAY_RANGE="0-$((N_TASKS - 1))%${MAX_PARALLEL}"

echo "[grid] project    : $PROJECT_DIR"
echo "[grid] log dir    : $LOG_DIR"
echo "[grid] configs    : $CONFIGS_FILE ($N_TASKS lignes)"
echo "[grid] array      : $ARRAY_RANGE   (= $N_TASKS tâches, max $MAX_PARALLEL en //)"
echo "[grid] time       : $TIME_LIMIT par tâche"
echo "[grid] runs       : ${#VALID_RUNS[@]}"
echo "[grid] modes      : ${MODES[*]}"
echo "[grid] also-pathw : $ALSO_PATHWAYS"
echo

# --- Génération du sbatch script (job array) -------------------------------
SBATCH_SCRIPT="$LOG_DIR/sbatch_perturb.sh"
cat > "$SBATCH_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=vgae_perturb
#SBATCH --comment="VGAE V3.6 batch perturbation"
#SBATCH --output=$LOG_DIR/%x_%A_%a.out
#SBATCH --error=$LOG_DIR/%x_%A_%a.err
#SBATCH --time=$TIME_LIMIT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --qos=short
#SBATCH --array=$ARRAY_RANGE

set -euo pipefail

cd "$PROJECT_DIR"

LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIGS_FILE")
RUN_DIR=\$(echo "\$LINE" | cut -f1)
MODE=\$(echo "\$LINE" | cut -f2)
TARGET_FLAG=\$(echo "\$LINE" | cut -f3)

echo "[\$(date +%T)] task \$SLURM_ARRAY_TASK_ID"
echo "  run_dir : \$RUN_DIR"
echo "  mode    : \$MODE"
echo "  target  : \$TARGET_FLAG"
echo "  node    : \$(hostname)"

python3 src/perturb_top_genes.py \\
    --run-dir "\$RUN_DIR" \\
    "\$TARGET_FLAG" \\
    --modes "\$MODE"

echo "[\$(date +%T)] task \$SLURM_ARRAY_TASK_ID terminée."
EOF

chmod +x "$SBATCH_SCRIPT"
echo "[grid] sbatch script : $SBATCH_SCRIPT"
echo

# --- Soumission -------------------------------------------------------------
# NB : pas de -M nautilus côté sbatch — SLURMDBD n'est pas joignable depuis
# le frontal Nautilus (le cluster est implicite). Cf. doc GLiCID.
if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] sbatch $SBATCH_SCRIPT"
    echo
    echo "[DRY-RUN] Aperçu du configs.tsv :"
    head -20 "$CONFIGS_FILE"
    echo "  ..."
    echo
    echo "[DRY-RUN] Aperçu du sbatch script (head) :"
    head -25 "$SBATCH_SCRIPT"
    exit 0
fi

JOB_ID=$(sbatch --parsable "$SBATCH_SCRIPT")
echo "[grid] soumis : job array $JOB_ID (= ${N_TASKS} tâches, ${MAX_PARALLEL} max en //)"
echo "[grid] suivi   : squeue -j $JOB_ID    (ou squeue -u \$USER)"
echo "[grid] cancel  : scancel $JOB_ID"
echo "[grid] logs    : $LOG_DIR/vgae_perturb_${JOB_ID}_<task>.{out,err}"
