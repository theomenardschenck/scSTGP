#!/usr/bin/env bash
# =============================================================================
# run_ablation_grid.sh — Génère un job array SLURM pour la grille d'ablation V3.6
# =============================================================================
# Soumet une SEULE commande sbatch qui crée un job array SLURM. Le cluster
# (Nautilus, GLiCID) gère le parallélisme via `--array=0-(N-1)%MAX_PARALLEL`.
# Tu peux fermer ta session après la soumission, c'est SLURM qui pilote.
#
# Usage :
#   bash scripts/run_ablation_grid.sh                # toutes configs × 3 seeds
#   bash scripts/run_ablation_grid.sh --dry-run      # imprime sbatch sans soumettre
#   bash scripts/run_ablation_grid.sh --seeds "1 2"  # override seeds
#   bash scripts/run_ablation_grid.sh --max-parallel 5
#   bash scripts/run_ablation_grid.sh --time 0-04:00:00
#
# Pré-requis : exécuter depuis gnn_huvec/ (cwd = racine du projet).
# Doc GLiCID : https://doc.glicid.fr/GLiCID-PUBLIC/quickstart_advanced_user.html
# =============================================================================

set -euo pipefail

# --- Paramètres par défaut --------------------------------------------------
DEFAULT_SEEDS=(1 2 3)
MAX_PARALLEL=10                  # cap "max 10 simultanés" via --array=...%N
TIME_LIMIT="0-02:00:00"          # 2h par run d'entraînement
DRY_RUN=0

# --- Parsing CLI ------------------------------------------------------------
SEEDS=("${DEFAULT_SEEDS[@]}")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --seeds)          IFS=' ' read -r -a SEEDS <<< "$2"; shift 2 ;;
        --max-parallel)   MAX_PARALLEL="$2"; shift 2 ;;
        --time)           TIME_LIMIT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Argument inconnu : $1"; exit 1 ;;
    esac
done

# --- Configurations d'ablation (cf. §17 V3.6 du rapport) -------------------
# Format : "<tag>::<flags>" — flags passés tels quels à gnn_vgae.py.
# RUN_TAG côté Python = "<tag>.s<seed>" (le launcher passe --run-tag explicite).
CONFIGS=(
    "full::"
    "no-humess::--no-humess"
    "no-coexpr::--no-coexpr"
    "no-reactome::--no-reactome"
    "no-ppi::--no-ppi"
    "no-scenic-regulons::--no-scenic-regulons"
    "no-cgrp::--no-cell-group-edges"
    "ex-degrees::--exclude-features ppi_degree,reg_degree"
)

# --- Préparation du dossier de logs + fichier de config par tâche ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs/ablation_${TS}"
mkdir -p "$LOG_DIR"
CONFIGS_FILE="$LOG_DIR/configs.tsv"

# Génération du TSV (une ligne par tâche array) : tag<TAB>seed<TAB>flags
> "$CONFIGS_FILE"
for cfg in "${CONFIGS[@]}"; do
    tag="${cfg%%::*}"
    flags="${cfg#*::}"
    for seed in "${SEEDS[@]}"; do
        printf '%s\t%s\t%s\n' "$tag" "$seed" "$flags" >> "$CONFIGS_FILE"
    done
done

N_TASKS=$(wc -l < "$CONFIGS_FILE")
ARRAY_RANGE="0-$((N_TASKS - 1))%${MAX_PARALLEL}"

echo "[grid] project   : $PROJECT_DIR"
echo "[grid] log dir   : $LOG_DIR"
echo "[grid] configs   : $CONFIGS_FILE ($N_TASKS lignes)"
echo "[grid] array     : $ARRAY_RANGE   (= $N_TASKS tâches, max $MAX_PARALLEL en //)"
echo "[grid] time      : $TIME_LIMIT par tâche"
echo "[grid] seeds     : ${SEEDS[*]}"
echo "[grid] configs   : ${#CONFIGS[@]}"
echo

# --- Génération du script sbatch (job array) -------------------------------
# Le worker lit la ligne $SLURM_ARRAY_TASK_ID du configs.tsv et lance le run.
SBATCH_SCRIPT="$LOG_DIR/sbatch_ablation.sh"
cat > "$SBATCH_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=vgae_ablation
#SBATCH --comment="VGAE V3.6 ablation grid"
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

# Lecture de la ligne correspondant à cette tâche array
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIGS_FILE")
TAG=\$(echo "\$LINE" | cut -f1)
SEED=\$(echo "\$LINE" | cut -f2)
FLAGS=\$(echo "\$LINE" | cut -f3)
RUN_TAG="\${TAG}.s\${SEED}"

echo "[\$(date +%T)] task \$SLURM_ARRAY_TASK_ID : tag=\$TAG seed=\$SEED run_tag=\$RUN_TAG"
echo "[\$(date +%T)] flags: \$FLAGS"
echo "[\$(date +%T)] node : \$(hostname)"

# shellcheck disable=SC2086
python3 src/gnn_vgae.py \\
    --run-tag "\$RUN_TAG" \\
    --seed "\$SEED" \\
    \$FLAGS

echo "[\$(date +%T)] task \$SLURM_ARRAY_TASK_ID terminée."
EOF

chmod +x "$SBATCH_SCRIPT"
echo "[grid] sbatch script : $SBATCH_SCRIPT"
echo

# --- Soumission -------------------------------------------------------------
if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] sbatch -M nautilus $SBATCH_SCRIPT"
    echo
    echo "[DRY-RUN] Aperçu du configs.tsv :"
    cat "$CONFIGS_FILE"
    echo
    echo "[DRY-RUN] Aperçu du sbatch script (head) :"
    head -25 "$SBATCH_SCRIPT"
    exit 0
fi

JOB_ID=$(sbatch -M nautilus --parsable "$SBATCH_SCRIPT")
echo "[grid] soumis : job array \$JOB_ID (= ${N_TASKS} tâches, ${MAX_PARALLEL} max en //)"
echo "[grid] suivi   : squeue -M nautilus -j $JOB_ID"
echo "[grid] cancel  : scancel -M nautilus $JOB_ID"
echo "[grid] logs    : $LOG_DIR/vgae_ablation_${JOB_ID}_<task>.{out,err}"
