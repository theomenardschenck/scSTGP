#!/usr/bin/env bash
# =============================================================================
# run_ablation_grid.sh — Job array SLURM pour grilles d'ablation VGAE
# =============================================================================
# Soumet UN seul job array SLURM. Le cluster (Nautilus, GLiCID) gère le
# parallélisme via `--array=0-(N-1)%MAX_PARALLEL`. Tu peux fermer la
# session après soumission, c'est SLURM qui pilote.
#
# 3 modes au choix (le défaut est v4_validation, plus rapide à lancer) :
#
# --mode v4_validation   [DÉFAUT] : 4 configs × N seeds = validation OmniPath
#     baseline (= V3.6 plein) ; +sig (SIGNOR) ; +tf (CollecTRI) ; +sig+tf (V4)
#     OBJECTIF : confirmer que SIGNOR / CollecTRI apportent un gain mesurable
#     avant de paralléliser sur toutes les ablations. Tags utilisés :
#         v4-baseline.s<seed>     (no omnipath, full V3.6)
#         v4-sig.s<seed>          (+SIGNOR signaling)
#         v4-tf.s<seed>           (+CollecTRI tf_curated)
#         v4-full.s<seed>         (+sig +tf)
#
# --mode v3_legacy       : 8 configs V3.6 historiques (sans omnipath).
#     Reproduit la grille d'ablation V3.6 décrite §17 du rapport.
#
# --mode full_grid       : cross-product 8 ablations × {sans, avec omnipath}
#     = 16 configs × N seeds. À lancer APRÈS validation v4_validation
#     ET correction du biais GRNBoost2 (cf. TODO).
#
# Usage :
#   bash scripts/run_ablation_grid.sh                          # mode v4_validation, 3 seeds
#   bash scripts/run_ablation_grid.sh --dry-run                # imprime sans soumettre
#   bash scripts/run_ablation_grid.sh --mode v3_legacy --seeds "1 2 3 4 5"
#   bash scripts/run_ablation_grid.sh --mode full_grid --seeds "1 2 3"
#   bash scripts/run_ablation_grid.sh --max-parallel 5 --time 0-04:00:00
#
# Pré-requis : cwd = racine du projet (gnn_huvec/).
#              data/omnipath/ doit contenir tf_collectri.tsv.gz +
#              signed_ppi_signor.tsv.gz si mode utilise omnipath.
# Doc GLiCID : https://doc.glicid.fr/GLiCID-PUBLIC/quickstart_advanced_user.html
# =============================================================================

set -euo pipefail

# --- Paramètres par défaut --------------------------------------------------
DEFAULT_SEEDS=(1 2 3)
MODE="v4_validation"
MAX_PARALLEL=10
TIME_LIMIT="0-02:00:00"
DRY_RUN=0

# --- Parsing CLI ------------------------------------------------------------
SEEDS=("${DEFAULT_SEEDS[@]}")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=1; shift ;;
        --mode)          MODE="$2"; shift 2 ;;
        --seeds)         IFS=' ' read -r -a SEEDS <<< "$2"; shift 2 ;;
        --max-parallel)  MAX_PARALLEL="$2"; shift 2 ;;
        --time)          TIME_LIMIT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "Argument inconnu : $1"; exit 1 ;;
    esac
done

# --- Définition des grilles selon $MODE -------------------------------------
# Format : "<tag>::<flags>" — flags passés tels quels à gnn_vgae.py.
# RUN_TAG côté Python = "<tag>.s<seed>" (--run-tag explicite override l'auto).

# OmniPath flags réutilisés
OP_SIG="--use-omnipath-signaling"
OP_TF="--use-omnipath-tf-curated"
OP_FULL="$OP_SIG $OP_TF"

V4_VALIDATION_CONFIGS=(
    "v4-baseline::"
    "v4-sig::$OP_SIG"
    "v4-tf::$OP_TF"
    "v4-full::$OP_FULL"
)

V3_LEGACY_CONFIGS=(
    "full::"
    "no-humess::--no-humess"
    "no-coexpr::--no-coexpr"
    "no-reactome::--no-reactome"
    "no-ppi::--no-ppi"
    "no-scenic-regulons::--no-scenic-regulons"
    "no-cgrp::--no-cell-group-edges"
    "ex-degrees::--exclude-features ppi_degree,reg_degree"
)

# full_grid = chaque ablation V3 × {no-omnipath, +sig+tf}
# Le tag suffixe "+op" identifie les variantes OmniPath.
FULL_GRID_CONFIGS=()
for cfg in "${V3_LEGACY_CONFIGS[@]}"; do
    tag="${cfg%%::*}"
    flags="${cfg#*::}"
    FULL_GRID_CONFIGS+=("${tag}::${flags}")
    FULL_GRID_CONFIGS+=("${tag}+op::${flags} $OP_FULL")
done

case "$MODE" in
    v4_validation) CONFIGS=("${V4_VALIDATION_CONFIGS[@]}") ;;
    v3_legacy)     CONFIGS=("${V3_LEGACY_CONFIGS[@]}") ;;
    full_grid)     CONFIGS=("${FULL_GRID_CONFIGS[@]}") ;;
    *) echo "Mode inconnu : $MODE (v4_validation | v3_legacy | full_grid)"; exit 1 ;;
esac

# --- Préparation logs + configs.tsv ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs/ablation_${MODE}_${TS}"
mkdir -p "$LOG_DIR"
CONFIGS_FILE="$LOG_DIR/configs.tsv"

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

echo "[grid] mode      : $MODE"
echo "[grid] project   : $PROJECT_DIR"
echo "[grid] log dir   : $LOG_DIR"
echo "[grid] configs   : $CONFIGS_FILE ($N_TASKS lignes)"
echo "[grid] array     : $ARRAY_RANGE   (= $N_TASKS tâches, max $MAX_PARALLEL en //)"
echo "[grid] time      : $TIME_LIMIT par tâche"
echo "[grid] seeds     : ${SEEDS[*]}"
echo "[grid] configs   : ${#CONFIGS[@]}"
echo

# --- Sécurité : prévenir si le cache OmniPath manque pour les modes qui en ont besoin
if [[ "$MODE" == "v4_validation" || "$MODE" == "full_grid" ]]; then
    CACHE_TF="$PROJECT_DIR/data/omnipath/tf_collectri.tsv.gz"
    CACHE_SIG="$PROJECT_DIR/data/omnipath/signed_ppi_signor.tsv.gz"
    if [[ ! -f "$CACHE_TF" ]]; then
        echo "[warn] cache OmniPath TF absent : $CACHE_TF"
        echo "       lance d'abord : python scripts/cache_omnipath.py --cache-dir data/omnipath"
    fi
    if [[ ! -f "$CACHE_SIG" ]]; then
        echo "[warn] cache OmniPath SIGNOR absent : $CACHE_SIG"
        echo "       (les arêtes signaling seront vides — runs +sig dégradés)"
    fi
fi

# --- Génération sbatch -----------------------------------------------------
SBATCH_SCRIPT="$LOG_DIR/sbatch_ablation.sh"
cat > "$SBATCH_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=vgae_ablation_${MODE}
#SBATCH --comment="VGAE ablation ${MODE}"
#SBATCH --output=$LOG_DIR/%x_%A_%a.out
#SBATCH --error=$LOG_DIR/%x_%A_%a.err
#SBATCH --time=$TIME_LIMIT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --qos=quick
#SBATCH --array=$ARRAY_RANGE

set -euo pipefail

cd "$PROJECT_DIR"

LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIGS_FILE")
TAG=\$(echo "\$LINE" | cut -f1)
SEED=\$(echo "\$LINE" | cut -f2)
FLAGS=\$(echo "\$LINE" | cut -f3)
RUN_TAG="\${TAG}.s\${SEED}"

echo "[\$(date +%T)] task \$SLURM_ARRAY_TASK_ID : tag=\$TAG seed=\$SEED run_tag=\$RUN_TAG"
echo "[\$(date +%T)] flags: \$FLAGS"
echo "[\$(date +%T)] node : \$(hostname)"

# shellcheck disable=SC2086
# NB cluster Nautilus : tous les .py sont déployés à plat sous src/
# (pas de sous-dossiers gnn/, perturbation/, validation/ comme en local).
python3 src/gnn_vgae.py \\
    --run-tag "\$RUN_TAG" \\
    --seed "\$SEED" \\
    \$FLAGS

echo "[\$(date +%T)] task \$SLURM_ARRAY_TASK_ID terminée."
EOF

chmod +x "$SBATCH_SCRIPT"
echo "[grid] sbatch script : $SBATCH_SCRIPT"
echo

# --- Soumission ------------------------------------------------------------
if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] sbatch $SBATCH_SCRIPT"
    echo
    echo "[DRY-RUN] Aperçu du configs.tsv :"
    cat "$CONFIGS_FILE"
    echo
    echo "[DRY-RUN] Aperçu du sbatch script (head) :"
    head -25 "$SBATCH_SCRIPT"
    exit 0
fi

JOB_ID=$(sbatch --parsable "$SBATCH_SCRIPT")
echo "[grid] soumis : job array $JOB_ID (= ${N_TASKS} tâches, ${MAX_PARALLEL} max en //)"
echo "[grid] suivi   : squeue -j $JOB_ID    (ou squeue -u \$USER)"
echo "[grid] cancel  : scancel $JOB_ID"
echo "[grid] logs    : $LOG_DIR/vgae_ablation_${MODE}_${JOB_ID}_<task>.{out,err}"
