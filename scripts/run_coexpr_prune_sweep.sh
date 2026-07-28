#!/usr/bin/env bash
# =============================================================================
# run_coexpr_prune_sweep.sh — Sweep d'élagage coexpr-diff (V4.2.1)
# =============================================================================
# Objectif : trouver le bon trade-off densité-coexpr ↔ AUC.
#
# Constat (run V4.2 du 2026-05-19) : avec K=5 / floor q=0.5 / universe=all
# → AUC 0.9051 vs V4.1 0.97 (perte 0.06). Le canal coexpr noie le signal
# PPI signé dans le décodeur InnerProduct. On sweep K et le floor sans
# relancer GRNBoost2 (les adjacencies P4/P16 existants suffisent — on
# refait JUSTE le merge-adjacencies en quelques minutes).
#
# 3 configs produites depuis les MÊMES adjacencies (universe=all) :
#   * baseline (rappel)  : K=5, floor q=0.5    → coexpr_diff.k5q5.tsv
#   * tight              : K=3, floor q=0.7    → coexpr_diff.k3q7.tsv
#   * very-tight         : K=2, floor q=0.8    → coexpr_diff.k2q8.tsv
#
# Config C (universe=graph) = chemin SÉPARÉ : il faut relancer
# extract-matrices --gene-universe graph PUIS GRNBoost2 (~8h). Géré
# par run_diff_coexpr.sh, pas par ce script. Recommandé en parallèle
# si on veut vraiment trancher "K trop élevé" vs "universe=all trop large".
#
# Workflow :
#   1. bash scripts/run_coexpr_prune_sweep.sh                # 3 merges
#   2. squeue -j <JID>                                        # attendre ~5 min
#   3. bash scripts/run_ablation_grid.sh --version V4.2-sweep --seeds "1 2 3"
#
# Options :
#   --dry-run | --diff-dir <path> | --baseline-only (ne refait pas k5q5)
# =============================================================================
set -euo pipefail

DRY_RUN=0
SKIP_BASELINE=0
DIFF_DIR="/LAB-DATA/GLiCID/users/${GNN_CLUSTER_USER:-$USER}/gnn/data/pyscenic/diff_coexpr"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --baseline-only)  SKIP_BASELINE=0; shift ;;   # garde k5q5 + 2 nouveaux
        --skip-baseline)  SKIP_BASELINE=1; shift ;;   # ne refait QUE k3q7 + k2q8
        --diff-dir)       DIFF_DIR="$2"; shift 2 ;;
        *) echo "Option inconnue : $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs/coexpr_sweep_${TS}"
mkdir -p "$LOG_DIR"

# --- Vérif prérequis : adjacencies P4 + P16 doivent exister ---------------
for cond in P4 P16; do
    A="$DIFF_DIR/adjacencies_${cond}.csv"
    if [[ ! -s "$A" ]]; then
        echo "[err] $A absent/vide. Lance d'abord :"
        echo "      bash scripts/run_diff_coexpr.sh --step grnboost2"
        exit 1
    fi
    n=$(wc -l < "$A")
    echo "[guard] $A : $n lignes ✓"
done

# --- Définition des configs du sweep --------------------------------------
# Format : "<tag>::<per-target-k>::<min-imax-quantile>"
SWEEP_CONFIGS=()
if [[ $SKIP_BASELINE -eq 0 ]]; then
    SWEEP_CONFIGS+=("k5q5::5::0.5")    # rappel du run 2026-05-19
fi
SWEEP_CONFIGS+=("k3q7::3::0.7")        # tight
SWEEP_CONFIGS+=("k2q8::2::0.8")        # very-tight

N_CONFIGS=${#SWEEP_CONFIGS[@]}
ARRAY_MAX=$((N_CONFIGS - 1))

# Écrit la table des configs (lue par SLURM_ARRAY_TASK_ID)
CONFIGS_FILE="$LOG_DIR/sweep_configs.tsv"
: > "$CONFIGS_FILE"
for c in "${SWEEP_CONFIGS[@]}"; do
    TAG=$(echo "$c"  | cut -d: -f1)
    K=$(echo "$c"    | cut -d: -f3)
    FQ=$(echo "$c"   | cut -d: -f5)
    echo -e "${TAG}\t${K}\t${FQ}" >> "$CONFIGS_FILE"
done

echo "[sweep] configs ($N_CONFIGS) :"
cat "$CONFIGS_FILE"
echo

# --- sbatch array ---------------------------------------------------------
SBATCH_SCRIPT="$LOG_DIR/sbatch_sweep.sh"
cat > "$SBATCH_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=coexpr_sweep
#SBATCH --comment="V4.2.1 sweep d'élagage merge-adjacencies"
#SBATCH --output=$LOG_DIR/%x_%A_%a.out
#SBATCH --error=$LOG_DIR/%x_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --qos=quick
#SBATCH --array=0-${ARRAY_MAX}

set -euo pipefail
cd "$PROJECT_DIR"

LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIGS_FILE")
TAG=\$(echo "\$LINE" | cut -f1)
K=\$(echo "\$LINE"   | cut -f2)
FQ=\$(echo "\$LINE"  | cut -f3)

OUT="$DIFF_DIR/coexpr_diff.\${TAG}.tsv"
echo "[\$(date +%T)] sweep \$TAG : K=\$K floor-q=\$FQ → \$OUT"

# NB cluster Nautilus : .py à plat sous src/.
python3 src/build_diff_coexpr.py merge-adjacencies \\
    --adj-p4  "$DIFF_DIR/adjacencies_P4.csv" \\
    --adj-p16 "$DIFF_DIR/adjacencies_P16.csv" \\
    --prune-mode per-target-topk \\
    --per-target-k \$K \\
    --min-imax-quantile \$FQ \\
    --out "\$OUT"

echo "[\$(date +%T)] \$TAG : sortie générée :"
wc -l "\$OUT"
head -3 "\$OUT"
EOF
chmod +x "$SBATCH_SCRIPT"

echo "[sweep] sbatch : $SBATCH_SCRIPT"
echo "[sweep] log dir : $LOG_DIR"
echo

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] sbatch $SBATCH_SCRIPT   # array 0-$ARRAY_MAX"
    sed -n '1,40p' "$SBATCH_SCRIPT"
    exit 0
fi

JID=$(sbatch --parsable "$SBATCH_SCRIPT")
echo "[sweep] soumis : job array $JID (0-$ARRAY_MAX)"
echo "[sweep] suivi  : squeue -j $JID"
echo "[sweep] sorties: $DIFF_DIR/coexpr_diff.{k5q5,k3q7,k2q8}.tsv"
echo
echo "[sweep] ====== ÉTAPE SUIVANTE (après COMPLETED) ======"
echo "  bash scripts/run_ablation_grid.sh --version V4.2-sweep --seeds \"1 2 3\""
echo "  (3 variants × 3 seeds = 9 runs VGAE — produit driver_score cross-seed comparable)"
