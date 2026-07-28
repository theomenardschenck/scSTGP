#!/usr/bin/env bash
# =============================================================================
# run_v6_train.sh — Entraînement VGAE généralisé (dataset bulk/scRNA) sur cluster.
# 1 job SLURM par seed. GPU-si-possible/fallback CPU (l'ENTRAÎNEMENT profite du
# GPU, contrairement à la perturbation → cf. run_v6_de_axis.sh qui reste CPU).
# =============================================================================
# Câble la généralisation V6 de gnn_vgae via env (cf. docs/technical/gnn_vgae_paths.md) :
#   GNN_EXPR_MATRIX / GNN_GROUP_META  → matrice + sample→group (chemin BULK)
#   GNN_CELL_GROUPS / GNN_HUMESS_CONDITIONS → conditions du dataset (ex. pro,sen)
#   GNN_HUMESS_DIR → importance HuMess ; GNN_OUT_DIR_BASE → /scratch (GLiCID)
# + flags config C2-signée bulk (OmniPath/FI signés, coexpr diff, no-scenic).
#
# GLiCID : sorties sur /scratch (rapide), runs FIGÉS transférés ensuite vers
# LAB-DATA (rsync). Le clone (code+data) est lu depuis LAB-DATA (défaut portable).
#
# Usage (défauts = GSE163251, conditions pro/sen) :
#   bash scripts/run_v6_train.sh --run-tag gse163 --seeds "1 2 3" \
#       --matrix   data/pyscenic/GSE163251/expr_all.csv \
#       --group-meta data/pyscenic/GSE163251/samplesheet.tsv \
#       --coexpr-file data/pyscenic/GSE163251/coexpr_diff.tsv \
#       --humess-dir data/humess/GSE163251 \
#       --cell-groups pro,sen --humess-conditions pro,sen
#   [--no-gpu] [--n-epochs 1200] [--patience 150] [--dry-run]
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_TAG="gse163"; SEEDS="1 2 3"
MATRIX="data/pyscenic/GSE163251/expr_all.csv"
GROUP_META="data/pyscenic/GSE163251/samplesheet.tsv"
COEXPR="data/pyscenic/GSE163251/coexpr_diff.tsv"
HUMESS_DIR_REL="data/humess/GSE163251"
CELL_GROUPS="pro,sen"; HUMESS_CONDS="pro,sen"
DATA_DIR="data"
N_EPOCHS=1200; PATIENCE=150; CPUS=8; MEM="16G"; USE_GPU=1; DRY=0
# flags config C2-signée bulk (sans pySCENIC ; is_tf/TF-cible via OmniPath CollecTRI)
FLAGS="--no-scenic-regulons --use-omnipath-signaling --use-omnipath-tf-curated --include-omnipath-genes --use-reactome-fi --signed-message --signed-decoder --coexpr-mode differential"
CLUSTER="${V6_CLUSTER:-nautilus}"
# OUT sur scratch (bonne pratique GLiCID). Surchargeable.
OUT_BASE="${V6_OUT_BASE:-/scratch/nautilus/users/${GNN_CLUSTER_USER:-$USER}/gnn_vgae}"
GPU_TIME="${V6_GPU_TIME:-0-03:00:00}"; GPU_PARTITION="${V6_GPU_PARTITION:-gpu}"
GPU_QOS="${V6_GPU_QOS:-gpus}"; GPU_GRES="${V6_GPU_GRES:-gpu:1}"; GPU_CONSTRAINT="${V6_GPU_CONSTRAINT:-}"
CPU_TIME="${V6_CPU_TIME:-1-00:00:00}"; CPU_PARTITION="${V6_CPU_PARTITION:-standard}"; CPU_QOS="${V6_CPU_QOS:-short}"

while [[ $# -gt 0 ]]; do case "$1" in
  --run-tag) RUN_TAG="$2"; shift 2;;
  --seeds) SEEDS="$2"; shift 2;;
  --matrix) MATRIX="$2"; shift 2;;
  --group-meta) GROUP_META="$2"; shift 2;;
  --coexpr-file) COEXPR="$2"; shift 2;;
  --humess-dir) HUMESS_DIR_REL="$2"; shift 2;;
  --cell-groups) CELL_GROUPS="$2"; shift 2;;
  --humess-conditions) HUMESS_CONDS="$2"; shift 2;;
  --data-dir) DATA_DIR="$2"; shift 2;;
  --out-base) OUT_BASE="$2"; shift 2;;
  --n-epochs) N_EPOCHS="$2"; shift 2;;
  --patience) PATIENCE="$2"; shift 2;;
  --extra-flags) FLAGS="$2"; shift 2;;
  --cpus) CPUS="$2"; shift 2;;
  --mem-per-cpu) MEM="$2"; shift 2;;
  --no-gpu) USE_GPU=0; shift;;
  --gpu-constraint) GPU_CONSTRAINT="$2"; shift 2;;
  --dry-run) DRY=1; shift;;
  -h|--help) sed -n '2,30p' "$0"; exit 0;;
  *) echo "arg inconnu : $1"; exit 1;;
esac; done

TS="$(date +%Y%m%d_%H%M%S)"; LOG_DIR="scripts/logs/v6_train_${TS}"; mkdir -p "$LOG_DIR"
CL_ARG=(); [[ -n "$CLUSTER" ]] && CL_ARG=(--clusters="$CLUSTER")

submit_seed() {  # $1 = seed
  local seed="$1"
  local sb="$LOG_DIR/sbatch_train_s${seed}.sh"
  cat > "$sb" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=v6train_s${seed}
#SBATCH --output=$LOG_DIR/%x_%j.out
#SBATCH --error=$LOG_DIR/%x_%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=$CPUS --mem-per-cpu=$MEM
set -euo pipefail
echo "[train] \$(date) node=\$(hostname)"; nvidia-smi -L 2>/dev/null || echo "  (CPU)"
# module load python/3.12 cuda/12 2>/dev/null || true   # adapter à GLiCID
PY="\${V6_PY:-python3}"
export GNN_DATA_DIR="$DATA_DIR"
export GNN_OUT_DIR_BASE="$OUT_BASE"
export GNN_EXPR_MATRIX="\$(readlink -f "$MATRIX")"
export GNN_GROUP_META="\$(readlink -f "$GROUP_META")"
export GNN_CELL_GROUPS="$CELL_GROUPS"
export GNN_HUMESS_CONDITIONS="$HUMESS_CONDS"
export GNN_HUMESS_DIR="\$(readlink -f "$HUMESS_DIR_REL")"
\$PY src/gnn/gnn_vgae.py --seed ${seed} --run-tag ${RUN_TAG}.s${seed} \\
    --n-epochs $N_EPOCHS --patience $PATIENCE \\
    --diff-coexpr-file "\$(readlink -f "$COEXPR")" $FLAGS
echo "[train] DONE → $OUT_BASE/${RUN_TAG}.s${seed}"
EOF
  if [[ $DRY -eq 1 ]]; then echo "[DRY] --- $sb ---"; cat "$sb"; return; fi
  local jid=""
  if [[ $USE_GPU -eq 1 ]]; then
    GA=(--partition="$GPU_PARTITION" --gres="$GPU_GRES" --time="$GPU_TIME")
    [[ -n "$GPU_QOS" ]] && GA+=(--qos="$GPU_QOS")
    [[ -n "$GPU_CONSTRAINT" ]] && GA+=(--constraint="$GPU_CONSTRAINT")
    if jid=$(sbatch --parsable "${CL_ARG[@]}" "${GA[@]}" "$sb" 2>"$LOG_DIR/gpu_s${seed}.err"); then
      echo "[train] seed ${seed} → GPU job ${jid%%;*}"; return; fi
    echo "[train] seed ${seed} GPU refusé ($(tail -1 "$LOG_DIR/gpu_s${seed}.err")) → CPU"
  fi
  local CA=(--partition="$CPU_PARTITION" --time="$CPU_TIME"); [[ -n "$CPU_QOS" ]] && CA+=(--qos="$CPU_QOS")
  jid=$(sbatch --parsable "${CL_ARG[@]}" "${CA[@]}" "$sb"); echo "[train] seed ${seed} → CPU job ${jid%%;*}"
}

echo "[train] run_tag=$RUN_TAG seeds=($SEEDS) → OUT=$OUT_BASE"
echo "[train] ⚠️ qos GPU GLiCID 'gpus' = 3h / MaxJobsPU 1 (les seeds se sérialisent ; sinon V6_GPU_QOS=quick)"
for s in $SEEDS; do submit_seed "$s"; done
echo "[train] logs : $LOG_DIR/ ; suivi : squeue --me"
echo "[train] runs figés → archiver vers LAB-DATA :"
echo "    rsync -av $OUT_BASE/${RUN_TAG}.s* /LAB-DATA/GLiCID/users/<user>/gnn/output/gnn_vgae/"
