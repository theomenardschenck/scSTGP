#!/usr/bin/env bash
# =============================================================================
# run_v6_de_axis.sh — Perturbation V6 (axe DE-ancré bulk) sur cluster.
# GPU si possible, fallback CPU automatique.
# =============================================================================
# Lance `perturb_top_genes.py --all-genes --de-axis-file <DE>` (axe défini
# depuis un DE bulk, design_log §14bis.8.1) puis chaîne `perturb_report --all`.
#
# GPU/CPU : UN seul sbatch body (python `--device auto`). La soumission essaie
# d'abord la partition GPU (`--gres=gpu:1`) ; si sbatch la refuse (pas de GPU /
# partition invalide), elle retombe sur la partition CPU. `--device auto` côté
# python ⇒ cuda sur nœud GPU, cpu sur nœud CPU, sans changer le code.
#
# Partitions/qos = SPÉCIFIQUES GLiCID → surcharger via env (cf. défauts) :
#   V6_GPU_PARTITION (déf. gpu)   V6_GPU_QOS (déf. vide)   V6_GPU_GRES (déf. gpu:1)
#   V6_CPU_PARTITION (déf. standard) V6_CPU_QOS (déf. short)
#
# Usage :
#   bash scripts/run_v6_de_axis.sh \
#       --run-dir <OUT>/v5.4.baseline.s1 \
#       --de-axis-file data/RNAseq/GSE98440_diff_expr_analysis_afterNorm_HUVEC_2reps.txt \
#       --de-axis-label sen_vs_pro --out-suffix _axisDEbulk
#   [--modes "knockout knockdown overexpress"] [--quiescent-groups P4,P16_cluster_0]
#   [--time 0-12:00:00] [--cpus 4] [--mem-per-cpu 24G] [--no-gpu] [--dry-run]
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR=""; DE_FILE=""; DE_LABEL="sen_vs_pro"; DE_TOPN=200; DE_RANK="stat"
MODES="knockout knockdown overexpress"; QGROUPS="P4,P16_cluster_0"
OUT_SUFFIX="_axisDEbulk"; CPUS=4; MEM="24G"; USE_GPU=1; DRY=0
CLUSTER="${V6_CLUSTER:-nautilus}"                          # GLiCID : -M nautilus
CPU_TIME="${V6_CPU_TIME:-0-12:00:00}"
CPU_PARTITION="${V6_CPU_PARTITION:-standard}"; CPU_QOS="${V6_CPU_QOS:-short}"
# GPU GLiCID : SEULES les partitions 'gpu' (gnode[1-5]) et 'visu' ont des GPU
# (sinon erreur "Requested node configuration is not available" → fallback CPU).
# qos 'gpus' ⚠️ MaxWall 3h, MaxJobsPU 1. Carte via --constraint=gpu_a100 ou
# gres gpu:A100:1 (V6_GPU_CONSTRAINT / V6_GPU_GRES). V6_GPU_PARTITION=visu possible.
GPU_TIME="${V6_GPU_TIME:-0-03:00:00}"
GPU_PARTITION="${V6_GPU_PARTITION:-gpu}"; GPU_QOS="${V6_GPU_QOS:-gpus}"
GPU_GRES="${V6_GPU_GRES:-gpu:1}"; GPU_CONSTRAINT="${V6_GPU_CONSTRAINT:-}"

while [[ $# -gt 0 ]]; do case "$1" in
  --run-dir) RUN_DIR="$2"; shift 2;;
  --de-axis-file) DE_FILE="$2"; shift 2;;
  --de-axis-label) DE_LABEL="$2"; shift 2;;
  --de-axis-top-n) DE_TOPN="$2"; shift 2;;
  --de-axis-rank) DE_RANK="$2"; shift 2;;
  --modes) MODES="$2"; shift 2;;
  --quiescent-groups) QGROUPS="$2"; shift 2;;
  --out-suffix) OUT_SUFFIX="$2"; shift 2;;
  --time) CPU_TIME="$2"; shift 2;;
  --gpu-time) GPU_TIME="$2"; shift 2;;
  --gpu-constraint) GPU_CONSTRAINT="$2"; shift 2;;
  --cpus) CPUS="$2"; shift 2;;
  --mem-per-cpu) MEM="$2"; shift 2;;
  --no-gpu) USE_GPU=0; shift;;
  --dry-run) DRY=1; shift;;
  -h|--help) sed -n '2,30p' "$0"; exit 0;;
  *) echo "arg inconnu : $1"; exit 1;;
esac; done

[[ -z "$RUN_DIR" || -z "$DE_FILE" ]] && { echo "--run-dir et --de-axis-file requis"; exit 1; }
[[ -f "$RUN_DIR/hetero_graph_vgae.pt" ]] || { echo "run dir invalide : $RUN_DIR"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"; LOG_DIR="scripts/logs/v6_deaxis_${TS}"; mkdir -p "$LOG_DIR"
SBATCH="$LOG_DIR/sbatch_v6.sh"

# --- sbatch body (commun GPU/CPU ; --device auto) ---------------------------
cat > "$SBATCH" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=v6_deaxis
#SBATCH --output=$LOG_DIR/%x_%j.out
#SBATCH --error=$LOG_DIR/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem-per-cpu=$MEM
set -euo pipefail
echo "[v6] node=\$(hostname) ; nvidia-smi :" ; nvidia-smi -L 2>/dev/null || echo "  (pas de GPU visible → CPU)"
# adapter le module/venv GLiCID si besoin :
# module load python/3.12 cuda/12 2>/dev/null || true
PY="\${V6_PY:-python3}"
\$PY src/perturbation/perturb_top_genes.py --run-dir "$RUN_DIR" \\
    --all-genes --modes $MODES --device auto \\
    --quiescent-groups "$QGROUPS" --out-suffix "$OUT_SUFFIX" \\
    --de-axis-file "$DE_FILE" --de-axis-label "$DE_LABEL" \\
    --de-axis-top-n $DE_TOPN --de-axis-rank $DE_RANK
# chaînage report cross-mode (single-seed ; cross-seed = run_analysis.sh ensuite)
\$PY src/validation/reports/perturb_report.py \\
    --all "$RUN_DIR"/perturbation_all_genes${OUT_SUFFIX}_*.tsv \\
    --report-dir "$RUN_DIR/report${OUT_SUFFIX}" --axis-tag "" \\
    || echo "[v6] report (single-seed) optionnel échoué"
echo "[v6] DONE"
EOF

echo "[v6] sbatch script : $SBATCH"
if [[ $DRY -eq 1 ]]; then echo "[DRY-RUN]"; cat "$SBATCH"; exit 0; fi

# --- soumission : GPU d'abord, fallback CPU (GLiCID : -M nautilus) ----------
CL_ARG=(); [[ -n "$CLUSTER" ]] && CL_ARG=(--clusters="$CLUSTER")
submit() {  # $1 = label, $2.. = args sbatch
  local label="$1"; shift
  echo "[v6] tentative $label : sbatch ${CL_ARG[*]} $*"
  sbatch --parsable "${CL_ARG[@]}" "$@" "$SBATCH"
}
JID=""
if [[ $USE_GPU -eq 1 ]]; then
  GPU_ARGS=(--partition="$GPU_PARTITION" --gres="$GPU_GRES" --time="$GPU_TIME")
  [[ -n "$GPU_QOS" ]] && GPU_ARGS+=(--qos="$GPU_QOS")
  [[ -n "$GPU_CONSTRAINT" ]] && GPU_ARGS+=(--constraint="$GPU_CONSTRAINT")
  echo "[v6] ⚠️ qos GPU GLiCID 'gpus' = MaxWall 3h, MaxJobsPU 1 (--time=$GPU_TIME)."
  if JID=$(submit "GPU" "${GPU_ARGS[@]}" 2>"$LOG_DIR/gpu_submit.err"); then
    echo "[v6] soumis sur GPU : job $JID"
  else
    echo "[v6] GPU refusé ($(tail -1 "$LOG_DIR/gpu_submit.err" 2>/dev/null)) → fallback CPU"
    JID=""
  fi
fi
if [[ -z "$JID" ]]; then
  CPU_ARGS=(--partition="$CPU_PARTITION" --time="$CPU_TIME")
  [[ -n "$CPU_QOS" ]] && CPU_ARGS+=(--qos="$CPU_QOS")
  JID=$(submit "CPU" "${CPU_ARGS[@]}")
  echo "[v6] soumis sur CPU : job $JID"
fi
echo "[v6] suivi : squeue --me  (ou squeue -M $CLUSTER -j $JID) ; logs : $LOG_DIR/"
echo "[v6] ressources : seff $JID  (après fin)"
