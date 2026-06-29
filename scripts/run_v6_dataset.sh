#!/usr/bin/env bash
# =============================================================================
# run_v6_dataset.sh — Orchestrateur V6 PAR DATASET (cluster).
# =============================================================================
# Enchaîne, pour un dataset :
#   [0] (optionnel) matrix_to_de.R : matrice counts/FPKM → table DE canonique ;
#   [1] run_v6_de_axis.sh par seed : perturbation axe DE-ancré (+ rôle DE +
#       fan-out signé) → report → probe (Tête B).  GPU-si-possible/fallback CPU.
#   [2] (optionnel --analysis) imprime la commande cross-seed à lancer ENSUITE
#       (run_analysis.sh, local, une fois les jobs SLURM terminés).
#
# ⚠️ AUCUN ré-entraînement : tout réutilise l'encodeur de --run-base.<seed>.
#    L'étape [0] (R) tourne en synchrone sur le nœud de login (rapide) AVANT de
#    soumettre, pour garantir que le DE existe quand les jobs démarrent.
#
# Usage :
#   bash scripts/run_v6_dataset.sh --name GSE163251 \
#       --matrix data/bulkRNAseq/GSE163251_huvec/GSE163251_fpkm_all.txt \
#       --gene-col Tracking_id --data-type fpkm \
#       --metadata data/bulkRNAseq/GSE163251_huvec/GSE163251_metadata.tsv \
#       --de-file data/bulkRNAseq/GSE163251_huvec/GSE163251_DE_sen_vs_pro.tsv \
#       --label sen_vs_pro \
#       --run-base output/gnn_vgae/V5.4.1/v5.4.baseline --seeds "s1 s2 s3" \
#       --out-suffix _axisGSE163251 --analysis
#   # sans --matrix : --de-file doit déjà exister (DE prêt, ex. GSE98440).
#   # tout ce qui suit `--` est passé tel quel à run_v6_de_axis.sh
#   #   (ex. --probe-proteo …, --no-gpu, --gpu-constraint gpu_a100, --no-probe).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="dataset"; MATRIX=""; DE_FILE=""; LABEL="sen_vs_pro"
GENE_COL=""; DATA_TYPE="auto"; METADATA=""; YOUNG_MAX=""; SENE_MIN=""
RUN_BASE=""; SEEDS="s1 s2 s3"; OUT_SUFFIX=""; ANALYSIS=0; DECOY=0; DRY=0
PASS=()   # args passés tels quels à run_v6_de_axis.sh (après `--`)

while [[ $# -gt 0 ]]; do case "$1" in
  --name) NAME="$2"; shift 2;;
  --matrix) MATRIX="$2"; shift 2;;
  --de-file) DE_FILE="$2"; shift 2;;
  --label) LABEL="$2"; shift 2;;
  --gene-col) GENE_COL="$2"; shift 2;;
  --data-type) DATA_TYPE="$2"; shift 2;;
  --metadata) METADATA="$2"; shift 2;;
  --young-max-day) YOUNG_MAX="$2"; shift 2;;
  --sene-min-day) SENE_MIN="$2"; shift 2;;
  --run-base) RUN_BASE="$2"; shift 2;;
  --seeds) SEEDS="$2"; shift 2;;
  --out-suffix) OUT_SUFFIX="$2"; shift 2;;
  --analysis) ANALYSIS=1; shift;;
  --decoy) DECOY=1; shift;;
  --dry-run) DRY=1; shift;;
  --) shift; PASS=("$@"); break;;
  -h|--help) sed -n '2,40p' "$0"; exit 0;;
  *) echo "arg inconnu : $1"; exit 1;;
esac; done

[[ -z "$DE_FILE" || -z "$RUN_BASE" ]] && { echo "--de-file et --run-base requis"; exit 1; }
[[ -z "$OUT_SUFFIX" ]] && OUT_SUFFIX="_axis${NAME}"
echo "=== [V6 dataset: $NAME] suffix=$OUT_SUFFIX label=$LABEL ==="

# --- [0] matrix_to_de.R (si --matrix) ---------------------------------------
if [[ -n "$MATRIX" ]]; then
  R_ARGS=(--matrix "$MATRIX" --out "$DE_FILE" --data-type "$DATA_TYPE"
          --condition-label "$LABEL")
  [[ -n "$GENE_COL" ]] && R_ARGS+=(--gene-col "$GENE_COL")
  [[ -n "$METADATA" ]] && R_ARGS+=(--metadata "$METADATA")
  [[ -n "$YOUNG_MAX" ]] && R_ARGS+=(--young-max-day "$YOUNG_MAX")
  [[ -n "$SENE_MIN" ]] && R_ARGS+=(--sene-min-day "$SENE_MIN")
  echo "[0] matrix_to_de.R : $MATRIX → $DE_FILE"
  if [[ $DRY -eq 1 ]]; then echo "   [DRY] Rscript src/data/preprocess/matrix_to_de.R ${R_ARGS[*]}";
  else Rscript src/data/preprocess/matrix_to_de.R "${R_ARGS[@]}"; fi
fi
[[ $DRY -eq 0 && ! -f "$DE_FILE" ]] && { echo "DE introuvable : $DE_FILE (fournir --matrix ?)"; exit 1; }

# --- [1] run_v6_de_axis.sh par seed -----------------------------------------
for SEED in $SEEDS; do
  RUN_DIR="$RUN_BASE.$SEED"
  echo "[1] seed=$SEED → $RUN_DIR"
  AX_ARGS=(--run-dir "$RUN_DIR" --de-axis-file "$DE_FILE" --de-axis-label "$LABEL"
           --out-suffix "$OUT_SUFFIX")
  [[ $DRY -eq 1 ]] && AX_ARGS+=(--dry-run)
  [[ ${#PASS[@]} -gt 0 ]] && AX_ARGS+=("${PASS[@]}")
  bash scripts/run_v6_de_axis.sh "${AX_ARGS[@]}"
done

# --- [2] analyse cross-seed (à lancer APRÈS la fin des jobs) -----------------
if [[ $ANALYSIS -eq 1 ]]; then
  OUT_PARENT="$(dirname "$RUN_BASE")"
  echo ""
  echo "=== [2] quand les jobs SLURM sont finis (squeue --me vide), lance : ==="
  SEED_DIRS=""; for s in $SEEDS; do SEED_DIRS+=" $RUN_BASE.$s"; done
  echo "bash scripts/run_analysis.sh --out $OUT_PARENT --seeds$SEED_DIRS \\"
  echo "    --axis-tag $OUT_SUFFIX --skip-interpret$([[ $DECOY -eq 1 ]] && echo ' --decoy')"
fi
echo "=== [V6 dataset: $NAME] soumission terminée ==="
