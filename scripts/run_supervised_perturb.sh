#!/usr/bin/env bash
# =============================================================================
# run_supervised_perturb.sh — Perturbation du modèle V-sup sur cluster (GLiCID).
# =============================================================================
# Soumet un job SLURM par VOIE de perturbation KO/KD/OE sur un run supervisé
# DÉJÀ ENTRAÎNÉ (output/gnn_vgae/<tag>, produit par run_supervised.sh) :
#   • native  (2a) : perturb_supervised.py  → Δ proba DEG PAR CLUSTER (tête)
#                    → perturbation/{perturbation_supervised.tsv, driver_supervised.tsv}
#   • vgae    (2b) : perturb_top_genes.py   → Δμ axe sénescence (encodeur seul)
#                    → perturbation/perturbation_all_*.tsv (puis perturb_report)
#
# AUTONOME : les deux voies lisent le run-dir sauvé (graphe + modèle + artefacts) ;
# PAS besoin des données brutes (LAB-DATA) ni de GNN_DATA_DIR.
#
# Partitions/qos GLiCID (surcharge via env) :
#   SUP_CPU_PARTITION (standard)  SUP_CPU_QOS (long)  SUP_PY (python/venv)
#
# Usage :
#   bash scripts/run_supervised_perturb.sh --run-dir output/gnn_vgae/vsup_full
#   [--which native|vgae|both]         # défaut both
#   [--all-genes | --top-n 300 | --genes HMGB1,HMGB2,ENO1]   # défaut --top-n 300
#   [--per-mode]                       # 1 job SLURM PAR mode (//) + finalize afterok
#                                      #   → évite le time-limit sur --all-genes
#   [--modes "knockout knockdown overexpress"]
#   [--time 0-12:00:00] [--cpus 8] [--mem 48G] [--dry-run]
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR=""; WHICH="both"; SCOPE="--top-n 300"
MODES="knockout knockdown overexpress"
TIME="0-12:00:00"; CPUS=8; MEM="48G"; DRY=0; PER_MODE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2;;
    --which) WHICH="$2"; shift 2;;
    --all-genes) SCOPE="--all-genes"; shift;;
    --top-n) SCOPE="--top-n $2"; shift 2;;
    --genes) SCOPE="--genes $2"; shift 2;;
    --modes) MODES="$2"; shift 2;;
    --per-mode) PER_MODE=1; shift;;
    --time) TIME="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --mem) MEM="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "[perturb-sup] argument inconnu : $1" >&2; exit 2;;
  esac
done
[[ -z "$RUN_DIR" ]] && { echo "[perturb-sup] --run-dir requis" >&2; exit 2; }
[[ ! -f "$RUN_DIR/supervised_config.json" ]] && \
  echo "[perturb-sup] WARN : $RUN_DIR/supervised_config.json absent (run entraîné ?)"

CPU_PARTITION="${SUP_CPU_PARTITION:-standard}"
CPU_QOS="${SUP_CPU_QOS:-long}"
LOG_DIR="output/gnn_supervised/_slurm"; mkdir -p "$LOG_DIR"

# Hyperparams encoder lus depuis la config du run (pour la voie VGAE / load_run).
read_hp() { python3 -c "import json;print(json.load(open('$RUN_DIR/supervised_config.json'))['hyperparams'].get('$1',$2))" 2>/dev/null || echo "$2"; }
HID=$(read_hp hidden 128); LAT=$(read_hp latent 64)
NL=$(read_hp n_layers 3); NH=$(read_hp n_heads 4)

LAST_JID=""
emit_and_submit() {  # $1=jobname $2=python-command $3=dependency(job:id|"")
  local f="$LOG_DIR/sbatch_$1.sh"
  cat > "$f" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$1
#SBATCH --output=$LOG_DIR/%x_%j.out
#SBATCH --error=$LOG_DIR/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --partition=$CPU_PARTITION
#SBATCH --time=$TIME
$( [[ -n "$CPU_QOS" ]] && echo "#SBATCH --qos=$CPU_QOS" )
set -euo pipefail
cd "\$SLURM_SUBMIT_DIR"
PY="\${SUP_PY:-python3}"
export OMP_NUM_THREADS=$CPUS
$2
EOF
  echo "[perturb-sup] sbatch : $f"
  if [[ "$DRY" -eq 1 ]]; then echo "--- DRY-RUN ($1) ---"; cat "$f"; LAST_JID="DRY_$1"; return; fi
  local dep=(); [[ -n "${3:-}" ]] && dep=(--dependency="$3")
  LAST_JID=$(sbatch --parsable "${dep[@]}" "$f")
  echo "[perturb-sup] soumis : job $LAST_JID ($1)${3:+  [dep=$3]}"
}

echo "[perturb-sup] run-dir=$RUN_DIR which=$WHICH scope='$SCOPE' modes='$MODES' per_mode=$PER_MODE"
echo "[perturb-sup] hyperparams (voie VGAE) : hidden=$HID latent=$LAT n_layers=$NL n_heads=$NH"

# perturb_top_genes gère --all-genes / --top-n / --genes-file ; on mappe --genes → --extra-genes.
VGAE_SCOPE="$SCOPE"; case "$SCOPE" in --genes*) VGAE_SCOPE="--extra-genes ${SCOPE#--genes }";; esac
NATIVE="src/perturbation/perturb_supervised.py"
VGAE="src/perturbation/perturb_top_genes.py --hidden $HID --latent $LAT --n-layers $NL --n-heads $NH"

do_native() {
  if [[ "$PER_MODE" -eq 0 ]]; then
    emit_and_submit "vsup_pert_native" \
      "\$PY $NATIVE --run-dir '$RUN_DIR' --modes $MODES $SCOPE --device cpu"
  else
    local ids=()
    for m in $MODES; do
      emit_and_submit "vsup_native_$m" \
        "\$PY $NATIVE --run-dir '$RUN_DIR' --modes $m $SCOPE --device cpu --no-aggregate"
      ids+=("$LAST_JID")
    done
    local dep; dep="afterok:$(IFS=:; echo "${ids[*]}")"
    emit_and_submit "vsup_native_finalize" \
      "\$PY $NATIVE --run-dir '$RUN_DIR' --finalize" "$dep"
  fi
}
do_vgae() {
  if [[ "$PER_MODE" -eq 0 ]]; then
    emit_and_submit "vsup_pert_vgae" \
      "\$PY $VGAE --run-dir '$RUN_DIR' --modes $MODES $VGAE_SCOPE --device cpu"
  else
    local ids=()
    for m in $MODES; do
      emit_and_submit "vsup_vgae_$m" \
        "\$PY $VGAE --run-dir '$RUN_DIR' --modes $m $VGAE_SCOPE --device cpu"
      ids+=("$LAST_JID")
    done
    local dep; dep="afterok:$(IFS=:; echo "${ids[*]}")"
    emit_and_submit "vsup_vgae_finalize" \
      "\$PY src/validation/reports/perturb_report.py --perturb-dir '$RUN_DIR/perturbation'" "$dep"
  fi
}

[[ "$WHICH" == "native" || "$WHICH" == "both" ]] && do_native
[[ "$WHICH" == "vgae"   || "$WHICH" == "both" ]] && do_vgae

echo "[perturb-sup] terminé. Sorties → $RUN_DIR/perturbation/"
[[ "$PER_MODE" -eq 1 ]] && echo "[perturb-sup] --per-mode : 1 job/mode (//), + finalize en afterok."
