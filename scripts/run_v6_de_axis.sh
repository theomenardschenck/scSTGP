#!/usr/bin/env bash
# =============================================================================
# run_v6_de_axis.sh — Perturbation V6 (axe DE-ancré bulk) sur cluster.
# =============================================================================
# Lance `perturb_top_genes.py --all-genes --de-axis-file <DE>` (axe défini
# depuis un DE bulk, design_log §14bis.8.1), chaîne `perturb_report --all`,
# puis (défaut) la PROBE linéaire (Tête B) sur l'embedding figé du run.
#
# ⚠️ AUCUN RÉ-ENTRAÎNEMENT : axe DE + probe réutilisent l'encodeur déjà
#    entraîné (--run-dir). Un nouveau modèle n'est nécessaire QUE pour S3.
#
# DEUX MODES DE SOUMISSION :
#   • défaut (1 job)   : les 3 modes + report + probe dans UN job. GPU-si-
#     possible/fallback CPU (`--device auto`).
#   • --per-mode (CPU) : 1 job CPU PAR mode (parallélisables — GLiCID autorise
#     ~10 jobs CPU vs 1 seul GPU) + 1 job `finalize` (report+probe) en
#     dépendance `afterok`. RECOMMANDÉ : le GPU est contre-productif ici
#     (petit graphe, ~33k forward séquentiels → l'overhead host↔device domine).
#
# Partitions/qos GLiCID (surcharge via env) :
#   V6_GPU_PARTITION (gpu) V6_GPU_QOS (gpus, 3h/MaxJobsPU 1) V6_GPU_GRES (gpu:1)
#   V6_CPU_PARTITION (standard) V6_CPU_QOS (short, 1 jour)   V6_PY (python/venv)
#
# Usage :
#   bash scripts/run_v6_de_axis.sh --run-dir <OUT>/v5.4.baseline.s1 \
#       --de-axis-file <DE.tsv> --de-axis-label sen_vs_pro --out-suffix _axisX
#   [--per-mode]                          # 1 job CPU/mode + finalize (afterok)
#   [--modes "knockout knockdown overexpress"] [--quiescent-groups P4,P16_cluster_0]
#   [--time 0-12:00:00] [--cpus 4] [--mem-per-cpu 24G] [--no-gpu] [--dry-run]
#   [--no-probe] [--probe-proteo <f>] [--probe-proteo-label mutant_vs_wt]
#   [--no-de-role] [--de-sign 1]
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR=""; DE_FILE=""; DE_LABEL="sen_vs_pro"; DE_TOPN=200; DE_RANK="stat"
MODES="knockout knockdown overexpress"; QGROUPS="P4,P16_cluster_0"
OUT_SUFFIX="_axisDEbulk"; CPUS=4; MEM="24G"; USE_GPU=1; DRY=0; PER_MODE=0
PROBE=1; PROTEO_FILE=""; PROTEO_LABEL="mutant_vs_wt"
DE_ROLE=1; DE_SIGN=1
CLUSTER="${V6_CLUSTER:-nautilus}"
CPU_TIME="${V6_CPU_TIME:-0-12:00:00}"
CPU_PARTITION="${V6_CPU_PARTITION:-standard}"; CPU_QOS="${V6_CPU_QOS:-short}"
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
  --per-mode) PER_MODE=1; shift;;
  --time) CPU_TIME="$2"; shift 2;;
  --gpu-time) GPU_TIME="$2"; shift 2;;
  --gpu-constraint) GPU_CONSTRAINT="$2"; shift 2;;
  --cpus) CPUS="$2"; shift 2;;
  --mem-per-cpu) MEM="$2"; shift 2;;
  --no-gpu) USE_GPU=0; shift;;
  --no-de-role) DE_ROLE=0; shift;;
  --de-sign) DE_SIGN="$2"; shift 2;;
  --no-probe) PROBE=0; shift;;
  --probe-proteo) PROTEO_FILE="$2"; shift 2;;
  --probe-proteo-label) PROTEO_LABEL="$2"; shift 2;;
  --dry-run) DRY=1; shift;;
  -h|--help) sed -n '2,40p' "$0"; exit 0;;
  *) echo "arg inconnu : $1"; exit 1;;
esac; done

[[ -z "$RUN_DIR" || -z "$DE_FILE" ]] && { echo "--run-dir et --de-axis-file requis"; exit 1; }
[[ -f "$RUN_DIR/hetero_graph_vgae.pt" ]] || { echo "run dir invalide : $RUN_DIR"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"; LOG_DIR="scripts/logs/v6_deaxis_${TS}"; mkdir -p "$LOG_DIR"
PREFIX="$RUN_DIR/perturbation_all_genes${OUT_SUFFIX}"
DE_ROLE_ARG=""
[[ $DE_ROLE -eq 1 ]] && DE_ROLE_ARG="--de-file \"$DE_FILE\" --de-sign $DE_SIGN"

# nom du fan-out (doit matcher perturb_top_genes : tag par mode si != 3 modes)
fanout_path() {  # $1 = modes string
  local m; m="$(echo "$1" | tr ' ' '\n' | sort | tr '\n' ' ')"
  if [[ "$m" == "knockdown knockout overexpress " ]]; then
    echo "${PREFIX}_signed_fanout.tsv"
  else
    echo "${PREFIX}_signed_fanout_$(echo "$1" | tr ' ' '_').tsv"
  fi
}
# liste des fan-outs pour le report (1 par mode en --per-mode, sinon 1 global)
if [[ $PER_MODE -eq 1 ]]; then
  FANOUT_LIST=""; for m in $MODES; do FANOUT_LIST+=" \"$(fanout_path "$m")\""; done
else
  FANOUT_LIST="\"$(fanout_path "$MODES")\""
fi

# bloc probe (Tête B) — \$PY = expansion runtime
if [[ $PROBE -eq 1 ]]; then
  PROTEO_ARGS=""
  [[ -n "$PROTEO_FILE" ]] && PROTEO_ARGS="--proteo \"$PROTEO_FILE\" --proteo-label \"$PROTEO_LABEL\""
  PROBE_BLOCK="echo '[v6] probe (Tête B)' ; \$PY src/validation/probe/probe.py --emb \"$RUN_DIR/gene_embeddings_vgae.csv\" --rna-de \"$DE_FILE\" --rna-label \"$DE_LABEL\" $PROTEO_ARGS --out \"$RUN_DIR/probe${OUT_SUFFIX}.tsv\" || echo '[v6] probe optionnelle échouée'"
else
  PROBE_BLOCK="echo '[v6] probe désactivée (--no-probe)'"
fi

# -------- générateurs de scripts sbatch --------------------------------------
sbatch_header() {  # $1=file $2=jobname
  cat > "$1" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$2
#SBATCH --output=$LOG_DIR/%x_%j.out
#SBATCH --error=$LOG_DIR/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem-per-cpu=$MEM
set -euo pipefail
echo "[v6] \$(date) node=\$(hostname)" ; nvidia-smi -L 2>/dev/null || echo "  (pas de GPU → CPU)"
# module load python/3.12 cuda/12 2>/dev/null || true
PY="\${V6_PY:-python3}"
EOF
}
append_perturb() {  # $1=file $2=modes $3=device
  cat >> "$1" <<EOF
\$PY src/perturbation/perturb_top_genes.py --run-dir "$RUN_DIR" \\
    --all-genes --modes $2 --device $3 \\
    --quiescent-groups "$QGROUPS" --out-suffix "$OUT_SUFFIX" \\
    --de-axis-file "$DE_FILE" --de-axis-label "$DE_LABEL" \\
    --de-axis-top-n $DE_TOPN --de-axis-rank $DE_RANK $DE_ROLE_ARG
echo "[v6] perturb ($2) DONE"
EOF
}
append_finalize() {  # $1=file
  cat >> "$1" <<EOF
\$PY src/validation/reports/perturb_report.py \\
    --all "${PREFIX}_knockout.tsv" "${PREFIX}_knockdown.tsv" "${PREFIX}_overexpress.tsv" \\
    --signed-fanout$FANOUT_LIST \\
    --report-dir "$RUN_DIR/report${OUT_SUFFIX}" --axis-tag "" \\
    || echo "[v6] report (single-seed) optionnel échoué"
$PROBE_BLOCK
echo "[v6] finalize DONE"
EOF
}

# -------- soumission ---------------------------------------------------------
CL_ARG=(); [[ -n "$CLUSTER" ]] && CL_ARG=(--clusters="$CLUSTER")
submit_cpu() {  # $1=file ; $2=dépendance afterok (jids ':'-séparés, optionnel)
  local f="$1" dep="${2:-}" a=(--partition="$CPU_PARTITION" --time="$CPU_TIME")
  [[ -n "$CPU_QOS" ]] && a+=(--qos="$CPU_QOS")
  [[ -n "$dep" ]] && a+=(--dependency="afterok:$dep")
  sbatch --parsable "${CL_ARG[@]}" "${a[@]}" "$f"
}

if [[ $PER_MODE -eq 1 ]]; then
  # ---- 1 job CPU par mode + finalize (afterok) ----
  echo "[v6] --per-mode : 1 job CPU/mode ($MODES) + finalize"
  declare -a JIDS=()
  for m in $MODES; do
    f="$LOG_DIR/sbatch_perturb_${m}.sh"; sbatch_header "$f" "v6_${m}"
    append_perturb "$f" "$m" "cpu"
    if [[ $DRY -eq 1 ]]; then echo "[DRY] --- $f ---"; cat "$f"; JIDS+=("DRY_$m"); continue; fi
    jid=$(submit_cpu "$f"); jid="${jid%%;*}"   # strip ';nautilus' (--clusters)
    echo "[v6] mode $m → job $jid"; JIDS+=("$jid")
  done
  ff="$LOG_DIR/sbatch_finalize.sh"; sbatch_header "$ff" "v6_finalize"; append_finalize "$ff"
  dep="$(IFS=:; echo "${JIDS[*]}")"
  if [[ $DRY -eq 1 ]]; then echo "[DRY] --- $ff (afterok:$dep) ---"; cat "$ff"; exit 0; fi
  fjid=$(submit_cpu "$ff" "$dep"); fjid="${fjid%%;*}"
  echo "[v6] finalize → job $fjid (afterok:$dep)"
else
  # ---- 1 seul job : tout dedans ; GPU-si-possible/fallback CPU ----
  f="$LOG_DIR/sbatch_v6.sh"; sbatch_header "$f" "v6_deaxis"
  append_perturb "$f" "$MODES" "auto"; append_finalize "$f"
  if [[ $DRY -eq 1 ]]; then echo "[DRY-RUN] --- $f ---"; cat "$f"; exit 0; fi
  JID=""
  if [[ $USE_GPU -eq 1 ]]; then
    GA=(--partition="$GPU_PARTITION" --gres="$GPU_GRES" --time="$GPU_TIME")
    [[ -n "$GPU_QOS" ]] && GA+=(--qos="$GPU_QOS")
    [[ -n "$GPU_CONSTRAINT" ]] && GA+=(--constraint="$GPU_CONSTRAINT")
    echo "[v6] ⚠️ qos GPU 'gpus' = 3h / MaxJobsPU 1 ; le GPU est souvent + LENT ici → --per-mode CPU conseillé."
    if JID=$(sbatch --parsable "${CL_ARG[@]}" "${GA[@]}" "$f" 2>"$LOG_DIR/gpu_submit.err"); then
      echo "[v6] soumis sur GPU : job $JID"
    else
      echo "[v6] GPU refusé ($(tail -1 "$LOG_DIR/gpu_submit.err" 2>/dev/null)) → fallback CPU"; JID=""
    fi
  fi
  [[ -z "$JID" ]] && { JID=$(submit_cpu "$f"); echo "[v6] soumis sur CPU : job $JID"; }
fi
echo "[v6] suivi : squeue --me ; logs : $LOG_DIR/ ; seff <jobID> après fin."
