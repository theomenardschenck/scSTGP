#!/usr/bin/env bash
# =============================================================================
# run_decoy_grid.sh — Job array SLURM pour les null-models décoy (industrialisé)
# =============================================================================
# Industrialise `perturb_decoy.py confidence` sur une grille de configs : 1
# tâche d'array par config (= 1 run-dir + son cross_seed_gene_ranking.tsv).
# Le décoy est CPU-only (load_run device=cpu, pas de flag --device) → partition
# CPU, AUCUN GPU demandé. Cf. gnn_futur §7.12 + perturb_decoy.py.
#
# Pour chaque config découverte sous --rankings-dir/<config>/cross_seed_gene_ranking.tsv :
#   - run-dir résolu = <runs-base>/<run-prefix><config><seed-suffix>
#     (fallback : 1er glob <runs-base>/*<config>*.s*)
#   - sortie         = <rankings-dir>/<config>/decoy_confidence.tsv
#
# Colonne produite : decoy_confidence = 1 − |proj_N2_moy| / |proj_réel| (mode KO,
# degré total) ⊥ degré ; ≈0 = marqueur diffusé, ↗1 = network-causal (type TP53).
#
# Usage :
#   bash scripts/run_decoy_grid.sh                       # défauts V5.4.1
#   bash scripts/run_decoy_grid.sh --rankings-dir output/interpretation/V5.4.1/cross_seed \
#       --runs-base /scratch/nautilus/users/${GNN_CLUSTER_USER:-$USER}/gnn_vgae
#   bash scripts/run_decoy_grid.sh --top-n 50 --n-rewires 5 --mode knockout
#   bash scripts/run_decoy_grid.sh --only baseline,ex-degrees,no-rfi
#   bash scripts/run_decoy_grid.sh --after 1234567        # chaîne afterok:<jobid>
#   bash scripts/run_decoy_grid.sh --dry-run
#
# Partitions/qos = SPÉCIFIQUES GLiCID → surcharger via env (cf. défauts) :
#   DECOY_CLUSTER (déf. nautilus)   DECOY_PARTITION (déf. standard)
#   DECOY_QOS (déf. short)          DECOY_PY (déf. python3 ; local : .venv-local/bin/python)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# Shebang bash robuste pour le job (GLiCID/Guix : `/usr/bin/env bash` n'est PAS
# résolu côté nœud, `/bin/bash` absent). On bake le chemin absolu résolu sur le
# frontal (FS partagé ⇒ valide sur les nœuds). Override : env BASH_BIN.
BASH_BIN="${BASH_BIN:-$(command -v bash || true)}"
# Guix : déréférencer le symlink de profil (node-local) vers le vrai chemin
# /gnu/store/… partagé sur tous les nœuds (sinon `bad interpreter` côté calcul).
[[ -n "$BASH_BIN" ]] && BASH_BIN="$(readlink -f "$BASH_BIN" 2>/dev/null || echo "$BASH_BIN")"
[[ -z "$BASH_BIN" ]] && BASH_BIN="/bin/bash"

# --- Défauts ----------------------------------------------------------------
RANKINGS_DIR="output/interpretation/V5.4.1/cross_seed"
RUNS_BASE="${DECOY_RUNS_BASE:-/scratch/nautilus/users/${GNN_CLUSTER_USER:-$USER}/gnn_vgae}"
RUN_PREFIX="v5.4."          # <run-prefix><config><seed-suffix>
SEED_SUFFIX=".s1"           # seed servant de support au décoy (encodeur gelé)
MODE="knockout"             # mode de perturbation pour la confidence (KO recommandé)
TOP_N=40
N_REWIRES=3
MIN_SIGNAL="5.0"
ONLY=""                     # liste de configs (CSV) à restreindre ; vide = toutes
OUT_NAME="decoy_confidence.tsv"
MAX_PARALLEL=10
TIME_LIMIT="0-02:00:00"
AFTER=""                    # job id pour --dependency afterok:<id>
DRY_RUN=0

CLUSTER="${DECOY_CLUSTER:-nautilus}"
PARTITION="${DECOY_PARTITION:-standard}"
QOS="${DECOY_QOS:-short}"
PY="${DECOY_PY:-python3}"

# --- Parsing CLI ------------------------------------------------------------
while [[ $# -gt 0 ]]; do case "$1" in
  --rankings-dir) RANKINGS_DIR="$2"; shift 2;;
  --runs-base)    RUNS_BASE="$2"; shift 2;;
  --run-prefix)   RUN_PREFIX="$2"; shift 2;;
  --seed-suffix)  SEED_SUFFIX="$2"; shift 2;;
  --mode)         MODE="$2"; shift 2;;
  --top-n)        TOP_N="$2"; shift 2;;
  --n-rewires)    N_REWIRES="$2"; shift 2;;
  --min-signal)   MIN_SIGNAL="$2"; shift 2;;
  --only)         ONLY="$2"; shift 2;;
  --out-name)     OUT_NAME="$2"; shift 2;;
  --max-parallel) MAX_PARALLEL="$2"; shift 2;;
  --time)         TIME_LIMIT="$2"; shift 2;;
  --after)        AFTER="$2"; shift 2;;
  --dry-run)      DRY_RUN=1; shift;;
  -h|--help)      sed -n '2,33p' "$0"; exit 0;;
  *) echo "[decoy] arg inconnu : $1"; exit 1;;
esac; done

# --- Résolution du script perturb_decoy (nested local / flat Nautilus) ------
DECOY_SCRIPT="src/validation/explain/perturb_decoy.py"
[[ -f "$DECOY_SCRIPT" ]] || DECOY_SCRIPT="src/perturb_decoy.py"
[[ -f "$DECOY_SCRIPT" ]] || { echo "[decoy] perturb_decoy.py introuvable (ni nested ni flat)"; exit 1; }

[[ -d "$RANKINGS_DIR" ]] || { echo "[decoy] --rankings-dir introuvable : $RANKINGS_DIR"; exit 1; }

# --- Helper : restriction --only (CSV) --------------------------------------
_in_only() {
    [[ -z "$ONLY" ]] && return 0
    local cfg="$1"; IFS=',' read -ra _o <<< "$ONLY"
    for x in "${_o[@]}"; do [[ "$x" == "$cfg" ]] && return 0; done
    return 1
}

# --- Helper : résolution du run-dir pour une config -------------------------
_resolve_run() {
    local cfg="$1"
    local cand="$RUNS_BASE/${RUN_PREFIX}${cfg}${SEED_SUFFIX}"
    if [[ -f "$cand/hetero_graph_vgae.pt" ]]; then echo "$cand"; return 0; fi
    # fallback : 1er glob *<config>*.s* portant un modèle
    for g in "$RUNS_BASE"/*"${cfg}"*.s*; do
        [[ -f "$g/hetero_graph_vgae.pt" ]] && { echo "$g"; return 0; }
    done
    return 1
}

# --- Découverte + configs.tsv (config<TAB>run-dir<TAB>ranking<TAB>out) -------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs/decoy_${TS}"; mkdir -p "$LOG_DIR"
CONFIGS_FILE="$LOG_DIR/configs.tsv"; : > "$CONFIGS_FILE"

for rk in "$RANKINGS_DIR"/*/cross_seed_gene_ranking.tsv; do
    [[ -f "$rk" ]] || continue
    cfg="$(basename "$(dirname "$rk")")"
    _in_only "$cfg" || continue
    if run="$(_resolve_run "$cfg")"; then
        out="$(dirname "$rk")/$OUT_NAME"
        printf '%s\t%s\t%s\t%s\n' "$cfg" "$run" "$rk" "$out" >> "$CONFIGS_FILE"
    else
        echo "[decoy] [skip] $cfg : aucun run-dir (cherché ${RUN_PREFIX}${cfg}${SEED_SUFFIX} + glob)"
    fi
done

N_TASKS=$(wc -l < "$CONFIGS_FILE")
[[ "$N_TASKS" -eq 0 ]] && { echo "[decoy] aucune config résolue (rankings + run-dirs)."; exit 1; }
ARRAY_RANGE="0-$((N_TASKS - 1))%${MAX_PARALLEL}"

echo "[decoy] rankings  : $RANKINGS_DIR"
echo "[decoy] runs-base : $RUNS_BASE"
echo "[decoy] script    : $DECOY_SCRIPT (py=$PY)"
echo "[decoy] configs   : $CONFIGS_FILE ($N_TASKS tâches, max $MAX_PARALLEL en //)"
echo "[decoy] params    : mode=$MODE top-n=$TOP_N n-rewires=$N_REWIRES min-signal=$MIN_SIGNAL"
echo "[decoy] cluster   : -M $CLUSTER  partition=$PARTITION  qos=$QOS  time=$TIME_LIMIT"
echo

# --- Génération sbatch ------------------------------------------------------
SBATCH_SCRIPT="$LOG_DIR/sbatch_decoy.sh"
cat > "$SBATCH_SCRIPT" <<EOF
#!${BASH_BIN}
#SBATCH --job-name=decoy
#SBATCH --comment="null-models décoy (confidence) industrialisé"
#SBATCH --output=$LOG_DIR/%x_%A_%a.out
#SBATCH --error=$LOG_DIR/%x_%A_%a.err
#SBATCH --time=$TIME_LIMIT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --array=$ARRAY_RANGE

set -euo pipefail
# GLiCID/Guix : PATH minimal sur le nœud → bake le PATH du frontal (python3 etc.)
export PATH="$PATH"
cd "$PROJECT_DIR"

# Lecture ligne + split TSV via builtins bash (pas de sed/cut sur le nœud).
mapfile -t _CFG_LINES < "$CONFIGS_FILE"
LINE="\${_CFG_LINES[\$SLURM_ARRAY_TASK_ID]}"
IFS=\$'\t' read -r CFG RUN RANK OUT <<< "\$LINE"

echo "[\$(date +%T)] task \$SLURM_ARRAY_TASK_ID : cfg=\$CFG node=\$(hostname)"
echo "  run=\$RUN"
echo "  rank=\$RANK"
echo "  out=\$OUT"

$PY $DECOY_SCRIPT confidence \\
    --run-dir "\$RUN" --ranking "\$RANK" \\
    --top-n $TOP_N --n-rewires $N_REWIRES --mode $MODE \\
    --min-signal $MIN_SIGNAL --out "\$OUT"

echo "[\$(date +%T)] task \$SLURM_ARRAY_TASK_ID terminée → \$OUT"
EOF
chmod +x "$SBATCH_SCRIPT"
echo "[decoy] sbatch script : $SBATCH_SCRIPT"

# --- Soumission -------------------------------------------------------------
if [[ $DRY_RUN -eq 1 ]]; then
    echo; echo "[DRY-RUN] configs.tsv :"; cat "$CONFIGS_FILE"
    echo; echo "[DRY-RUN] sbatch (head) :"; head -25 "$SBATCH_SCRIPT"
    exit 0
fi

CL_ARG=(); [[ -n "$CLUSTER" ]] && CL_ARG=(--clusters="$CLUSTER")
DEP_ARG=(); [[ -n "$AFTER" ]] && DEP_ARG=(--dependency=afterok:"$AFTER")
JOB_ID=$(sbatch --parsable "${CL_ARG[@]}" "${DEP_ARG[@]}" "$SBATCH_SCRIPT")
echo "[decoy] soumis : job array $JOB_ID ($N_TASKS tâches)${AFTER:+ (afterok:$AFTER)}"
echo "[decoy] suivi  : squeue --me  (ou squeue -M $CLUSTER -j $JOB_ID)"
echo "[decoy] cancel : scancel $JOB_ID"
echo "[decoy] logs   : $LOG_DIR/"
