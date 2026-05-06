#!/usr/bin/env bash
# =============================================================================
# run_perturbation_grid.sh — Job array SLURM pour les perturbations VGAE
# =============================================================================
# Découvre les run_dirs entraînés (auto-discovery via glob) et soumet UN job
# array sbatch — le cluster gère le parallélisme.
#
# Nouveauté V4 : --axis {v3,v4,both} pour basculer l'axe de sénescence
# entre P4-only (V3.x) et P4+P16_cluster_0 (V4 — c0 quiescent-like, cf.
# §V4 du rapport). Avec --axis both, chaque run est perturbé deux fois
# avec des fichiers de sortie distincts (--out-suffix) → permet de
# comparer V3.7 (= baseline + axe V4) à V3.6 (= baseline + axe V3) sur le
# MÊME modèle entraîné.
#
# Usage typique post-V4 :
#   bash scripts/run_perturbation_grid.sh \\
#       --pattern '<base>/v4-*.s*' --axis both --also-pathways
#       → perturbe les 4 trainings V4 avec axe V3 ET V4 (8 perturbations × seeds × modes)
#
#   bash scripts/run_perturbation_grid.sh \\
#       --pattern '<base>/full.s*' --axis v4
#       → V3.7 : perturbe la baseline V3.6 (déjà entraînée) avec le NOUVEL axe
#
# Chaînage automatique perturb_report --all :
#   Après chaque perturb_top_genes, on lance `src/perturb_report.py --all`
#   sur le TSV produit pour générer le dossier report/ (ou report_axisV4/
#   selon l'axe). Désactive avec --skip-report si tu veux générer les
#   reports en batch séparé après coup.
#
# Usage générique :
#   bash scripts/run_perturbation_grid.sh                          # axe V3 (legacy)
#   bash scripts/run_perturbation_grid.sh --modes "knockout knockdown overexpress"
#   bash scripts/run_perturbation_grid.sh --runs "V3.3_Run1 V3.3_Run2"
#   bash scripts/run_perturbation_grid.sh --pattern '<base>/no-humess.s*'
#   bash scripts/run_perturbation_grid.sh --also-pathways
#   bash scripts/run_perturbation_grid.sh --dry-run
#   bash scripts/run_perturbation_grid.sh --max-parallel 5
#   bash scripts/run_perturbation_grid.sh --time 1-00:00:00
#   bash scripts/run_perturbation_grid.sh --skip-report             # désactive le chaînage
#
# Doc GLiCID : https://doc.glicid.fr/GLiCID-PUBLIC/quickstart_advanced_user.html
# =============================================================================

set -euo pipefail

# --- Paramètres par défaut --------------------------------------------------
OUT_DIR_BASE_DEFAULT="/scratch/nautilus/users/USER@univ-nantes.fr/gnn_vgae"
OUT_DIR_BASE="${OUT_DIR_BASE:-$OUT_DIR_BASE_DEFAULT}"
DEFAULT_PATTERN="${OUT_DIR_BASE}/*.s*"

DEFAULT_MODES=(knockout)
DEFAULT_AXIS="v3"           # backward-compat : axe historique P4 quiescent
MAX_PARALLEL=10
TIME_LIMIT="0-12:00:00"
DRY_RUN=0
ALSO_PATHWAYS=0
SKIP_REPORT=0               # par défaut : on chaîne perturb_report --all après

# --- Parsing CLI ------------------------------------------------------------
PATTERN=""
EXPLICIT_RUNS=()
MODES=("${DEFAULT_MODES[@]}")
AXIS="$DEFAULT_AXIS"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --max-parallel)   MAX_PARALLEL="$2"; shift 2 ;;
        --time)           TIME_LIMIT="$2"; shift 2 ;;
        --pattern)        PATTERN="$2"; shift 2 ;;
        --runs)           IFS=' ' read -r -a EXPLICIT_RUNS <<< "$2"; shift 2 ;;
        --modes)          IFS=' ' read -r -a MODES <<< "$2"; shift 2 ;;
        --also-pathways)  ALSO_PATHWAYS=1; shift ;;
        --axis)           AXIS="$2"; shift 2 ;;
        --skip-report)    SKIP_REPORT=1; shift ;;
        -h|--help)
            sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "Argument inconnu : $1"; exit 1 ;;
    esac
done

# --- Résolution de l'axe → quiescent_groups + suffixe -----------------------
# AXES est un tableau parallèle : pour chaque entrée, on a un suffixe et un
# argument --quiescent-groups. Avec axis=both, on en a 2 (V3 et V4).
case "$AXIS" in
    v3)
        AXIS_SPECS=("axisV3::P4")
        ;;
    v4)
        AXIS_SPECS=("axisV4::P4,P16_cluster_0")
        ;;
    both)
        AXIS_SPECS=(
            "axisV3::P4"
            "axisV4::P4,P16_cluster_0"
        )
        ;;
    *)
        echo "Axis inconnu : $AXIS (v3 | v4 | both)"; exit 1 ;;
esac

# Pour --axis v3 (legacy), on conserve le comportement historique : pas de
# suffixe sur les TSV pour ne pas casser les pipelines aval (perturb_report,
# cross_seed). Les autres modes ajoutent un suffixe explicite.
if [[ "$AXIS" == "v3" ]]; then
    AXIS_SPECS=("::P4")    # suffixe vide = backward-compat total
fi

# --- Découverte des run_dirs -----------------------------------------------
RUN_DIRS=()
if [[ ${#EXPLICIT_RUNS[@]} -gt 0 ]]; then
    RUN_DIRS=("${EXPLICIT_RUNS[@]}")
else
    # shellcheck disable=SC2206
    RUN_DIRS=( ${PATTERN:-$DEFAULT_PATTERN} )
fi

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

# --- Préparation logs + configs.tsv ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs/perturb_${AXIS}_${TS}"
mkdir -p "$LOG_DIR"
CONFIGS_FILE="$LOG_DIR/configs.tsv"

# Une ligne par tâche : run_dir<TAB>mode<TAB>target_flag<TAB>suffixe<TAB>quiescent_groups
> "$CONFIGS_FILE"
for run_dir in "${VALID_RUNS[@]}"; do
    for mode in "${MODES[@]}"; do
        for spec in "${AXIS_SPECS[@]}"; do
            sfx="${spec%%::*}"          # "axisV4" ou ""
            qg="${spec#*::}"            # "P4,P16_cluster_0" ou "P4"
            # Le suffixe d'output prend une underscore prefix si non vide
            out_sfx=""
            [[ -n "$sfx" ]] && out_sfx="_${sfx}"
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$run_dir" "$mode" "--all-genes" "$out_sfx" "$qg" >> "$CONFIGS_FILE"
            if [[ $ALSO_PATHWAYS -eq 1 ]]; then
                printf '%s\t%s\t%s\t%s\t%s\n' \
                    "$run_dir" "$mode" "--all-pathways" "$out_sfx" "$qg" >> "$CONFIGS_FILE"
            fi
        done
    done
done

N_TASKS=$(wc -l < "$CONFIGS_FILE")
ARRAY_RANGE="0-$((N_TASKS - 1))%${MAX_PARALLEL}"

echo "[grid] axis       : $AXIS  (specs : ${AXIS_SPECS[*]})"
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
#SBATCH --job-name=vgae_perturb_${AXIS}
#SBATCH --comment="VGAE perturbation grid (axis=${AXIS})"
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
OUT_SUFFIX=\$(echo "\$LINE" | cut -f4)
QGROUPS=\$(echo "\$LINE" | cut -f5)

echo "[\$(date +%T)] task \$SLURM_ARRAY_TASK_ID"
echo "  run_dir       : \$RUN_DIR"
echo "  mode          : \$MODE"
echo "  target        : \$TARGET_FLAG"
echo "  out-suffix    : '\$OUT_SUFFIX'"
echo "  quiescent     : \$QGROUPS"
echo "  node          : \$(hostname)"

EXTRA_ARGS=()
if [[ -n "\$OUT_SUFFIX" ]]; then
    EXTRA_ARGS+=(--out-suffix "\$OUT_SUFFIX")
fi
EXTRA_ARGS+=(--quiescent-groups "\$QGROUPS")

# NB cluster Nautilus : tous les .py sont déployés à plat sous src/
# (pas de sous-dossiers gnn/, perturbation/, validation/ comme en local).
python3 src/perturb_top_genes.py \\
    --run-dir "\$RUN_DIR" \\
    "\$TARGET_FLAG" \\
    --modes "\$MODE" \\
    "\${EXTRA_ARGS[@]}"

# --- Chaînage perturb_report --all après chaque perturbation ----------
# Reprend les TSV agrégés produits ci-dessus et matérialise un dossier
# report/ avec figures + drivers. Le rapport va dans un sous-dossier
# distinct par axe pour ne pas écraser le report V3 quand on relance V4.
if [[ $SKIP_REPORT -eq 0 ]]; then
    # Nom du TSV produit par perturb_top_genes selon TARGET_FLAG :
    #   --all-genes     → perturbation_all_genes\${OUT_SUFFIX}_\${MODE}.tsv
    #   --all-pathways  → perturbation_all_pathways\${OUT_SUFFIX}_\${MODE}.tsv
    case "\$TARGET_FLAG" in
        --all-genes)    PREFIX="perturbation_all_genes" ;;
        --all-pathways) PREFIX="perturbation_all_pathways" ;;
        *) PREFIX="" ;;
    esac
    if [[ -n "\$PREFIX" ]]; then
        TSV="\$RUN_DIR/\${PREFIX}\${OUT_SUFFIX}_\${MODE}.tsv"
        if [[ -f "\$TSV" ]]; then
            # Report dir séparé par axe : report (V3 par défaut) ou report_axisV4
            REPORT_SUFFIX="\${OUT_SUFFIX}"   # "" pour V3, "_axisV4" pour V4
            REPORT_DIR="\$RUN_DIR/report\${REPORT_SUFFIX}"
            mkdir -p "\$REPORT_DIR"
            echo "[\$(date +%T)] perturb_report --all → \$REPORT_DIR"
            python3 src/perturb_report.py \\
                --all "\$TSV" \\
                --report-dir "\$REPORT_DIR"
        else
            echo "[\$(date +%T)] [warn] TSV introuvable, perturb_report skip : \$TSV"
        fi
    fi
fi

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
echo "[grid] logs    : $LOG_DIR/vgae_perturb_${AXIS}_${JOB_ID}_<task>.{out,err}"
