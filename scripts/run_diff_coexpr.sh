#!/usr/bin/env bash
# =============================================================================
# run_diff_coexpr.sh — Prérequis V4.2 : GRNBoost2 différentiel P4/P16
# =============================================================================
# DEUX ÉTAPES MANUELLES DÉCOUPLÉES (plus de --dependency=afterok, qui
# laissait des coexpr_merge bloqués en DependencyNeverSatisfied quand
# GRNBoost2 échouait — il fallait les scancel à la main) :
#   --step grnboost2 : soumet l'array (P4,P16) GRNBoost2 sklearn pur.
#   --step merge     : garde-fou (refuse si adjacencies absents/vides)
#                      puis soumet merge SANS dépendance.
#
# Prérequis étape grnboost2 (en local ou frontal) — RECOMMANDÉ mode
# graph (couverture coexpr 100%% du graphe VGAE vs 40%% en HVG-5000) :
#   python src/preprocess/build_diff_coexpr.py extract-matrices \
#       --gene-universe graph \
#       --graph-genes output/gnn_vgae/V4.1/full/cross_seed_v4.1-full_axisV4/cross_seed_gene_ranking.tsv
#   → data/pyscenic/diff_coexpr/expr_matrix_{P4,P16}.csv (~11k gènes)
# (legacy : --gene-universe hvg --n-hvg 5000 ; ou all = ~15779 gènes)
#
# Sortie finale : data/pyscenic/diff_coexpr/coexpr_diff.tsv
# (consommé par gnn_vgae.py --coexpr-mode differential).
#
# Workflow :
#   1. bash scripts/run_diff_coexpr.sh --step grnboost2
#   2. squeue -j <JID>   # attendre COMPLETED
#   3. bash scripts/run_diff_coexpr.sh --step merge
#   4. bash scripts/run_ablation_grid.sh --version V4.2 --seeds "1 2 3"
#
# Options : --dry-run | --n-workers 16 | --prune-mode {per-target-topk
#   (défaut),global-quantile,hybrid} | --per-target-k 5 |
#   --top-quantile 0.98 | --time-grn 08:00:00 | --diff-dir <path>
#
# Implémentation GRNBoost2 : sklearn pur via
# `build_diff_coexpr.py grnboost2-local` (Moerman 2019, mêmes
# hyperparams SGBM). On N'UTILISE PLUS arboreto/dask : il est cassé
# sur l'env GLiCID (dask ≥ 2025.1 a retiré l'API legacy dask.dataframe
# → NotImplementedError "The legacy implementation is no longer
# supported"). Zéro dépendance dask → robuste sur n'importe quel nœud.
#
# NB cluster Nautilus : .py déployés à plat sous src/ (pas de sous-dossiers).
# Donc src/build_diff_coexpr.py (PAS src/preprocess/...).
# =============================================================================
set -euo pipefail

DRY_RUN=0
# --step : étapes MANUELLES découplées (plus de --dependency=afterok,
# qui laissait des jobs bloqués en DependencyNeverSatisfied quand
# GRNBoost2 échouait). Workflow :
#   1. bash scripts/run_diff_coexpr.sh --step grnboost2   (soumet l'array)
#   2. … attendre la fin (squeue) + vérifier adjacencies_{P4,P16}.csv …
#   3. bash scripts/run_diff_coexpr.sh --step merge        (garde-fou +
#      soumet merge SANS dépendance ; refuse si adjacencies absents)
STEP="grnboost2"
N_WORKERS=8
# Élagage : per-target-topk (DÉFAUT). Le filtre global-quantile est
# DOMINÉ par les hubs Tier-1 (HMGB1/H2AFZ/ENO1…) → à q=0.995 seulement
# 67% des cibles couvertes et ASNS/IL6/IL1B/DDIT3 = 0 arête (même à
# q=0.98). per-target top-K garde les K meilleurs régulateurs de
# CHAQUE cible → 100% couverture, hubs déconcentrés (= élagage SCENIC
# standard). Cf. §14bis.6undecies du rapport.
PRUNE_MODE="per-target-topk"
PER_TARGET_K=5
TOP_QUANTILE=0.995   # utilisé seulement si --prune-mode global-quantile|hybrid
# 08:00:00 : avec --gene-universe graph (~11k cibles vs 5k HVG) le
# GRNBoost2-local sklearn prend ~2× plus longtemps. 4h ne suffit pas.
TIME_GRN="08:00:00"
TIME_MERGE="00:30:00"
DIFF_DIR="/LAB-DATA/GLiCID/users/USER@univ-nantes.fr/gnn/data/pyscenic/diff_coexpr"
TF_LIST="/LAB-DATA/GLiCID/users/USER@univ-nantes.fr/gnn/data/pyscenic/scenic_refs/allTFs_hg38.txt"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --step)           STEP="$2"; shift 2 ;;
        --n-workers)      N_WORKERS="$2"; shift 2 ;;
        --prune-mode)     PRUNE_MODE="$2"; shift 2 ;;
        --per-target-k)   PER_TARGET_K="$2"; shift 2 ;;
        --top-quantile)   TOP_QUANTILE="$2"; shift 2 ;;
        --time-grn)       TIME_GRN="$2"; shift 2 ;;
        --diff-dir)       DIFF_DIR="$2"; shift 2 ;;
        *) echo "Option inconnue : $1"; exit 1 ;;
    esac
done

if [[ "$STEP" != "grnboost2" && "$STEP" != "merge" ]]; then
    echo "[err] --step doit être 'grnboost2' ou 'merge' (reçu : $STEP)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs/diff_coexpr_${TS}"
mkdir -p "$LOG_DIR"

# --- Vérif prérequis SELON l'étape -----------------------------------------
if [[ "$STEP" == "grnboost2" ]]; then
    # GRNBoost2 a besoin des matrices d'expression P4/P16.
    for cond in P4 P16; do
        M="$DIFF_DIR/expr_matrix_${cond}.csv"
        if [[ ! -f "$M" ]]; then
            echo "[err] $M absent."
            echo "      Lance d'abord (local/frontal) :"
            echo "      python src/preprocess/build_diff_coexpr.py extract-matrices \\"
            echo "          --gene-universe graph --graph-genes <cross_seed_gene_ranking.tsv>"
            exit 1
        fi
    done
else
    # merge a besoin des adjacencies produits par l'étape grnboost2.
    # GARDE-FOU : on REFUSE de soumettre si elles sont absentes/vides
    # (remplace le --dependency=afterok fragile).
    missing=0
    for cond in P4 P16; do
        A="$DIFF_DIR/adjacencies_${cond}.csv"
        if [[ ! -s "$A" ]]; then
            echo "[err] $A absent ou vide."
            missing=1
        else
            n=$(wc -l < "$A")
            echo "[merge-guard] $A : $n lignes ✓"
            if [[ "$n" -lt 100 ]]; then
                echo "[err] $A suspicieusement petit ($n lignes) — "
                echo "      l'étape grnboost2 a probablement échoué."
                missing=1
            fi
        fi
    done
    if [[ $missing -eq 1 ]]; then
        echo
        echo "[err] Prérequis merge non satisfaits. L'étape grnboost2"
        echo "      n'est pas terminée (ou a échoué). Vérifie :"
        echo "        squeue -u \$USER -n grnboost2_diff"
        echo "        tail scripts/logs/diff_coexpr_*/grnboost2_diff_*.err"
        echo "      Puis relance : bash scripts/run_diff_coexpr.sh --step merge"
        exit 1
    fi
fi

echo "[diff-coexpr] project    : $PROJECT_DIR"
echo "[diff-coexpr] log dir    : $LOG_DIR"
echo "[diff-coexpr] n_workers  : $N_WORKERS (sklearn grnboost2-local + joblib)"
echo "[diff-coexpr] prune      : $PRUNE_MODE (k=$PER_TARGET_K, q=$TOP_QUANTILE)"

# --- 1. sbatch GRNBoost2 (array P4/P16) ------------------------------------
GRN_SBATCH="$LOG_DIR/sbatch_grnboost2.sh"
cat > "$GRN_SBATCH" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=grnboost2_diff
#SBATCH --comment="V4.2 GRNBoost2 différentiel P4/P16"
#SBATCH --output=$LOG_DIR/%x_%A_%a.out
#SBATCH --error=$LOG_DIR/%x_%A_%a.err
#SBATCH --time=$TIME_GRN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$N_WORKERS
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=short
#SBATCH --array=0-1

set -euo pipefail
cd "$PROJECT_DIR"

CONDS=(P4 P16)
COND=\${CONDS[\$SLURM_ARRAY_TASK_ID]}
echo "[\$(date +%T)] GRNBoost2-local \$COND sur \$(hostname) (\$SLURM_CPUS_PER_TASK jobs)"

# arboreto est CASSÉ sur l'env GLiCID : dask ≥ 2025.1 a retiré l'API
# legacy dask.dataframe que arboreto importe → NotImplementedError
# "The legacy implementation is no longer supported". Au lieu de pinner
# dask (fragile, risque de casser torch/pyg dans l'env `gnn`), on
# utilise notre réimplémentation GRNBoost2 en sklearn pur (Moerman 2019,
# mêmes hyperparams SGBM) : zéro dask, zéro arboreto, parallélisé joblib.
python3 src/build_diff_coexpr.py grnboost2-local \\
    --expr "$DIFF_DIR/expr_matrix_\${COND}.csv" \\
    --tf-list "$TF_LIST" \\
    --out "$DIFF_DIR/adjacencies_\${COND}.csv" \\
    --n-jobs $N_WORKERS \\
    --seed 42

echo "[\$(date +%T)] GRNBoost2-local \$COND terminé."
EOF
chmod +x "$GRN_SBATCH"

# --- 2. sbatch merge (soumis SANS dépendance via --step merge) -------------
MERGE_SBATCH="$LOG_DIR/sbatch_merge.sh"
cat > "$MERGE_SBATCH" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=coexpr_merge
#SBATCH --comment="V4.2 merge-adjacencies → coexpr_diff.tsv"
#SBATCH --output=$LOG_DIR/%x_%j.out
#SBATCH --error=$LOG_DIR/%x_%j.err
#SBATCH --time=$TIME_MERGE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --qos=quick

set -euo pipefail
cd "$PROJECT_DIR"

python3 src/build_diff_coexpr.py merge-adjacencies \\
    --adj-p4  "$DIFF_DIR/adjacencies_P4.csv" \\
    --adj-p16 "$DIFF_DIR/adjacencies_P16.csv" \\
    --prune-mode "$PRUNE_MODE" \\
    --per-target-k $PER_TARGET_K \\
    --top-quantile $TOP_QUANTILE \\
    --out "$DIFF_DIR/coexpr_diff.tsv"

echo "[\$(date +%T)] coexpr_diff.tsv généré :"
wc -l "$DIFF_DIR/coexpr_diff.tsv"
head -2 "$DIFF_DIR/coexpr_diff.tsv"
EOF
chmod +x "$MERGE_SBATCH"

echo "[diff-coexpr] sbatch GRNBoost2 : $GRN_SBATCH"
echo "[diff-coexpr] sbatch merge     : $MERGE_SBATCH"

# ── Soumission SANS afterok : une seule étape par invocation ─────────────
if [[ "$STEP" == "grnboost2" ]]; then
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] sbatch $GRN_SBATCH   # array 0-1 (P4,P16)"
        sed -n '1,32p' "$GRN_SBATCH"
        exit 0
    fi
    GRN_JID=$(sbatch --parsable "$GRN_SBATCH")
    echo "[diff-coexpr] GRNBoost2 soumis : job array $GRN_JID (0-1 = P4,P16)"
    echo "[diff-coexpr] suivi   : squeue -j $GRN_JID"
    echo "[diff-coexpr] sortie  : $DIFF_DIR/adjacencies_{P4,P16}.csv"
    echo
    echo "[diff-coexpr] ====== ÉTAPE SUIVANTE (manuelle, après fin) ======"
    echo "  Quand le job $GRN_JID est COMPLETED et que les 2 adjacencies"
    echo "  existent, lance :"
    echo "    bash scripts/run_diff_coexpr.sh --step merge"
    echo "  (le garde-fou refusera si les adjacencies sont absents/vides —"
    echo "   plus de job bloqué en DependencyNeverSatisfied)"
else  # STEP == merge  (les adjacencies ont déjà été vérifiés plus haut)
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] sbatch $MERGE_SBATCH   # SANS dépendance"
        sed -n '1,30p' "$MERGE_SBATCH"
        exit 0
    fi
    MERGE_JID=$(sbatch --parsable "$MERGE_SBATCH")
    echo "[diff-coexpr] merge soumis : job $MERGE_JID (aucune dépendance)"
    echo "[diff-coexpr] suivi   : squeue -j $MERGE_JID"
    echo "[diff-coexpr] sortie  : $DIFF_DIR/coexpr_diff.tsv"
    echo
    echo "[diff-coexpr] ====== ÉTAPE SUIVANTE (après coexpr_diff.tsv) ======"
    echo "    bash scripts/run_ablation_grid.sh --version V4.2 --seeds \"1 2 3\""
fi
