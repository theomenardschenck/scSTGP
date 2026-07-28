#!/usr/bin/env bash
# =============================================================================
# run_coexpr_method_compare.sh — Comparaison méthodes×prune coexpr (V4.3)
# =============================================================================
# Objectif : trancher entre 4 méthodes d'inférence GRN × 4 modes d'élagage
# pour la couche coexpression (V4.3, post V4.2.1-sweep).
#
# Pourquoi V4.3 : la V4.2 (coexpr différentielle merge-first + per-target-topk)
# plafonne AUC ~0.91 vs V4.1 0.97 et les variantes de pruning (k5q5, k3q7,
# k2q8) testées dans V4.2.1-sweep n'ont pas redressé driver_score. On veut
# maintenant tester DEUX axes orthogonaux :
#
#   GRN   ∈ {sklearn (V4.2 défaut), arboreto (canonique, local seulement),
#            corr (Pearson/Spearman), mi (mutual information)}
#   prune ∈ {topk (V4.2 défaut), quantile (baseline -), mr (mutual rank),
#            zscore (per-target adaptatif)}
#
# Sorties :
#   data/pyscenic/diff_coexpr/coexpr_diff.<METHOD>.<PRUNE>.tsv  (16 fichiers)
#
# Workflow (3 étapes) :
#   1. Vérifier que adjacencies_<COND>.<METHOD>.csv existe pour chaque
#      méthode. Si arboreto absent (cluster GLiCID, dask ≥ 2025.1) → skip
#      proprement avec un warning et continuer les 3 autres méthodes.
#   2. SLURM array : merge-adjacencies pour chaque (method, prune)
#      → 16 fichiers, ~5 min total (pas de re-grnboost2).
#   3. Lancer run_ablation_grid.sh --version V4.3-method-compare
#      --seeds "1 2 3" (16 × 3 = 48 jobs VGAE).
#
# Strategy recommandée (2 temps) :
#   Phase A : sweep PRUNE sur method=sklearn (4 × 3 = 12 runs) → prune gagnant
#   Phase B : sweep METHOD sur prune-gagnant (4 × 3 = 12 runs) → method gagnant
# Permet d'éviter la grille 16×3=48 et de décorréler les deux effets.
# Sélectionner --only-method/--only-prune pour cibler une phase.
# =============================================================================
set -euo pipefail

DRY_RUN=0
DIFF_DIR="/LAB-DATA/GLiCID/users/${GNN_CLUSTER_USER:-$USER}/gnn/data/pyscenic/diff_coexpr"
ONLY_METHOD=""
ONLY_PRUNE=""
METHODS=(arboreto sklearn)            # corr/mi non générés en routine (V4.3 figé 2026-05-29)
PRUNES=(topk quantile mr)             # zscore exclu (trop d'arêtes, 113k arb / 149k skl, noie le décodeur)
OVERWRITE=0

# Hyperparams par défaut des modes (alignés sur build_diff_coexpr.py).
# NB : MR_K=10 (et non 5) — à K=5 le mode mr perd ASNS/IL6/IL1B des drivers
# Tier-1. K=10 préserve TOUS les drivers V3.3 sur sklearn, perd seulement
# CDKN2A sur arboreto. Cf. analyse adjacencies 2026-05-29.
TOPK=5
TOPK_FLOOR_Q=0.5
QUANTILE=0.98
MR_K=10
Z_THRESH=2.0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)         DRY_RUN=1; shift ;;
        --diff-dir)        DIFF_DIR="$2"; shift 2 ;;
        --only-method)     ONLY_METHOD="$2"; shift 2 ;;
        --only-prune)      ONLY_PRUNE="$2"; shift 2 ;;
        --overwrite)       OVERWRITE=1; shift ;;
        --topk)            TOPK="$2"; shift 2 ;;
        --topk-floor)      TOPK_FLOOR_Q="$2"; shift 2 ;;
        --quantile)        QUANTILE="$2"; shift 2 ;;
        --mr-k)            MR_K="$2"; shift 2 ;;
        --z-thresh)        Z_THRESH="$2"; shift 2 ;;
        -h|--help)
            grep '^# ' "$0" | head -50; exit 0 ;;
        *) echo "Option inconnue : $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$ONLY_METHOD" ]]; then
    METHODS=("$ONLY_METHOD")
fi
if [[ -n "$ONLY_PRUNE" ]]; then
    PRUNES=("$ONLY_PRUNE")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs/coexpr_method_compare_${TS}"
mkdir -p "$LOG_DIR"

# Compat. fichier d'adjacencies : la convention V4.3 est <name>.<method>.csv
# mais on accepte le fichier legacy <name>.csv pour la méthode 'sklearn'
# (rétrocompat. avec ce qui existe déjà sur le cluster).
adj_file() {
    local cond="$1" method="$2"
    local f1="$DIFF_DIR/adjacencies_${cond}.${method}.csv"
    local f2="$DIFF_DIR/adjacencies_${cond}.csv"   # legacy
    if [[ -s "$f1" ]]; then
        echo "$f1"; return 0
    fi
    if [[ "$method" == "sklearn" && -s "$f2" ]]; then
        echo "$f2"; return 0
    fi
    return 1
}

# --- 1. Filtrer les méthodes selon les adjacencies disponibles ---------------
echo "[compare] vérification des adjacencies dans $DIFF_DIR"
AVAILABLE_METHODS=()
for m in "${METHODS[@]}"; do
    ok=1
    for cond in P4 P16; do
        if ! adj_file "$cond" "$m" >/dev/null; then
            echo "  [skip] method=$m : adjacencies_${cond}.${m}.csv absent."
            ok=0; break
        fi
    done
    if [[ $ok -eq 1 ]]; then
        AVAILABLE_METHODS+=("$m")
        echo "  [ok]   method=$m : adjacencies P4+P16 prêts."
    fi
done

if [[ ${#AVAILABLE_METHODS[@]} -eq 0 ]]; then
    echo "[err] aucune méthode disponible. Lancer :" >&2
    echo "  - sklearn  : bash scripts/run_diff_coexpr.sh --step grnboost2" >&2
    echo "  - arboreto : bash scripts/run_grnboost2_diff_arboreto.sh (local)" >&2
    echo "  - corr/mi  : python src/data/preprocess/build_diff_coexpr.py {correlation,mutual-info} ..." >&2
    exit 1
fi

# --- 2. Construire la table des configs (method × prune) ---------------------
CONFIGS_FILE="$LOG_DIR/configs.tsv"
: > "$CONFIGS_FILE"
echo -e "tag\tmethod\tprune\tparams" >> "$CONFIGS_FILE"
for m in "${AVAILABLE_METHODS[@]}"; do
    for p in "${PRUNES[@]}"; do
        case "$p" in
            topk)     params="--prune-mode per-target-topk --per-target-k $TOPK --min-imax-quantile $TOPK_FLOOR_Q" ;;
            quantile) params="--prune-mode global-quantile --top-quantile $QUANTILE" ;;
            mr)       params="--prune-mode mutual-rank --per-target-k $MR_K" ;;
            zscore)   params="--prune-mode z-score --z-thresh $Z_THRESH" ;;
            *) echo "[err] prune inconnu : $p"; exit 1 ;;
        esac
        echo -e "${m}.${p}\t${m}\t${p}\t${params}" >> "$CONFIGS_FILE"
    done
done

N_CONFIGS=$(($(wc -l < "$CONFIGS_FILE") - 1))
ARRAY_MAX=$((N_CONFIGS - 1))
echo "[compare] ${N_CONFIGS} configs (method × prune) :"
column -t -s $'\t' "$CONFIGS_FILE"
echo

# --- 3. sbatch array : un job par (method, prune) ----------------------------
OVERWRITE_FLAG=""
[[ $OVERWRITE -eq 1 ]] && OVERWRITE_FLAG="--overwrite"

SBATCH_SCRIPT="$LOG_DIR/sbatch_compare.sh"
cat > "$SBATCH_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=coexpr_compare
#SBATCH --comment="V4.3 grille méthodes×prune coexpr"
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

LINE_NO=\$((SLURM_ARRAY_TASK_ID + 2))    # ligne 1 = header
LINE=\$(sed -n "\${LINE_NO}p" "$CONFIGS_FILE")
TAG=\$(echo "\$LINE"    | cut -f1)
METHOD=\$(echo "\$LINE" | cut -f2)
PRUNE=\$(echo "\$LINE"  | cut -f3)
PARAMS=\$(echo "\$LINE" | cut -f4)

# Résolution des adjacencies amont (V4.3 ou legacy pour sklearn).
ADJ_P4="$DIFF_DIR/adjacencies_P4.\${METHOD}.csv"
ADJ_P16="$DIFF_DIR/adjacencies_P16.\${METHOD}.csv"
if [[ ! -s "\$ADJ_P4" && "\$METHOD" == "sklearn" ]]; then
    ADJ_P4="$DIFF_DIR/adjacencies_P4.csv"
    ADJ_P16="$DIFF_DIR/adjacencies_P16.csv"
fi

OUT="$DIFF_DIR/coexpr_diff.\${TAG}.tsv"
echo "[\$(date +%T)] [\$TAG] method=\$METHOD prune=\$PRUNE → \$OUT"

# NB chemin script : sur le cluster le code est aplati sous src/ (les
# imports internes utilisent src.data.preprocess.build_diff_coexpr en
# package, mais le CLI direct fonctionne avec le module à plat).
python3 src/build_diff_coexpr.py merge-adjacencies \\
    --adj-p4 "\$ADJ_P4" --adj-p16 "\$ADJ_P16" \\
    \$PARAMS \\
    $OVERWRITE_FLAG \\
    --out "\$OUT"

echo "[\$(date +%T)] \$TAG : sortie générée :"
wc -l "\$OUT"
head -3 "\$OUT"
EOF
chmod +x "$SBATCH_SCRIPT"

echo "[compare] sbatch : $SBATCH_SCRIPT"
echo "[compare] log dir : $LOG_DIR"
echo

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] sbatch $SBATCH_SCRIPT   # array 0-$ARRAY_MAX"
    sed -n '1,40p' "$SBATCH_SCRIPT"
    exit 0
fi

JID=$(sbatch --parsable "$SBATCH_SCRIPT")
echo "[compare] soumis : job array $JID (0-$ARRAY_MAX)"
echo "[compare] suivi  : squeue -j $JID"
echo
echo "[compare] ====== ÉTAPES SUIVANTES ======"
echo "  1. Attendre COMPLETED (~5 min)"
echo "  2. bash scripts/run_ablation_grid.sh --version V4.3-method-compare --seeds \"1 2 3\""
echo "     → un run VGAE par config × 3 seeds"
echo "  3. Cross-seed driver_score, choisir la config gagnante."
