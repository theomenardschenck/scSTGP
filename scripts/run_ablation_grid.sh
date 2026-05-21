#!/usr/bin/env bash
# =============================================================================
# run_ablation_grid.sh — Job array SLURM pour grilles d'ablation VGAE
# =============================================================================
# Soumet UN seul job array SLURM. Le cluster (Nautilus, GLiCID) gère le
# parallélisme via `--array=0-(N-1)%MAX_PARALLEL`. Tu peux fermer la
# session après soumission, c'est SLURM qui pilote.
#
# Interface : produit cartésien `--version × --ablations`.
#
# --version {V3, V4, V4.1, V4.2, V4.2-sweep, V4_1, V4_2}   (défaut : V4.1)
#     Définit les configs de base à tester.
#         V3   : 1 config (full, no OmniPath, no flags). Reproduit V3.6 baseline.
#         V4   : 4 configs avec edge_types `signaling` / `tf_curated` opt-in.
#                tags v4-{baseline,sig,tf,full}.
#         V4.1 : V4 + `--include-omnipath-genes` (étend gene_to_idx avec
#                endpoints OmniPath). tags v4.1-{baseline,sig,tf,full}.
#         V4.2 : 5 configs sur base V4.1-full + 3 leviers toggleable :
#                coexpr différentielle P4∪P16, γ_t par edge_type,
#                Reactome FI. tags v4.2-{full,coexdiff,coexdiff.gw,
#                coexdiff.rfi,coexdiff.gw.rfi}. PRÉREQUIS :
#                bash scripts/run_diff_coexpr.sh + cache_reactome_fi.sh.
#                Cf. §14bis.6octies du rapport.
#         V4.2-sweep : 3 configs comparant l'élagage du canal coexpr
#                (constat 2026-05-19 : AUC 0.91 vs V4.1 0.97 — coexpr
#                trop dense noie le décodeur). tags v4.2-coex.{k5q5,
#                k3q7,k2q8}. PRÉREQUIS : bash scripts/run_coexpr_prune_sweep.sh
#                (génère les 3 coexpr_diff.*.tsv depuis adjacencies existants).
#                Cf. §14bis.6sexdecies du rapport.
#
# --ablations <list>     (défaut : "" = juste les 4 configs base, AKA validation)
#     Liste séparée par virgules. Chaque entrée peut être :
#       (a) une ablation simple : no-coexpr, no-humess, no-reactome, no-ppi,
#           no-scenic-regulons, no-cgrp, ex-degrees
#       (b) composite par "+"      : no-coexpr+no-humess → flags cumulés sur 1 config
#       (c) pseudo "all-standard"  : applique chaque des 7 ablations standards
#           SÉPARÉMENT, restreint au sub-config `*-full` (sinon ça explose).
#       (d) pseudo "all-other-standard" : idem mais sans no-coexpr et no-humess
#           (= no-reactome, no-ppi, no-scenic-regulons, no-cgrp, ex-degrees)
#
# Cardinalité produite :
#     # Configs = (4 si V4/V4.1, 1 si V3) × (n entrées dans --ablations OU 1)
#     # Tâches  = # Configs × n_seeds
#
# Exemples (les 4 cas demandés sur V4.1, 3 seeds chacun) :
#     # 1. 3 seeds no-coexpr seul → 4 base × 1 ablation × 3 seeds = 12 tâches
#     bash scripts/run_ablation_grid.sh --version V4.1 --ablations no-coexpr \
#         --seeds "1 2 3"
#
#     # 2. 3 seeds no-humess seul → 12 tâches
#     bash scripts/run_ablation_grid.sh --version V4.1 --ablations no-humess \
#         --seeds "1 2 3"
#
#     # 3. 3 seeds no-coexpr ET no-humess (composite, flags cumulés) → 12 tâches
#     bash scripts/run_ablation_grid.sh --version V4.1 --ablations "no-coexpr+no-humess" \
#         --seeds "1 2 3"
#
#     # 4. 3 seeds des 5 autres ablations standard (no-reactome / no-ppi /
#     #    no-scenic-regulons / no-cgrp / ex-degrees), appliquées sur le
#     #    sub-config `v4.1-full` uniquement → 5 × 3 = 15 tâches.
#     bash scripts/run_ablation_grid.sh --version V4.1 --ablations all-other-standard \
#         --seeds "1 2 3"
#
# Validation de version (rejoue la phase 1) :
#     bash scripts/run_ablation_grid.sh --version V4.1 --seeds "1 2 3"
#     # = 4 configs × 3 seeds = 12 tâches, pas d'ablation extra
#
# Aliases backward-compat :
#     --mode v4_validation     == --version V4    --ablations ""
#     --mode v4_1_validation   == --version V4.1  --ablations ""
#     --mode v4_no_coexpr      == --version V4    --ablations no-coexpr
#     --mode v3_legacy         == --version V3    --ablations all-standard
#     --mode full_grid         == complexe — gardé via le case legacy
#
# Pré-requis : cwd = racine du projet (gnn_huvec/).
#              data/omnipath/ doit contenir tf_collectri.tsv.gz +
#              signed_ppi_signor.tsv.gz pour V4/V4.1.
# Doc GLiCID : https://doc.glicid.fr/GLiCID-PUBLIC/quickstart_advanced_user.html
# =============================================================================

set -euo pipefail

# --- Paramètres par défaut --------------------------------------------------
DEFAULT_SEEDS=(1 2 3)
VERSION="V4.1"
ABLATIONS=""
MODE=""                        # legacy alias
MAX_PARALLEL=10
TIME_LIMIT="0-02:00:00"
DRY_RUN=0
DATA_ROOT_CLI=""               # --data-root (override DATA_ROOT)

# --- Parsing CLI ------------------------------------------------------------
SEEDS=("${DEFAULT_SEEDS[@]}")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --version)        VERSION="$2"; shift 2 ;;
        --ablations)      ABLATIONS="$2"; shift 2 ;;
        --mode)           MODE="$2"; shift 2 ;;   # legacy
        --seeds)          IFS=' ' read -r -a SEEDS <<< "$2"; shift 2 ;;
        --max-parallel)   MAX_PARALLEL="$2"; shift 2 ;;
        --time)           TIME_LIMIT="$2"; shift 2 ;;
        --data-root)      DATA_ROOT_CLI="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,75p' "$0"; exit 0 ;;
        *) echo "Argument inconnu : $1"; exit 1 ;;
    esac
done

# --- Backward-compat : si --mode passé, remap sur --version + --ablations --
case "$MODE" in
    "")                ;;
    v4_validation)     VERSION="V4";    ABLATIONS="" ;;
    v4_1_validation)   VERSION="V4.1";  ABLATIONS="" ;;
    v4_no_coexpr)      VERSION="V4";    ABLATIONS="no-coexpr" ;;
    v3_legacy)         VERSION="V3";    ABLATIONS="all-standard" ;;
    full_grid)         VERSION="V4.1";  ABLATIONS="all-standard" ;;
    *) echo "Mode legacy inconnu : $MODE"; exit 1 ;;
esac

# Normalisation : V4_1 → V4.1 / V4_2 → V4.2 pour éviter pièges shell
[[ "$VERSION" == "V4_1" ]] && VERSION="V4.1"
[[ "$VERSION" == "V4_2" ]] && VERSION="V4.2"
[[ "$VERSION" == "V4_2-sweep" || "$VERSION" == "V4_2_sweep" ]] && VERSION="V4.2-sweep"

# --- Définition des sub-configs par version ---------------------------------
# Format : "<tag>::<flags>" — flags passés tels quels à gnn_vgae.py.
# Le sub-tag est COMBINÉ avec un tag d'ablation pour former le RUN_TAG final.
OP_SIG="--use-omnipath-signaling"
OP_TF="--use-omnipath-tf-curated"
OP_INC="--include-omnipath-genes"

case "$VERSION" in
    V3)
        BASE_CONFIGS=( "v3-full::" )
        ;;
    V4)
        BASE_CONFIGS=(
            "v4-baseline::"
            "v4-sig::$OP_SIG"
            "v4-tf::$OP_TF"
            "v4-full::$OP_SIG $OP_TF"
        )
        ;;
    V4.1)
        BASE_CONFIGS=(
            "v4.1-baseline::"
            "v4.1-sig::$OP_SIG $OP_INC"
            "v4.1-tf::$OP_TF $OP_INC"
            "v4.1-full::$OP_SIG $OP_TF $OP_INC"
        )
        ;;
    V4.2)
        # V4.2 = V4.1-full + 3 leviers toggleable (cf. §14bis.6octies).
        # Prérequis : coexpr_diff.tsv (bash scripts/run_diff_coexpr.sh)
        # + cache Reactome FI (bash scripts/cache_reactome_fi.sh).
        V41_FULL="$OP_SIG $OP_TF $OP_INC"
        COEXDIFF="--coexpr-mode differential"
        GW="--edge-type-weights ppi=0.1,coexpression=0.5"
        RFI="--use-reactome-fi"
        BASE_CONFIGS=(
            "v4.2-full::$V41_FULL"
            "v4.2-coexdiff::$V41_FULL $COEXDIFF"
            "v4.2-coexdiff.gw::$V41_FULL $COEXDIFF $GW"
            "v4.2-coexdiff.rfi::$V41_FULL $COEXDIFF $RFI"
            "v4.2-coexdiff.gw.rfi::$V41_FULL $COEXDIFF $GW $RFI"
        )
        ;;
    V4.2-sweep)
        # Sweep d'élagage coexpr (cf. §14bis.6sexdecies). Run V4.2 du
        # 2026-05-19 : AUC 0.9051 (vs V4.1 0.97). Hypothèse : K=5 / floor
        # q=0.5 / universe=all = trop d'arêtes coexpr (~150-200k, 1.5×PPI)
        # qui noient le signal PPI signé dans le décodeur.
        # 3 configs avec --diff-coexpr-file vers fichiers pré-générés par
        # run_coexpr_prune_sweep.sh (depuis MÊMES adjacencies, juste re-merge).
        V41_FULL="$OP_SIG $OP_TF $OP_INC"
        COEXDIFF="--coexpr-mode differential"
        DD_BASE="data/pyscenic/diff_coexpr/coexpr_diff"
        BASE_CONFIGS=(
            "v4.2-coex.k5q5::$V41_FULL $COEXDIFF --diff-coexpr-file ${DD_BASE}.k5q5.tsv"
            "v4.2-coex.k3q7::$V41_FULL $COEXDIFF --diff-coexpr-file ${DD_BASE}.k3q7.tsv"
            "v4.2-coex.k2q8::$V41_FULL $COEXDIFF --diff-coexpr-file ${DD_BASE}.k2q8.tsv"
        )
        ;;
    *)
        echo "Version inconnue : $VERSION (V3 | V4 | V4.1 | V4.2 | V4.2-sweep)"; exit 1 ;;
esac

# --- Parsing de --ablations en variantes -----------------------------------
# Chaque variante = "<tag>::<flags>". Une variante vide ("::") = pas d'ablation.
# Pour les pseudos all-* on restreint plus tard aux sub-configs `*-full`.
declare -a ABL_VARIANTS=()
RESTRICT_TO_FULL=0   # vrai pour all-standard / all-other-standard

# Helper : mappe une ablation simple → ses flags CLI
_abl_flags() {
    case "$1" in
        no-coexpr)          echo "--no-coexpr" ;;
        no-humess)          echo "--no-humess" ;;
        no-reactome)        echo "--no-reactome" ;;
        no-ppi)             echo "--no-ppi" ;;
        no-scenic-regulons) echo "--no-scenic-regulons" ;;
        no-cgrp)            echo "--no-cell-group-edges" ;;
        ex-degrees)         echo "--exclude-features ppi_degree,reg_degree" ;;
        "")                 echo "" ;;
        *) echo "__INVALID__" ;;
    esac
}

case "$ABLATIONS" in
    ""|none)
        ABL_VARIANTS=( "::" )
        ;;
    all-standard)
        RESTRICT_TO_FULL=1
        for abl in no-coexpr no-humess no-reactome no-ppi no-scenic-regulons no-cgrp ex-degrees; do
            ABL_VARIANTS+=( "${abl}::$(_abl_flags "$abl")" )
        done
        ;;
    all-other-standard)
        RESTRICT_TO_FULL=1
        for abl in no-reactome no-ppi no-scenic-regulons no-cgrp ex-degrees; do
            ABL_VARIANTS+=( "${abl}::$(_abl_flags "$abl")" )
        done
        ;;
    *)
        # Custom : split par ',' → variantes séparées.
        # Chaque variante peut elle-même être composite via '+'.
        IFS=',' read -ra _ABL_LIST <<< "$ABLATIONS"
        for v in "${_ABL_LIST[@]}"; do
            tag="$v"          # ex : "no-coexpr+no-humess"
            flags=""
            IFS='+' read -ra _COMPS <<< "$v"
            for c in "${_COMPS[@]}"; do
                f="$(_abl_flags "$c")"
                if [[ "$f" == "__INVALID__" ]]; then
                    echo "Ablation inconnue : '$c' (dans '$v')."
                    echo "Valides : no-coexpr no-humess no-reactome no-ppi no-scenic-regulons no-cgrp ex-degrees"
                    echo "Pseudos : all-standard all-other-standard"
                    exit 1
                fi
                flags="${flags}${flags:+ }${f}"
            done
            ABL_VARIANTS+=( "${tag}::${flags}" )
        done
        ;;
esac

# --- Produit cartésien BASE × ABL_VARIANTS → CONFIGS finales ---------------
CONFIGS=()
for base in "${BASE_CONFIGS[@]}"; do
    base_tag="${base%%::*}"
    base_flags="${base#*::}"
    for av in "${ABL_VARIANTS[@]}"; do
        abl_tag="${av%%::*}"
        abl_flags="${av#*::}"
        # Restriction all-* : seules les configs *-full passent
        if [[ $RESTRICT_TO_FULL -eq 1 && "$base_tag" != *"-full" ]]; then
            continue
        fi
        # Compose tag final ; on saute le "+" si pas d'ablation
        if [[ -n "$abl_tag" ]]; then
            tag="${base_tag}+${abl_tag}"
        else
            tag="${base_tag}"
        fi
        # Compose flags (squeeze whitespace)
        flags="${base_flags} ${abl_flags}"
        flags="$(echo $flags | xargs -n1 | xargs)"   # trim multiple spaces
        CONFIGS+=( "${tag}::${flags}" )
    done
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "[grid] aucune config produite. Vérifie --version et --ablations."
    exit 1
fi

# --- Préparation logs + configs.tsv ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# DATA_ROOT : racine des données d'ENTRÉE. DOIT correspondre à ce que
# gnn_vgae.py lit réellement (cf. gnn_vgae.py:381-382, BASE_DIR/data sur
# LAB-DATA). PROJECT_DIR est sur /scratch (code), mais les données
# (omnipath, pyscenic, reactome_fi) sont sur /LAB-DATA → le garde-fou
# DOIT vérifier LAB-DATA, pas $PROJECT_DIR/data. Override : --data-root
# ou env GNN_DATA_ROOT.
# Précédence : --data-root > env GNN_DATA_ROOT > défaut (= gnn_vgae.py:381)
DATA_ROOT="${DATA_ROOT_CLI:-${GNN_DATA_ROOT:-/LAB-DATA/GLiCID/users/USER@univ-nantes.fr/gnn/data}}"
TS="$(date +%Y%m%d_%H%M%S)"
# Slug court pour identifier le run dans le LOG_DIR
ABL_SLUG="${ABLATIONS//,/_}"
ABL_SLUG="${ABL_SLUG//+/-}"
[[ -z "$ABL_SLUG" ]] && ABL_SLUG="validation"
LOG_DIR="$SCRIPT_DIR/logs/ablation_${VERSION}_${ABL_SLUG}_${TS}"
mkdir -p "$LOG_DIR"
CONFIGS_FILE="$LOG_DIR/configs.tsv"

> "$CONFIGS_FILE"
for cfg in "${CONFIGS[@]}"; do
    tag="${cfg%%::*}"
    flags="${cfg#*::}"
    for seed in "${SEEDS[@]}"; do
        printf '%s\t%s\t%s\n' "$tag" "$seed" "$flags" >> "$CONFIGS_FILE"
    done
done

N_TASKS=$(wc -l < "$CONFIGS_FILE")
ARRAY_RANGE="0-$((N_TASKS - 1))%${MAX_PARALLEL}"

echo "[grid] version   : $VERSION"
echo "[grid] ablations : ${ABLATIONS:-<none>}"
echo "[grid] project   : $PROJECT_DIR"
echo "[grid] log dir   : $LOG_DIR"
echo "[grid] configs   : $CONFIGS_FILE (${#CONFIGS[@]} configs × ${#SEEDS[@]} seeds = $N_TASKS tâches)"
echo "[grid] array     : $ARRAY_RANGE   (max $MAX_PARALLEL en //)"
echo "[grid] time      : $TIME_LIMIT par tâche"
echo "[grid] seeds     : ${SEEDS[*]}"
echo

# --- Sécurité : prérequis vérifiés sur DATA_ROOT (= ce que gnn_vgae.py
#     lit réellement, LAB-DATA), PAS $PROJECT_DIR/data (scratch, code) ---
echo "[grid] data root : $DATA_ROOT  (doit = gnn_vgae.py BASE_DIR/data)"
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "[warn] DATA_ROOT introuvable : $DATA_ROOT"
    echo "       Vérifie qu'il correspond à gnn_vgae.py:381 (LAB_DIR/gnn/data)."
    echo "       Override : --data-root <path> ou export GNN_DATA_ROOT=<path>"
fi
if [[ "$VERSION" == "V4" || "$VERSION" == "V4.1" || "$VERSION" == "V4.2" || "$VERSION" == "V4.2-sweep" ]]; then
    CACHE_TF="$DATA_ROOT/omnipath/tf_collectri.tsv.gz"
    CACHE_SIG="$DATA_ROOT/omnipath/signed_ppi_signor.tsv.gz"
    if [[ ! -f "$CACHE_TF" ]]; then
        echo "[warn] cache OmniPath TF absent : $CACHE_TF"
        echo "       lance d'abord : python scripts/cache_omnipath.py --cache-dir $DATA_ROOT/omnipath"
    fi
    if [[ ! -f "$CACHE_SIG" ]]; then
        echo "[warn] cache OmniPath SIGNOR absent : $CACHE_SIG"
        echo "       (les arêtes signaling seront vides — runs +sig dégradés)"
    fi
fi

# --- Sécurité V4.2 : prérequis coexpr différentiel + Reactome FI -----------
if [[ "$VERSION" == "V4.2" ]]; then
    DIFF_FILE="$DATA_ROOT/pyscenic/diff_coexpr/coexpr_diff.tsv"
    RFI_FILE="$DATA_ROOT/reactome_fi/FIsInGene_with_annotations.txt"
    if [[ ! -f "$DIFF_FILE" ]]; then
        echo "[err] coexpr_diff.tsv absent : $DIFF_FILE"
        echo "      Lance d'abord (workflow en 2 étapes découplées) :"
        echo "        1. python src/preprocess/build_diff_coexpr.py extract-matrices \\"
        echo "             --gene-universe graph --graph-genes <cross_seed_gene_ranking.tsv>"
        echo "        2. bash scripts/run_diff_coexpr.sh --step grnboost2"
        echo "        3. (attendre) bash scripts/run_diff_coexpr.sh --step merge"
        exit 1
    fi
    if [[ ! -f "$RFI_FILE" ]]; then
        echo "[warn] Reactome FI absent : $RFI_FILE"
        echo "       Les configs *.rfi auront un edge_type reactome_fi vide."
        echo "       Lance : bash scripts/cache_reactome_fi.sh $DATA_ROOT/reactome_fi"
    fi
fi

# --- Sécurité V4.2-sweep : 3 fichiers coexpr_diff.{k5q5,k3q7,k2q8}.tsv ----
if [[ "$VERSION" == "V4.2-sweep" ]]; then
    DD="$DATA_ROOT/pyscenic/diff_coexpr"
    missing=0
    for v in k5q5 k3q7 k2q8; do
        F="$DD/coexpr_diff.${v}.tsv"
        if [[ ! -f "$F" ]]; then
            echo "[err] coexpr_diff.${v}.tsv absent : $F"
            missing=1
        fi
    done
    if [[ $missing -eq 1 ]]; then
        echo "      Lance d'abord : bash scripts/run_coexpr_prune_sweep.sh"
        exit 1
    fi
fi

# --- Génération sbatch -----------------------------------------------------
SBATCH_SCRIPT="$LOG_DIR/sbatch_ablation.sh"
cat > "$SBATCH_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=vgae_${VERSION}_${ABL_SLUG}
#SBATCH --comment="VGAE ablation ${VERSION} ${ABL_SLUG}"
#SBATCH --output=$LOG_DIR/%x_%A_%a.out
#SBATCH --error=$LOG_DIR/%x_%A_%a.err
#SBATCH --time=$TIME_LIMIT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --qos=quick
#SBATCH --array=$ARRAY_RANGE

set -euo pipefail

cd "$PROJECT_DIR"

LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIGS_FILE")
TAG=\$(echo "\$LINE" | cut -f1)
SEED=\$(echo "\$LINE" | cut -f2)
FLAGS=\$(echo "\$LINE" | cut -f3)
RUN_TAG="\${TAG}.s\${SEED}"

echo "[\$(date +%T)] task \$SLURM_ARRAY_TASK_ID : tag=\$TAG seed=\$SEED run_tag=\$RUN_TAG"
echo "[\$(date +%T)] flags: \$FLAGS"
echo "[\$(date +%T)] node : \$(hostname)"

# shellcheck disable=SC2086
# NB cluster Nautilus : tous les .py sont déployés à plat sous src/
# (pas de sous-dossiers gnn/, perturbation/, validation/ comme en local).
python3 src/gnn_vgae.py \\
    --run-tag "\$RUN_TAG" \\
    --seed "\$SEED" \\
    \$FLAGS

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
    cat "$CONFIGS_FILE"
    echo
    echo "[DRY-RUN] Aperçu du sbatch script (head) :"
    head -25 "$SBATCH_SCRIPT"
    exit 0
fi

JOB_ID=$(sbatch --parsable "$SBATCH_SCRIPT")
echo "[grid] soumis : job array $JOB_ID ($N_TASKS tâches, ${MAX_PARALLEL} max en //)"
echo "[grid] suivi   : squeue -j $JOB_ID    (ou squeue -u \$USER)"
echo "[grid] cancel  : scancel $JOB_ID"
echo "[grid] logs    : $LOG_DIR/vgae_${VERSION}_${ABL_SLUG}_${JOB_ID}_<task>.{out,err}"
