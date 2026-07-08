#!/usr/bin/env bash
# =============================================================================
# run_perturbation_grid.sh — Job array SLURM pour les perturbations VGAE
# =============================================================================
# Découvre les run_dirs entraînés (auto-discovery via glob) et soumet UN job
# array sbatch — le cluster gère le parallélisme.
#
# --axis {v3,v4,both,effector} bascule l'axe de sénescence :
#   v3       : P4-only (axe V3.x)                       → suffixe vide (legacy)
#   v4       : P4+P16_cluster_0 (c0 quiescent-like)     → _axisV4
#   both     : v3 ET v4 (2 perturbations / run, fichiers distincts)
#   effector : axe effecteur-ancré unit(μ[pro]−μ[anti]) → _axisEffector
#              (référence causale stable, ne pivote pas avec les hyperparams,
#               cf. gnn_futur §2.2/§7.8 ; pôles via env EFFECTOR_PRO/ANTI).
#   multi    : axe global V4 (P4+c0) + axes P4→chaque cluster + transitions
#              inter-cluster c0→c1→c2→c3 (LOG #log-multiaxis) → _axisMULTI.
#              driver_score PAR AXE côté cross-seed (cross_seed_gene_ranking__
#              <axis>.tsv + long-format). Réglages via env :
#                MULTI_TRANSITIONS  (default|all-pairs ; défaut default)
#                MULTI_ANCHOR_MODE  (none|de-markers|manual ; défaut none —
#                  'manual' requiert data/gnn_data/ahn_cluster_anchors.tsv).
#                MULTI_CACHE_DZ     (1|0 ; défaut 1) — persiste le cache Δz →
#                  re-projection d'axes ultérieure sans re-perturber
#                  (reproject_axes.py).
# Avec --axis both, chaque run est perturbé deux fois (--out-suffix distinct)
# → compare V3.7 (axe V4) à V3.6 (axe V3) sur le MÊME modèle entraîné.
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
# Chaînage automatique :
#   1. Après chaque perturb_top_genes : `src/perturb_report.py --all` →
#      report par seed dans report/ (ou report_axisV4/ selon l'axe).
#      Désactive avec --skip-report.
#   2. Après que TOUTES les perturbations soient finies (dépendance SLURM
#      afterok) : `src/perturb_report.py --cross-seed` par catégorie de
#      modèles. Une catégorie = même tag sans le seed (v4-full.s1, .s2,
#      .s3 → catégorie "v4-full"). Le cross-seed est lancé pour chaque
#      (catégorie × axe) avec --axis-tag pour ne pas mélanger V3 et V4.
#      Sortie : `<OUT_DIR_BASE>/cross_seed_<category>[_axisV4]/`.
#      Désactive avec --skip-cross-seed.
#
# Usage générique :
#   bash scripts/run_perturbation_grid.sh                          # axe V3 (legacy)
#   bash scripts/run_perturbation_grid.sh --axis effector --pattern '<base>/v5.4.*.s*'
#   bash scripts/run_perturbation_grid.sh --modes "knockout knockdown overexpress"
#   bash scripts/run_perturbation_grid.sh --runs "V3.3_Run1 V3.3_Run2"
#   bash scripts/run_perturbation_grid.sh --pattern '<base>/no-humess.s*'
#   bash scripts/run_perturbation_grid.sh --also-pathways
#   bash scripts/run_perturbation_grid.sh --dry-run
#   bash scripts/run_perturbation_grid.sh --max-parallel 5
#   bash scripts/run_perturbation_grid.sh --time 1-00:00:00
#   bash scripts/run_perturbation_grid.sh --skip-report             # pas de report/ par seed
#   bash scripts/run_perturbation_grid.sh --skip-cross-seed          # pas de cross_seed_<cat>/
#
# Doc GLiCID : https://doc.glicid.fr/GLiCID-PUBLIC/quickstart_advanced_user.html
# =============================================================================

set -euo pipefail

# Shebang bash robuste pour les jobs (GLiCID/Guix : `/usr/bin/env bash` n'est
# PAS résolu côté nœud de calcul, et `/bin/bash` est absent — seul `/bin/sh`
# existe). Le corps des jobs utilise des tableaux bash → on bake le chemin
# absolu de bash résolu sur le frontal (FS partagé Guix/scratch ⇒ valide sur
# les nœuds). Override : env BASH_BIN.
BASH_BIN="${BASH_BIN:-$(command -v bash || true)}"
# Guix : déréférencer le symlink de profil (node-local, ex.
# /run/current-system/profile/bin/bash) vers le vrai chemin /gnu/store/… qui,
# lui, est partagé sur tous les nœuds (sinon `bad interpreter` côté calcul).
[[ -n "$BASH_BIN" ]] && BASH_BIN="$(readlink -f "$BASH_BIN" 2>/dev/null || echo "$BASH_BIN")"
[[ -z "$BASH_BIN" ]] && BASH_BIN="/bin/bash"

# --- Paramètres par défaut --------------------------------------------------
OUT_DIR_BASE_DEFAULT="/scratch/nautilus/users/USER@univ-nantes.fr/gnn_vgae"
OUT_DIR_BASE="${OUT_DIR_BASE:-$OUT_DIR_BASE_DEFAULT}"
DEFAULT_PATTERN="${OUT_DIR_BASE}/*.s*"

DEFAULT_MODES=(knockout knockdown overexpress)   # 3 modes — cohérent avec V3.3 cross-seed
DEFAULT_AXIS="v4"           # backward-compat : axe historique P4 quiescent
MAX_PARALLEL=10
TIME_LIMIT="0-12:00:00"
DRY_RUN=0
ANALYSIS=0
ANALYSIS_DECOY=0
ANALYSIS_FIGURES=0
GPU=0
ALSO_PATHWAYS=0
SKIP_REPORT=0               # par défaut : on chaîne perturb_report --all après
SKIP_CROSS_SEED=0           # par défaut : cross-seed report auto post-perturb
# V5.5 — sémantique KO. mask (DÉFAUT) = feature=0 sans coupure d'arêtes
# (corrige le bug KO-cut : choc topologique). cut = legacy (coupe arêtes).
# soft = feature × ko_soft_factor. NB : perturb_top_genes applique aussi en
# V5.5 un KD soft (×0.15) par défaut (le grid ne plumbe pas --legacy ; pour
# un run 100% legacy, appeler perturb_top_genes.py directement avec --legacy).
KO_MODE="mask"
KO_SOFT_FACTOR="0.1"

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
        --skip-cross-seed) SKIP_CROSS_SEED=1; shift ;;
        --ko-mode)        KO_MODE="$2"; shift 2 ;;
        --ko-soft-factor) KO_SOFT_FACTOR="$2"; shift 2 ;;
        --analysis)         ANALYSIS=1; shift ;;       # ne lance QUE l'analyse (local)
        --analysis-decoy)   ANALYSIS_DECOY=1; shift ;; # + confidence décoy (lourd)
        --analysis-figures) ANALYSIS_FIGURES=1; shift ;; # + figures (visualize_global/viz_explorer)
        --gpu)              GPU=1; shift ;;             # essaie GPU (gres+partition) ; python --device auto
        -h|--help)
            sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "Argument inconnu : $1"; exit 1 ;;
    esac
done

# --- Résolution de l'axe → quiescent_groups + suffixe (+ flags extra) --------
# AXIS_SPECS est un tableau parallèle. Format d'une entrée :
#     "<suffixe>::<quiescent_groups>[::<flags extra perturb_top_genes>]"
# Le 3e champ (optionnel) est passé tel quel à perturb_top_genes.py — c'est
# ainsi que l'axe `effector` injecte `--effector-axis` (axis = unit(μ[pro]−
# μ[anti]), référence causale stable, cf. gnn_futur §2.2/§7.8) là où v3/v4
# définissent l'axe par le contraste de cell_groups (--quiescent-groups).
# Avec axis=both, on en a 2 (V3 et V4).
#
# Axe effecteur : pôles surchargés via env EFFECTOR_PRO / EFFECTOR_ANTI
# (sinon défauts de perturb_top_genes : pro=CDKN2A,CDKN1A,CDKN2B,SERPINE1,GLB1
#  anti=LMNB1,MKI67,CCNB1,CCNA2,PCNA,TOP2A). --quiescent-groups reste passé
# (sert au baseline cell-group ; l'axe lui-même est remplacé par les effecteurs).
EFF_FLAGS="--effector-axis"
[[ -n "${EFFECTOR_PRO:-}"  ]] && EFF_FLAGS+=" --effector-pro ${EFFECTOR_PRO}"
[[ -n "${EFFECTOR_ANTI:-}" ]] && EFF_FLAGS+=" --effector-anti ${EFFECTOR_ANTI}"

# Axe multi-état : global V4 + P4→cluster + transitions. Le mode d'ancrage des
# centroïdes (none/de-markers/manual) et le jeu de transitions sont pilotés par
# env pour ne pas multiplier les cas. Défaut = none (axe global V4 INCHANGÉ, on
# ne fait qu'AJOUTER les sous-axes → comparaison propre).
MULTI_FLAGS="--transition-axes ${MULTI_TRANSITIONS:-default}"
[[ -n "${MULTI_ANCHOR_MODE:-}" ]] && MULTI_FLAGS+=" --cluster-anchor-mode ${MULTI_ANCHOR_MODE}"
# Persiste le cache Δz (dz_mean + w_diff_sum, axis-indépendant) → tout NOUVEL
# axe (ancres, transitions, cross-dataset) devient une simple re-projection
# via reproject_axes.py, SANS re-perturber. Désactivable : MULTI_CACHE_DZ=0.
[[ "${MULTI_CACHE_DZ:-1}" != "0" ]] && MULTI_FLAGS+=" --cache-delta-z"

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
    effector)
        AXIS_SPECS=("axisEffector::P4,P16_cluster_0::${EFF_FLAGS}")
        ;;
    multi|multiaxis)
        AXIS_SPECS=("axisMULTI::P4,P16_cluster_0::${MULTI_FLAGS}")
        ;;
    *)
        echo "Axis inconnu : $AXIS (v3 | v4 | both | effector | multi)"; exit 1 ;;
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

# --- Mode --analysis : ne lance QUE la partie ANALYSE (local, aucun SLURM) ---
# Requiert des runs DÉJÀ perturbés. Groupe par catégorie (tag sans .s<N>) et
# enchaîne run_analysis.sh (cross-seed report + baselines + interpret [+ décoy]).
if [[ $ANALYSIS -eq 1 ]]; then
    case "$AXIS" in v3) ATAG="";; effector) ATAG="axisEffector";; multi|multiaxis) ATAG="axisMULTI";; *) ATAG="axisV4";; esac
    EXTRA_FLAGS=""
    [[ $ANALYSIS_DECOY -eq 1 ]] && EXTRA_FLAGS+=" --decoy"
    [[ $ANALYSIS_FIGURES -eq 1 ]] && EXTRA_FLAGS+=" --figures"
    declare -A A_CATS=()
    for r in "${VALID_RUNS[@]}"; do
        nm="$(basename "$r")"; A_CATS["${nm%.s*}"]+="$r "
    done
    for cat in "${!A_CATS[@]}"; do
        runs=(${A_CATS[$cat]})
        out="$(dirname "${runs[0]}")/analysis/$cat"
        echo "[grid][analysis] $cat : ${#runs[@]} seed(s) → $out"
        bash scripts/run_analysis.sh --out "$out" --axis-tag "$ATAG" \
            $EXTRA_FLAGS --seeds "${runs[@]}" || echo "[grid][analysis] $cat ÉCHEC"
    done

    # --- Cross-config (≥2 catégories) : ablation_attribution + compare_runs ---
    # Référence = catégorie 'baseline' (sinon la 1ère) ; sorties sous son interpret/.
    cats=("${!A_CATS[@]}")
    if [[ ${#cats[@]} -ge 2 ]]; then
        f0=(${A_CATS[${cats[0]}]}); APAR="$(dirname "${f0[0]}")/analysis"
        REF=""; for c in "${cats[@]}"; do [[ "$c" == *baseline* ]] && REF="$c" && break; done
        [[ -z "$REF" ]] && REF="${cats[0]}"
        ABL=(); for c in "${cats[@]}"; do [[ "$c" != "$REF" ]] && ABL+=("$APAR/$c"); done
        REFINT="$APAR/$REF/interpret"
        echo "[grid][analysis] cross-config : réf=$REF vs ${#ABL[@]} ablation(s) → $REFINT"
        python3 src/validation/reports/ablation_attribution.py \
            --reference "$APAR/$REF" --ablations "${ABL[@]}" \
            --out-dir "$REFINT/ablation_attribution" || echo "  ablation_attribution ÉCHEC"
        python3 src/validation/reports/compare_runs.py --source perturb_driver \
            --base-dir "$APAR" --report-subdir "" --runs "${cats[@]}" \
            --baseline "$REF" --output-dir "$REFINT/compare_runs" || echo "  compare_runs ÉCHEC"
    fi
    echo "[grid][analysis] terminé (catégories : ${!A_CATS[*]} ; aucun job SLURM)."
    exit 0
fi

# --- Préparation logs + configs.tsv ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs/perturb_${AXIS}_${TS}"
mkdir -p "$LOG_DIR"
CONFIGS_FILE="$LOG_DIR/configs.tsv"

# Une ligne par tâche :
#   run_dir<TAB>mode<TAB>target_flag<TAB>suffixe<TAB>quiescent_groups<TAB>axis_extra
> "$CONFIGS_FILE"
for run_dir in "${VALID_RUNS[@]}"; do
    for mode in "${MODES[@]}"; do
        for spec in "${AXIS_SPECS[@]}"; do
            sfx="${spec%%::*}"          # "axisV4" / "axisEffector" / ""
            rest="${spec#*::}"          # "<qg>[::<extra>]"
            qg="${rest%%::*}"           # "P4,P16_cluster_0" ou "P4"
            # 3e champ optionnel = flags extra (ex. --effector-axis). Word-split
            # voulu côté python → écrit non-quoté dans configs.tsv.
            axis_extra=""
            [[ "$rest" == *::* ]] && axis_extra="${rest#*::}"
            # Le suffixe d'output prend une underscore prefix si non vide
            out_sfx=""
            [[ -n "$sfx" ]] && out_sfx="_${sfx}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$run_dir" "$mode" "--all-genes" "$out_sfx" "$qg" "$axis_extra" >> "$CONFIGS_FILE"
            if [[ $ALSO_PATHWAYS -eq 1 ]]; then
                printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$run_dir" "$mode" "--all-pathways" "$out_sfx" "$qg" "$axis_extra" >> "$CONFIGS_FILE"
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
# GPU optionnel (--gpu) : directives gres+partition+clusters (GLiCID) + python
# --device auto. ⚠️ qos GPU 'gpus' = MaxWall 3h + MaxJobsPU 1 → le job array se
# SÉRIALISE (1 tâche GPU à la fois). Pour du parallélisme GPU, préférer 1 job
# par config (run_v6_de_axis.sh) ou la qos 'quick' (3h, 50 jobs).
GPU_DIRECTIVES=""
DEVICE_ARG=""
QOS="short"
if [[ ${GPU:-0} -eq 1 ]]; then
    QOS="${V6_GPU_QOS:-gpus}"
    TIME_LIMIT="${V6_GPU_TIME:-0-03:00:00}"        # qos gpus plafonne à 3h
    GPU_DIRECTIVES="#SBATCH --partition=${V6_GPU_PARTITION:-gpu}
#SBATCH --gres=${V6_GPU_GRES:-gpu:1}
#SBATCH --clusters=${V6_CLUSTER:-nautilus}"
    [[ -n "${V6_GPU_CONSTRAINT:-}" ]] && GPU_DIRECTIVES+="
#SBATCH --constraint=${V6_GPU_CONSTRAINT}"
    DEVICE_ARG="--device auto"
    echo "[grid] GPU → partition=${V6_GPU_PARTITION:-gpu} gres=${V6_GPU_GRES:-gpu:1} qos=$QOS time=$TIME_LIMIT (cap 3h) ; --device auto"
    echo "[grid] ⚠️ qos gpus : MaxJobsPU=1 → l'array se sérialise."
fi

SBATCH_SCRIPT="$LOG_DIR/sbatch_perturb.sh"
cat > "$SBATCH_SCRIPT" <<EOF
#!${BASH_BIN}
#SBATCH --job-name=vgae_perturb_${AXIS}
#SBATCH --comment="VGAE perturbation grid (axis=${AXIS})"
#SBATCH --output=$LOG_DIR/%x_%A_%a.out
#SBATCH --error=$LOG_DIR/%x_%A_%a.err
#SBATCH --time=$TIME_LIMIT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --qos=$QOS
#SBATCH --array=$ARRAY_RANGE
$GPU_DIRECTIVES

set -euo pipefail

# GLiCID/Guix : le PATH du job sur le nœud est minimal (ni sed/cut, ni python3 —
# le PATH de soumission n'est pas propagé). On bake le PATH du frontal (env
# conda + profil Guix, sur FS partagé) → fournit python3/mkdir/date.
export PATH="$PATH"

cd "$PROJECT_DIR"

# Lecture de la ligne SLURM_ARRAY_TASK_ID + split TSV via BUILTINS bash
# (pas de sed/cut, indisponibles dans le PATH minimal du nœud).
mapfile -t _CFG_LINES < "$CONFIGS_FILE"
LINE="\${_CFG_LINES[\$SLURM_ARRAY_TASK_ID]}"
IFS=\$'\t' read -r RUN_DIR MODE TARGET_FLAG OUT_SUFFIX QGROUPS AXIS_EXTRA <<< "\$LINE"

echo "[\$(date +%T)] task \$SLURM_ARRAY_TASK_ID"
echo "  run_dir       : \$RUN_DIR"
echo "  mode          : \$MODE"
echo "  target        : \$TARGET_FLAG"
echo "  out-suffix    : '\$OUT_SUFFIX'"
echo "  quiescent     : \$QGROUPS"
echo "  axis-extra    : '\$AXIS_EXTRA'"
echo "  node          : \$(hostname)"

EXTRA_ARGS=()
if [[ -n "\$OUT_SUFFIX" ]]; then
    EXTRA_ARGS+=(--out-suffix "\$OUT_SUFFIX")
fi
EXTRA_ARGS+=(--quiescent-groups "\$QGROUPS")
# V5.4 — ko_mode (cut=statu quo / mask / soft). Passé même en --mode knockdown
# ou --mode overexpress : perturb_top_genes l'ignorera (uniquement utilisé
# si knockout).
EXTRA_ARGS+=(--ko-mode "$KO_MODE")
if [[ "$KO_MODE" == "soft" ]]; then
    EXTRA_ARGS+=(--ko-soft-factor "$KO_SOFT_FACTOR")
fi

# NB cluster Nautilus : tous les .py sont déployés à plat sous src/
# (pas de sous-dossiers gnn/, perturbation/, validation/ comme en local).
python3 src/perturb_top_genes.py \\
    --run-dir "\$RUN_DIR" \\
    "\$TARGET_FLAG" \\
    --modes "\$MODE" \\
    $DEVICE_ARG \\
    \$AXIS_EXTRA \\
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

# --- Préparation du job cross-seed dépendant -------------------------------
# Une "catégorie" = même tag de config sans le seed.
# Ex : v4-full.s1, v4-full.s2, v4-full.s3 → catégorie "v4-full"
# Pour chaque catégorie ayant ≥ 2 seeds × chaque axe, on lance un
# `perturb_report.py --cross-seed`.
declare -A CATEGORIES=()
for r in "${VALID_RUNS[@]}"; do
    name=$(basename "$r")
    cat="${name%.s*}"               # strip suffixe ".s<N>"
    CATEGORIES["$cat"]+="$r "
done

CS_CONFIGS_FILE="$LOG_DIR/cross_seed_configs.tsv"
> "$CS_CONFIGS_FILE"
if [[ $SKIP_CROSS_SEED -eq 0 ]]; then
    for cat in "${!CATEGORIES[@]}"; do
        runs="${CATEGORIES[$cat]}"
        # shellcheck disable=SC2206
        runs_arr=($runs)
        n_seeds=${#runs_arr[@]}
        if [[ $n_seeds -lt 2 ]]; then
            echo "[grid cross-seed] $cat : $n_seeds seed (besoin ≥ 2) → skip"
            continue
        fi
        for spec in "${AXIS_SPECS[@]}"; do
            sfx="${spec%%::*}"      # "axisV4" ou ""
            out_sfx=""
            [[ -n "$sfx" ]] && out_sfx="_${sfx}"
            # axis_tag passé à perturb_report : "" en V3, "axisV4" en V4
            printf '%s\t%s\t%s\t%s\n' \
                "$cat" "$runs" "$out_sfx" "$sfx" >> "$CS_CONFIGS_FILE"
        done
    done
fi
CS_N=$(wc -l < "$CS_CONFIGS_FILE")
echo "[grid] cross-seed : $CS_N tâche(s) (catégories ${!CATEGORIES[@]})"

# --- Génération du sbatch cross-seed (dépendant) ---------------------------
CS_SBATCH_SCRIPT=""
if [[ $CS_N -gt 0 ]]; then
    CS_SBATCH_SCRIPT="$LOG_DIR/sbatch_cross_seed.sh"
    cat > "$CS_SBATCH_SCRIPT" <<EOF
#!${BASH_BIN}
#SBATCH --job-name=vgae_cs_${AXIS}
#SBATCH --comment="VGAE cross-seed report (axis=${AXIS})"
#SBATCH --output=$LOG_DIR/cs_%x_%A_%a.out
#SBATCH --error=$LOG_DIR/cs_%x_%A_%a.err
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --qos=short
#SBATCH --array=0-$((CS_N - 1))%5

set -euo pipefail

# GLiCID/Guix : PATH minimal sur le nœud → bake le PATH du frontal (python3 etc.)
export PATH="$PATH"

cd "$PROJECT_DIR"

# Lecture ligne + split TSV via builtins bash (pas de sed/cut sur le nœud).
mapfile -t _CS_LINES < "$CS_CONFIGS_FILE"
LINE="\${_CS_LINES[\$SLURM_ARRAY_TASK_ID]}"
IFS=\$'\t' read -r CAT RUNS OUT_SUFFIX AXIS_TAG <<< "\$LINE"

# Dossier de sortie : <OUT_DIR_BASE>/cross_seed_<category>[_axisV4]/
REPORT_DIR="${OUT_DIR_BASE}/cross_seed_\${CAT}\${OUT_SUFFIX}"
mkdir -p "\$REPORT_DIR"

echo "[\$(date +%T)] cross-seed task \$SLURM_ARRAY_TASK_ID"
echo "  category    : \$CAT"
echo "  runs        : \$RUNS"
echo "  axis-tag    : '\$AXIS_TAG'"
echo "  report-dir  : \$REPORT_DIR"
echo "  node        : \$(hostname)"

EXTRA_ARGS=()
# AXIS_TAG="" pour V3 (legacy, pas de filtre) ; "axisV4" pour V4
# (filtre les TSV pour ne pas mélanger axes côte à côte).
if [[ -n "\$AXIS_TAG" ]]; then
    EXTRA_ARGS+=(--axis-tag "\$AXIS_TAG")
fi

# shellcheck disable=SC2086
python3 src/perturb_report.py \\
    --cross-seed \$RUNS \\
    --report-dir "\$REPORT_DIR" \\
    "\${EXTRA_ARGS[@]}"

echo "[\$(date +%T)] cross-seed task \$SLURM_ARRAY_TASK_ID terminée."
EOF
    chmod +x "$CS_SBATCH_SCRIPT"
    echo "[grid] sbatch cross-seed : $CS_SBATCH_SCRIPT"
fi
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
    if [[ -n "$CS_SBATCH_SCRIPT" ]]; then
        echo
        echo "[DRY-RUN] cross-seed configs ($CS_N lignes) :"
        cat "$CS_CONFIGS_FILE"
        echo
        echo "[DRY-RUN] cross-seed sbatch (head) :"
        head -30 "$CS_SBATCH_SCRIPT"
    fi
    exit 0
fi

JOB_ID=$(sbatch --parsable "$SBATCH_SCRIPT")
echo "[grid] soumis : job array $JOB_ID (= ${N_TASKS} tâches, ${MAX_PARALLEL} max en //)"
echo "[grid] suivi   : squeue -j $JOB_ID    (ou squeue -u \$USER)"
echo "[grid] cancel  : scancel $JOB_ID"
echo "[grid] logs    : $LOG_DIR/vgae_perturb_${AXIS}_${JOB_ID}_<task>.{out,err}"

# Soumission du cross-seed dépendant — ne tourne que si tous les jobs perturb
# ont réussi (afterok). Si tu veux le lancer même en cas d'échec partiel,
# remplace par "afterany" ; ou re-soumets manuellement plus tard.
if [[ -n "$CS_SBATCH_SCRIPT" ]]; then
    CS_JOB_ID=$(sbatch --parsable --dependency=afterok:$JOB_ID "$CS_SBATCH_SCRIPT")
    echo "[grid] cross-seed soumis : job $CS_JOB_ID (dépendance afterok:$JOB_ID)"
    echo "[grid] cross-seed logs   : $LOG_DIR/cs_vgae_cs_${AXIS}_${CS_JOB_ID}_<task>.{out,err}"
fi
