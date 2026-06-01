#!/usr/bin/env bash
# =============================================================================
# run_grnboost2_diff_arboreto.sh — GRNBoost2 différentiel P4/P16 via arboreto
# =============================================================================
# Pendant de run_diff_coexpr.sh, mais qui utilise la sous-commande
# `scenic_from_r.py grnboost2-diff` (arboreto/dask, canonique Moerman 2019)
# au lieu de `build_diff_coexpr.py grnboost2-local` (sklearn pur).
#
# Objectif : COMPARAISON canonique vs sklearn-local (cohérence top-TF,
# stabilité du delta P4 vs P16).
#
# Différences clés vs run_diff_coexpr.sh :
#   - lit DIRECTEMENT le merged_P4_P16_normalized.csv (filtre par
#     colonne `passage` en interne) : pas besoin d'expr_matrix_{P4,P16}.csv
#     pré-extraits.
#   - opère sur TOUS les gènes QC (~15779) = équivalent --gene-universe all.
#   - exige l'env conda/micromamba `arboreto` (dask<2024.3, arboreto=0.1.6,
#     python=3.11) — l'env `gnn` du cluster a dask ≥ 2025.1 qui a retiré
#     l'API legacy dask.dataframe (NotImplementedError à l'import).
#
# Prérequis (à exécuter UNE FOIS sur le frontal GLiCID) :
#
#   1. Créer l'env arboreto :
#      micromamba create -n arboreto -c conda-forge -c bioconda -y \
#          python=3.11 dask=2024.2.1 distributed=2024.2.1 \
#          arboreto=0.1.6 pandas=2.* numpy=1.26.* scikit-learn
#
#   2. Uploader le merged depuis WSL2 vers LAB-DATA :
#      scp data/gnn_data/merged_P4_P16_normalized.csv \
#          USER@nautilus.glicid.fr:/LAB-DATA/GLiCID/users/USER@univ-nantes.fr/gnn/data/gnn_data/
#
#   3. Vérifier que TF_LIST est bien sur LAB-DATA :
#      ls /LAB-DATA/GLiCID/users/USER@univ-nantes.fr/gnn/data/pyscenic/scenic_refs/allTFs_hg38.txt
#
# Workflow :
#   1. bash scripts/run_grnboost2_diff_arboreto.sh             # array P4+P16
#   2. squeue -j <JID>                                          # attendre
#   3. Adjacencies → data/pyscenic/diff_coexpr/adjacencies_{P4,P16}.arboreto.csv
#      (convention V4.3 — consommée par run_coexpr_method_compare.sh)
#   4. (optionnel) merge arboreto :
#        bash scripts/run_diff_coexpr.sh --step merge \
#          --diff-dir <dir>  # après renommage des adjacencies_*.csv
#
# Options : --dry-run | --n-workers 8 | --mem-per-cpu 10G | --time 06:00:00
#
# NB cluster Nautilus : .py déployés à plat sous src/. Donc
# src/scenic_from_r.py (pas src/data/extract/scenic_from_r.py).
# =============================================================================
set -euo pipefail

DRY_RUN=0
N_WORKERS=8
MEM_PER_CPU="10G"   # 8 × 10G = 80G/job ; arboreto pic ~50G sur 15779 gènes
TIME_GRN="06:00:00"
ENV_NAME="arboreto"
MICROMAMBA_BIN=""   # opt : chemin absolu vers le binaire (sinon auto-détection)
DIFF_DIR="/LAB-DATA/GLiCID/users/USER@univ-nantes.fr/gnn/data/pyscenic/diff_coexpr"
MERGED="/LAB-DATA/GLiCID/users/USER@univ-nantes.fr/gnn/data/gnn_data/merged_P4_P16_normalized.csv"
TF_LIST="/LAB-DATA/GLiCID/users/USER@univ-nantes.fr/gnn/data/pyscenic/scenic_refs/allTFs_hg38.txt"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --n-workers)      N_WORKERS="$2"; shift 2 ;;
        --mem-per-cpu)    MEM_PER_CPU="$2"; shift 2 ;;
        --time)           TIME_GRN="$2"; shift 2 ;;
        --env-name)       ENV_NAME="$2"; shift 2 ;;
        --micromamba-bin) MICROMAMBA_BIN="$2"; shift 2 ;;
        --diff-dir)       DIFF_DIR="$2"; shift 2 ;;
        --merged)         MERGED="$2"; shift 2 ;;
        --tf-list)        TF_LIST="$2"; shift 2 ;;
        *) echo "Option inconnue : $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs/grnboost2_diff_arboreto_${TS}"
mkdir -p "$LOG_DIR"

# --- Vérif prérequis -------------------------------------------------------
if [[ ! -f "$MERGED" ]]; then
    echo "[err] merged absent : $MERGED"
    echo "      Upload depuis WSL2 :"
    echo "        scp data/gnn_data/merged_P4_P16_normalized.csv \\"
    echo "            USER@nautilus.glicid.fr:$MERGED"
    exit 1
fi
if [[ ! -f "$TF_LIST" ]]; then
    echo "[err] TF list absente : $TF_LIST"
    exit 1
fi
mkdir -p "$DIFF_DIR"

echo "[grn-arboreto] project   : $PROJECT_DIR"
echo "[grn-arboreto] log dir   : $LOG_DIR"
echo "[grn-arboreto] env       : $ENV_NAME (micromamba)"
echo "[grn-arboreto] merged    : $MERGED"
echo "[grn-arboreto] tf_list   : $TF_LIST"
echo "[grn-arboreto] out dir   : $DIFF_DIR"
echo "[grn-arboreto] n_workers : $N_WORKERS × mem-per-cpu=$MEM_PER_CPU (= $((N_WORKERS)) × $MEM_PER_CPU)"

# --- sbatch ----------------------------------------------------------------
GRN_SBATCH="$LOG_DIR/sbatch_grnboost2_arboreto.sh"
cat > "$GRN_SBATCH" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=grnboost2_arboreto
#SBATCH --comment="V4.2 GRNBoost2 différentiel arboreto (comparaison sklearn-local)"
#SBATCH --output=$LOG_DIR/%x_%A_%a.out
#SBATCH --error=$LOG_DIR/%x_%A_%a.err
#SBATCH --time=$TIME_GRN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$N_WORKERS
#SBATCH --mem-per-cpu=$MEM_PER_CPU
#SBATCH --qos=short
#SBATCH --array=0-1

set -euo pipefail
cd "$PROJECT_DIR"

# Activation env arboreto via micromamba — les nœuds de calcul GLiCID
# n'héritent PAS du PATH du shell login → on cherche micromamba dans
# l'ordre :  (0) --micromamba-bin si fourni ; (1) déjà dans le PATH ;
# (2) ~/.bashrc ; (3) chemins d'installation standards. Sortie
# explicite si rien ne marche.
if [[ -n "$MICROMAMBA_BIN" ]]; then
    export MAMBA_ROOT_PREFIX="\${MAMBA_ROOT_PREFIX:-\$HOME/micromamba}"
    export PATH="$(dirname "$MICROMAMBA_BIN"):\$PATH"
fi
if ! command -v micromamba >/dev/null 2>&1; then
    if [[ -f "\$HOME/.bashrc" ]]; then
        # shellcheck disable=SC1091
        source "\$HOME/.bashrc" || true
    fi
fi
if ! command -v micromamba >/dev/null 2>&1; then
    for _cand in "\$HOME/.local/bin/micromamba" "\$HOME/micromamba/bin/micromamba" \\
                 "\$HOME/bin/micromamba" "/opt/micromamba/bin/micromamba"; do
        if [[ -x "\$_cand" ]]; then
            export MAMBA_ROOT_PREFIX="\${MAMBA_ROOT_PREFIX:-\$HOME/micromamba}"
            export PATH="\$(dirname "\$_cand"):\$PATH"
            echo "[env] micromamba trouvé : \$_cand"
            break
        fi
    done
fi
if ! command -v micromamba >/dev/null 2>&1; then
    echo "[env] ERREUR : micromamba introuvable sur le compute node." >&2
    echo "      Trouve son chemin sur le frontal avec 'which micromamba'," >&2
    echo "      puis ajoute-le à --env-micromamba-bin <chemin>." >&2
    exit 1
fi
eval "\$(micromamba shell hook --shell bash)"
micromamba activate $ENV_NAME

CONDS=(P4 P16)
COND=\${CONDS[\$SLURM_ARRAY_TASK_ID]}
OUT="$DIFF_DIR/adjacencies_\${COND}.arboreto.csv"

echo "[\$(date +%T)] GRNBoost2 arboreto \$COND sur \$(hostname) (\$SLURM_CPUS_PER_TASK workers)"
echo "[\$(date +%T)] python : \$(which python)"
echo "[\$(date +%T)] dask   : \$(python -c 'import dask;print(dask.__version__)')"

# NB Nautilus : .py à plat sous src/ (pas de sous-dossier extraction/).
python src/scenic_from_r.py grnboost2-diff \\
    --condition \$COND \\
    --merged "$MERGED" \\
    --tf-list "$TF_LIST" \\
    --out "\$OUT" \\
    --n-workers $N_WORKERS \\
    --memory-limit $MEM_PER_CPU \\
    --seed 42

echo "[\$(date +%T)] GRNBoost2 arboreto \$COND terminé → \$OUT"
wc -l "\$OUT"
EOF
chmod +x "$GRN_SBATCH"

echo "[grn-arboreto] sbatch    : $GRN_SBATCH"

# --- Soumission ------------------------------------------------------------
if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] sbatch $GRN_SBATCH   # array 0-1 (P4,P16)"
    sed -n '1,40p' "$GRN_SBATCH"
    exit 0
fi

GRN_JID=$(sbatch --parsable "$GRN_SBATCH")
echo "[grn-arboreto] soumis : job array $GRN_JID (0-1 = P4,P16)"
echo "[grn-arboreto] suivi  : squeue -j $GRN_JID"
echo "[grn-arboreto] logs   : tail -f $LOG_DIR/grnboost2_arboreto_${GRN_JID}_0.out"
echo "[grn-arboreto] sortie : $DIFF_DIR/adjacencies_{P4,P16}.arboreto.csv"
echo
echo "[grn-arboreto] ====== APRÈS COMPLETED ======"
echo "  Comparaison avec adjacencies_{P4,P16}.csv (sklearn-local) :"
echo "    head adjacencies_P4.csv adjacencies_P4.arboreto.csv"
echo "    wc -l adjacencies_P4.csv adjacencies_P4.arboreto.csv"
echo "  Grille V4.3 méthode×prune (consomme directement le suffixe .arboreto.csv) :"
echo "    bash scripts/run_coexpr_method_compare.sh --only-method arboreto"
echo "    bash scripts/run_ablation_grid.sh --version V4.3-method-compare \\"
echo "        --seeds \"1 2 3\"   # avec V43_METHODS=\"arboreto\" si Phase B"
