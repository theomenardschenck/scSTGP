#!/usr/bin/env bash
# =============================================================================
# run_supervised.sh — V-sup : plafond supervisé circulaire sur cluster (GLiCID).
# =============================================================================
# Soumet UN job SLURM qui lance `gnn_vgae.py --supervised` : reconstruit le
# graphe hétérogène depuis GSE102090 (HUVEC scRNA P4/P16) PUIS entraîne
# l'encodeur end-to-end sur les labels DEG multi-label (P4_vs_P16 + cluster_0..3)
# + features DE, et calcule l'importance PAR CLUSTER. Cf. docs/technical/
# gnn_supervised.md, LOG #log-supervised.
#
# ⚠️ CPU : gnn_vgae n'a pas de support GPU (entraînement plein-graphe sur CPU).
#    Le cluster fournit assez de RAM pour l'étape d'importance (4 backward
#    plein-graphe) qui est réapée en local (4 Go). Prévoir --mem généreux.
#
# Config graphe par défaut = référence V5.4.1 (OmniPath signaling + tf_curated +
# Reactome FI) → le plafond supervisé est DIRECTEMENT comparable aux drivers du
# VGAE headline. Surcharge via --graph-flags.
#
# Partitions/qos GLiCID (surcharge via env) :
#   SUP_CPU_PARTITION (standard)  SUP_CPU_QOS (long)  SUP_PY (python/venv)
#
# Usage :
#   bash scripts/run_supervised.sh --run-tag vsup_full
#   [--no-de-features]                 # ablation : topologie seule (pas de DE)
#   [--graph-flags "--use-omnipath-signaling --use-omnipath-tf-curated --use-reactome-fi"]
#   [--reuse-graph]                    # réutilise le cache de build si valide
#   [--epochs 300] [--seed 1] [--time 1-00:00:00] [--cpus 8] [--mem 64G]
#   [--dry-run]
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_TAG="vsup_full"
DE_FEATURES=1
GRAPH_FLAGS="--use-omnipath-signaling --use-omnipath-tf-curated --use-reactome-fi"
REUSE_GRAPH=0
EPOCHS=1500          # large : l'early-stopping (patience) coupe au vrai plateau
PATIENCE=200         # epochs sans amélioration de l'AUC recon avant arrêt
SEED=1
TIME="1-00:00:00"
CPUS=8
MEM="64G"
DRY=0
# DATA_ROOT : racine des DONNÉES d'entrée (= gnn_vgae BASE_DIR/data). Sur GLiCID
# les données sont sur /LAB-DATA (PAS rsync vers /scratch où vit le code) → il
# FAUT pointer gnn_vgae dessus, sinon merged_P4_P16_metadata.csv introuvable.
DATA_ROOT="${GNN_DATA_ROOT:-/LAB-DATA/GLiCID/users/USER@univ-nantes.fr/gnn/data}"
OUT_BASE="${GNN_OUT_DIR_BASE:-$PWD/output/gnn_vgae}"
HEAD=1   # tête de classification jointe (--supervised) ; --no-head = VGAE+DE seul

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-tag) RUN_TAG="$2"; shift 2;;
    --no-de-features) DE_FEATURES=0; shift;;
    --no-head) HEAD=0; shift;;
    --graph-flags) GRAPH_FLAGS="$2"; shift 2;;
    --reuse-graph) REUSE_GRAPH=1; shift;;
    --epochs) EPOCHS="$2"; shift 2;;
    --patience) PATIENCE="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --data-root) DATA_ROOT="$2"; shift 2;;
    --out-base) OUT_BASE="$2"; shift 2;;
    --time) TIME="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --mem) MEM="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "[run_supervised] argument inconnu : $1" >&2; exit 2;;
  esac
done

CPU_PARTITION="${SUP_CPU_PARTITION:-standard}"
CPU_QOS="${SUP_CPU_QOS:-long}"
LOG_DIR="output/gnn_supervised/_slurm"
mkdir -p "$LOG_DIR"
BASE_DIR_ENV="$(dirname "$DATA_ROOT")"   # gnn_vgae BASE_DIR (relatif reactome_fi/diff_coexpr)

# Le VGAE RECONSTRUIT toujours (perturbation-ready). --de-features = features DE
# circulaires ; --supervised = tête classif jointe (multi-tâche). Sortie =
# output/gnn_vgae/<run_tag> (le supervisé EST un run VGAE circulaire).
DE_FLAG=""; [[ "$DE_FEATURES" -eq 1 ]] && DE_FLAG="--de-features"
HEAD_FLAG=""; [[ "$HEAD" -eq 1 ]] && HEAD_FLAG="--supervised"
REUSE_FLAG=""; [[ "$REUSE_GRAPH" -eq 1 ]] && REUSE_FLAG="--reuse-graph"

SBATCH_FILE="$LOG_DIR/sbatch_${RUN_TAG}.sh"
cat > "$SBATCH_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${RUN_TAG}
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
# module load python/3.12 2>/dev/null || true
PY="\${SUP_PY:-python3}"
export OMP_NUM_THREADS=$CPUS
# Données sur LAB-DATA (code sur scratch) : pointer gnn_vgae dessus.
export GNN_DATA_DIR="$DATA_ROOT"          # DATA_DIR (metadata, gnn_data/, omnipath/)
export GNN_BASE_DIR="$BASE_DIR_ENV"       # BASE_DIR (résout reactome_fi/diff_coexpr relatifs)
export GNN_OUT_DIR_BASE="$OUT_BASE"       # sorties (scratch)
echo "[run_supervised] GNN_DATA_DIR=\$GNN_DATA_DIR"
echo "[run_supervised] GNN_BASE_DIR=\$GNN_BASE_DIR"
echo "[run_supervised] GNN_OUT_DIR_BASE=\$GNN_OUT_DIR_BASE"
srun \$PY src/gnn/gnn_vgae.py \\
    $DE_FLAG $HEAD_FLAG \\
    $GRAPH_FLAGS \\
    $REUSE_FLAG \\
    --n-epochs $EPOCHS \\
    --patience $PATIENCE \\
    --seed $SEED \\
    --run-tag $RUN_TAG
EOF

echo "[run_supervised] script sbatch : $SBATCH_FILE"
echo "[run_supervised] tag=$RUN_TAG de_features=$DE_FEATURES head=$HEAD reuse=$REUSE_GRAPH"
echo "[run_supervised] graph_flags: $GRAPH_FLAGS"
echo "[run_supervised] sorties → ${OUT_BASE}/$RUN_TAG/ (run VGAE circulaire)"
if [[ "$DRY" -eq 1 ]]; then
  echo "[run_supervised] --dry-run : script généré, non soumis."; cat "$SBATCH_FILE"; exit 0
fi
JID=$(sbatch --parsable "$SBATCH_FILE")
echo "[run_supervised] soumis : job $JID"
