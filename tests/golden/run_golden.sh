#!/usr/bin/env bash
# =============================================================================
# run_golden.sh — filet iso-comportement pour le split de gnn_vgae.py.
#
# Lance gnn_vgae.py sur le CACHE scRNA local (build sauté, déterministe CPU,
# epochs courts) et compare gene_ranking_vgae.csv + gene_embeddings_vgae.csv
# à une référence figée. À exécuter après CHAQUE stage du split.
#
# Sous-commandes :
#   capture   — (re)génère la référence figée (à faire 1× sur le monolithe).
#   check     — génère un run neuf et le compare à la référence (exit≠0 si écart).
#   probe     — sonde de déterminisme : 2 runs neufs comparés bit-exact (atol=0)
#               → dit si le golden peut être bit-exact ou doit tolérer.
#
# Tolérance (check) : GOLDEN_ATOL / GOLDEN_RTOL (défaut 0 = bit-exact).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."          # → racine gnn_huvec

PY="${GOLDEN_PY:-.venv-local/bin/python}"
CACHE="${GOLDEN_CACHE:-output/gnn_vgae/_graph_cache_scrna.pkl}"
EPOCHS="${GOLDEN_EPOCHS:-2}"
REF_DIR="tests/golden/reference"
CMP="tests/golden/compare.py"

# --- Mode DÉTERMINISTE (obligatoire) : le pipeline est non-reproductible en
# multi-thread (réductions scatter-add CPU non-associatives → variance run-to-run
# même à seed fixe, mesuré : embeddings max|Δ|≈1.6 sur 2 runs). On force donc
# threads=1 + algos déterministes + PYTHONHASHSEED pour un golden bit-exact.
export GNN_DETERMINISTIC=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Flags-graphe = config scRNA baseline (DOIT matcher la signature du cache).
GRAPH_FLAGS=(--use-omnipath-signaling --use-omnipath-tf-curated --include-omnipath-genes
             --use-reactome-fi --signed-message --signed-decoder --decoder-split
             --kl-beta-max 0.0001 --coexpr-mode p16_only)
# Baselines OFF par défaut (rapide) ; GOLDEN_BASELINES=1 pour couvrir mlp/node2vec.
[[ "${GOLDEN_BASELINES:-0}" == "1" ]] && BASE_FLAG=() || BASE_FLAG=(--no-baselines)

export GNN_HUMESS_DIR="$PWD/data/humess/output_huvec"
export GNN_HUMESS_CONDITIONS="P4,P16"
export GNN_ALLOW_DOWNLOADS=0

run_one() {   # $1 = run_tag → écrit dans output/gnn_vgae/<run_tag>/
  local tag="$1"
  rm -rf "output/gnn_vgae/$tag"
  "$PY" -u src/gnn/gnn_vgae.py --seed 1 --run-tag "$tag" \
    --reuse-graph --graph-cache "$CACHE" --n-epochs "$EPOCHS" \
    "${GRAPH_FLAGS[@]}" "${BASE_FLAG[@]}" > "output/gnn_vgae/${tag}.log" 2>&1
}

collect() {   # copie les 2 sorties clés d'un run vers $2
  local tag="$1" dest="$2"
  mkdir -p "$dest"
  cp "output/gnn_vgae/$tag/gene_ranking_vgae.csv" "$dest/"
  cp "output/gnn_vgae/$tag/gene_embeddings_vgae.csv" "$dest/"
}

case "${1:-check}" in
  capture)
    echo "[golden] capture référence (epochs=$EPOCHS, baselines=${GOLDEN_BASELINES:-0})"
    run_one golden_ref
    collect golden_ref "$REF_DIR"
    echo "[golden] référence figée → $REF_DIR"
    ls -la "$REF_DIR"
    ;;
  check)
    echo "[golden] check (epochs=$EPOCHS, atol=${GOLDEN_ATOL:-0})"
    run_one golden_new
    collect golden_new tests/golden/_new
    "$PY" "$CMP" "$REF_DIR" tests/golden/_new
    ;;
  probe)
    echo "[golden] sonde déterminisme : 2 runs neufs, comparaison bit-exact"
    run_one golden_p1; collect golden_p1 tests/golden/_p1
    run_one golden_p2; collect golden_p2 tests/golden/_p2
    GOLDEN_ATOL=0 GOLDEN_RTOL=0 "$PY" "$CMP" tests/golden/_p1 tests/golden/_p2
    ;;
  *)
    echo "usage: $0 {capture|check|probe}"; exit 2 ;;
esac
