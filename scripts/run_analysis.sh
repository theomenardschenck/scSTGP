#!/usr/bin/env bash
# =============================================================================
# run_analysis.sh — Pipeline d'ANALYSE post-perturbation (sans (re)perturber).
# =============================================================================
# Enchaîne, sur des runs DÉJÀ perturbés (perturbation_all_*.tsv présents) :
#   1. perturb_report --cross-seed   → cross_seed_gene_ranking.tsv (+ figures)
#   2. driver_baselines.py           → driver_baselines.tsv (coexpr⊥ / CellAge / Tier-1)
#   3. interpret_embedding.py        → communautés + UMAP + ORA + ShinyGO (axe-libre)
#   4. perturb_decoy.py confidence   → decoy_confidence.tsv  (optionnel --decoy, lourd)
#
# C'est la "partie analyse" du pipeline (cf. run_perturbation_grid.sh --analysis).
#
# Usage :
#   bash scripts/run_analysis.sh --out DIR --seeds RUNDIR1 [RUNDIR2 ...] [options]
#     --out DIR            dossier de sortie (OBLIGATOIRE)
#     --seeds RUNDIR...    1+ run dirs (seeds) d'une même catégorie
#     --axis-tag TAG       défaut axisV4
#     --coexpr-file TSV    défaut data/pyscenic/diff_coexpr/coexpr_diff.tsv
#     --decoy              lance aussi la confidence décoy (top-N, lourd CPU)
#     --decoy-top-n N      défaut 40
#     --skip-interpret     saute interpret_embedding
#     --skip-cross-seed    réutilise un cross_seed_gene_ranking.tsv déjà dans --out
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${ANALYSIS_PY:-python3}"                 # report + baselines (pandas/scipy)
TPY="${ANALYSIS_TORCH_PY:-.venv-local/bin/python}"   # interpret + decoy (umap/torch)
COEXPR="data/pyscenic/diff_coexpr/coexpr_diff.tsv"
DE="data/gnn_data/DEGs_P4_vs_P16_MAST.csv"
AXIS="axisV4"; OUT=""; SEEDS=(); DECOY=0; DECOY_N=40; ATTENTION=0
FIGURES=0; ORA_TOPN=100; SKIP_INTERPRET=0; SKIP_CS=0

while [[ $# -gt 0 ]]; do case "$1" in
  --out) OUT="$2"; shift 2;;
  --seeds) shift; while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done;;
  --axis-tag) AXIS="$2"; shift 2;;
  --coexpr-file) COEXPR="$2"; shift 2;;
  --decoy) DECOY=1; shift;;
  --decoy-top-n) DECOY_N="$2"; shift 2;;
  --attention) ATTENTION=1; shift;;          # edge_attention (α GAT, gros TSV)
  --figures) FIGURES=1; shift;;              # visualize_global + viz_explorer
  --ora-top-n) ORA_TOPN="$2"; shift 2;;
  --skip-interpret) SKIP_INTERPRET=1; shift;;
  --skip-cross-seed) SKIP_CS=1; shift;;
  *) echo "[run_analysis] arg inconnu : $1"; exit 1;;
esac; done

[[ -z "$OUT" ]] && { echo "[run_analysis] --out requis"; exit 1; }
mkdir -p "$OUT"
RANK="$OUT/cross_seed_gene_ranking.tsv"
# Toutes les sorties d'ANALYSE (baselines + interpret + décoj) → <OUT>/interpret/.
# Le cross-seed report (perturb_report) reste à la racine <OUT>.
INTERP="$OUT/interpret"
mkdir -p "$INTERP"

# --- 1. cross-seed driver_score ---------------------------------------------
if [[ "$SKIP_CS" -eq 0 ]]; then
  [[ ${#SEEDS[@]} -lt 1 ]] && { echo "[run_analysis] --seeds requis (≥1)"; exit 1; }
  echo "[run_analysis] (1) cross-seed sur ${#SEEDS[@]} seed(s) → $OUT"
  $PY src/validation/reports/perturb_report.py --cross-seed "${SEEDS[@]}" \
      --report-dir "$OUT" --axis-tag "$AXIS" \
      --de-magnitude-csv "$DE" --coexpr-degree-file "$COEXPR"
fi
[[ -f "$RANK" ]] || { echo "[run_analysis] $RANK manquant"; exit 1; }

# --- 2. baselines (coexpr⊥ / CellAge degré-contrôlé / Tier-1) ----------------
echo "[run_analysis] (2) driver_baselines → interpret/"
$PY src/validation/reports/driver_baselines.py --ranking "$RANK" \
    --coexpr-file "$COEXPR" --out "$INTERP/driver_baselines.tsv"

# --- 3. interprétation non-supervisée (axe-libre) ---------------------------
if [[ "$SKIP_INTERPRET" -eq 0 ]]; then
  echo "[run_analysis] (3) interpret_embedding (communautés + UMAP + ORA + ShinyGO)"
  $TPY src/validation/figures/interpret_embedding.py \
      --run-dir "${SEEDS[0]}" --ranking "$RANK" --shinygo \
      --out-dir "$INTERP" || echo "[run_analysis] interpret a échoué (deps ?)"
fi

# --- 4. confidence décoy (optionnel, lourd) ---------------------------------
if [[ "$DECOY" -eq 1 ]]; then
  echo "[run_analysis] (4) perturb_decoy confidence (top-$DECOY_N) → interpret/"
  $TPY src/validation/explain/perturb_decoy.py confidence \
      --run-dir "${SEEDS[0]}" --ranking "$RANK" --top-n "$DECOY_N" \
      --out "$INTERP/decoy_confidence.tsv" || echo "[run_analysis] décoj a échoué"
fi

# --- 5. attention GAT (optionnel, gros TSV ~150 MB) -------------------------
if [[ "$ATTENTION" -eq 1 ]]; then
  echo "[run_analysis] (5) edge_attention (α GAT) → interpret/attention/"
  $TPY src/validation/explain/edge_attention.py extract \
      --run-dir "${SEEDS[0]}" --out-dir "$INTERP/attention" \
      || echo "[run_analysis] edge_attention a échoué"
fi

# --- 6. ORA des top-drivers (Reactome + aging) → interpret/ora/ -------------
echo "[run_analysis] (6) ora_consensus top-$ORA_TOPN drivers → interpret/ora/"
mkdir -p "$INTERP/ora"
tail -n +2 "$RANK" | sort -t$'\t' -k2,2 -gr | head -"$ORA_TOPN" | cut -f1 \
    > "$INTERP/ora/top${ORA_TOPN}_drivers.txt"
$PY src/validation/ora/ora_consensus.py --genes "$INTERP/ora/top${ORA_TOPN}_drivers.txt" \
    --db reactome --out "$INTERP/ora/top${ORA_TOPN}_reactome.tsv" || echo "  ora reactome ÉCHEC"
$PY src/validation/ora/ora_consensus.py --genes "$INTERP/ora/top${ORA_TOPN}_drivers.txt" \
    --db aging --out "$INTERP/ora/top${ORA_TOPN}_aging.tsv" || echo "  ora aging ÉCHEC"

# --- 7. figures (optionnel) : visualize_global + viz_explorer → interpret/figures/
if [[ "$FIGURES" -eq 1 ]]; then
  echo "[run_analysis] (7) figures → interpret/figures/"
  FIG="$INTERP/figures"; mkdir -p "$FIG"
  $TPY src/validation/figures/visualize_global.py --out-dir "$FIG" umap \
      --run-dir "${SEEDS[0]}" --version-dir "$OUT" || echo "  visualize_global umap ÉCHEC"
  # ranking-seul (pas besoin des runs bruts)
  for cmd in heatmap-clusters tier-distributions aging-bubbles; do
    $TPY src/validation/figures/viz_explorer.py "$cmd" \
        --version-dir "$OUT" --out-dir "$FIG" 2>/dev/null || echo "  viz_explorer $cmd skip"
  done
  # nécessitent les runs bruts (cosinus per-cluster agrégé / group_expression)
  for cmd in radar lollipop-clusters quadrant heatmap-expression clustermap; do
    $TPY src/validation/figures/viz_explorer.py "$cmd" --version-dir "$OUT" \
        --raw-runs "${SEEDS[@]}" --out-dir "$FIG" 2>/dev/null || echo "  viz_explorer $cmd skip"
  done
fi

echo "[run_analysis] DONE → cross-seed report: $OUT ; analyses: $INTERP"
