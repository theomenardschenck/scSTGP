#!/usr/bin/env bash
# =============================================================================
# run_suite_v613.sh — COMPLETE post-perturbation validation suite for the three
# V6.1.3 waves. Idempotent (per-output skip) so restarts resume. Reuses the
# 3-seed cross_seed_gene_ranking.tsv already in <cfg>/analysis (never re-perturbs).
#
# Stages (each independent, each skips finished outputs):
#   A driver_baselines      — GNN vs trivial baselines            (all 17)
#   B hyper downstream       — decoy + interpret + ora (run_analysis) (7 cplx.*)
#   C pipeline_qc            — repro gate rho>=0.95 + 5 checks      (per wave)
#   D purity_source mediate  — source attribution of the readout   (all 17, s1)
#   E head_to_head_baselines — does a simple statistic reproduce it (all 17)
#   F readout_specificity    — off-axis / structured-but-not-DE    (all 17, s1)
#   G driver_pattern (NULL)  — GBM on raw descriptors vs known drivers (all 17)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
# Interpreter carrying torch + torch-geometric. This used to be a hardcoded path
# into one machine's conda prefix, which made the suite unrunnable anywhere else.
# The suite spawns sub-processes that do not necessarily inherit a `conda
# activate`, so point SUITE_PY at the right interpreter when `python` on PATH has
# no torch: SUITE_PY=/path/to/env/bin/python bash scripts/run_suite_v613.sh
PY="${SUITE_PY:-$(command -v python3 || command -v python)}"
[ -x "$PY" ] || { echo "no python found — export SUITE_PY" >&2; exit 1; }
export ANALYSIS_PY="$PY" ANALYSIS_TORCH_PY="$PY"
ROOT=output/gnn_vgae/V6.1.3
COEXPR=data/pyscenic/diff_coexpr/coexpr_diff.tsv
TGT="HMGB1 HMGB2 H2AFZ KMT2A OCRL SYNJ2 SMPD1 TP53 ISG15 CYCS"
FAILED=()
seeds_of(){ for s in s1 s2 s3; do [ -f "$1/$s/perturbation_all_genes_knockout.tsv" ] && echo "$1/$s"; done; }
have(){ [ -f "$1" ]; }

for w in output_de output_fi output_hyper; do
 for cfg in "$ROOT/$w"/*/; do
  cfg="${cfg%/}"; a="$cfg/analysis"; name=$(basename "$cfg")
  rk="$a/cross_seed_gene_ranking.tsv"; have "$rk" || { echo "SKIP $name (no ranking)"; continue; }
  s1="$cfg/s1"; emb="$s1/gene_embeddings_vgae.csv"; graph="$s1/hetero_graph_vgae.pt"

  # A — driver_baselines
  if ! have "$a/driver_baselines.tsv" && ! have "$a/interpret/driver_baselines.tsv"; then
    echo "A driver_baselines $name"
    "$PY" src/validation/reports/driver_baselines.py --ranking "$rk" \
      --coexpr-file "$COEXPR" --out "$a/driver_baselines.tsv" >>"$a/driver_baselines.log" 2>&1 || FAILED+=("A:$name")
  fi

  # B — hyper downstream (decoy+interpret+ora) only for cplx.* lacking decoy
  if [[ "$w" == output_hyper ]] && ! have "$a/interpret/decoy_confidence.tsv"; then
    mapfile -t sd < <(seeds_of "$cfg")
    echo "B run_analysis $name (seeds=${#sd[@]})"
    bash scripts/run_analysis.sh --out "$a" --seeds "${sd[@]}" --skip-cross-seed \
      --decoy --decoy-top-n 40 --coexpr-file "$COEXPR" >>"$a/run_analysis_suite.log" 2>&1 || FAILED+=("B:$name")
  fi

  # D — purity_source mediate (s1)
  if ! have "$a/purity_mediation.tsv"; then
    echo "D purity $name"
    "$PY" src/validation/explain/purity_source_attribution.py mediate \
      --run-dir "$s1" --ranking "$rk" --targets $TGT --out "$a/purity_mediation.tsv" \
      >>"$a/purity.log" 2>&1 || FAILED+=("D:$name")
  fi

  # E — head_to_head_baselines (no humess = robust across pure/rich)
  if ! have "$a/head_to_head_baselines.tsv"; then
    echo "E head_to_head $name"
    "$PY" src/validation/reports/head_to_head_baselines.py --ranking "$rk" \
      --no-humess --coexpr-file "$COEXPR" --out "$a/head_to_head_baselines.tsv" \
      >>"$a/head_to_head.log" 2>&1 || FAILED+=("E:$name")
  fi

  # F — readout_specificity (s1)
  if ! have "$a/readout_specificity.tsv"; then
    echo "F readout_spec $name"
    "$PY" src/validation/explain/readout_specificity.py --run-dir "$s1" --ranking "$rk" \
      --targets "$TGT" --out "$a/readout_specificity.tsv" >>"$a/readout_spec.log" 2>&1 || FAILED+=("F:$name")
  fi

  # G — driver_pattern NULL (needs graph + embeddings)
  if ! have "$a/driver_pattern_importance.tsv" && have "$graph" && have "$emb"; then
    echo "G driver_pattern $name"
    "$PY" src/validation/reports/driver_pattern_classifier.py --graph "$graph" \
      --embeddings "$emb" --ranking "$rk" --label-sets cellage,genage \
      --out "$a/driver_pattern_importance.tsv" >>"$a/driver_pattern.log" 2>&1 || FAILED+=("G:$name")
  fi
 done
done

# C — pipeline_qc per wave (runs once all rankings in the wave exist)
for w in output_de output_fi output_hyper; do
  echo "C pipeline_qc $w"
  "$PY" src/validation/qc/pipeline_qc.py all --wave "$ROOT/$w" >"$ROOT/$w/pipeline_qc.txt" 2>&1 || FAILED+=("C:$w")
done

if [ ${#FAILED[@]} -gt 0 ]; then echo "SUITE done WITH FAILURES: ${FAILED[*]}" >&2; exit 1; fi
echo "SUITE done (all OK)."
