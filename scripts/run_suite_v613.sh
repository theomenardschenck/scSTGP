#!/usr/bin/env bash
# =============================================================================
# run_suite_v613.sh — Complete the post-perturbation analyses for the three
# V6.1.3 waves (output_de / output_fi / output_hyper) once the 3-seed
# cross_seed_gene_ranking.tsv files are regenerated.
#
# Fills the gaps found on 2026-07-30:
#   - driver_baselines : MISSING everywhere  (GNN vs trivial baselines)
#   - pipeline_qc      : MISSING everywhere  (repro gate rho>=0.95 + 5 checks)
#   - HYPER downstream : MISSING (decoy + interpret + ora for the 7 cplx.* arms)
#   - purity_source    : MISSING everywhere  (optional, Stage C — commented)
#
# It REUSES the 3-seed ranking already in <cfg>/analysis (--skip-cross-seed), so
# it never re-perturbs and never overwrites the ranking.
#
# Usage (detached):
#   setsid bash scripts/run_suite_v613.sh >logs/suite_v613.log 2>&1 &
#   # or gate on the ranking batch first:
#   #   until [ -f .../xseed3_batch.log ] && grep -q '^ALL DONE' .../xseed3_batch.log; do sleep 60; done
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/USER/miniforge3/envs/gnn/bin/python
export ANALYSIS_PY="$PY" ANALYSIS_TORCH_PY="$PY"
ROOT=output/gnn_vgae/V6.1.3
COEXPR=data/pyscenic/diff_coexpr/coexpr_diff.tsv
FAILED=()

seeds_of() { for s in s1 s2 s3; do [ -f "$1/$s/perturbation_all_genes_knockout.tsv" ] && echo "$1/$s"; done; }

echo "==================== STAGE A : driver_baselines (all 17) ===================="
for w in output_de output_fi output_hyper; do
  for cfg in "$ROOT/$w"/*/; do
    cfg="${cfg%/}"; a="$cfg/analysis"; name=$(basename "$cfg")
    [ -f "$a/cross_seed_gene_ranking.tsv" ] || { echo "SKIP $name (no ranking yet)"; continue; }
    echo "-- driver_baselines $name"
    "$PY" src/validation/reports/driver_baselines.py \
        --ranking "$a/cross_seed_gene_ranking.tsv" \
        --coexpr-file "$COEXPR" --out "$a/driver_baselines.tsv" \
        >>"$a/driver_baselines.log" 2>&1 || FAILED+=("baselines:$name")
  done
done

echo "==================== STAGE B : HYPER downstream (decoy+interpret+ora) ========"
# DE/FI already have decoy/interpret/ora from the cluster; only the 7 cplx.* need it.
for cfg in "$ROOT/output_hyper"/*/; do
  cfg="${cfg%/}"; a="$cfg/analysis"; name=$(basename "$cfg")
  [ -f "$a/cross_seed_gene_ranking.tsv" ] || { echo "SKIP $name"; continue; }
  mapfile -t sd < <(seeds_of "$cfg")
  echo "-- run_analysis (baselines+interpret+decoy) $name  seeds=${#sd[@]}"
  bash scripts/run_analysis.sh --out "$a" --seeds "${sd[@]}" \
      --skip-cross-seed --decoy --decoy-top-n 40 --coexpr-file "$COEXPR" \
      >>"$a/run_analysis_suite.log" 2>&1 || FAILED+=("hyper-analysis:$name")
done

echo "==================== STAGE C : pipeline_qc per wave (repro gate) ============="
for w in output_de output_fi output_hyper; do
  echo "-- pipeline_qc $w"
  "$PY" src/validation/qc/pipeline_qc.py all --wave "$ROOT/$w" \
      >"$ROOT/$w/pipeline_qc.txt" 2>&1 || FAILED+=("qc:$w")
done

# ---- STAGE D (optional, targeted) : purity_source_attribution --------------
# Heavier (frozen-encoder re-projection). Uncomment to run on the headline arms.
# for cfg in "$ROOT/output_fi/rfi2.rich-dir" "$ROOT/output_hyper/cplx.rich-ctrl"; do
#   "$PY" src/validation/explain/purity_source_attribution.py mediate \
#       --run-dir "$cfg/s1" --targets OCRL SYNJ2 SMPD1 HMGB1 GCLC \
#       --out "$cfg/analysis/purity_source.tsv" || FAILED+=("purity:$(basename $cfg)")
# done

if [ ${#FAILED[@]} -gt 0 ]; then echo "SUITE done WITH FAILURES: ${FAILED[*]}" >&2; exit 1; fi
echo "SUITE done (all OK)."
