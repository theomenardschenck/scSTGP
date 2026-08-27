#!/usr/bin/env bash
# =============================================================================
# run_decoy_newtargets.sh -- degree-preserving decoy (N2, n=50) on the candidate
# genes that the v3 rescoring promoted, across the four memoir views.
# =============================================================================
# One invocation per (view, gene) so the wave is resumable: an interrupted run
# loses at most one gene. Each result lands in its own TSV; merge_decoy_new.py
# concatenates them. Sequential on purpose -- the decoy is CPU-only and each
# rewiring reloads the frozen encoder.
#
#   bash scripts/run_decoy_newtargets.sh            # run (resumes)
#   bash scripts/run_decoy_newtargets.sh --force    # recompute everything
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

# Needs torch: the decoy re-encodes the rewired graph. Point DECOY_PY at the
# right interpreter when `python3` on PATH has none (this used to default to one
# machine's conda prefix, which made the script local-only).
PY="${DECOY_PY:-$(command -v python3 || command -v python)}"
N_REWIRES="${N_REWIRES:-50}"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

VIEWS="rfi2.pure-dir rfi2.pure-legacy rfi2.rich-dir rfi2.rich-legacy"
GENES="HMMR CDK1 CDK2 CD59 AKT1 LNPEP POLR2L XYLT1 FAM20C"

for v in $VIEWS; do
  run="output/gnn_vgae/V6.1.3/output_fi/$v/s1"
  out="output/gnn_vgae/V6.1.3/output_fi/$v/analysis/decoy_new"
  mkdir -p "$out"
  for g in $GENES; do
    dst="$out/${g}.tsv"
    if [[ -s "$dst" && $FORCE -eq 0 ]]; then
      echo "[skip] $v/$g"
      continue
    fi
    echo "[run ] $v/$g"
    "$PY" src/validation/explain/perturb_decoy.py per-gene \
        --run-dir "$run" --genes "$g" --modes knockout \
        --n-rewires "$N_REWIRES" --out "$dst" \
        >> "$out/${g}.log" 2>&1 \
      || echo "[FAIL] $v/$g (see $out/${g}.log)"
  done
done
echo "[done] decoy wave finished"
