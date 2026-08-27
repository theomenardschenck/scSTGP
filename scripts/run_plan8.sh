#!/usr/bin/env bash
# run_plan8.sh — launch the 2x2x2 wave in priority order, resumably.
#
# WHY A WRAPPER AROUND THE WRAPPER
# The wave runner takes the ablation grid as one block. Two things must not be
# left to chance here:
#
#   1. PRIORITY. The four non-DE configurations are the ones the thesis ranks
#      on; the four `-de` ones only measure how much of that ranking the
#      differential analysis could have produced. If the cluster budget runs
#      out, it must run out on the second group. `--only` enforces that, and
#      phase 2 is a separate command that can simply never be run.
#   2. SEED EXTENSION. Phase 3 re-runs the exact same thing with six seeds.
#      Snakemake skips s1-s3 (their outputs exist) and trains s4-s6 only, then
#      re-aggregates over the six. This is legitimate ONLY because the runs are
#      deterministic: re-aggregating seeds 1-3 gives bit-identical per-seed
#      inputs, so the six-seed aggregate differs for one reason, not two.
#
# Usage
#   bash scripts/run_plan8.sh phase1        # 4 configs sans features DE
#   bash scripts/run_plan8.sh phase2        # 4 configs avec features DE
#   bash scripts/run_plan8.sh seeds6        # etend phase1+2 a 6 graines
#   bash scripts/run_plan8.sh downstream    # regenere tout l'aval du memoire
#   bash scripts/run_plan8.sh -n phase1     # dry-run
#
# Run it detached: the orchestrator must survive to submit the downstream jobs.
#   tmux new -s plan8 'bash scripts/run_plan8.sh phase1'
set -uo pipefail
cd "$(dirname "$0")/.."

PROFILE_YAML=workflow/config/ablations_plan8.yaml
WAVE=scripts/run_omnipath_ablation_wave.sh
SNAKE_ARGS=(--profile workflow/profiles/slurm -j 8 --parallel 2)

DRY=""
if [[ "${1:-}" == "-n" ]]; then DRY="-n"; shift; fi
PHASE="${1:-}"

NODE=(T-o T-c S-o S-c)
DEFT=(T-o-de T-c-de S-o-de S-c-de)
join() { local IFS=,; echo "$*"; }

case "$PHASE" in
  phase1)
    echo "== phase 1 : les 4 configurations SANS features DE (prioritaires) =="
    bash "$WAVE" "$PROFILE_YAML" --only "$(join "${NODE[@]}")" $DRY "${SNAKE_ARGS[@]}"
    ;;
  phase2)
    echo "== phase 2 : les 4 configurations AVEC features DE (facteur 3) =="
    echo "   Ces runs ne sont PAS des candidats au classement du mémoire :"
    echo "   ils mesurent la part redérivable de l'analyse différentielle."
    bash "$WAVE" "$PROFILE_YAML" --only "$(join "${DEFT[@]}")" $DRY "${SNAKE_ARGS[@]}"
    ;;
  seeds6)
    # Sed on a copy: never mutate the profile in place, the header documents
    # the 3-seed first pass and must stay readable.
    TMP=$(mktemp /tmp/plan8_seeds6_XXXX.yaml)
    sed 's/^seeds: \[1, 2, 3\]$/seeds: [1, 2, 3, 4, 5, 6]/' "$PROFILE_YAML" > "$TMP"
    grep -q '1, 2, 3, 4, 5, 6' "$TMP" || { echo "!! substitution des graines ratée"; exit 1; }
    echo "== extension a 6 graines (profil temporaire : $TMP) =="
    echo "   s1-s3 existent -> sautees ; s4-s6 entrainees ; agregat recalcule."
    bash "$WAVE" "$TMP" $DRY "${SNAKE_ARGS[@]}"
    ;;
  downstream)
    # The thesis reads the cross-seed aggregate, never the per-seed runs, so
    # this chain must be replayed after ANY change of seeds or of the score.
    #
    # Some steps of this chain build the artefacts of one specific write-up and
    # are therefore NOT part of the published repository (see .gitignore and
    # workflow/rules/memoire.local.smk). A missing step is skipped with a notice
    # rather than aborting the chain: on a clone the generic steps must still
    # run, and locally every step is present so nothing changes.
    echo "== aval : table de reference -> ORA -> figures -> annexes -> synthese =="
    run_step() {  # run_step <script> [args…] — skip (not fail) if absent
      local script="$1"; shift
      if [ ! -f "$script" ]; then
        echo "   -- $script absent (etape locale non publiee) : ignoree"
        return 0
      fi
      python "$script" "$@" || exit 1
    }
    run_step scripts/gene_reference_table.py
    run_step scripts/build_ora_memoire.py
    run_step scripts/module_ora.py
    run_step src/validation/figures/memoire_figures.py --regime-default \
        --ora-dir output/ora_memoire \
        --out output/gnn_vgae/V6.1.3/global_figures --manifest
    run_step scripts/build_annexes.py
    run_step scripts/wave_synthesis.py
    ;;
  *)
    sed -n '2,30p' "$0"
    exit 1
    ;;
esac
