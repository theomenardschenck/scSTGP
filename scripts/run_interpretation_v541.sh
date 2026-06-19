#!/usr/bin/env bash
# =============================================================================
# run_interpretation_v541.sh — Pipeline d'interprétation UNIFIÉ V5.4.1
# =============================================================================
# COMBINE l'analyse PAR-CONFIG (run_analysis.sh) et la synthèse CROSS-CONFIG en
# un seul point d'entrée. Remplace l'ancien duo {run_analysis.sh × N + couche
# cross-config}.
#
#   PHASE 1 — par config (cœur délégué à run_analysis.sh --skip-interpret) :
#       cross-seed report (perturb_report) → driver_baselines → ORA top-drivers
#       + viz_explorer            → cross_seed/<config>/figures/
#       + interpret_embedding     → cross_seed/<config>/embed/   (baseline + ex-degrees)
#       [+ perturb_decoy] [+ edge_attention]   (optionnels, lourds)
#   PHASE 2 — cross-config :
#       ablation_attribution (réf=baseline) → ablation_attribution/
#       compare_runs (configs fiables)      → compare_configs/
#       plot_pathway_heatmap                → pathway_summary/
#
# Usage :
#   bash scripts/run_interpretation_v541.sh [options]
#     --reuse-cross-seed   réutilise les cross_seed_gene_ranking.tsv existants
#                          (passe --skip-cross-seed à run_analysis ; pas de re-perturb_report)
#     --skip-perconfig     saute la PHASE 1 (synthèse cross-config seule)
#     --skip-embed         saute interpret_embedding (UMAP/communautés)
#     --skip-compare       saute compare_runs
#     --decoy              ajoute la confidence décoy (lourd CPU, torch)
#     --attention          ajoute edge_attention (gros TSV ~150 MB)
#
# Reproduction « de zéro » (depuis runs bruts perturbés) : sans --reuse-cross-seed.
# Reproduction « figures seules » (rapports déjà là) : --reuse-cross-seed.
#
# Layout (source unique depuis la consolidation 2026-06-12) :
#   output/interpretation/V5.4.1/cross_seed/<config>/   (SANS préfixe v5.4.)
#   runs bruts (embeddings) : output/gnn_vgae/V5.4.1/v5.4.<config>.sN  (gardent v5.4.)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY="python3"
TOPY=".venv-local/bin/python"            # torch + umap (interpret_embedding)
ANALYSIS="scripts/run_analysis.sh"        # cœur par-config réutilisé
RUNS="output/gnn_vgae/V5.4.1"             # runs bruts (gardent le préfixe v5.4.)
OUT="output/interpretation/V5.4.1"
CROSS="$OUT/cross_seed"                    # rapports cross-seed par config (sans préfixe)
AXIS="axisV4"

REUSE_CS=0; SKIP_PERCFG=0; SKIP_EMBED=0; SKIP_COMPARE=0; DECOY=0; ATTENTION=0
while [[ $# -gt 0 ]]; do case "$1" in
  --reuse-cross-seed) REUSE_CS=1; shift;;
  --skip-perconfig)   SKIP_PERCFG=1; shift;;
  --skip-embed)       SKIP_EMBED=1; shift;;
  --skip-compare)     SKIP_COMPARE=1; shift;;
  --decoy)            DECOY=1; shift;;
  --attention)        ATTENTION=1; shift;;
  *) echo "Arg inconnu : $1"; exit 1;;
esac; done
mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# Configs : clé = nom cross_seed (sans préfixe) ; valeur = runs bruts (seeds)
# ---------------------------------------------------------------------------
declare -A SEEDS=(
  # === s1+s2 cross-seed RÉEL (16 configs, robustesse/stabilité non triviales) ===
  [baseline]="$RUNS/v5.4.baseline.s1 $RUNS/v5.4.baseline.s2"
  [ex-degrees]="$RUNS/v5.4.ex-degrees.s1 $RUNS/v5.4.ex-degrees.s2"
  [coexpr-legacy-arboreto]="$RUNS/v5.4.coexpr-legacy-arboreto.s1 $RUNS/v5.4.coexpr-legacy-arboreto.s2"
  [coexpr-arboreto.quantile-limited]="$RUNS/v5.4.coexpr-arboreto.quantile-limited.s1 $RUNS/v5.4.coexpr-arboreto.quantile-limited.s2"
  [coexpr-arboreto.mr]="$RUNS/v5.4.coexpr-arboreto.mr.s1 $RUNS/v5.4.coexpr-arboreto.mr.s2"
  [coexpr-arboreto.mr5]="$RUNS/v5.4.coexpr-arboreto.mr5.s1 $RUNS/v5.4.coexpr-arboreto.mr5.s2"
  [no-dedup]="$RUNS/v5.4.no-dedup.s1 $RUNS/v5.4.no-dedup.s2"
  [no-humess]="$RUNS/v5.4.no-humess.s1 $RUNS/v5.4.no-humess.s2"
  [no-rfi]="$RUNS/v5.4.no-rfi.s1 $RUNS/v5.4.no-rfi.s2"
  [no-rfi+ex-degrees]="$RUNS/v5.4.no-rfi+ex-degrees.s1 $RUNS/v5.4.no-rfi+ex-degrees.s2"
  [no-reactome]="$RUNS/v5.4.no-reactome.s1 $RUNS/v5.4.no-reactome.s2"
  [no-coexpr]="$RUNS/v5.4.no-coexpr.s1 $RUNS/v5.4.no-coexpr.s2"
  [no-coexpr+no-humess]="$RUNS/v5.4.no-coexpr+no-humess.s1 $RUNS/v5.4.no-coexpr+no-humess.s2"
  [no-coexpr+no-humess+no-scenic]="$RUNS/v5.4.no-coexpr+no-humess+no-scenic.s1 $RUNS/v5.4.no-coexpr+no-humess+no-scenic.s2"
  [no-coexpr+no-humess+no-scenic+ex-degrees]="$RUNS/v5.4.no-coexpr+no-humess+no-scenic+ex-degrees.s1 $RUNS/v5.4.no-coexpr+no-humess+no-scenic+ex-degrees.s2"
  [backbone]="$RUNS/v5.4.backbone.s1 $RUNS/v5.4.backbone.s2"
  # === s1 seulement (bug cluster : s2 incomplet/absent) ===
  [no-scenic]="$RUNS/v5.4.no-scenic.s1"      # s2 = KO seul (KD/OE manquants) → s1 (modes complets)
  [signed-only]="$RUNS/v5.4.signed-only.s1"  # s2 absent
)
# interpret_embedding (coûteux : UMAP+Louvain+ORA/communauté+ShinyGO) — configs clés
EMBED_CONFIGS="baseline ex-degrees no-humess no-coexpr no-reactome"
ABLATION_CONFIGS="no-humess no-rfi no-scenic no-dedup no-reactome coexpr-legacy-arboreto ex-degrees no-coexpr"
REF="baseline"

# ===========================================================================
# PHASE 1 — par config
# ===========================================================================
if [[ "$SKIP_PERCFG" -eq 0 ]]; then
  echo ""; echo "████ PHASE 1 — analyse par config ████"
  reuse_flag=(); [[ "$REUSE_CS" -eq 1 ]] && reuse_flag=(--skip-cross-seed)
  decoy_flag=(); [[ "$DECOY" -eq 1 ]] && decoy_flag=(--decoy)
  attn_flag=();  [[ "$ATTENTION" -eq 1 ]] && attn_flag=(--attention)

  for cfg in "${!SEEDS[@]}"; do
    read -ra rr <<< "${SEEDS[$cfg]}"
    cdir="$CROSS/$cfg"
    echo ""; echo "──── $cfg ────"

    # 1a. Cœur (cross-seed report + driver_baselines + ORA) — délégué, sans interpret/figures
    bash "$ANALYSIS" --out "$cdir" --seeds "${rr[@]}" --axis-tag "$AXIS" \
         --skip-interpret "${reuse_flag[@]}" "${decoy_flag[@]}" "${attn_flag[@]}" \
         2>&1 | grep -E "^\[run_analysis\]|wrote|ORA|Tier-1" | head -20 || true

    [[ -f "$cdir/cross_seed_gene_ranking.tsv" ]] || { echo "  [skip figures] pas de ranking"; continue; }

    # 1b. Figures biologiques (viz_explorer) → cross_seed/<config>/figures/
    fdir="$cdir/figures"; mkdir -p "$fdir"
    $PY src/validation/viz/viz_explorer.py all \
        --version-dir "$cdir" --raw-runs "${rr[@]}" --out-dir "$fdir" \
        2>&1 | grep -E "écrit|Saved|Error" || true
    echo "  figures → $fdir"

    # 1c. interpret_embedding (UMAP + communautés + ShinyGO) → cross_seed/<config>/embed/
    if [[ "$SKIP_EMBED" -eq 0 ]] && echo " $EMBED_CONFIGS " | grep -q " $cfg "; then
      edir="$cdir/embed"; mkdir -p "$edir"
      $TOPY src/validation/viz/interpret_embedding.py \
          --run-dir "${rr[0]}" --ranking "$cdir/cross_seed_gene_ranking.tsv" --shinygo \
          --out-dir "$edir" 2>&1 | grep -E "wrote|umap|community|Error" | head -8 || true
      echo "  embed → $edir"
    fi
  done
else
  echo "  [--skip-perconfig] PHASE 1 ignorée"
fi

# ===========================================================================
# PHASE 2 — synthèse cross-config
# ===========================================================================
echo ""; echo "████ PHASE 2 — synthèse cross-config ████"

# 2a. ablation_attribution (réf=baseline)
ATTR_OUT="$OUT/ablation_attribution"; mkdir -p "$ATTR_OUT"
ABLATIONS=()
for cfg in $ABLATION_CONFIGS; do
  [[ -f "$CROSS/$cfg/cross_seed_gene_ranking.tsv" ]] && ABLATIONS+=("$CROSS/$cfg")
done
echo "  → ablation_attribution (réf=$REF, ${#ABLATIONS[@]} ablations)"
$PY src/validation/reports/ablation_attribution.py \
    --reference "$CROSS/$REF" --ablations "${ABLATIONS[@]}" \
    --out-dir "$ATTR_OUT" --top-k 50 2>&1 | grep -E "wrote|\[ora|\[delta" | head -12 || true

# 2b. compare_runs (configs fiables : exclut backbone / no-coexpr* / signed-only)
if [[ "$SKIP_COMPARE" -eq 0 ]]; then
  COMP_OUT="$OUT/compare_configs"; mkdir -p "$COMP_OUT"
  GOOD_RUNS=()
  for d in "$CROSS"/*/; do
    n=$(basename "$d")
    [[ "$n" == *backbone* || "$n" == *no-coexpr* || "$n" == *signed-only* ]] && continue
    [[ -f "$d/cross_seed_gene_ranking.tsv" ]] && GOOD_RUNS+=("$n")
  done
  echo "  → compare_runs (${#GOOD_RUNS[@]} configs fiables)"
  $PY src/validation/reports/compare_runs.py \
      --base-dir "$CROSS" --runs "${GOOD_RUNS[@]}" --source perturb_driver \
      --report-subdir "" --baseline "$REF" --top-n 100 \
      --output-dir "$COMP_OUT" 2>&1 | grep -E "✓|wrote|Error" | head -8 || true
else
  echo "  [--skip-compare] compare_runs ignoré"
fi

# 2c. pathway × config heatmap
PW_OUT="$OUT/pathway_summary"; mkdir -p "$PW_OUT"
echo "  → plot_pathway_heatmap"
$PY src/validation/viz/plot_pathway_heatmap.py --reports-dir "$CROSS" --out-dir "$PW_OUT" \
    2>&1 | grep -E "Saved|chargées|Error" | head -12 || true

# ---------------------------------------------------------------------------
echo ""; echo "████ DONE — $OUT/ ████"
find "$OUT" -name "*.png" -o -name "*.svg" | wc -l | xargs -I{} echo "  {} fichiers image"

# =============================================================================
# Rôle vs run_analysis.sh
# =============================================================================
# run_analysis.sh        = worker PAR-CONFIG (1 config × N seeds), réutilisable seul
#                          (ex. run_perturbation_grid.sh --analysis sur cluster).
# run_interpretation_v541.sh = orchestrateur UNIFIÉ : boucle run_analysis.sh sur
#                          toutes les configs (PHASE 1) PUIS ajoute la synthèse
#                          cross-config (PHASE 2). Point d'entrée unique pour la
#                          présentation. Cf. RES §13.11.1 + SYNTHESIS.md.
# =============================================================================
