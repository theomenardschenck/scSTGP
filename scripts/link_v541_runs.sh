#!/usr/bin/env bash
# =============================================================================
# link_v541_runs.sh — expose les runs V5.4.1 dans le layout attendu par Snakemake
# =============================================================================
# Les runs de juin 2026 suivent le nommage historique
#     output/gnn_vgae/V5.4.1/v5.4.<config>.s<seed>/
# alors que le Snakefile (réorg 2026-07-15) attend
#     <out_base>/<run_tag>/s<seed>/
# Ce script pose les liens symboliques qui réconcilient les deux, SANS copier
# ni déplacer quoi que ce soit (les sorties du nouveau run atterrissent donc
# dans les dossiers d'origine, sous des noms de fichiers distincts).
#
#   bash scripts/link_v541_runs.sh baseline 1 2
#   bash scripts/link_v541_runs.sh no-rfi 1 2
#   RUNS_ROOT=/scratch/.../output/gnn_vgae/V5.4.1 bash scripts/link_v541_runs.sh baseline 1 2
#
# Résultat pour `baseline 1 2` :
#   output/gnn_vgae/V5.4.1/v5.4.baseline/s1 -> ../v5.4.baseline.s1
#   output/gnn_vgae/V5.4.1/v5.4.baseline/s2 -> ../v5.4.baseline.s2
# = exactement le `run_tag: "V5.4.1/v5.4.baseline"` de config.V5.4.1-dz.yaml.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS_ROOT="${RUNS_ROOT:-output/gnn_vgae/V5.4.1}"

if [[ $# -lt 2 ]]; then
    echo "usage : bash scripts/link_v541_runs.sh <config> <seed> [<seed>...]"
    echo "        <config> sans le préfixe v5.4. (ex. baseline, no-rfi, backbone)"
    exit 1
fi

CONFIG="$1"; shift
DEST="$RUNS_ROOT/v5.4.$CONFIG"
mkdir -p "$DEST"

for SEED in "$@"; do
    SRC="$RUNS_ROOT/v5.4.$CONFIG.s$SEED"
    if [[ ! -d "$SRC" ]]; then
        echo "[link] ABSENT : $SRC — seed ignoré."
        continue
    fi
    # Lien RELATIF (../v5.4.<config>.sN) : survit à un déplacement de l'arbre
    # et à un montage différent entre frontal et nœud de calcul.
    ln -sfn "../v5.4.$CONFIG.s$SEED" "$DEST/s$SEED"
    echo "[link] $DEST/s$SEED -> ../v5.4.$CONFIG.s$SEED"
done

echo "[link] prêt : run_tag = \"$(basename "$RUNS_ROOT")/v5.4.$CONFIG\""
