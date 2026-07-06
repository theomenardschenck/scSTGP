#!/usr/bin/env bash
# =============================================================================
# run.sh — lanceur simple du pipeline VGAE (Snakemake), backend local OU cluster.
#
# C'est LA commande d'entrée pour un utilisateur : elle cache les incantations
# Snakemake et choisit l'exécuteur.
#
#   bash workflow/run.sh [--backend local|cluster] [--configfile FILE]
#                        [--cores N] [--jobs N] [--dry-run] [-- <args snakemake>]
#
#   --backend local   (DÉFAUT) → snakemake --cores N        (+ WARNING : les vraies
#                                charges — build ~40 min, entraînement — devraient
#                                tourner sur le cluster ; utiliser --backend cluster)
#   --backend cluster         → snakemake --executor slurm --profile <profil slurm>
#
# Le backend par défaut peut aussi être fixé dans config.yaml (compute.backend).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."               # → racine du dépôt (gnn_huvec)

SNAKEFILE="workflow/Snakefile"
CONFIGFILE="workflow/config/config.yaml"
PROFILE="workflow/profiles/slurm"
BACKEND=""                            # vide → lu depuis config.yaml, sinon 'local'
CORES=8
JOBS=20
DRY=0
PASS=()                               # args bruts passés à snakemake (après --)

while [[ $# -gt 0 ]]; do case "$1" in
  --backend)     BACKEND="$2"; shift 2 ;;
  --configfile)  CONFIGFILE="$2"; shift 2 ;;
  --cores)       CORES="$2"; shift 2 ;;
  --jobs)        JOBS="$2"; shift 2 ;;
  --dry-run|-n)  DRY=1; shift ;;
  --)            shift; PASS=("$@"); break ;;
  -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
  *)             echo "[run] arg inconnu : $1 (utilise -- pour passer des args à snakemake)"; exit 1 ;;
esac; done

# Backend par défaut : --backend > config.yaml:compute.backend > 'local'.
if [[ -z "$BACKEND" ]]; then
  BACKEND="$(python - "$CONFIGFILE" <<'PY' 2>/dev/null || echo local
import sys, yaml
try:
    print((yaml.safe_load(open(sys.argv[1])).get("compute", {}) or {}).get("backend", "local"))
except Exception:
    print("local")
PY
)"
fi

command -v snakemake >/dev/null || { echo "[run] snakemake introuvable dans le PATH."; exit 1; }

COMMON=(-s "$SNAKEFILE" --configfile "$CONFIGFILE")
[[ $DRY -eq 1 ]] && COMMON+=(-n)

case "$BACKEND" in
  local)
    echo "⚠️  [run] BACKEND=local — exécution sur CETTE machine. Les vraies charges"
    echo "        (build du graphe ~40 min, entraînement) devraient tourner sur le"
    echo "        cluster : relance avec --backend cluster (ou compute.backend: cluster)."
    set -x
    snakemake "${COMMON[@]}" --cores "$CORES" "${PASS[@]}"
    ;;
  cluster)
    echo "[run] BACKEND=cluster — soumission SLURM via le profil $PROFILE (jobs=$JOBS)."
    set -x
    snakemake "${COMMON[@]}" --executor slurm --profile "$PROFILE" --jobs "$JOBS" "${PASS[@]}"
    ;;
  *)
    echo "[run] backend inconnu : '$BACKEND' (attendu : local | cluster)"; exit 1 ;;
esac
