#!/usr/bin/env bash
# =============================================================================
# run_scenic.sh — régulons pySCENIC (cisTarget + AUCell) pour un jeu quelconque.
# =============================================================================
# Pourquoi ce script : la commande brute enchaîne un `python -c` multiligne et
# un `sbatch --wrap` à guillemets imbriqués — increcopiable sans faute. Tout est
# ici, paramétré par --dir et --conditions.
#
#   bash scripts/run_scenic.sh --dir data/pyscenic/GSE96583 --conditions ctrl,stim
#   bash scripts/run_scenic.sh --dir … --conditions … --cluster    # soumet en job
#   bash scripts/run_scenic.sh --dir … --conditions … --dry-run
#
# CE QU'IL FAIT
#   1. fusionne les adjacences GRNBoost2 des deux conditions (moyenne des
#      importances, top-K par cible) → <dir>/scenic_out/adjacencies_arb.csv.
#      scenic_from_r.py DÉTECTE ce fichier et saute son propre GRNBoost2 :
#      celui-ci passe par arboreto+dask, qui se bloque sur GLiCID.
#   2. lance scenic_from_r.py (modules → cisTarget → AUCell) avec les variables
#      STGP_SCENIC_* qui le détournent des chemins HUVEC par défaut.
#
# PRÉREQUIS
#   • étape co-expression faite : adjacencies_<cond>.csv pour les 2 conditions
#   • bases cisTarget dans data/pyscenic/scenic_refs/ (allTFs, ≥1 feather, motifs)
#   • pyscenic installé dans l'env visé :
#       conda run -n arboreto pip install "setuptools<81" pyscenic
#
# SORTIES (lues par le graphe si --use-scenic-regulons)
#   <dir>/scenic_out/regulon_edges_TF_to_gene.csv
#   <dir>/scenic_out/mean_TF_activity_per_cluster.csv
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

DIR=""; CONDITIONS=""; ENV_NAME="${SCENIC_ENV:-arboreto}"; WORKERS=8
MEM="64G"; TIME="03:00:00"; PARTITION="standard"; QOS="short"
TOPK=50; CLUSTER=0; DRY=0
PY_LOCAL="${SCENIC_PY:-python}"

while [[ $# -gt 0 ]]; do case "$1" in
  --dir)         DIR="$2"; shift 2 ;;
  --conditions)  CONDITIONS="$2"; shift 2 ;;
  --env)         ENV_NAME="$2"; shift 2 ;;
  --workers)     WORKERS="$2"; shift 2 ;;
  --top-k)       TOPK="$2"; shift 2 ;;
  --mem)         MEM="$2"; shift 2 ;;
  --time)        TIME="$2"; shift 2 ;;
  --partition)   PARTITION="$2"; shift 2 ;;
  --qos)         QOS="$2"; shift 2 ;;
  --cluster)     CLUSTER=1; shift ;;
  --dry-run)     DRY=1; shift ;;
  -h|--help)     sed -n '2,32p' "$0"; exit 0 ;;
  *) echo "[scenic] option inconnue : $1"; exit 1 ;;
esac; done

[[ -n "$DIR" && -n "$CONDITIONS" ]] || { echo "[scenic] --dir et --conditions requis"; exit 1; }
IFS=',' read -r COND_A COND_B <<< "$CONDITIONS"
OUT="$DIR/scenic_out"
ADJ_A="$DIR/adjacencies_${COND_A}.csv"; ADJ_B="$DIR/adjacencies_${COND_B}.csv"
EXPR="$DIR/expr_all.csv"; META="$DIR/cell_metadata.csv"

# En --dry-run on AVERTIT au lieu de sortir : le but est de voir ce qui serait
# lancé, y compris avant que la co-expression ait produit ses adjacences.
missing=0
for f in "$ADJ_A" "$ADJ_B" "$EXPR" "$META"; do
  [[ -s "$f" ]] || { echo "[scenic] manquant : $f"; missing=1; }
done
REFS=data/pyscenic/scenic_refs
ls "$REFS"/*.feather >/dev/null 2>&1 || {
  echo "[scenic] aucun feather cisTarget dans $REFS"
  echo "         resources.aertslab.org/cistarget/databases/homo_sapiens/hg38/"; missing=1; }
if [[ $missing -eq 1 ]]; then
  echo "         adjacences → scripts/run_diff_coexpr.sh ;"
  echo "         expr/metadata → scripts/sc_to_inputs.py"
  [[ $DRY -eq 1 ]] || exit 1
  echo "[scenic] --dry-run : on continue malgré les manquants"
fi

mkdir -p "$OUT"
echo "[scenic] jeu=$DIR  conditions=$COND_A,$COND_B  env=$ENV_NAME  workers=$WORKERS"

# --- 1. adjacences fusionnées (top-K par cible) -----------------------------
MERGE_PY=$(cat <<PYEOF
import pandas as pd
a = pd.read_csv("$ADJ_A"); b = pd.read_csv("$ADJ_B")
m = (pd.concat([a, b]).groupby(["TF", "target"], as_index=False)["importance"].mean()
       .sort_values("importance", ascending=False).groupby("target").head($TOPK))
m.to_csv("$OUT/adjacencies_arb.csv", index=False)
print(f"[scenic] adjacencies_arb.csv : {len(m)} paires, {m.TF.nunique()} TFs")
PYEOF
)

# --- 2. corps SCENIC (identique local/cluster) ------------------------------
BODY=$(cat <<BEOF
set -euo pipefail
cd "$PWD"
[ -f scripts/cluster_env.glicid.sh ] && source scripts/cluster_env.glicid.sh || true
export STGP_SCENIC_EXPR="\$(readlink -f "$EXPR")"
export STGP_SCENIC_META="\$(readlink -f "$META")"
export STGP_SCENIC_RESULTS="\$(readlink -f "$OUT")"
export STGP_SCENIC_CLUSTER_COL=group
export STGP_SCENIC_WORKERS=$WORKERS
export STGP_SCENIC_MEM_LIMIT=$MEM
conda run -n $ENV_NAME python src/data/extract/scenic_from_r.py
BEOF
)

if [[ $DRY -eq 1 ]]; then
  echo "--- fusion des adjacences ---"; echo "$MERGE_PY"
  echo "--- corps SCENIC ---";          echo "$BODY"
  [[ $CLUSTER -eq 1 ]] && echo "--- soumission : sbatch -p $PARTITION --qos $QOS -t $TIME -c $WORKERS --mem $MEM"
  exit 0
fi

"$PY_LOCAL" -c "$MERGE_PY"

if [[ $CLUSTER -eq 1 ]]; then
  mkdir -p logs/slurm
  JOB="$OUT/_scenic_job.sh"
  { echo '#!/usr/bin/env bash'; echo "$BODY"; } > "$JOB"; chmod +x "$JOB"
  jid=$(sbatch --parsable --partition="$PARTITION" --qos="$QOS" --time="$TIME" \
        --cpus-per-task="$WORKERS" --mem="$MEM" --job-name=scenic \
        --output="logs/slurm/scenic_%j.log" "$JOB")
  echo "[scenic] job $jid soumis — suivi : tail -f logs/slurm/scenic_${jid}.log"
else
  echo "[scenic] exécution LOCALE (⚠ sur un frontal à 7 Go, OOM garanti)"
  bash -c "$BODY"
fi
