#!/usr/bin/env bash
# =============================================================================
# run_v6_build.sh — Construction des features/arêtes data-dérivées V6 depuis
# une matrice d'expression quelconque : coexpr (GRNBoost2) + SCENIC + HuMess.
# Tourne EN LOCAL (défaut) ou sur le CLUSTER (--cluster, SLURM GLiCID).
# =============================================================================
# Chaîne (couche coexpr/SCENIC-GRNBoost2, PRÊTE, sklearn pur — zéro dep lourde) :
#   [1] prep-matrices : matrice (counts/FPKM, genes×samples) + metadata
#                       (sample→groupe) → expr_<groupe>.csv (échantillons×gènes)
#   [2] grnboost2-local (young + sen) → adjacencies_<groupe>.csv
#       (= aussi l'étape 1 de SCENIC : adjacences TF→cible)
#   [3] merge-adjacencies → coexpr_diff.tsv   (consommé par gnn_vgae --coexpr-mode differential)
#
# DÉPENDANCES EXTERNES (détectées, sinon SKIP avec message) :
#   • SCENIC regulons (regulon_edges/TF_activity) = cisTarget → exige les
#     bases de ranking motif (feathers, plusieurs Go) dans scenic_refs/.
#     Sans elles : seules les adjacences GRNBoost2 sont produites (step 1).
#   • HuMess (cs_gene_to_importance_<cond>.tsv) = modèle métabolique externe
#     (GEM + Corner Sampling). PAS générable ici. Si absent → entraîner le
#     graphe avec `--no-humess` (couche ablatable, retirée de toute façon en no-circ).
#
# ⚠️ n faible : GRNBoost2 sur peu d'échantillons (bulk) = réseau peu fiable
#    (p≫n). prep-matrices avertit si <20 échantillons/groupe.
#
# Usage :
#   bash scripts/run_v6_build.sh \
#       --matrix data/bulkRNAseq/GSE163251_huvec/GSE163251_fpkm_all.txt \
#       --gene-col Tracking_id --group-col 2 \
#       --metadata data/bulkRNAseq/GSE163251_huvec/GSE163251_metadata.tsv \
#       --young-group pro --sen-group sen \
#       --out-dir data/pyscenic/GSE163251 [--cluster] [--dry-run]
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MATRIX=""; METADATA=""; GENE_COL=""; GROUP_COL=1
YOUNG="pro"; SEN="sen"; OUT_DIR="data/pyscenic/diff_coexpr"
TF_LIST="${V6_TF_LIST:-data/pyscenic/scenic_refs/allTFs_hg38.txt}"
# HuMess (modèle métabolique) : repo + sortie. Cache local 'humess/' ou cluster
# /LAB-DATA/.../humess ; version finale = gnn_huvec/data/humess.
HUMESS_REPO="${HUMESS_REPO:-$(ls -d ../humess /LAB-DATA/GLiCID/users/*/humess 2>/dev/null | head -1 || true)}"
HUMESS_OUT="${HUMESS_DIR:-data/humess}"   # = HUMESS_DIR lu par gnn_vgae
RUN_HUMESS=1; GENETYPE=symbol; HUMESS_SCORE=sample-ratio; HUMESS_SCORE_THRESH=0.6
# corner sampling REQUIS pour cs_gene_to_importance (lu par gnn_vgae). make_humess_config
# ne l'active pas par défaut → on l'injecte. ⚠️ corner_sampling.py fait int(n_samples)
# ('auto' crashe) ; le run HUVEC validé = 1000 (cf. config_used.yaml).
HUMESS_CS_NSAMPLES="${V6_HUMESS_CS:-1000}"
HUMESS_COMP_DE="${V6_HUMESS_COMP_DE:-}"   # DE optionnel (gene/logFC/padj) → comparaison DGE pro vs sen
NWORK=8; PRUNE="per-target-topk"; PTK=5
CLUSTER=0; DRY=0; CPUS=8; MEM="8G"
CL="${V6_CLUSTER:-nautilus}"; CPU_TIME="${V6_CPU_TIME:-0-08:00:00}"
CPU_PART="${V6_CPU_PARTITION:-standard}"; CPU_QOS="${V6_CPU_QOS:-short}"

while [[ $# -gt 0 ]]; do case "$1" in
  --matrix) MATRIX="$2"; shift 2;;
  --metadata) METADATA="$2"; shift 2;;
  --gene-col) GENE_COL="$2"; shift 2;;
  --group-col) GROUP_COL="$2"; shift 2;;
  --young-group) YOUNG="$2"; shift 2;;
  --sen-group) SEN="$2"; shift 2;;
  --out-dir) OUT_DIR="$2"; shift 2;;
  --tf-list) TF_LIST="$2"; shift 2;;
  --n-workers) NWORK="$2"; shift 2;;
  --per-target-k) PTK="$2"; shift 2;;
  --cpus) CPUS="$2"; shift 2;;
  --mem-per-cpu) MEM="$2"; shift 2;;
  --cluster) CLUSTER=1; shift;;
  --no-humess) RUN_HUMESS=0; shift;;
  --humess-repo) HUMESS_REPO="$2"; shift 2;;
  --humess-out) HUMESS_OUT="$2"; shift 2;;
  --dry-run) DRY=1; shift;;
  -h|--help) sed -n '2,40p' "$0"; exit 0;;
  *) echo "arg inconnu : $1"; exit 1;;
esac; done
[[ -z "$MATRIX" || -z "$METADATA" ]] && { echo "--matrix et --metadata requis"; exit 1; }

GC_ARG=""; [[ -n "$GENE_COL" ]] && GC_ARG="--gene-col \"$GENE_COL\""
BD="src/data/preprocess/build_diff_coexpr.py"
ADJ_Y="$OUT_DIR/adjacencies_${YOUNG}.csv"; ADJ_S="$OUT_DIR/adjacencies_${SEN}.csv"
COEXPR="$OUT_DIR/coexpr_diff.tsv"

# Décision diff vs poolé : GRNBoost2 (GBM) exige n≥~5/groupe. En-dessous, le
# différentiel par groupe est vide/instable → réseau UNIQUE poolé (tous
# échantillons) = adjacencies.csv (gnn_vgae --coexpr-mode p16_only).
MIN_N="${V6_MIN_COEXPR_N:-5}"
count_grp() { python3 - "$METADATA" "$GROUP_COL" "$1" <<'PY'
import sys, csv
f, gc, g = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sep = "\t" if f.endswith(".tsv") else ","
n = 0
with open(f) as fh:
    r = csv.reader(fh, delimiter=sep); next(r, None)
    for row in r:
        if len(row) > gc and row[gc] == g:
            n += 1
print(n)
PY
}
NY=$(count_grp "$YOUNG"); NS=$(count_grp "$SEN")
MODE=diff; [[ "$NY" -lt "$MIN_N" || "$NS" -lt "$MIN_N" ]] && MODE=pooled
echo "[build] groupes : $YOUNG=$NY $SEN=$NS (seuil $MIN_N) → mode coexpr = $MODE"

# corps de la chaîne (commun local/cluster) — \$PY résolu au runtime
read -r -d '' BODY <<EOF || true
set -euo pipefail
PY="\${V6_PY:-python3}"
echo "[build] (1) prep-matrices (+ entrées HuMess)"
\$PY $BD prep-matrices --matrix "$MATRIX" --metadata "$METADATA" \\
    --group-col $GROUP_COL $GC_ARG --emit-humess --out-dir "$OUT_DIR"
if [[ -f "$COEXPR" ]]; then
  echo "[build] (2-3) coexpr CACHÉE ($COEXPR) → skip GRNBoost2/merge"
elif [[ "$MODE" == "diff" ]]; then
  for G in "$YOUNG" "$SEN"; do
    echo "[build] (2) grnboost2-local groupe \$G"
    \$PY $BD grnboost2-local --expr "$OUT_DIR/expr_\${G}.csv" \\
        --tf-list "$TF_LIST" --n-jobs $NWORK --out "$OUT_DIR/adjacencies_\${G}.csv"
  done
  echo "[build] (3) merge-adjacencies → coexpr_diff.tsv (différentiel)"
  \$PY $BD merge-adjacencies --adj-p4 "$ADJ_Y" --adj-p16 "$ADJ_S" \\
      --prune-mode $PRUNE --per-target-k $PTK --overwrite --out "$COEXPR"
  echo "[build] coexpr (différentiel) OK → $COEXPR"
else
  echo "[build] (2) grnboost2-local POOLÉ (n faible) → adjacencies.csv"
  \$PY $BD grnboost2-local --expr "$OUT_DIR/expr_all.csv" \\
      --tf-list "$TF_LIST" --n-jobs $NWORK --out "$OUT_DIR/adjacencies.csv"
  echo "[build] (3) élagage per-target-K → coexpr_diff.tsv (poolé symétrique)"
  # même réseau en 2 'conditions' → imp_p4==imp_p16, élagué top-K/cible
  # (sinon l'adjacencies brut ~22M arêtes à n faible noie le graphe).
  \$PY $BD merge-adjacencies --adj-p4 "$OUT_DIR/adjacencies.csv" \\
      --adj-p16 "$OUT_DIR/adjacencies.csv" \\
      --prune-mode $PRUNE --per-target-k $PTK --overwrite --out "$COEXPR"
  rm -f "$OUT_DIR/adjacencies.csv"   # brut volumineux, plus nécessaire
  echo "[build] coexpr (poolé, élagué) OK → $COEXPR (--coexpr-mode differential)"
fi
# --- SCENIC regulons (cisTarget) : détecté, sinon SKIP ---
if ls "$(dirname "$TF_LIST")"/*.feather >/dev/null 2>&1; then
  echo "[build] SCENIC : bases motif présentes → regulons (cisTarget) à lancer (pyscenic ctx/aucell)"
else
  echo "[build] SCENIC : bases ranking motif (*.feather) absentes de $(dirname "$TF_LIST") → seules les adjacences GRNBoost2 sont produites (regulons SKIP)"
fi
# --- HuMess : cache → sinon RUN (modèle métabolique ; bulk OK, robuste petit-n) ---
HUMESS_IMP="$HUMESS_OUT/models/$YOUNG/cs/cs_gene_to_importance_$YOUNG.tsv"
if [[ -f "\$HUMESS_IMP" ]]; then
  echo "[build] HuMess : CACHÉ (\$HUMESS_IMP) → gnn_vgae HUMESS_DIR=$HUMESS_OUT"
elif [[ "$RUN_HUMESS" -eq 1 && -n "$HUMESS_REPO" && -f "$HUMESS_REPO/Snakefile" ]]; then
  echo "[build] HuMess : ABSENT → génération (repo $HUMESS_REPO ; conda env 'humess' + cplex requis)"
  mkdir -p "$HUMESS_OUT"
  # CHEMINS ABSOLUS obligatoires : le Snakemake HuMess tourne depuis son repo
  # (cd) → tout chemin relatif baké dans le config y serait invalide.
  HUMESS_ABUN="\$(readlink -f "$OUT_DIR/abundance_table.tsv")"
  HUMESS_SHEET="\$(readlink -f "$OUT_DIR/samplesheet.tsv")"
  HUMESS_OUT_ABS="\$(readlink -f "$HUMESS_OUT")"
  HUMESS_REPO_ABS="\$(readlink -f "$HUMESS_REPO")"
  # comparaison DGE optionnelle : DE → 3 colonnes (gene/logFC/padj) + comp_file
  COMP_ARG=""
  if [[ -n "$HUMESS_COMP_DE" ]]; then
    \$PY -c "import pandas as pd,sys; d=pd.read_csv(sys.argv[1],sep='\t'); d[['gene_symbol','log2FoldChange','padj']].dropna().to_csv(sys.argv[2],sep='\t',index=False)" "$HUMESS_COMP_DE" "$OUT_DIR/humess_de3.tsv"
    printf '%s\\t%s\\t%s\\n' "$YOUNG" "$SEN" "\$(readlink -f "$OUT_DIR/humess_de3.tsv")" > "$OUT_DIR/comp_file.tsv"
    COMP_ARG="--comparisons \$(readlink -f "$OUT_DIR/comp_file.tsv")"
  fi
  \$PY "\$HUMESS_REPO_ABS/scripts/make_humess_config.py" \\
      -s "\$HUMESS_SHEET" -a "\$HUMESS_ABUN" \\
      -o "\$HUMESS_OUT_ABS" --genetype $GENETYPE --solver cplex \\
      --score $HUMESS_SCORE -t $HUMESS_SCORE_THRESH \$COMP_ARG > "\$HUMESS_OUT_ABS/config.json" \\
    && \$PY -c "import json,sys;c=sys.argv[1];n=sys.argv[2];d=json.load(open(c));d['corner_sampling']={'n_samples': int(n) if n.isdigit() else n};open(c,'w').write(json.dumps(d,indent=4))" "\$HUMESS_OUT_ABS/config.json" "$HUMESS_CS_NSAMPLES"
  # --keep-going : les règles d'annotation finales (HumanGEM/WGCNA/KEGG) ont besoin
  # d'internet (ensembl/KEGG) ; leur échec ne doit PAS bloquer l'importance par
  # condition (cs_gene_to_importance), seule chose lue par gnn_vgae.
  ( cd "\$HUMESS_REPO_ABS" && conda run -n humess snakemake -p --keep-going \\
       --configfile "\$HUMESS_OUT_ABS/config.json" -j $NWORK ) || true
  if [[ -f "\$HUMESS_OUT_ABS/models/$YOUNG/cs/cs_gene_to_importance_$YOUNG.tsv" \\
     && -f "\$HUMESS_OUT_ABS/models/$SEN/cs/cs_gene_to_importance_$SEN.tsv" ]]; then
    echo "[build] HuMess OK → importance produite (annotation online optionnelle ignorée) ; HUMESS_DIR=\$HUMESS_OUT_ABS"
  else
    echo "[build] HuMess ÉCHEC : pas d'importance produite → entraîner avec --no-humess"
  fi
else
  echo "[build] HuMess : ABSENT + non lancé (--no-humess ou repo introuvable) → --no-humess"
fi
echo "[build] DONE"
EOF

if [[ $DRY -eq 1 ]]; then echo "[DRY-RUN] corps :"; echo "$BODY"; exit 0; fi

if [[ $CLUSTER -eq 0 ]]; then
  echo "[build] mode LOCAL"
  V6_PY="${V6_PY:-python3}" bash -c "$BODY"
else
  TS="$(date +%Y%m%d_%H%M%S)"; LOG="scripts/logs/v6_build_${TS}"; mkdir -p "$LOG"
  SB="$LOG/sbatch_build.sh"
  { echo "#!/usr/bin/env bash"
    echo "#SBATCH --job-name=v6_build"
    echo "#SBATCH --output=$LOG/%x_%j.out"
    echo "#SBATCH --error=$LOG/%x_%j.err"
    echo "#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=$CPUS --mem-per-cpu=$MEM"
    echo "# module load python/3.12 2>/dev/null || true"
    echo "$BODY"; } > "$SB"
  CLA=(); [[ -n "$CL" ]] && CLA=(--clusters="$CL")
  A=(--partition="$CPU_PART" --time="$CPU_TIME"); [[ -n "$CPU_QOS" ]] && A+=(--qos="$CPU_QOS")
  JID=$(sbatch --parsable "${CLA[@]}" "${A[@]}" "$SB"); JID="${JID%%;*}"
  echo "[build] soumis CPU : job $JID ; logs $LOG/ ; squeue --me"
fi
