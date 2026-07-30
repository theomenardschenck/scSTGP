# Prep GSE252921 — coexpr + SCENIC + HuMess (single-nucleus, système externe)

Workflow Snakemake qui produit les **3 entrées data-dérivées** du graphe V6
depuis les RDS snRNA de GSE252921 (test de capacité de l'encodeur sur un
système ≠ HUVEC). Le graphe réutilise PPI/Reactome/reactome_fi/OmniPath tels
quels ; seuls coexpr + SCENIC + HuMess + features viennent de ce dataset.

## Sorties (dans `out_dir`)
| Fichier | Rôle graphe | Produit par |
|---|---|---|
| `coexpr_diff.tsv` | arêtes coexpression | GRNBoost2-local diff (base env) |
| `scenic_out/regulon_edges_TF_to_gene.csv` + `mean_TF_activity_per_cluster.csv` | regulons + features TF | pySCENIC ctx+AUCell (env `arboreto`) |
| `<humess_dir>/models/{base,adv}/cs/cs_gene_to_importance_*.tsv` | importance métabolique | HuMess (env `humess`, cplex) |

## Lancer
```bash
# depuis la racine gnn_huvec/
snakemake -s workflow/prep_gse252921/Snakefile \
    --configfile workflow/prep_gse252921/config.endo.yaml -j 8 -p
# contrôle négatif Oligo (⚠ cluster : count_all_celltype ~6 Go) :
snakemake -s workflow/prep_gse252921/Snakefile \
    --configfile workflow/prep_gse252921/config.oligo.yaml -j 8 -p
```

## Prérequis
- envs conda : `arboreto` (+ `pip install "setuptools<81" pyscenic`) et `humess` (+ cplex).
- feathers cisTarget dans `data/pyscenic/scenic_refs/*.feather`.
- `Rscript` + paquet `Matrix` ; `scipy/pandas` pour la conversion mtx→csv.

## Chaînage vers l'entraînement
Une fois `all` atteint, entraîner le VGAE (config sn complète, regulons SCENIC ON) :
```bash
bash scripts/run_v6_train.sh --run-tag gse252921_endo --seeds "1 2 3" \
  --matrix      data/pyscenic/GSE252921_endo/expr_all.csv \
  --group-meta  data/pyscenic/GSE252921_endo/samplesheet.tsv \
  --coexpr-file data/pyscenic/GSE252921_endo/coexpr_diff.tsv \
  --humess-dir  data/humess/GSE252921_endo \
  --scenic-dir  data/pyscenic/GSE252921_endo/scenic_out \
  --cell-groups CT,AD --humess-conditions CT,AD \
  --extra-flags "--use-scenic-regulons --use-omnipath-signaling --use-omnipath-tf-curated --include-omnipath-genes --use-reactome-fi --signed-message --signed-decoder --coexpr-mode differential"
```

## Notes
- `baseline`/`advanced` = valeurs de groupe (ex. CT/AD) ; ordre = axe
  référence→avancé (imp_delta HuMess, sens de l'axe readout).
- Le contrôle **Oligo** rejoue tout depuis le tar (extract→export inclus) ;
  l'endo saute les étapes déjà faites (HuMess, export).
- SCENIC lance son propre GRNBoost2 (arboreto) sur `expr_all.csv` — indépendant
  du coexpr différentiel (qui, lui, est baseline-vs-advanced).
