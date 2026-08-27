# Preprocess — coexpr + SCENIC + HuMess (générique, data-adaptatif)

Produit les **3 entrées data-dérivées** du graphe V6 depuis les données
single-nucleus/single-cell de l'utilisateur. Le graphe réutilise
PPI/Reactome/reactome_fi/OmniPath tels quels ; seuls coexpr + SCENIC + HuMess +
features viennent du dataset.

## 1. Situer les données — le wizard
```bash
python workflow/preprocess/init.py
```
Il demande : source (tar GEO+RDS, ou RDS directs), colonnes de metadata
(condition / donneur / type cellulaire), valeurs baseline→advanced, sous-
échantillon… → écrit `workflow/preprocess/config.<dataset>.yaml` et affiche la
commande de lancement adaptée.

## 2. Le découpage LOCAL / CLUSTER (contrainte : HuMess = cplex = local)

| Étape | Où | Pourquoi |
|---|---|---|
| extract → export → mtx_to_expr | là où sont les données + la RAM | gros RDS sn → cluster ; petit → local |
| **coexpr** (GRNBoost2), **SCENIC** | **cluster** | compute lourd |
| **HuMess** (`localrule`) | **local uniquement** | cplex ; jamais soumis à SLURM |

I/O de HuMess = petites (`abundance_table`+`samplesheet` → `cs_gene_to_importance`
~80 Ko) → faciles à synchroniser entre les deux machines.

### Tout en local (petites données + cplex)
```bash
snakemake -s workflow/preprocess/Snakefile \
    --configfile workflow/preprocess/config.<dataset>.yaml -j 8
```

### Split gros volume (2 machines)
```bash
# 1) CLUSTER — export + coexpr + SCENIC en jobs SLURM, SANS HuMess :
snakemake -s workflow/preprocess/Snakefile \
    --configfile workflow/preprocess/config.<dataset>.yaml \
    --profile workflow/profiles/slurm --jobs 20 --omit-from humess

# 2) LOCAL — HuMess seul (cplex) ; ne nécessite que abundance_table+samplesheet :
snakemake -s workflow/preprocess/Snakefile \
    --configfile workflow/preprocess/config.<dataset>.yaml -j 4 \
    data/humess/<dataset>/models/<baseline>/cs/cs_gene_to_importance_<baseline>.tsv \
    data/humess/<dataset>/models/<advanced>/cs/cs_gene_to_importance_<advanced>.tsv

# 3) rsync des cs_gene_to_importance (local) → data/humess/<dataset>/ (cluster)
```

## Sorties (dans `out_dir` / `humess_dir`)
| Fichier | Rôle graphe |
|---|---|
| `coexpr_diff.tsv` | arêtes coexpression |
| `scenic_out/{regulon_edges_TF_to_gene.csv, mean_TF_activity_per_cluster.csv}` | regulons + features TF |
| `<humess_dir>/models/{base,adv}/cs/cs_gene_to_importance_*.tsv` | importance métabolique |

## Prérequis
- Env conda `arboreto` (**+ `pip install "setuptools<81" pyscenic`** — même
  sur le cluster) et `humess` (+ cplex, local).
- Feathers cisTarget dans `data/pyscenic/scenic_refs/*.feather`.
- La règle `scenic` **réutilise les adjacences coexpr** (`adjacencies_arb.csv`
  poolé top-50/cible) → `scenic_from_r.py` saute son GRNBoost2 arboreto+dask
  (qui hangue sur GLiCID). Rien à faire, c'est câblé.

## Chaînage → entraînement
Une fois les 3 sorties là, entraîner via le workflow principal :
```bash
bash workflow/run.sh --backend cluster \
    --configfile workflow/config/config.<dataset>.yaml
```
(le config principal pointe `paths.{coexpr_file,scenic_dir,humess_dir}` +
`dataset.{cell_groups,expr_matrix,group_meta}` sur les sorties du preprocess).

## Configs d'exemple
- `config.GSE252921_endo.yaml` — EC AD vs CT.
- `config.GSE252921_oligo.yaml` — contrôle négatif Oligo (⚠ export sur cluster,
  153k noyaux > 7 Go).
