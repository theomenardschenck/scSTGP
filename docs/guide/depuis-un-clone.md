# Utilisation depuis un clone git

Ce guide décrit l'usage **depuis un dépôt cloné** : mise en place, chaîne de
prétraitement des données, exécution locale et sur cluster. C'est le mode de
travail de l'auteur et celui des vingt scripts SLURM historiques.

Le guide [execution.md](execution.md) couvre l'usage par la commande
`stateshift` ; les deux mènent au même DAG. Le présent document ajoute ce que le
paquet installé **ne peut pas** faire.

## Pourquoi un clone

| Besoin | Paquet pip | Clone |
|---|---|---|
| lancer le pipeline sur des données déjà préparées | ✅ | ✅ |
| **construire les features amont** (co-expression, pySCENIC, HuMess) | ❌ | ✅ |
| grilles d'ablation, vagues, Optuna sur SLURM (`scripts/*.sh`) | ❌ | ✅ |
| jeu jouet (`tests/fixtures/make_tiny_dataset.py`) | ❌ | ✅ |
| caches OmniPath / HGNC | ❌ | ✅ |
| suite de tests | ❌ | ✅ |

La raison est simple : `scripts/` et `tests/` ne sont pas empaquetés, et la règle
Snakemake `build_features` appelle `bash scripts/run_v6_build.sh` en chemin
relatif. **Tout le prétraitement suppose donc un clone.**

## Mise en place

```bash
git clone https://github.com/theomenardschenck/scSTGP.git
cd scSTGP

# environnement de référence — c'est lui qui tourne sur le cluster
micromamba create -n stateshift -f environment.yml
micromamba activate stateshift

pip install -e .          # paquet en éditable : les deux modes cohabitent
stateshift doctor
```

`pip install -e .` n'est pas obligatoire pour le mode script, mais il rend le
Snakefile capable de localiser les scripts par le paquet plutôt que par un
repli, et donne accès à `stateshift path`. Vivement recommandé.

### Disposition attendue des données

```
data/
├── gnn_data/              matrices, métadonnées, tables DE
├── omnipath/              caches OmniPath + HGNC (versionnés, ~8 Mo)
├── ppi/  databases/       STRING, Reactome, bases de vieillissement
├── pyscenic/              adjacences, régulons, coexpr_diff.tsv
└── humess/                importance métabolique par condition
```

Les chemins sont configurables (`paths.*`) ; ce n'est pas une camisole. Ce qui
compte est que `data_root` contienne `omnipath/` — sans lui le pipeline tourne à
vide.

---

# Prétraitement

Quatre couches de données alimentent le graphe. Aucune n'est orchestrée par le
DAG principal : elles sont **précalculées en amont**, volontairement, parce
qu'elles sont lentes, qu'elles dépendent d'outils externes et qu'on les rejoue
rarement.

| Étape | Produit | Où l'exécuter | Dépendances |
|---|---|---|---|
| 0. Extraction R | matrices + DE (MAST) | local | R, Seurat, MAST |
| 1. Matrices par groupe | `expr_<groupe>.csv` | local | — |
| 2. Co-expression GRNBoost2 | `coexpr_diff.tsv` | local **ou cluster** | sklearn |
| 3. pySCENIC (régulons) | `regulon_edges`, `TF_activity` | local ou cluster | bases cisTarget (plusieurs Go) |
| 4. HuMess | `cs_gene_to_importance_<cond>.tsv` | **local uniquement** | conda `humess` + **cplex** |

Les couches 2, 3 et 4 sont toutes **ablatables** : `--no-coexpr`,
`--no-scenic-regulons`, `--no-humess`. Si une source manque, entraînez sans elle
plutôt que de bloquer — et sachez que le régime de graphe change avec, ce que
[resultats.md](resultats.md#le-régime-de-graphe) détaille.

## 0. Extraction depuis R (données scRNA)

Chaîne d'origine du projet, propre au jeu HUVEC.

```bash
export STGP_ROOT="$PWD"
Rscript src/data/extract/sene_clusteringR.R    # QC Seurat + clustering P16
Rscript src/data/extract/export_gnn_full.R     # → CSV pour Python + DE MAST
```

`export_gnn_full.R` exporte **tous les gènes**, pas seulement les HVG, et
calcule la DE avec des seuils relâchés (`min.pct=0.1`, `logfc.threshold=0`) :
les gènes non-DE sont de vrais négatifs pour le GNN, il ne faut pas les perdre.
`STGP_ROOT` remplace les chemins WSL absolus d'origine.

Sur un jeu bulk déjà tabulé, cette étape n'existe pas : passez directement à
l'étape 1.

## 1 → 2. Matrices et co-expression

Le script `run_v6_build.sh` enchaîne les étapes 1, 2 et 4 depuis une matrice
d'expression quelconque :

```bash
bash scripts/run_v6_build.sh \
    --matrix     data/bulkRNAseq/GSE163251/GSE163251_fpkm_all.txt \
    --metadata   data/bulkRNAseq/GSE163251/GSE163251_metadata.tsv \
    --gene-col   Tracking_id --group-col 2 \
    --young-group pro --sen-group sen \
    --out-dir    data/pyscenic/GSE163251 \
    [--cluster] [--no-humess] [--dry-run]
```

Ce qu'il fait, dans l'ordre :

1. **`prep-matrices`** — matrice (genes×samples ou l'inverse, détecté) +
   metadata `sample→groupe` → `expr_<groupe>.csv`. Avec `--emit-humess`, produit
   aussi `abundance_table.tsv` et `samplesheet.tsv`, qui sont les entrées de
   HuMess.
2. **`grnboost2-local`** sur chaque groupe → `adjacencies_<groupe>.csv`. C'est
   aussi l'étape 1 de SCENIC.
3. **`merge-adjacencies`** → `coexpr_diff.tsv`, consommé par
   `gnn_vgae --coexpr-mode differential`.

> ⚠️ **GRNBoost2 sur peu d'échantillons est peu fiable** (p ≫ n).
> `prep-matrices` avertit sous 20 échantillons par groupe. Sur du bulk à faible
> effectif, la couche co-expression est à considérer comme du bruit structuré.

### Faire les étapes à la main

Utile pour rejouer une seule brique :

```bash
python src/data/preprocess/build_diff_coexpr.py prep-matrices \
    --matrix <matrice> --metadata <meta> --group-col 2 --emit-humess \
    --out-dir data/pyscenic/mon_jeu

python src/data/preprocess/build_diff_coexpr.py grnboost2-local \
    --expr data/pyscenic/mon_jeu/expr_pro.csv \
    --tf-list data/pyscenic/scenic_refs/allTFs_hg38.txt \
    --out data/pyscenic/mon_jeu/adjacencies_pro.csv --n-jobs 8

python src/data/preprocess/build_diff_coexpr.py merge-adjacencies \
    --adj-p4 data/pyscenic/mon_jeu/adjacencies_pro.csv \
    --adj-p16 data/pyscenic/mon_jeu/adjacencies_sen.csv \
    --prune-mode per-target-topk --per-target-k 5
```

`--seed` de `grnboost2-local` est le même pour les deux conditions **par
conception** : sinon la différence P4/P16 confond l'effet biologique avec la
variance du GBM.

L'élagage a une vraie conséquence : **`--prune-mode per-target-topk`** (défaut)
garde les K meilleurs régulateurs de *chaque* cible, comme le fait SCENIC.
L'alternative `global-quantile` est **dominée par les hubs** — même à q = 0.98,
des gènes comme ASNS, IL6, IL1B ou DDIT3 se retrouvent sans aucune arête.

### Le cas scRNA : `extract-matrices`

Quand on part d'une matrice scRNA déjà fusionnée plutôt que d'un bulk, la
première étape n'est pas `prep-matrices` mais `extract-matrices`, qui porte le
réglage décisif de l'univers de gènes :

```bash
python src/data/preprocess/build_diff_coexpr.py extract-matrices \
    --merged data/gnn_data/merged_P4_P16_normalized.csv \
    --gene-universe graph \
    --graph-genes <…>/cross_seed_gene_ranking.tsv
```

**`--gene-universe graph`** plutôt que `hvg` : les 5 000 HVG ne couvrent que 40 %
de l'univers du graphe VGAE, le mode `graph` monte à 100 %. Avec
`per-target-topk`, ce sont les deux corrections complémentaires et requises.

### Sur cluster, en deux temps

GRNBoost2 est la seule brique amont qui gagne vraiment à passer sur SLURM :

```bash
bash scripts/run_diff_coexpr.sh --step grnboost2   # soumet l'array
squeue -u $USER                                    # attendre COMPLETED
bash scripts/run_diff_coexpr.sh --step merge       # garde-fou puis merge
```

Les deux étapes sont **délibérément découplées** : l'ancien chaînage
`--dependency=afterok` laissait des jobs de merge coincés en
`DependencyNeverSatisfied` dès que GRNBoost2 échouait, et il fallait les
`scancel` un à un. L'étape `merge` refuse de partir si les adjacences sont
absentes ou vides.

> L'implémentation est **sklearn pur**, pas `arboreto`. Ce dernier est cassé sur
> l'environnement du cluster : dask ≥ 2025.1 a retiré l'API legacy dont il
> dépend. Le portage reprend les hyperparamètres SGBM d'arboreto (Moerman 2019)
> et a été validé par recoupement : 10 des 15 meilleurs TF sont communs.

## 3. pySCENIC — régulons

Étape **optionnelle et lourde**. Les adjacences GRNBoost2 de l'étape 2 sont déjà
l'étape 1 de SCENIC ; ce qui manque est le raffinement par motifs (cisTarget)
puis le scoring (AUCell).

Elle exige des bases de référence à télécharger une fois, plusieurs gigaoctets,
à placer dans `data/pyscenic/scenic_refs/` :

| Fichier | Source |
|---|---|
| `allTFs_hg38.txt` | `resources.aertslab.org/cistarget/tf_lists/` |
| `hg38_*_v10_clust.genes_vs_motifs.rankings.feather` | `resources.aertslab.org/cistarget/databases/homo_sapiens/hg38/` |
| `motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl` | `resources.aertslab.org/cistarget/motif2tf/` |

`run_v6_build.sh` **détecte leur absence et saute l'étape** avec un message : sans
les feathers, seules les adjacences sont produites. C'est un comportement voulu,
pas un échec.

Variante arboreto canonique, si l'environnement le permet :

```bash
python src/data/extract/scenic_from_r.py grnboost2-diff \
    --condition P4 --merged <matrice> --out <adjacencies.csv> --n-workers 6
```

> Sur cluster « offline », les bases doivent être mises en place depuis le nœud
> de connexion. Même logique que `scripts/cache_omnipath.py`, qui pré-télécharge
> OmniPath sur le frontal parce que les nœuds de calcul n'ont pas le réseau.

## 4. HuMess — métabolisme, **en local uniquement**

HuMess produit l'importance métabolique par gène et par condition, via un modèle
métabolique spécifique (GEM + Corner Sampling).

**Pourquoi local seulement :** HuMess repose sur le solveur **cplex**, sous
licence académique IBM, installé poste par poste et non déployé sur le cluster.
Le README HuMess est catégorique sur un autre point : ne pas lui substituer
`gurobi`, qui produit des résultats **faux** avec carveme.

### Mise en place, une fois

```bash
git clone https://gitlab.univ-nantes.fr/bird_pipeline_registry/humess.git ../humess
cd ../humess
conda env create -f conda/humess.yml
# puis installer cplex 22.1.1 (gratuit pour les académiques : http://ibm.biz/CPLEXonAI)
```

### Exécution

Le plus simple est de laisser `run_v6_build.sh` l'orchestrer — il génère la
configuration, injecte le Corner Sampling et appelle le Snakefile de HuMess :

```bash
HUMESS_REPO=../humess bash scripts/run_v6_build.sh \
    --matrix <matrice> --metadata <meta> --group-col 2 \
    --young-group pro --sen-group sen \
    --out-dir data/pyscenic/mon_jeu \
    --humess-out data/humess/mon_jeu
```

Le script est **idempotent** : si `cs_gene_to_importance_<cond>.tsv` existe déjà,
il le signale et ne relance rien.

Détails qui comptent :

- **Corner Sampling est requis** et n'est pas activé par défaut par
  `make_humess_config.py` — le script l'injecte. `n_samples` doit être un entier
  (`auto` fait planter `corner_sampling.py`) ; le run HUVEC validé utilise
  **1000** (`V6_HUMESS_CS`).
- Solveur forcé à `cplex`, score `sample-ratio` au seuil 0.6.
- Sortie attendue : `data/humess/<jeu>/models/<cond>/cs/cs_gene_to_importance_<cond>.tsv`,
  c'est-à-dire `HUMESS_DIR` tel que le lit `gnn_vgae`.

### Si HuMess n'est pas disponible

C'est un cas normal, pas un blocage :

```bash
python src/gnn/gnn_vgae.py --no-humess …
# ou, dans la configuration :  models.vgae.extra_flags: "--no-humess"
```

La couche est ablatable, et elle est de toute façon retirée dans les
configurations « non circulaires ». Sachez seulement que l'ablation
`no-humess` déplace le classement — c'est mesuré, pas anodin.

## Récapitulatif : quoi tourne où

```
   LOCAL (poste de travail)              CLUSTER (SLURM)
   ─────────────────────────             ────────────────
   [0] Extraction R (Seurat, MAST)
   [1] prep-matrices
   [4] HuMess  ← cplex, local only
                                          [2] GRNBoost2 (array)
                                          [3] pySCENIC (si bases présentes)
                                          build_graph → train × graines
                                          perturb × (graine, mode)
                                          agrégation + validations
```

Les sorties du prétraitement sont de simples fichiers : produisez-les où vous
voulez, puis synchronisez-les vers `data_root` du côté où vous entraînez.

---

# Exécution depuis le clone

## En local

```bash
bash workflow/run.sh --init                                    # assistant
bash workflow/run.sh --dry-run --configfile workflow/config/config.<nom>.yaml
bash workflow/run.sh --backend local --cores 8 --configfile workflow/config/config.<nom>.yaml
```

`run.sh` se repère par `dirname $0` : il doit être lancé depuis le clone, et il
s'y replace de lui-même. Le jeu jouet, qui ne demande aucune donnée :

```bash
python tests/fixtures/make_tiny_dataset.py --out data_tiny
bash workflow/run.sh --backend local --configfile workflow/config/config.tiny.yaml
```

> **Piège d'environnement.** `compute.python: "python"` signifie « le python de
> l'environnement actif ». Si votre `PATH` pointe ailleurs — le venv d'un autre
> projet, par exemple — les règles s'exécutent avec un interpréteur sans torch.
> Le Snakefile choisit désormais son interpréteur par capacité et le dit au
> lancement, mais le plus sûr reste d'activer l'environnement du projet. Pour
> forcer : `export STATESHIFT_PYTHON=$(which python)`.

## Sur cluster

Deux voies, complémentaires.

### Voie 1 — Snakemake pilote tout

```bash
tmux new -s stateshift
bash workflow/run.sh --backend cluster --jobs 20 \
     --configfile workflow/config/config.<nom>.yaml
```

Adaptez d'abord partition et QOS dans `workflow/profiles/slurm/config.yaml`
(`sinfo`, `sacctmgr show qos format=Name,MaxWall`). Sans le profil, tout
s'exécute sur le nœud de connexion — rien n'apparaît dans `squeue`.

### Voie 2 — job arrays SLURM, puis analyse

C'est la voie des vagues d'ablation et de l'entraînement GPU. Les scripts
soumettent **un seul job array** ; vous pouvez fermer la session, SLURM pilote.

```bash
# entraînement d'une grille d'ablations (produit cartésien version × ablations)
bash scripts/run_ablation_grid.sh --version V5.4 --seeds "1 2 3"

# entraînement généralisé V6 (bulk/scRNA), 1 job par graine, GPU si possible
bash scripts/run_v6_train.sh --run-tag mon_jeu --seeds "1 2 3" \
     --matrix data/pyscenic/mon_jeu/expr_all.csv \
     --group-meta data/pyscenic/mon_jeu/samplesheet.tsv \
     --coexpr-file data/pyscenic/mon_jeu/coexpr_diff.tsv \
     --humess-dir data/humess/mon_jeu \
     --cell-groups pro,sen --humess-conditions pro,sen

# perturbation sur des runs entraînés
bash scripts/run_perturbation_grid.sh --pattern 'output/gnn_vgae/V6/*.s*' --axis v4

# analyse post-perturbation, sans SLURM, sur des runs déjà perturbés
bash scripts/run_analysis.sh --out <dir> --seeds <rundir1> <rundir2> [--decoy] [--figures]
```

Puis, pour laisser Snakemake reprendre l'aval sans ré-entraîner, mettez
`models.vgae.enabled: false` dans la configuration et relancez `run.sh`.

Tous ces scripts acceptent `--dry-run`. Servez-vous-en : une vague mal
paramétrée coûte des heures de file d'attente.

### Sorties sur scratch

```bash
export GNN_OUT_DIR_BASE=/scratch/$USER/stateshift-output
```

Prime sur `paths.out_base`. La pratique établie : calculer sur scratch, puis
`rsync` les runs figés vers le stockage permanent.

## Mode script pur

Rien n'oblige à passer par Snakemake. Chaque étape est un point d'entrée :

```bash
python src/gnn/gnn_vgae.py --build-only --graph-cache <cache.pkl> [ablations]
python src/gnn/gnn_vgae.py --reuse-graph --graph-cache <cache.pkl> --seed 1
python src/perturbation/perturb_top_genes.py --all-genes --run-dir <run>
python src/validation/reports/perturb_report.py --cross-seed <run1> <run2> <run3>
```

C'est ce mode qu'utilisent les scripts SLURM, et il reste couvert par les tests
(`tests/test_cli_contract.py` vérifie que chaque point d'entrée répond à
`--help`). Les chemins des données se pilotent par les variables
`GNN_DATA_DIR`, `GNN_OUT_DIR_BASE`, `GNN_HUMESS_DIR`, `GNN_SCENIC_DIR`.

## Tests

```bash
pytest                  # structure, ~5 min
pytest -m slow          # la chaîne réelle sur jeu jouet, ~2 min
ruff check .
```

## Rester à jour

```bash
git pull
pip install -e .        # si pyproject a changé
stateshift doctor
```

Après un `git pull` touchant au code, Snakemake ré-exécutera les règles
concernées : il déclenche aussi sur la provenance, pas seulement sur les dates.
Pour ne considérer que les dates : `-- --rerun-triggers mtime`.
