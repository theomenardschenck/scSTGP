# Configuration

Un run est entièrement décrit par un fichier YAML. Le plus simple est de le
générer avec `stateshift init` puis de l'ajuster — les blocs ci-dessous
documentent ce que vous y trouverez.

Les chemins relatifs sont résolus **par rapport au répertoire de travail**, pas
au paquet ni à la configuration.

## Blocs

### `run` — identité

```yaml
run:
  name: "mon_etude"
  description: "HUVEC pro vs sen — 3 graines"
```

Détermine l'étiquette du run et le dossier de sortie.

### `paths` — où sont les données

```yaml
paths:
  data_root: "data"                              # racine des entrées
  out_base: "output/mon_etude"                   # racine des sorties
  humess_dir: "data/humess/output_huvec"
  scenic_dir: "output/pyscenic"
  de_magnitude_csv: "data/gnn_data/DE_sen_vs_pro.csv"
  coexpr_file: "data/pyscenic/diff_coexpr/coexpr_diff.tsv"
```

Ces clés deviennent des variables d'environnement (`GNN_DATA_DIR`,
`GNN_OUT_DIR_BASE`, …) — cf. [execution.md](execution.md). `GNN_OUT_DIR_BASE`
défini dans l'environnement **prime** sur `out_base`, ce qui permet d'écrire sur
un scratch sans toucher au fichier.

### `compute` — comment ça tourne

```yaml
compute:
  deterministic: true      # OBLIGATOIRE pour interpréter une ablation
  device: "cpu"            # cpu | cuda | auto
  python: "python"
  python_torch: "python"
  backend: "local"         # local | cluster — défaut, surchargé par --backend
```

**`deterministic: true` n'est pas une option de confort.** Sans lui, deux runs
d'une configuration identique à graine fixe divergent au point qu'aucune
ablation n'est lisible. Le coût est `threads=1`.

`python` / `python_torch` désignent l'interpréteur des règles. Ils gardent leur
sens historique — « le python de l'environnement actif » — mais ne sont plus un
point de rupture : si la valeur n'est pas exécutable, l'interpréteur qui fait
tourner la commande prend le relais, et `STATESHIFT_PYTHON` /
`STATESHIFT_PYTHON_TORCH` l'emportent sur tout.

### `dataset` / `input` — décrire vos états

```yaml
dataset:
  cell_groups: "pro,sen"                     # les deux états contrastés
  expr_matrix: "gnn_data/expr.csv"
  group_meta:  "gnn_data/samplesheet.tsv"
input:
  rna_type: scrna                            # scrna | bulk
  degs_path: "data/gnn_data/DE_sen_vs_pro.csv"
```

**Décrire son dataset n'est pas reconstruire ses variables** : ce bloc est
indépendant de `build`, ce qui permet de pointer des variables déjà calculées.

### `build` — reconstruire co-expression et HuMess

```yaml
build:
  enabled: false          # true = recalcule coexpression + HuMess en amont
```

Coûteux, et **exige un clone** : la règle appelle `scripts/run_v6_build.sh`, qui
n'est pas empaqueté. HuMess exige en outre le solveur `cplex` et ne tourne qu'en
local. Laissez à `false` si vos features sont déjà calculées ; la chaîne de
prétraitement est décrite dans [depuis-un-clone.md](depuis-un-clone.md#prétraitement).

### `models.vgae` — l'encodeur

```yaml
models:
  vgae:
    enabled: true         # false = repartir de points de reprise existants
    run_tag: "v6.baseline"
    seeds: [1, 2, 3]      # TROIS graines minimum
    extra_flags: ""       # drapeaux passés tels quels à gnn_vgae.py
```

`extra_flags` est la porte d'entrée des ablations : `--no-coexpr`,
`--no-humess`, `--use-reactome-fi`, `--signed-message --signed-decoder`, etc.

### `perturbation` — le readout

```yaml
perturbation:
  enabled: true
  modes: [knockdown, knockout, overexpress]
  scope: all_genes
  genes_file: ""            # restreindre à un sous-ensemble de cibles
  axis: phenotypic          # phenotypic | de | effector
  cache_delta_z: true       # permet de re-projeter des axes sans re-perturber
  axis_method: diff         # estimateur de l'axe principal
  axis_method_compare: ""   # ex. "diff,lda,cav,pca"
```

C'est **ici** que la transition se définit, et nulle part ailleurs :

| `axis` | Comment l'axe est construit |
|---|---|
| `phenotypic` | contraste des groupes de cellules A→B (défaut) |
| `de` | pôles = top-N up/down d'une table d'analyse différentielle |
| `effector` | ancré sur deux listes de gènes effecteurs pro/anti |

`cache_delta_z: true` est peu coûteux et très rentable : il permet de rejouer
**n'importe quel nouvel axe en quelques secondes**, sans re-perturber.

### `scoring` — ce qui compte comme « DE-significatif »

```yaml
scoring:
  de_significance: "pvalue"   # pvalue | magnitude-rank
  de_padj_max: 0.05
  de_abs_lfc_min: 0.5
```

N'affecte que les colonnes de bonus et les paliers, jamais `driver_score`.

### `validation` — les contrôles

```yaml
validation:
  ora_top_n: 100
  qc: {enabled: true}
  decoy: {enabled: false, n_rewires: 50, random_axis: 0}
  purity_source: {enabled: false, targets: []}
  head_to_head: {enabled: false}
  readout_specificity: {enabled: false}
  ora_de_baseline: {enabled: false}
  signed_cascade: {enabled: false}
  cluster_annotation: {enabled: false}
```

Détail de chaque contrôle et de la question à laquelle il répond dans
[resultats.md](resultats.md#la-validation-est-de-première-classe).

Deux pièges :

- `purity_source` **exige** une liste `targets`, sinon la règle échoue ;
- `decoy.n_rewires: 50` est un minimum. À 3–5 réassignations, les écarts-types
  sont sous-estimés d'un facteur ~10 et la nulle raconte n'importe quoi.

## Éditer sans casser

```bash
stateshift run --dry-run --configfile ma-config.yaml
```

Le dry-run résout le DAG entier sans rien exécuter : c'est le contrôle à passer
après **chaque** modification, surtout avant de soumettre au cluster.
