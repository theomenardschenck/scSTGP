# Exécution — local et cluster

Le pipeline est un DAG Snakemake. Il est identique dans les deux backends :
seul l'exécuteur change.

```
build_graph  →  train_vgae × graine  →  perturb × (graine, mode)
             →  aggregate_cross_seed  →  validations  →  report
```

## Les deux modes

| | mode paquet | mode script |
|---|---|---|
| lancement | `stateshift run …` | `bash workflow/run.sh …` |
| exige un clone | non | oui |
| répertoire de travail | libre | racine du clone |
| Snakefile utilisé | celui du paquet installé | celui du clone |
| scripts utilisés | ceux du paquet | `src/` du clone |

Les deux construisent la **même invocation Snakemake**. Le mode script n'est pas
déprécié : les vingt scripts SLURM historiques (grilles d'ablation, vagues,
Optuna) reposent dessus et continuent de fonctionner tels quels.

Depuis un clone où le paquet est installé en éditable, les deux modes désignent
les mêmes fichiers — il n'y a alors aucune différence observable.

Ce document couvre l'usage par la commande. Pour le travail depuis un clone —
**construction des features amont (co-expression, pySCENIC, HuMess), grilles
d'ablation SLURM, mode script pur** — voir
[depuis-un-clone.md](depuis-un-clone.md).

## Local

```bash
stateshift run --backend local --cores 8 --configfile ma-config.yaml
```

Le local convient à trois usages, et pas à un quatrième :

| Usage | Config |
|---|---|
| vérifier que la chaîne s'exécute | `config.tiny.yaml` |
| vérifier que le DAG se résout sur les vraies entrées | `config.smoke.yaml --dry-run` |
| outils de diagnostic hors DAG | cf. [commandes.md](commandes.md) |
| **un run de production** | **non — le cluster** |

La commande affiche d'ailleurs un avertissement : la construction du graphe
prend environ 40 minutes et l'entraînement plusieurs heures en CPU.

## Cluster SLURM

```bash
stateshift run --backend cluster --jobs 20 --configfile ma-config.yaml
```

Quatre points décident du succès ou de l'échec, tous appris à la dure.

### 1. Un processus contrôleur qui survit

Snakemake doit rester vivant pendant toute la durée du DAG pour soumettre et
surveiller les jobs. Lancez-le dans `tmux` ou `nohup`, jamais dans une session
SSH nue :

```bash
tmux new -s stateshift
stateshift run --backend cluster --configfile ma-config.yaml
# Ctrl-b d pour détacher
```

### 2. Sans profil, tout tourne sur le nœud de connexion

C'est l'erreur la plus coûteuse : sans profil SLURM, `snakemake -j N` exécute
**tout en local**, séquentiellement, sur le frontal — rien n'apparaît dans
`squeue`. `stateshift run --backend cluster` fixe le profil pour vous ; c'est
la raison d'être du backend.

Vérification : `squeue -u $USER` doit se remplir dans la minute.

### 3. Partition et QOS sont propres à votre cluster

Les valeurs par défaut viennent de GLiCID/Nautilus. Pour les adapter :

```bash
stateshift profile                       # afficher le profil courant
stateshift profile --copy ./mon-profil   # en obtenir une copie éditable
$EDITOR mon-profil/config.yaml           # partition, QOS, ressources
stateshift run --backend cluster --profile-dir ./mon-profil --configfile ma-config.yaml
```

Les valeurs à confronter à votre site :

```bash
sinfo                                        # partitions disponibles
sacctmgr show qos format=Name,MaxWall        # plafonds de durée
```

⚠️ La QOS par défaut `normal` plafonne souvent à quelques minutes, ce qui tue
un entraînement en cours de route ; le profil livré utilise `short` (24 h).

Le profil livré route **tout en CPU** (partition `standard`, QOS `short`,
`train_vgae` à 480 min). C'est un choix délibéré de robustesse : l'entraînement
GPU est plus rapide (~1 h contre 2–4 h) mais se heurte aux limites de la QOS
GPU. Pour re-router vers le GPU, remettez `slurm_partition='gpu'`,
`slurm_qos='gpus'`, `slurm_gres='gpu:1'` et une durée sous le plafond.

### 4. Chemins et sorties

Les chemins de données sont pilotés par le bloc `paths` de la configuration, que
le Snakefile réinjecte en variables d'environnement :

| Variable | Clé de config | Rôle |
|---|---|---|
| `GNN_DATA_DIR` | `paths.data_root` | données d'entrée |
| `GNN_OUT_DIR_BASE` | `paths.out_base` | racine des runs et analyses |
| `GNN_HUMESS_DIR` | `paths.humess_dir` | sorties HuMess |
| `GNN_SCENIC_DIR` | `paths.scenic_dir` | sorties pySCENIC |

Pour écrire sur un scratch, l'environnement **prime** sur la configuration :

```bash
export GNN_OUT_DIR_BASE=/scratch/$USER/stateshift-output
```

### Entraînement GPU hors Snakemake

Voie recommandée quand le GPU est disponible mais la QOS contraignante :
entraîner avec les scripts array éprouvés, puis laisser Snakemake faire la
perturbation et les validations en CPU.

```bash
# (a) entraînement GPU, hors DAG — exige un clone
bash scripts/run_ablation_grid.sh --version V5.4 --seeds "1 2 3"

# (b) analyse orchestrée, avec models.vgae.enabled: false dans la config
stateshift run --backend cluster --configfile ma-config.yaml
```

`models.vgae.enabled: false` fait repartir Snakemake des points de reprise et
des tables de perturbation déjà présents, sans ré-entraîner.

## Options utiles

```bash
# passer des arguments bruts à snakemake, après --
stateshift run --configfile c.yaml -- --set-resources build_graph:runtime=30
stateshift run --configfile c.yaml -- --rerun-triggers mtime
stateshift run --configfile c.yaml -- --unlock          # après une interruption brutale
```

`--keep-going` est activé par défaut, dans les deux modes : l'échec d'une
**validation** ne doit pas emporter la branche qui produit les livrables
scientifiques.

## Ce qu'il faut savoir avant d'interpréter

Trois graines sont un **minimum**, et `compute.deterministic: true` est
obligatoire pour qu'une ablation soit interprétable. Les raisons, chiffrées,
sont dans [resultats.md](resultats.md#le-plancher-de-bruit-dabord).

## Reproduire un run publié

Deux configurations rejouent la version V5.4.1 de juin 2026, selon le besoin :

```bash
# (a) reproduction complète : build → 3 graines → perturbation → analyses
stateshift run --backend cluster --configfile workflow/config/config.V5.4.1.yaml

# (b) cache Δz SUR les points de reprise de juin, sans ré-entraîner — la seule
#     voie dont les Δz correspondent EXACTEMENT aux classements publiés
bash scripts/link_v541_runs.sh baseline 1 2
stateshift run --backend cluster --configfile workflow/config/config.V5.4.1-dz.yaml
```

Deux pièges à ne pas rater :

- `extra_flags` ajoute `--no-omnipath-hgnc-alias --no-dedup-ppi-mirror`, absents
  de la ligne de juin : ce sont des défauts qui ont basculé depuis. Sans eux le
  graphe est meilleur mais **n'est plus V5.4.1** (11 168 nœuds contre 11 133).
  Contrôle après run : `n_genes == 11133`.
- Ré-entraîner ne redonne pas les poids de juin. Pour tout ce qui doit
  s'aligner sur les résultats publiés, passez par la voie (b).
