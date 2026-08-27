# Commandes et points d'entrée

Deux niveaux : la **commande `stateshift`**, qui couvre l'usage normal, et les
**scripts individuels**, qui répondent à des questions ponctuelles. Ce fichier
est la référence des deux, et il dit surtout ce qui est *branché* et ce qui ne
l'est pas — un écart qui, non écrit, avait laissé cinq modules de validation
exister pendant des semaines sans jamais être appelés.

## La commande `stateshift`

```
stateshift init      [--preset quick|full] [--out-dir DIR]
stateshift run       [--backend local|cluster] [--configfile F] [--cores N]
                     [--jobs N] [--profile-dir DIR] [-n] [-- <args snakemake>]
stateshift configs
stateshift path      [package|workflow|snakefile|config|profile]
stateshift profile   [--copy DIR]
stateshift doctor
stateshift --version
```

| Commande | À quoi elle sert |
|---|---|
| `init` | assistant interactif : écrit une configuration à partir de questions |
| `run` | lance le pipeline ; c'est la commande principale |
| `configs` | liste les configurations livrées avec le paquet |
| `path` | chemin résolu d'un composant — **à utiliser dans vos scripts** |
| `profile` | affiche le profil SLURM, ou en donne une copie éditable |
| `doctor` | diagnostic d'environnement ; à joindre à toute demande d'aide |

`path` est le point d'entrée pour scripter sans supposer un clone :

```bash
SRC=$(stateshift path package)
python "$SRC/validation/reports/summarize_drivers.py" --help
```

Tout ce qui suit `--` part tel quel vers snakemake :

```bash
stateshift run --configfile c.yaml -- --set-resources build_graph:runtime=30
```

## Statut des modules

| Statut | Sens |
|---|---|
| **DAG** | appelé par le Snakefile. Tourne à chaque run. |
| **DAG opt-in** | règle existante, activée par une clé de configuration. |
| **Manuel** | outil de diagnostic lancé à la main. Volontaire — mais documenté ici. |

### Pipeline principal

| Module | Statut | Règle |
|---|---|---|
| `gnn/gnn_vgae.py` | DAG | `build_graph`, `train_vgae` |
| `perturbation/perturb_top_genes.py` | DAG | `perturb` |
| `validation/reports/perturb_report.py` | DAG | `aggregate_cross_seed` |
| `perturbation/reproject_axes.py` | DAG opt-in | `perturbation.axis_method_compare` |
| `data/preprocess/build_diff_coexpr.py` | DAG opt-in | `build.enabled` |

### Validation

| Module | Statut | Activation |
|---|---|---|
| `validation/reports/driver_baselines.py` | DAG | — |
| `validation/viz/interpret_embedding.py` | DAG | — |
| `validation/ora/ora_consensus.py` | DAG | — |
| `validation/qc/pipeline_qc.py` | DAG opt-in *(défaut ON)* | `validation.qc.enabled` |
| `validation/explain/perturb_decoy.py` | DAG opt-in | `validation.decoy.enabled` |
| `validation/cluster/cluster_annotation.py` | DAG opt-in | `validation.cluster_annotation.enabled` |
| `validation/explain/purity_source_attribution.py` | DAG opt-in | `validation.purity_source.enabled` + `targets` |
| `validation/reports/head_to_head_baselines.py` | DAG opt-in | `validation.head_to_head.enabled` |
| `validation/explain/readout_specificity.py` | DAG opt-in | `validation.readout_specificity.enabled` |
| `validation/ora/ora_de_baseline.py` | DAG opt-in | `validation.ora_de_baseline.enabled` |
| `perturbation/signed_cascade.py` | DAG opt-in | `validation.signed_cascade.enabled` |
| `validation/figures/memoire_figures.py` | DAG opt-in | `validation.memoire_figures.enabled` |

### Outils manuels — volontairement hors DAG

Ces modules portent sur **plusieurs configurations ou plusieurs vagues à la
fois** : les mettre dans le DAG d'une seule configuration n'aurait pas de sens.

| Module | Question posée |
|---|---|
| `validation/explain/rank_compare.py` | comment un gène bouge-t-il entre N configurations ? |
| `validation/explain/run_signed_auc_gate.py` | le gate signé passe-t-il cross-graine ? |
| `validation/reports/compare_runs.py` | comparaison cross-ablation / cross-version |
| `validation/reports/ablation_attribution.py` | Δrang + ORA par ablation |
| `validation/viz/viz_explorer.py`, `visualize_global.py`, `plot_pathway_heatmap.py` | figures de soutenance |
| `validation/explain/edge_attention.py` | poids d'attention du GAT |
| `validation/reports/summarize_drivers.py`, `graph_summary.py` | synthèses ponctuelles |
| `scripts/gene_reference_table.py` | une ligne par gène : rang, DE, degré par type d'arête |
| `scripts/module_ora.py` | corroboration des modules par des ensembles externes |
| `scripts/build_config_site.py` | site statique d'exploration des résultats |

Tous répondent à `--help`, ce que vérifie `pytest tests/test_cli_contract.py`.

### Scripts d'orchestration cluster

Dans `scripts/`, **exigent un clone** (ils précèdent la mise en paquet) :

| Script | Rôle |
|---|---|
| `run_ablation_grid.sh` | job array SLURM d'ablations |
| `run_perturbation_grid.sh` | job array de perturbation |
| `run_analysis.sh` | analyse post-perturbation sur des runs existants |
| `run_optuna.sh` | recherche d'hyperparamètres, job contrôleur |
| `run_v6_*.sh` | build / entraînement / axe DE en V6 |

## Sans CLI — bibliothèques

`validation/schema/method_comparison_schema.py`, `gnn/_config.py`, `_paths.py`,
`_vgae_model.py`, `hgnc_alias.py`, `omnipath_*.py`, `data/loaders/*`. Pas de
point d'entrée : c'est normal.

## Hors périmètre

- `src/coexpr_benchmark/` — comparatif WGCNA/hdWGCNA mené une fois. Fichiers
  préfixés par un chiffre (`03_compare.py`), donc non importables par
  construction : ce sont des scripts, pas des modules. Exclus du paquet.
- `archive/` — code historique gelé.
- `gnn/_config_derive.py`, `_graph_build_body.py`, `_score_body.py`,
  `_train_body.py` — **corps `exec`**, pas des modules : ils s'exécutent dans
  l'espace de noms de leur appelant et lèvent `NameError` si on les importe.
  C'est attendu.

## Optimisation automatique

Optuna pilote Snakemake — jamais l'inverse : un DAG doit connaître ses jobs à
l'avance, une recherche non.

```bash
# 1. MESURER LE BRUIT AVANT DE CHERCHER (obligatoire pour lire la suite)
python "$(stateshift path package)/optim/search.py" calibrate --repeats 3 \
       --objective cross_seed_stability

# 2. chercher
python "$(stateshift path package)/optim/search.py" search --n-trials 20 --seeds 3

# sur cluster (job contrôleur qui survit à la session) — exige un clone
bash scripts/run_optuna.sh calibrate --repeats 3
bash scripts/run_optuna.sh search    --n-trials 20
```

Trois objectifs branchables : `cross_seed_stability` (recommandé — vise le
plancher de bruit), `recon_auc` (peu coûteux, mais ne décide pas des drivers),
`known_driver_recall` (le plus proche du but biologique, et le plus circulaire).

`report` refuse de conclure si la calibration n'a pas été faite : une étude dont
l'amplitude n'excède pas le bruit n'a rien trouvé.

## Tests

```bash
pytest                                # tout sauf le end-to-end (~5 min)
pytest tests/test_package_layout.py   # le paquet s'importe, le workflow voyage
pytest tests/test_de_schema.py        # unitaires sur la définition de l'axe DE
pytest tests/test_workflow.py         # le DAG se résout, validations comprises
pytest tests/test_cli_contract.py     # chaque point d'entrée répond à --help
pytest -m slow                        # pipeline complet sur jeu jouet (~2 min)
```

Le test `slow` est le seul qui exécute réellement la chaîne scientifique ; les
autres vérifient la structure. Il est désélectionné par défaut.
