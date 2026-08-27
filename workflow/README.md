# workflow/ — le pipeline Snakemake

**La documentation d'utilisation vit dans [`../docs/guide/`](../docs/guide/README.md)**
(installation, exécution locale et SLURM, configuration, dépannage). Ce fichier
ne garde que ce qui décrit le contenu du répertoire.

Ce répertoire est embarqué dans le paquet sous `stateshift.workflow` : le
Snakefile, les configurations et le profil SLURM voyagent avec l'installation.
Leur chemin se résout par `stateshift.snakefile()`, jamais par un chemin relatif
au répertoire de travail.

## Contenu

```
workflow/
├── Snakefile                 # le DAG
├── run.sh                    # lanceur mode script (exige un clone)
├── init.py                   # assistant de configuration
├── config/                   # configurations livrées
│   ├── config.yaml           #   production HUVEC, 3 graines
│   ├── config.tiny.yaml      #   jeu jouet, exécutable sans données
│   ├── config.smoke.yaml     #   résolution du DAG sur les vraies entrées
│   ├── config.V5.4.1*.yaml   #   reproduction de la version publiée
│   └── ablations_*.yaml      #   profils de vagues d'ablation
└── profiles/slurm/           # profil de soumission SLURM
```

## Stages du DAG

| # | Stage | Script | Statut |
|---|---|---|---|
| 1-2 | Prétraitement scRNA + DE | (R, externe) | ✗ précalculé |
| 3 | pySCENIC | (externe) | ✗ précalculé |
| 4 | HuMess | (externe) | ✗ précalculé |
| 6a | Build du graphe, **une fois** | `gnn_vgae.py --build-only` | rule `build_graph` |
| 6b | Entraînement VGAE × graine | `gnn_vgae.py --reuse-graph` | rule `train_vgae` |
| 7 | Perturbation KO/KD/OE | `perturb_top_genes.py` | rule `perturb` |
| 8 | Agrégation cross-graine | `perturb_report.py` | rule `aggregate_cross_seed` |
| 8b | Baselines + interprétation [+ décoy] | `driver_baselines.py`, … | rules |
| 8c | Axes alternatifs (re-projection du cache Δz) | `reproject_axes.py` | rule `axis_method_compare` |
| 9 | ORA [+ annotation de clusters] | `ora_consensus.py` | rules |
| 10 | Validations post-modèle | 6 modules | rules opt-in |
| 11 | Rapport de synthèse | — | rule `report` |

Depuis le découpage modulaire de `gnn_vgae.py`, build et entraînement sont
**deux règles séparées** : le graphe est bâti une fois puis réutilisé par toutes
les graines.

Toutes les validations du stage 10 sont **post-hoc sur l'encodeur gelé** :
aucune ne ré-entraîne, aucune ne change le classement principal.

## Variables d'environnement reconnues

| Variable | Effet |
|---|---|
| `GNN_DATA_DIR`, `GNN_OUT_DIR_BASE`, `GNN_HUMESS_DIR`, `GNN_SCENIC_DIR` | chemins, injectés depuis `config.paths` |
| `STATESHIFT_SRC` | localise les scripts (sinon : paquet installé, sinon clone) |
| `STATESHIFT_WORKFLOW` | localise ce répertoire |
| `STATESHIFT_PYTHON`, `STATESHIFT_PYTHON_TORCH` | interpréteur des règles, prime sur `compute.python` |

## Synthèses hors DAG

La synthèse **cross-configuration** (`ablation_attribution`, `compare_runs`,
`plot_pathway_heatmap`) reste pilotée par `scripts/run_interpretation_v541.sh` :
elle opère sur plusieurs configurations à la fois, donc n'a pas sa place dans le
DAG d'une seule.

## Référence

Mölder F. et al. (2021) *Sustainable data analysis with Snakemake*,
F1000Research 10:33.
