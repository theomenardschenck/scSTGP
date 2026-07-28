# Catalogue des points d'entrée — quoi est branché, quoi ne l'est pas

Ce dépôt contient une cinquantaine de modules. Tous ne sont pas au même statut,
et l'écart n'était jusqu'ici écrit nulle part : cinq modules de validation
existaient depuis des semaines sans être appelés par quoi que ce soit. Ce
fichier est la référence à jour ; il est vérifié par
`tests/test_workflow.py::test_wired_validations_enter_the_dag`.

Trois statuts :

| Statut | Sens |
|---|---|
| **DAG** | Appelé par `workflow/Snakefile`. Tourne à chaque run. |
| **DAG opt-in** | Règle Snakemake existante, activée par une clé de `config.yaml`. |
| **Manuel** | Outil de diagnostic lancé à la main. Volontaire — mais alors documenté ici. |

---

## Pipeline principal

| Module | Statut | Règle / commande |
|---|---|---|
| `src/gnn/gnn_vgae.py` | **DAG** | `build_graph`, `train_vgae` |
| `src/perturbation/perturb_top_genes.py` | **DAG** | `perturb` |
| `src/validation/reports/perturb_report.py` | **DAG** | `aggregate_cross_seed` |
| `src/perturbation/reproject_axes.py` | **DAG opt-in** | `axis_method_compare` — `perturbation.axis_method_compare` |
| `src/data/preprocess/build_diff_coexpr.py` | **DAG opt-in** | `build_features` — `build.enabled` |

## Validation

| Module | Statut | Activation |
|---|---|---|
| `src/validation/reports/driver_baselines.py` | **DAG** | — |
| `src/validation/viz/interpret_embedding.py` | **DAG** | — |
| `src/validation/ora/ora_consensus.py` | **DAG** | — |
| `src/validation/qc/pipeline_qc.py` | **DAG opt-in** *(défaut ON)* | `validation.qc.enabled` |
| `src/validation/explain/perturb_decoy.py` | **DAG opt-in** | `validation.decoy.enabled` |
| `src/validation/cluster/cluster_annotation.py` | **DAG opt-in** | `validation.cluster_annotation.enabled` |
| `src/validation/explain/purity_source_attribution.py` | **DAG opt-in** | `validation.purity_source.enabled` + `targets` |
| `src/validation/reports/head_to_head_baselines.py` | **DAG opt-in** | `validation.head_to_head.enabled` |
| `src/validation/explain/readout_specificity.py` | **DAG opt-in** | `validation.readout_specificity.enabled` |
| `src/validation/ora/ora_de_baseline.py` | **DAG opt-in** | `validation.ora_de_baseline.enabled` |

> Les cinq dernières lignes ont été câblées le 2026-07-27. Avant cette date
> elles n'étaient appelées ni par le Snakefile ni par un `run_*.sh` — donc
> jamais exécutées dans une vague. `pipeline_qc.py` le constate lui-même dans sa
> docstring : *« None of them was part of the pipeline, so every wave shipped
> without them. »*

## Outils manuels — volontairement hors DAG

Ces modules répondent à une question ponctuelle, sur plusieurs configs ou
plusieurs vagues à la fois. Les mettre dans le DAG d'**une** config n'aurait pas
de sens.

| Module | Question posée | Invocation |
|---|---|---|
| `src/validation/explain/rank_compare.py` | Comment un gène bouge-t-il entre N configs ? | `--dir <vague>` ou `--seeds a b c` |
| `src/validation/explain/run_signed_auc_gate.py` | Le gate 1c.5 passe-t-il cross-seed ? | `--runs-glob 'output/.../v5-full.s*' --label v5-full` |
| `src/validation/reports/compare_runs.py` | Comparaison cross-ablation / cross-version | via `scripts/run_interpretation_v541.sh` |
| `src/validation/reports/ablation_attribution.py` | Δrang + ORA par ablation | via `scripts/run_perturbation_grid.sh` |
| `src/validation/viz/viz_explorer.py`, `visualize_global.py`, `plot_pathway_heatmap.py` | Figures de soutenance / publication | via `scripts/run_analysis.sh --figures` |
| `src/validation/explain/edge_attention.py` | Extraction des poids d'attention GAT | via `scripts/run_analysis.sh --attention` |
| `src/validation/reports/summarize_drivers.py`, `graph_summary.py` | Synthèses ponctuelles | à la main |

## Sans CLI — bibliothèques importées

`src/validation/schema/method_comparison_schema.py` (schéma de tables partagé),
`src/gnn/_config.py`, `_paths.py`, `_vgae_model.py`, `hgnc_alias.py`,
`omnipath_*.py`, `src/data/loaders/*`. Pas de point d'entrée : normal.

## Hors périmètre

- `src/coexpr_benchmark/` — comparatif WGCNA / hdWGCNA mené une fois. Les
  fichiers `03_compare.py` / `04_figures.py` ont un nom préfixé par un chiffre,
  donc non importables : ce sont des scripts, pas des modules. Exclus du paquet
  et du lint.
- `archive/` — code historique gelé. Voir `archive/README.md`.
- `src/gnn/_config_derive.py`, `_graph_build_body.py`, `_score_body.py`,
  `_train_body.py` — **corps `exec`**, pas des modules : ils sont exécutés dans
  l'espace de noms de leur appelant et lèvent `NameError` si on les importe.
  C'est attendu ; les convertir en vraies fonctions est un chantier à part.

---

## Vérifier soi-même

```bash
# le DAG complet, sans rien exécuter
bash workflow/run.sh --dry-run --configfile workflow/config/config.smoke.yaml

# le contrat CLI de chaque point d'entrée
pytest tests/test_cli_contract.py

# le câblage des validations
pytest tests/test_workflow.py

# la chaîne RÉELLEMENT exécutée, sur jeu jouet (~2 min, hors ligne)
python tests/fixtures/make_tiny_dataset.py --out data_tiny
bash workflow/run.sh --backend local --configfile workflow/config/config.tiny.yaml
```

## Les trois configurations de vérification

| Config | Ce qu'elle prouve | Données requises | Durée |
|---|---|---|---|
| `config.tiny.yaml` | la chaîne **s'exécute** de bout en bout | aucune (générées) | ~2 min |
| `config.smoke.yaml` | le DAG **se résout** sur les vraies entrées | `data/` complet | secondes |
| `config.yaml` | production | `data/` complet | heures |

`config.tiny.yaml` est le seul exécutable par un tiers. Il a fait remonter
quatre défauts réels le jour de son écriture — `latent_dim` non relu depuis le
checkpoint, `CELL_GROUPS` figé sur HUVEC côté perturbation, description du
dataset couplée à `build.enabled`, et `--no-baselines` qui faisait tomber
l'agrégation via une figure. Aucun n'était atteignable sans exécuter la chaîne.
