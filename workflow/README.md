# Workflow Snakemake — pipeline VGAE HUVEC

**Statut** : **fonctionnel** (2026-06-18). Le Snakefile est câblé sur les
CLI réelles des scripts (plus de stub `python -m gnn_huvec.cli…`). Les
chemins et paramètres sont pilotés par [`config/config.yaml`](config/config.yaml).

L'entraînement utilise toujours le monolithe [`gnn_vgae.py`](../src/gnn/gnn_vgae.py)
(qui construit le graphe ET entraîne) ; la modularisation `cli/` reste un
objectif Tier 2.5 mais n'est plus un prérequis pour exécuter le pipeline.

## Structure

```
workflow/
├── Snakefile                 # DAG fonctionnel (train → perturb → score → valid → report)
├── config/
│   └── config.yaml           # paramètres + chemins (édités par l'utilisateur)
└── profiles/
    └── slurm/config.yaml     # profil de soumission SLURM (à adapter)
```

## Stages orchestrés

| # | Stage | Script | Statut |
|---|---|---|---|
| 1-2 | Preprocessing scRNA + DE (MAST) | (R, externe) | ✗ précalculé → `data/gnn_data/` |
| 3 | pySCENIC | (externe) | ✗ précalculé → `paths.scenic_dir` |
| 4 | HuMess | (externe) | ✗ précalculé → `paths.humess_dir` |
| 6 | Train VGAE × seed | `gnn_vgae.py` | ✓ rule `train_vgae` |
| 7 | Perturbation KO/KD/OE × seed | `perturb_top_genes.py` | ✓ rule `perturb` |
| 8 | Agrégation cross-seed → driver_score | `perturb_report.py` | ✓ rule `aggregate_cross_seed` |
| 8b | driver_baselines + interpret_embedding [+ décoy] | `driver_baselines.py` / `viz/interpret_embedding.py` / `explain/perturb_decoy.py` | ✓ |
| 9 | ORA Reactome+aging [+ cluster_annotation] | `ora/ora_consensus.py` / `cluster/cluster_annotation.py` | ✓ |
| 11 | Report (synthèse markdown) | rule `report` | ✓ |

**Hors DAG single-run** : la synthèse **cross-config** (ablation_attribution,
compare_runs, `viz/plot_pathway_heatmap.py`) reste pilotée par
[`run_interpretation_v541.sh`](../scripts/run_interpretation_v541.sh)
car elle opère sur plusieurs configs. Le **cross-method** (stage 10) et le
GNN_Lite restent opt-in/non câblés end-to-end.

**Axe de readout (stage 7)** paramétrable via `perturbation.axis_tag` /
`out_suffix` : `V3/V4` = contraste phénotypique scRNA (P4→P16), `V6` =
axe DE-ancré bulk (`perturb_top_genes --de-axis-file`, cf.
[`run_v6_de_axis.sh`](../scripts/run_v6_de_axis.sh)), `effector` = ancré
effecteurs. Le DE reste un readout, jamais une feature de l'encodeur
(cf. [`../docs/pipeline_design.md`](../docs/pipeline_design.md) §2.4).

## Portabilité (clone sur le cluster)

`gnn_vgae.py` code en dur les chemins GLiCID par défaut, mais ils sont
**surchargeables par variables d'environnement**, que le Snakefile
injecte depuis `config.paths` :

| Variable | Config | Rôle |
|---|---|---|
| `GNN_DATA_DIR` | `paths.data_root` | données d'entrée (gnn_data, PPI, databases, pyscenic, omnipath) |
| `GNN_OUT_DIR_BASE` | `paths.out_base` | racine des runs + analyses |
| `GNN_HUMESS_DIR` | `paths.humess_dir` | sorties HuMess |
| `GNN_SCENIC_DIR` | `paths.scenic_dir` | sorties pySCENIC |

Après `git clone`, **stager les données** (non versionnées : cf.
`.gitignore`) sous `paths.data_root` puis ajuster `config/config.yaml`.

## Lancement

```bash
# 0. Environnement (depuis la racine du repo)
conda env create -f environment.yml && conda activate gnn   # fournit snakemake 7.32

# 1. Dry-run — visualise le DAG sans rien exécuter
snakemake -n -s workflow/Snakefile --configfile workflow/config/config.yaml

# 2. Run LOCAL (8 cœurs). Entraînement GPU si dispo (compute.device: auto)
snakemake --cores 8 -s workflow/Snakefile --configfile workflow/config/config.yaml

# 3. Analyse SEULE sur des runs déjà entraînés
#    → models.vgae.enabled: false dans config.yaml (Snakemake repart des
#      perturbation_all_genes_*.tsv / best_vgae.pt déjà présents dans out_base)
```

### Sur le cluster (SLURM, Snakemake 7.x)

```bash
# Éditer d'abord workflow/profiles/slurm/config.yaml (partition, --account…)
snakemake -s workflow/Snakefile --configfile workflow/config/config.yaml \
          --profile workflow/profiles/slurm
```

**Entraînement GPU** — recommandé : entraîner via le script array éprouvé
puis laisser Snakemake faire perturbation→validation (CPU) :

```bash
# (a) entraînement GPU hors Snakemake
bash scripts/run_ablation_grid.sh --version V5.4 --seeds "1 2 3"
# (b) analyse orchestrée par Snakemake (models.vgae.enabled: false)
snakemake -s workflow/Snakefile --configfile workflow/config/config.yaml \
          --profile workflow/profiles/slurm
```

Sinon, router `train_vgae` vers une partition GPU dans le profil
(`set-resources` + `--gres=gpu:1`).

## Référence

Mölder F. et al. (2021) *Sustainable data analysis with Snakemake*,
F1000Research 10:33.
