# Workflow Snakemake — pipeline VGAE HUVEC

**Statut** : **fonctionnel** — pipeline testé de bout en bout sur cluster SLURM
(build → train → perturb → analyse → report). Chemins et paramètres pilotés par
un `config/config*.yaml` ; le plus simple est de le générer avec l'assistant
(`bash workflow/run.sh --init`).

Depuis le refactor modulaire de [`gnn_vgae.py`](../src/gnn/gnn_vgae.py), la
construction du graphe (`--build-only`) et l'entraînement (`--reuse-graph`) sont
**deux règles séparées** : le graphe est bâti **une fois** puis réutilisé par tous
les seeds (au lieu d'être reconstruit à chaque seed).

## Structure

```
workflow/
├── Snakefile                 # DAG (build_graph → train × seed → perturb → analyse → report)
├── run.sh                    # LANCEUR : backend local/cluster + assistant --init
├── init.py                   # assistant interactif de génération de config
├── config/
│   ├── config.yaml           # config par défaut (HUVEC, 3 seeds)
│   └── config.smoke.yaml     # test fonctionnel rapide (1 seed, cibles)
└── profiles/
    └── slurm/config.yaml     # profil SLURM (partition + --qos à adapter)
```

## Stages orchestrés

| # | Stage | Script | Statut |
|---|---|---|---|
| 1-2 | Preprocessing scRNA + DE (MAST) | (R, externe) | ✗ précalculé → `data/gnn_data/` |
| 3 | pySCENIC | (externe) | ✗ précalculé → `paths.scenic_dir` |
| 4 | HuMess | (externe) | ✗ précalculé → `paths.humess_dir` |
| 6a | **Build graphe** (§1-7) 1× | `gnn_vgae.py --build-only` | ✓ rule `build_graph` |
| 6b | Train VGAE × seed (`--reuse-graph`) | `gnn_vgae.py` | ✓ rule `train_vgae` |
| 7 | Perturbation KO/KD/OE — **1 job / (seed, mode)** ; `perturbation.genes_file` = restreint aux cibles | `perturb_top_genes.py` | ✓ rule `perturb` |
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

## Lancement (recommandé : `run.sh`)

```bash
# 0. Environnement (fournit snakemake 7.32 — canal bioconda requis)
micromamba create -n gnn -f environment.yml && micromamba activate gnn

# 1. Générer une config (assistant : données, groupes A/B, backend, seeds,
#    perturbation ciblée/totale, ablations, presets quick|full)
bash workflow/run.sh --init

# 2. Dry-run (vérifie le DAG) puis run
bash workflow/run.sh --configfile workflow/config/config.<nom>.yaml --dry-run
bash workflow/run.sh --backend local   --configfile workflow/config/config.<nom>.yaml
bash workflow/run.sh --backend cluster --configfile workflow/config/config.<nom>.yaml

# Test fonctionnel rapide (1 seed, cibles) :
bash workflow/run.sh --configfile workflow/config/config.smoke.yaml
```

`run.sh` choisit l'exécuteur : `--backend local` → `snakemake --cores N` (+ warning) ;
`--backend cluster` → `snakemake --profile workflow/profiles/slurm` (soumission SLURM).
Le backend par défaut est lu depuis `compute.backend` du config. Passer des options
brutes à snakemake après `--` (ex. `-- --set-resources build_graph:runtime=30`).

**Sorties sur scratch (cluster)** : `export GNN_OUT_DIR_BASE=/scratch/.../output`
avant le run (prime sur `paths.out_base`).

**Analyse seule** sur des runs déjà entraînés : `models.vgae.enabled: false` dans le
config (Snakemake repart des `best_vgae.pt` / `perturbation_all_genes_*.tsv` présents).

### Détails cluster (SLURM, Snakemake 7.x)

Éditer d'abord `workflow/profiles/slurm/config.yaml` : **partition** (`sinfo`) et
**QOS** (`sacctmgr show qos format=Name,MaxWall` — la QOS par défaut `normal` plafonne
souvent à quelques minutes ; le profil met `short`). Invocation équivalente sans le
wrapper :

```bash
snakemake -s workflow/Snakefile --configfile workflow/config/config.<nom>.yaml \
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
