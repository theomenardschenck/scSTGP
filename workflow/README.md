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
│   ├── config.smoke.yaml     # test fonctionnel rapide (1 seed, cibles)
│   ├── config.V5.4.1.yaml    # reproduction v5.4.baseline (graphe + Δz, ré-entraîne)
│   └── config.V5.4.1-dz.yaml # Δz sur les checkpoints V5.4.1 de juin (pas de train)
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
| 8c | **Axes alternatifs** (F1) — re-projection du cache Δz sur N estimateurs | `perturbation/reproject_axes.py` | ✓ rule `axis_method_compare` |
| 9 | ORA Reactome+aging [+ cluster_annotation] | `ora/ora_consensus.py` / `cluster/cluster_annotation.py` | ✓ |
| 10 | **Validations post-modèle** (5 modules, cf. ci-dessous) | `qc/pipeline_qc.py` / `explain/purity_source_attribution.py` / `reports/head_to_head_baselines.py` / `explain/readout_specificity.py` / `ora/ora_de_baseline.py` / `perturbation/signed_cascade.py` | ✓ |
| 11 | Report (synthèse markdown) | rule `report` | ✓ |

### Stage 10 — validations post-modèle

Toutes **post-hoc sur l'encodeur gelé** : aucune ne ré-entraîne, aucune ne
change le ranking headline. `enabled: true` par défaut depuis le 2026-07-29
(elles étaient `false` depuis leur branchement, donc ne tournaient nulle part).

| clé `validation.*` | question à laquelle elle répond |
|---|---|
| `qc` | les 5 contrôles préalables : plancher de bruit, multiplicité d'arêtes, recouvrement de sources, confusion degré↔readout, spécificité d'axe |
| `purity_source` | d'où vient la purity d'une cible, et quelle **source** la porte ? (⚠️ `targets` obligatoire) |
| `head_to_head` | un outil **plus simple** (importance, betweenness) sort-il les mêmes cibles ? |
| `readout_specificity` | métriques de readout **affranchies du degré** |
| `ora_de_baseline` | l'ORA du top-drivers bat-elle l'ORA du **DE seul** ? |
| `signed_cascade` | rôle pro/anti par composition de signes multi-hop, **axis-free** |

**Deux nulles complémentaires, à ne pas confondre** :
`validation.decoy.enabled` = décoy N2 de **structure** (rewire degré-préservant,
`n_rewires: 50` obligatoire — cf. LOG §25bis, à n=3-5 les SD sont sous-estimées
~10×) ; `validation.decoy.random_axis` = nulle de **spécificité d'axe** (N axes
aléatoires, quasi gratuite car re-projetée du cache Δz).

**Depuis un profil de vague** : `run_omnipath_ablation_wave.sh` recopie tels
quels les blocs `validation:` et `perturbation:` du profil dans le config généré
(fusion récursive, appliquée en dernier). C'est la seule voie pour piloter ces
modules par vague — les raccourcis (`decoy:`, `axis_method:`, …) ne couvrent que
le décoy et l'axe.

**Hors DAG single-run** : la synthèse **cross-config** (ablation_attribution,
compare_runs, `viz/plot_pathway_heatmap.py`) reste pilotée par
[`run_interpretation_v541.sh`](../scripts/run_interpretation_v541.sh)
car elle opère sur plusieurs configs. Le **cross-method** et le
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

## Rejouer V5.4.1 (caches graphe + Δz)

Les runs V5.4.1 de juin 2026 précèdent le cache de build ET le cache Δz. Deux
configs les régénèrent, selon ce dont on a besoin (détail : [LOG §34](../docs/design_log.md#log-v541-repro)) :

```bash
# (a) reproduction complète — build_graph → 3 seeds → perturb → analyses
#     → _graph_cache.pkl + *_dz_cache_<mode>.npz sous V5.4.1_repro/v5.4.baseline/
bash workflow/run.sh --backend cluster --configfile workflow/config/config.V5.4.1.yaml --dry-run
bash workflow/run.sh --backend cluster --configfile workflow/config/config.V5.4.1.yaml

#     le graphe seul (~40 min, sans entraîner) :
snakemake -s workflow/Snakefile --configfile workflow/config/config.V5.4.1.yaml \
          --profile workflow/profiles/slurm \
          output/gnn_vgae/V5.4.1_repro/v5.4.baseline/_graph_cache.pkl

# (b) cache Δz SUR les checkpoints de juin (aucun ré-entraînement) — la seule voie
#     dont les Δz correspondent EXACTEMENT aux rankings V5.4.1 publiés
bash scripts/link_v541_runs.sh baseline 1 2      # layout <run_tag>/s<seed> (liens)
bash workflow/run.sh --backend cluster --configfile workflow/config/config.V5.4.1-dz.yaml
```

Deux points à ne pas rater :

- `extra_flags` ajoute **`--no-omnipath-hgnc-alias --no-dedup-ppi-mirror`**, absents
  de la ligne de juin : ce sont des défauts qui ont basculé ON depuis (2026-07-10 et
  2026-07-28). Sans eux le graphe est meilleur mais **n'est plus V5.4.1** (11168 vs
  11133 nœuds, `ppi_degree` divisé par 2). Contrôle post-run : `n_genes == 11133`.
- Ré-entraîner ne redonne **pas** les poids de juin (runs non déterministes,
  ρ 0.556-0.687 entre deux runs identiques) — d'où la voie (b) pour tout ce qui doit
  s'aligner sur les résultats publiés.

## Référence

Mölder F. et al. (2021) *Sustainable data analysis with Snakemake*,
F1000Research 10:33.
