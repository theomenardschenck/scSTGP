# gnn_huvec — Priorisation de gènes de la sénescence cellulaire par VGAE

Pipeline GNN (Graph Neural Network) pour identifier les gènes
**drivers** de la transition prolifératif → sénescent dans les cellules
endothéliales humaines (HUVEC), à partir de scRNA-seq P4 vs P16.

L'approche combine :

1. Un **VGAE** (Variational Graph AutoEncoder) sur graphe hétérogène
   multi-source (PPI STRING, REACTOME, co-expression, régulons
   pySCENIC, métabolisme HuMess).
2. Une **perturbation in silico** (KO / KD / OE) propagée par le
   modèle, projetée sur un axe sénescence P4 → P16.
3. Un scoring multi-tier (`driver` / `validation` / `discovery`) avec
   robustesse cross-seed et evidence_tier A–E.

**Statut** : projet de stage M2 — soutenance 16 septembre 2026. Repo
en cours de modularisation (cf. [TODO](TODO) Tier 2.5).

## Documentation

- **Méthodes scientifiques** : [`docs/vgae_report.md`](docs/vgae_report.md)
  — rapport méthodologique exhaustif (architecture, métriques,
  résultats par version, FAQ).
- **Architecture & pipeline cible** :
  [`docs/pipeline_design.md`](docs/pipeline_design.md) — DAG 11 stages,
  schéma `config.yaml`, plan de migration vers Snakemake.
- **Backlog priorisé** : [`TODO`](TODO) — Tier 1 (priorité défense)
  → Tier 4 (post-stage).
- **Outils apparentés** : §18 du rapport — positionnement vs
  CellOracle, scTenifoldKnk, GEARS, DREAMwalk, decoupler-py.

## Architecture (état actuel)

```
gnn_huvec/
├── src/
│   ├── gnn/                        # Modèles
│   │   ├── gnn_vgae.py             # VGAE V3.6 + baselines (MLP/Stat/DeepWalk)
│   │   ├── gnn_classification.py   # GNN_Lite supervisé (HeteroGNN multi-label)
│   │   ├── gnn_perturbation.py     # Perturbation core (KO/KD/OE + axes)
│   │   └── omnipath_integration.py # Loader OmniPath signed (V4)
│   ├── perturbation/
│   │   └── perturb_top_genes.py    # Batch all-genes × seeds
│   ├── extraction/                 # pySCENIC + Seurat → TSV
│   └── validation/
│       ├── perturb_report.py       # Cross-seed scoring + figures
│       ├── compare_runs.py         # Cross-ablation / cross-version
│       ├── ora_consensus.py        # Test hypergéométrique vs DBs
│       ├── cluster_annotation.py   # Annotation biologique cell_groups
│       ├── method_comparison_schema.py  # Schéma TSV unifié cross-method
│       └── visualize_global.py     # Figures de synthèse multi-versions
├── workflow/                       # Snakemake (ébauche, non fonctionnel)
├── scripts/                        # Helpers SLURM (cluster Nautilus)
└── docs/                           # Rapport + design pipeline
```

Le pipeline actuel est piloté à la main via les CLI argparse de chaque
script. La modularisation `src/gnn_huvec/` + `cli/` + `workflow/`
prévue par [`docs/pipeline_design.md`](docs/pipeline_design.md) §3 est
en cours (Tier 2.5 du TODO).

## Quickstart

> ⚠️ Le pipeline n'est pas encore packagé. Les commandes ci-dessous
> reflètent l'état actuel ; après modularisation Tier 2.5, elles
> deviendront `python -m gnn_huvec.cli.<command>`.

### Installation

```bash
git clone <repo> gnn_huvec
cd gnn_huvec
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt    # à figer (Tier 2.5)
```

Dépendances principales : `torch`, `torch-geometric`, `numpy`,
`pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `anndata`,
`pyscenic`, `mygene`, `omnipath` (optionnel V4).

### Données nécessaires

Téléchargements / pré-calculs à placer dans `data/` (gitignored) :

| Fichier | Source | Description |
|---|---|---|
| `gnn_data/HUVEC_seurat_processed.rds` | données HUVEC P4/P16 | matrice scRNA Drop-seq normalisée |
| `gnn_data/DEGs_P4_vs_P16_MAST.csv` | Seurat MAST | DEGs P4 vs P16 |
| `gnn_data/group_expression.tsv` | `gnn_vgae.py --export-only` | mean expr par cell_group |
| `dbs/SenMayo.tsv`, `CellAge.tsv`, `GenAge.tsv`, `Fridman.tsv` | bases aging | signatures sénescence |
| `omnipath/*.tsv.gz` | `scripts/cache_omnipath.py` | cache pour V4 (frontal-only) |

### Run baseline V3.6 (1 seed)

```bash
# 1. Entraîner le VGAE (1 seed, ~30 min sur 1 GPU)
python src/gnn/gnn_vgae.py --seed 42 --out-dir output/V3.6/run_s42

# 2. Perturbation in silico (all-genes × KO/KD/OE)
python src/perturbation/perturb_top_genes.py \
    --run-dir output/V3.6/run_s42 \
    --modes KO,KD,OE --all-genes

# 3. Single-seed report (figures + ranking)
python src/validation/perturb_report.py \
    --perturb-dir output/V3.6/run_s42/perturbation
```

### Run cross-seed V3.6 (10 seeds)

```bash
# Cluster SLURM Nautilus (job array sur 10 seeds)
bash scripts/run_ablation_grid.sh
bash scripts/run_perturbation_grid.sh

# Aggregation cross-seed
python src/validation/perturb_report.py \
    --cross-seed \
    --perturb-dirs output/V3.6/run_s4{2..51}/perturbation \
    --de-magnitude-csv data/gnn_data/DEGs_P4_vs_P16_MAST.csv \
    --out-dir output/V3.6/cross_seed_report
```

### Validation externe

```bash
# Test hypergéométrique vs aging DBs
python src/validation/ora_consensus.py \
    --db aging \
    --consensus-runs output/V3.6/run_s4{2..51} \
    --out-dir output/V3.6/ora

# Annotation biologique des cell_groups
python src/validation/cluster_annotation.py \
    --run-dir output/V3.6/run_s42 \
    --cross-seed-dirs output/V3.6/run_s4{2..51}
```

## Sortie principale

`cross_seed_gene_ranking.tsv` — un gène par ligne, trié par
`driver_score` :

| Colonne | Sens |
|---|---|
| `gene_symbol` | HGNC |
| `driver_score` ∈ [0,1] | composite multi-source post-perturbation |
| `discovery_score` | + bonus non-DE (graph-only findings) |
| `validation_score` | + bonus DE-sig + aging DBs |
| `evidence_tier` ∈ {A,B,C,D,E} | A=confirmé, B=découverte, C=effecteur, D=hub, E=bruit |
| `canon_diff`, `canon_cosine` | métriques signées (effet × directionalité) |
| `interpretation` | tag biologique calibré (cf. §11 du rapport) |

Top drivers V3.6 (10 seeds) : H2AFZ, HMGB1, FHL2 (anti-sénescence DE-sig),
ASNS (pro-sénescence non-DE), CEBPB / NFE2L2 / MYC (TFs). Cf. §10.11
et §13 du rapport pour l'interprétation biologique complète.

## Reproductibilité

- **Seeds** : entraînement déterministe pour seed donné (10 seeds
  V3.6 disponibles, 4/10 V3.3).
- **Manifest** : chaque run écrit `run_config.json` (CLI args + version
  Git + hash données) — cf. §V3.6 du rapport.
- **Cross-ablation** : `compare_runs.py --cross-ablation` reproduit
  les analyses V3.6.2 (club des 6 drivers universels).

## Citation

Travail M2 (université de Nantes, équipe Petry) — papier à venir.
En attendant :

```bibtex
@unpublished{menard2026vgae_huvec,
  author = {Menard, Théo and Maillasson, Mike},
  title  = {VGAE-based gene prioritization in HUVEC cellular senescence},
  year   = {2026},
  note   = {M2 internship, Université de Nantes}
}
```

## Licence

À définir (cf. TODO Tier 2.5). Vérifier compatibilité avec dépendances
(PyG MIT, OmniPath GPL-3).

## Articles méthodologiques de référence

- Kipf & Welling 2016, *Variational Graph Auto-Encoders*, NeurIPS BDL.
- Veličković et al. 2018, *Graph Attention Networks*, ICLR.
- Aibar et al. 2017 ; Van de Sande et al. 2020, *SCENIC / pySCENIC*,
  Nat Methods / Nat Protoc.
- Saul et al. 2022, *SenMayo: a transcriptomic biomarker of cellular
  senescence*, Nat Commun.
- Hernandez-Segura et al. 2018, *Hallmarks of cellular senescence*,
  Trends Cell Biol.

Liste complète : §"Articles de référence" du rapport.

## Contact

Théo Menard — `theo.menard@etu.univ-nantes.fr`
