# Workflow Snakemake — pipeline VGAE HUVEC

**Statut** : ébauche (2026-05-07). **Non fonctionnelle telle quelle.**

Ce dossier contient l'orchestration Snakemake cible. Beaucoup de rules
pointent vers les scripts monolithiques actuels en attendant la
modularisation (TODO Tier 2.5 — splitter `perturb_report.py`,
`gnn_vgae.py`, `gnn_classification.py`).

## Structure

```
workflow/
├── Snakefile                # DAG global (11 stages)
├── config/
│   └── config.yaml          # Config exemple HUVEC V3.6 baseline
└── rules/                   # (futur) rules par stage
```

## Stages couverts

| # | Stage | Statut |
|---|---|---|
| 0 | Config (`configfile`) | ✓ |
| 1 | Preprocessing (scRNA / bulk) | ✗ inputs précalculés |
| 2 | DE (MAST / DESeq2) | ✗ inputs précalculés |
| 3 | pySCENIC | ⏳ TODO wrapper |
| 4 | HuMess | ⏳ TODO wrapper sub-Snakefile |
| 4b | DB downloads | ⏳ TODO |
| 5 | Build hetero graph | ⏳ stub — TODO Tier 2.5 |
| 6.1 | Train VGAE × N seeds | ✓ via gnn_vgae.py monolithique |
| 6.2 | Train GNN_Lite | ✓ via gnn_classification.py |
| 6.3 | Baselines (MLP/Stat/DeepWalk) | ✓ inclus dans gnn_vgae.py |
| 7 | Perturbation × seed × mode | ✓ via perturb_top_genes.py |
| 8 | Aggregation cross-seed | ✓ via perturb_report.py |
| 9 | Validation ORA + cluster_annotation | ✓ |
| 10 | Cross-method comparison | ⏳ TODO Phase 1-4 |
| 11 | HTML report final | ⏳ stub |

## Prochaines étapes

Cf. [`../docs/pipeline_design.md`](../docs/pipeline_design.md) §6 (plan
de migration) :

1. **Phase B** — modularisation des scripts monolithiques (Tier 2.5).
2. **Phase C** — Snakefile fonctionnel sur les rules stubs (stage
   5 + 11).
3. **Phase D** — généralisation bulk + question utilisateur (post-soutenance).

## Lancement (futur)

```bash
# Dry-run (visualiser le DAG sans rien exécuter)
snakemake -n --configfile workflow/config/config.yaml

# Run local 8 cores
snakemake --use-conda --cores 8 --configfile workflow/config/config.yaml

# Run cluster SLURM (Nautilus)
snakemake --profile workflow/profiles/nautilus
```

## Référence

Mölder F. et al. (2021) *Sustainable data analysis with Snakemake*,
F1000Research 10:33.
