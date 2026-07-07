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

**Statut** : projet de stage M2 — soutenance 16 septembre 2026. Outil
**modularisé** (le monolithe `gnn_vgae.py` a été éclaté en modules
importables, refactor validé **bit-exact** par un golden test) et
**pipeline Snakemake fonctionnel** de bout en bout (local **ou** cluster
SLURM), avec un **assistant de configuration** (`bash workflow/run.sh --init`).

## Documentation

- **Usage du pipeline** : ce README (§Quickstart) + [`workflow/README.md`](workflow/README.md)
  — orchestration Snakemake, règles, backend local/cluster, assistant `init`.
- **Backlog priorisé** : [`TODO`](TODO).

> Le **rapport scientifique** détaillé (méthodes, métriques, résultats par
> version) et la doc technique par script vivent dans `docs/` — un **cahier
> de labo local non versionné** (le dépôt ne contient que l'outil). Demander
> à l'auteur pour y accéder.

## Architecture

Le VGAE, historiquement un monolithe de ~4800 lignes, est **éclaté en
modules importables** (refactor validé bit-exact) :

```
gnn_huvec/
├── src/gnn/
│   ├── gnn_vgae.py            # ORCHESTRATEUR mince (parse → build → train → score)
│   ├── _config.py            # parsing CLI + dérivations (modules/features/run_tag)
│   ├── _paths.py             # résolution des chemins (env + racine repo, layout-robuste)
│   ├── _graph_build.py       # §1-7 : construction du graphe hétérogène (+ cache)
│   ├── _train.py             # §8-10 : modèle VGAE + boucle d'entraînement
│   ├── _score.py             # §11-16 : embeddings + scoring + baselines + export
│   ├── _vgae_model.py        # classes VGAE (HeteroEncoder, décodeur signé…)
│   ├── gnn_perturbation.py   # perturbation core (KO/KD/OE + axes sénescence)
│   └── omnipath_integration.py
├── src/perturbation/perturb_top_genes.py   # perturbation batch (all-genes / cibles)
├── src/validation/           # scoring cross-seed, ORA, baselines, annotation, figures
├── workflow/                 # Snakemake FONCTIONNEL
│   ├── Snakefile             # DAG : build_graph → train × seed → perturb → analyse → report
│   ├── run.sh                # lanceur (backend local/cluster) + assistant --init
│   ├── init.py               # assistant interactif de génération de config
│   ├── config/config*.yaml   # config utilisateur (+ config.smoke.yaml = test rapide)
│   └── profiles/slurm/       # profil SLURM (partition/QOS à adapter)
├── scripts/                  # helpers SLURM (grilles d'ablation, etc.)
└── tests/golden/             # test de non-régression bit-exact
```

Chaque module `_*.py` s'importe indépendamment (`from _graph_build import build_graph`).
Le graphe est construit **une fois** (`build_graph`, `--build-only`) puis réutilisé
par tous les seeds (`--reuse-graph`).

## Quickstart

### 1. Installation (environnement conda)

```bash
git clone <repo> gnn_huvec && cd gnn_huvec
micromamba create -n gnn -f environment.yml   # ou conda/mamba (contient snakemake)
micromamba activate gnn
```

### 2. Générer sa config (assistant interactif)

```bash
bash workflow/run.sh --init          # pose des questions et écrit workflow/config/config.<nom>.yaml
```
L'assistant demande : préréglage (`quick`/`full`), type de données (`bulk`/`sc`),
contraste **A vs B** (à ta discrétion : pro/sen, sain/malade, WT/mutant…), chemins,
backend (local/cluster), nombre de seeds, **perturbation ciblée ou totale**, et
**ablations** (sources du graphe à désactiver).

### 3. Lancer le pipeline

```bash
# dry-run (vérifie le DAG sans exécuter)
bash workflow/run.sh --configfile workflow/config/config.<nom>.yaml --dry-run

# local (CPU)
bash workflow/run.sh --backend local   --configfile workflow/config/config.<nom>.yaml

# cluster SLURM (adapter la partition/QOS dans workflow/profiles/slurm/config.yaml ;
#  export GNN_OUT_DIR_BASE=/scratch/.../output pour écrire les sorties sur scratch)
bash workflow/run.sh --backend cluster --configfile workflow/config/config.<nom>.yaml
```

Un **test fonctionnel rapide** (1 seed, peu d'epochs, perturbation sur cibles) est
fourni : `--configfile workflow/config/config.smoke.yaml`.

Les prérequis (matrice scRNA/bulk, DE, pySCENIC, HuMess, bases aging, cache OmniPath)
sont des entrées externes précalculées dans `data/` (gitignored). Voir
[`workflow/README.md`](workflow/README.md) pour le détail des stages et des chemins.

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
