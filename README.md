# (sc)STGP — State Transition Gene Prediction

Prioriser les gènes qui **pilotent** le passage d'un état cellulaire à un
autre, à partir de données transcriptomiques (single-cell ou bulk) et de
réseaux biologiques curés.

L'application de référence est la sénescence réplicative endothéliale
(HUVEC P4 → P16, scRNA-seq Drop-seq GSE102090), mais la méthode ne lui est pas
propre : une transition se définit au **readout**, pas dans le modèle. Voir
[GENERALIZATION.md](GENERALIZATION.md).

## Le principe, et pourquoi il tient

La difficulté n'est pas de trouver les gènes qui **changent** entre deux états —
une analyse différentielle le fait. Elle est de trouver ceux qui **causent** le
changement, et de le faire sans se contenter de redécouvir la liste dont on est
parti.

L'architecture répond par une séparation stricte :

1. **L'encodeur ne voit jamais le contraste.** Un VGAE hétérogène apprend une
   représentation à partir de sources structurelles et condition-indépendantes
   (PPI STRING, Reactome, OmniPath signé, co-expression, régulons pySCENIC,
   métabolisme HuMess). Le logFC n'est jamais une feature.
2. **La perturbation est in silico.** Chaque gène est éteint (KO), atténué (KD)
   ou surexprimé (OE) ; l'encodeur gelé propage l'effet dans le graphe.
3. **L'axe seul porte les deux états.** Le déplacement Δz est projeté sur un axe
   défini par le contraste A→B. Changer de transition = changer d'axe.

C'est cette séparation qui rend le résultat non trivialement re-dérivable du DE
— et le dépôt fournit les outils pour le **vérifier** plutôt que l'affirmer
(voir Validation ci-dessous).

## Statut

Travail de stage M2 (Université de Nantes, équipe Petry) — soutenance
16 septembre 2026. Le pipeline tourne de bout en bout, en local comme sur SLURM.
Le rapport scientifique (méthodes détaillées, résultats par version, cahier de
conception) vit dans `docs/`, **non versionné** : demander à l'auteur.

## Installation

```bash
git clone <repo> && cd gnn_huvec

# Environnement de référence (pile scientifique épinglée : torch, PyG,
# pySCENIC, scanpy, snakemake 7.32)
micromamba create -n gnn -f environment.yml     # ou conda / mamba
micromamba activate gnn

# Le paquet lui-même, en mode éditable
pip install -e .
```

`environment.yml` fait foi pour un vrai run ; les dépendances de
`pyproject.toml` ne couvrent que l'import du paquet et les tests.

### Deux modes d'import, par conception

| Mode | Usage | Qui l'utilise |
|---|---|---|
| script | `python src/gnn/gnn_vgae.py …` | Snakefile, scripts SLURM |
| paquet | `import stgp.data.loaders.bulk_rna` | tests, code nouveau |

Les fichiers ne bougent pas : `src/` est monté comme paquet `stgp` via
`package-dir`. Déplacer l'arborescence aurait cassé les 20 scripts cluster et le
Snakefile sans rien apporter.

## Utilisation

```bash
# 1. générer une config (assistant interactif)
bash workflow/run.sh --init

# 2. vérifier le DAG sans rien exécuter
bash workflow/run.sh --dry-run --configfile workflow/config/config.<nom>.yaml

# 3. lancer
bash workflow/run.sh --backend local   --configfile workflow/config/config.<nom>.yaml
bash workflow/run.sh --backend cluster --configfile workflow/config/config.<nom>.yaml
```

L'assistant demande le contraste **A vs B** (pro/sen, sain/malade, WT/mutant…),
les chemins, le backend, le nombre de seeds, la portée de la perturbation et les
ablations.

### Essayer sans les données

Le pipeline tourne de bout en bout sur un jeu jouet auto-généré — aucune donnée
à demander, aucun réseau, environ deux minutes sur un portable :

```bash
python tests/fixtures/make_tiny_dataset.py --out data_tiny
bash workflow/run.sh --backend local --configfile workflow/config/config.tiny.yaml
```

Vous obtenez un `cross_seed_gene_ranking.tsv` complet (43 colonnes), un rapport
QC, l'ORA et le SUMMARY — la chaîne réelle, sur ~65 gènes et 8 échantillons.

> **Ce n'est pas de la biologie.** Les valeurs d'expression sortent d'un
> générateur et l'écart entre les deux « états » est injecté à la main. Le
> classement produit ne veut rien dire. Ce jeu démontre une seule chose : que la
> mécanique est branchée. Il sert à vérifier qu'un refactor n'a rien cassé, et à
> voir l'outil bouger avant de réclamer les vraies données.

Deux autres configs de vérification : `config.smoke.yaml` (résolution du DAG sur
les vraies données, sans rien exécuter) et `config.yaml` (production).

Sur cluster, lancer dans `tmux`/`nohup` : Snakemake a besoin d'un processus
contrôleur vivant pendant tout le DAG. Détail des règles et des chemins dans
[workflow/README.md](workflow/README.md).

> **Entrées non versionnées.** Matrice d'expression, DE, pySCENIC, HuMess et
> bases de vieillissement vivent dans `data/` et ne sont pas dans le dépôt. Les
> caches OmniPath/HGNC, eux, **le sont** (~7 Mo) : sans eux le pipeline tourne
> à vide (features à zéro, aucune arête supplémentaire).

## Sortie principale

`cross_seed_gene_ranking.tsv`, un gène par ligne, trié par `driver_score` :

| Colonne | Sens |
|---|---|
| `driver_score` | composite post-perturbation, agrégé cross-seed |
| `discovery_score` | + bonus non-DE (trouvailles portées par le graphe) |
| `validation_score` | + bonus DE-significatif et bases de vieillissement |
| `evidence_tier` A–E | A confirmé · B découverte · C effecteur · D hub · E bruit |
| `canon_diff`, `canon_cosine` | amplitude de l'effet × alignement sur l'axe |
| `mean_stability` | accord des seeds sur le **signe** de l'effet |

## Validation — l'essentiel du dépôt

Un score de driver est facile à produire et difficile à croire. Les contrôles
sont donc de première classe, pas des extras.

**Toujours actifs** : `pipeline_qc` (les cinq contrôles préalables à toute
lecture d'ablation : plancher de bruit, multiplicité d'arêtes, recouvrement des
sources, confusion degré↔readout, spécificité d'axe) et `driver_baselines` (le
GNN bat-il une statistique triviale, à degré contrôlé ?).

**Activables** dans `config.yaml` : `purity_source` (quelle source d'arête porte
réellement une cible), `head_to_head` (un outil simple sort-il les mêmes
cibles ?), `readout_specificity` (métriques affranchies du degré), `decoy`
(nulles de voisinage et d'axe), `ora_de_baseline` (l'enrichissement dépasse-t-il
un simple tri par DE ?).

Le catalogue complet — ce qui est dans le DAG, ce qui est manuel, et pourquoi —
est dans [TOOLS.md](TOOLS.md).

> **Le plancher de bruit avant tout.** Deux runs d'une configuration
> *identique* diffèrent de ρ 0.556–0.687 sur le ranking (écart médian : 942
> rangs). C'est plus grand que la plupart des effets qu'on cherche à mesurer.
> Trois seeds sont un minimum, et `compute.deterministic: true` réduit le
> plancher au prix de la vitesse.

## Optimisation automatique

Optuna pilote Snakemake — jamais l'inverse : un DAG doit connaître ses jobs à
l'avance, une recherche non.

```bash
# 1. MESURER LE BRUIT AVANT DE CHERCHER (obligatoire pour interpréter la suite)
python src/optim/search.py calibrate --repeats 3 --objective cross_seed_stability

# 2. chercher
python src/optim/search.py search --n-trials 20 --seeds 3

# sur cluster (job contrôleur qui survit à la session)
bash scripts/run_optuna.sh calibrate --repeats 3
bash scripts/run_optuna.sh search    --n-trials 20
```

Trois objectifs branchables (`src/optim/objectives.py`) :
`cross_seed_stability` (recommandé — vise le plancher de bruit),
`recon_auc` (pas cher, mais ne décide pas des drivers), `known_driver_recall`
(le plus proche du but biologique, et le plus circulaire).

`report` refuse de conclure si la calibration n'a pas été faite : une étude dont
l'amplitude n'excède pas le bruit n'a rien trouvé.

## Tests

```bash
pytest                                # tout sauf le end-to-end (~5 min)
pytest tests/test_package_layout.py   # le paquet s'importe, pas de nom racine
pytest tests/test_de_schema.py        # unitaires sur la définition de l'axe DE
pytest tests/test_workflow.py         # le DAG se résout, validations comprises
pytest tests/test_optim.py            # plomberie de la recherche
pytest tests/test_cli_contract.py     # chaque point d'entrée répond à --help
pytest -m slow                        # pipeline complet sur jeu jouet (~2 min)
```

Le test `slow` est le seul qui exécute réellement la chaîne scientifique ; les
autres vérifient la structure. Il est désélectionné par défaut et tourne dans
son propre job de CI.

`tests/golden/` est un comparateur bit-exact conservé pour les refactors lourds.
Il n'est **pas exécutable en l'état** : il dépend d'un cache de graphe absent du
dépôt. Le régénérer avec `tests/golden/run_golden.sh capture`.

## Structure

```
src/gnn/            encodeur : construction du graphe, entraînement, scoring
src/perturbation/   KO / KD / OE et re-projection d'axes
src/validation/     tout ce qui met le résultat à l'épreuve
src/data/           loaders (bulk, protéomique, schéma DE) et préprocessing
src/optim/          recherche d'hyperparamètres
workflow/           Snakemake : DAG, configs, profil SLURM
scripts/            lanceurs cluster (grilles d'ablation, vagues, Optuna)
tests/              suite pytest + golden
archive/            code historique gelé — voir archive/README.md
```

## Citation

```bibtex
@unpublished{menard2026scstgp,
  author = {Ménard, Théo and Maillasson, Mike},
  title  = {(sc)STGP: State Transition Gene Prediction by heterogeneous VGAE
            and in-silico perturbation},
  year   = {2026},
  note   = {Stage M2, Université de Nantes}
}
```

Voir aussi [CITATION.cff](CITATION.cff).

## Licence

MIT — voir [LICENSE](LICENSE). Les dépendances gardent la leur : PyTorch
Geometric (MIT), OmniPath (GPL-3), pySCENIC (GPL-3). Redistribuer un travail
dérivé incluant OmniPath ou pySCENIC impose leurs conditions.

## Références méthodologiques

- Kipf & Welling 2016, *Variational Graph Auto-Encoders*, NeurIPS BDL.
- Veličković et al. 2018, *Graph Attention Networks*, ICLR.
- Aibar et al. 2017 ; Van de Sande et al. 2020, *SCENIC / pySCENIC*.
- Türei et al. 2021, *OmniPath*, Mol Syst Biol.
- Zirkel et al. 2018, *HMGB2 loss upon senescence entry*, Mol Cell.
- Hernandez-Segura et al. 2018, *Hallmarks of cellular senescence*.

## Contact

Théo Ménard — `theo.menard@etu.univ-nantes.fr`
