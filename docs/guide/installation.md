# Installation

Trois chemins, du plus simple au plus complet. Le choix dépend de ce que vous
voulez faire, pas de vos goûts : **la pile scientifique lourde (torch, PyG,
scanpy, pySCENIC) n'est pas installée par défaut**, et un vrai run en a besoin.

| Vous voulez | Faites | Suffit pour |
|---|---|---|
| lire des résultats, écrire du code autour | `pip install stateshift` | loaders, schéma DE, CLI, analyses légères |
| faire tourner le pipeline | `pip install "stateshift[run]"` | tout, hors pySCENIC/HuMess amont |
| reproduire l'environnement de référence | `conda env create -f environment.yml` | tout, versions épinglées, cluster |

Python **3.12 minimum**.

## 1. Par pip

```bash
python -m pip install "stateshift[run]"
stateshift doctor
```

`stateshift doctor` est à lancer immédiatement : il vérifie l'installation, la
présence et la **cohérence** de snakemake, la pile scientifique et l'accès à
SLURM. Sa sortie est le premier élément à joindre à toute demande d'aide.

Extras disponibles, cumulables (`"stateshift[torch,omics]"`) :

| Extra | Contenu | Nécessaire pour |
|---|---|---|
| `torch` | torch, torch-geometric | entraînement, perturbation — donc tout run |
| `omics` | scanpy, anndata, omnipath | lecture scRNA, sources OmniPath |
| `workflow` | snakemake 7.32 (+ `pulp<2.8`) | orchestration du DAG |
| `optim` | optuna | recherche d'hyperparamètres |
| `run` | les quatre ci-dessus | **le raccourci recommandé** |
| `dev` | pytest, ruff | développement |

> **torch par pip, attention à la roue.** `pip install stateshift[torch]` prend
> la roue par défaut de PyPI, qui embarque CUDA. Sur une machine sans GPU c'est
> plusieurs gigaoctets pour rien :
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install "stateshift[workflow,omics,optim]"
> ```

## 2. Par conda — l'environnement de référence

C'est celui qui tourne sur le cluster, et le seul dont les versions sont
épinglées finement. **`environment.yml` fait foi pour un vrai run** ; les
dépendances déclarées dans `pyproject.toml` ne visent que l'import du paquet et
les tests.

```bash
git clone https://github.com/theomenardschenck/scSTGP.git
cd scSTGP

micromamba create -n stateshift -f environment.yml   # ou conda / mamba
micromamba activate stateshift

pip install -e .          # le paquet lui-même, en mode éditable
stateshift doctor
```

Le canal `bioconda` est requis (il fournit `snakemake-minimal`).

## 3. Depuis le dépôt, sans publication

```bash
pip install "git+https://github.com/theomenardschenck/scSTGP.git"
```

## Vérifier que ça marche vraiment

Un `import` réussi ne prouve rien : ce qui compte est que la chaîne s'exécute.
Le jeu jouet le démontre en deux minutes, sans aucune donnée à obtenir.

```bash
stateshift doctor          # environnement
cd /un/repertoire/de/travail
python -c "import stateshift; print(stateshift.snakefile())"   # le workflow a voyagé
```

Puis suivez [demarrage.md](demarrage.md) pour l'exécution réelle.

## Le piège classique : deux environnements

Le mode d'échec le plus coûteux observé sur ce projet n'est pas un bug de code,
c'est une **dérive d'environnement** : `snakemake` résolu depuis un
environnement, le paquet installé dans un autre. Le DAG démarre puis échoue au
milieu, ou pire, tourne avec un interpréteur périmé.

```bash
which snakemake && which python && python -c "import sys; print(sys.prefix)"
```

Les deux doivent pointer dans le même préfixe. `stateshift doctor` le signale
explicitement — c'est la ligne « snakemake est HORS de l'environnement
courant ». `stateshift run` corrige d'ailleurs la moitié du problème en
imposant aux règles l'interpréteur qui possède le paquet, mais mieux vaut un
environnement propre.

## Données d'entrée

Le paquet contient le **code et le pipeline**, pas les données. Sont attendus
en entrée, non versionnés :

- matrice d'expression + métadonnées de groupes ;
- table d'analyse différentielle (le *readout*, jamais une variable d'entrée) ;
- réseaux : STRING/PPI, Reactome, OmniPath, régulons pySCENIC, HuMess.

### Les caches OmniPath et HGNC

Cas particulier, à ne pas rater : ces caches (~8 Mo) sont **versionnés dans le
dépôt** mais **ne voyagent pas dans la roue pip**. Ce n'est pas un oubli — le
code les cherche sous `<data_root>/omnipath`, c'est-à-dire dans *votre*
répertoire de données, pas dans le paquet ; les embarquer serait du poids mort
que rien ne lirait.

Sans eux, le pipeline tourne à vide : variables à zéro, aucune arête
supplémentaire, et le rapport QC le signale. Après une installation par pip,
récupérez-les :

```bash
git clone --depth 1 https://github.com/theomenardschenck/scSTGP.git /tmp/ss
mkdir -p <votre-data_root>
cp -r /tmp/ss/data/omnipath <votre-data_root>/
```

Depuis un clone, ils sont déjà en place et il n'y a rien à faire.

Voir [configuration.md](configuration.md) pour la disposition attendue.
