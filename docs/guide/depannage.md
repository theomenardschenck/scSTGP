# Dépannage

**Premier réflexe, avant tout le reste :**

```bash
stateshift doctor
```

Il répond à la majorité des questions ci-dessous, et sa sortie est ce qu'il faut
joindre à une demande d'aide.

## Installation et environnement

### `stateshift: command not found`

Le paquet est installé dans un environnement qui n'est pas actif, ou le
répertoire des scripts n'est pas dans le `PATH`. Contournement immédiat :

```bash
python -m stateshift.cli doctor
```

### `snakemake introuvable dans le PATH`

Le pipeline exige snakemake **7.x** :

```bash
pip install "stateshift[workflow]"
# ou : conda env create -f environment.yml && conda activate stateshift
```

Attention au nom si vous l'installez à la main : côté **pip** le paquet
s'appelle `snakemake` (`snakemake-minimal` n'existe que sur **conda**), et il
faut y adjoindre `pulp<2.8` — snakemake 7.32 appelle `pulp.list_solvers`, retiré
en pulp 2.8, et sans ce pin `snakemake --version` plante de lui-même. L'extra
`[workflow]` s'en charge.

### « snakemake est HORS de l'environnement courant »

Le diagnostic le plus important que produit `doctor`. `snakemake` vient d'un
environnement, le paquet d'un autre : le Snakefile peut alors ne pas voir
`stateshift`, ou les règles tourner avec un interpréteur dépourvu de torch.

```bash
which snakemake
python -c "import sys; print(sys.prefix)"
```

Les deux doivent partager le même préfixe. Sinon, installez snakemake dans
l'environnement du paquet. `stateshift run` limite déjà les dégâts en imposant
aux règles l'interpréteur qui possède le paquet, mais la situation reste
fragile.

### `Workflow introuvable (ni dans le paquet, ni à côté)`

La roue a été construite sans ses données de paquet. Réinstallez depuis une
source saine ; en dépannage, désignez le workflow explicitement :

```bash
export STATESHIFT_WORKFLOW=/chemin/vers/workflow
```

### `Script introuvable : …`

Un sous-paquet manque à la roue installée. Vérifiez avec :

```bash
python -c "import stateshift; print(stateshift.script('validation/figures/memoire_figures.py'))"
```

Si le fichier existe dans un clone mais pas dans l'installation, c'est la liste
`packages` de `pyproject.toml` qui est incomplète.

## Exécution

### Le DAG ne se résout pas

Toujours diagnostiquer à vide avant de consommer du calcul :

```bash
stateshift run --dry-run --configfile ma-config.yaml
```

Causes fréquentes : un chemin de `paths` qui n'existe pas depuis le répertoire
courant (ils sont relatifs au **répertoire de travail**), ou une validation
activée sans son paramètre obligatoire — `purity_source` exige `targets`.

### `Directory cannot be locked`

Une exécution précédente s'est interrompue brutalement :

```bash
stateshift run --configfile ma-config.yaml -- --unlock
```

### Tout est ré-exécuté alors que rien n'a changé

Snakemake déclenche aussi sur la provenance (code modifié), pas seulement sur
les dates. Pour ne considérer que les dates :

```bash
stateshift run --configfile ma-config.yaml -- --rerun-triggers mtime
```

### Une validation échoue et emporte le run

Elle ne devrait pas : `--keep-going` est actif par défaut, les livrables
scientifiques se terminent et Snakemake sort en erreur à la fin. Si le DAG
s'arrête au premier échec, c'est que `--keep-going` a été neutralisé par un
argument passé après `--`.

## Cluster

### Rien n'apparaît dans `squeue`

Tout tourne sur le nœud de connexion : le profil SLURM n'a pas été pris en
compte. Utilisez `--backend cluster` (et non `-j N` passé à la main) et
vérifiez la ligne « profil résolu » affichée au lancement.

### Les jobs sont tués après quelques minutes

La QOS plafonne la durée. Regardez la vôtre et ajustez le profil :

```bash
sacctmgr show qos format=Name,MaxWall
stateshift profile --copy ./mon-profil
$EDITOR mon-profil/config.yaml          # slurm_qos, runtime
stateshift run --backend cluster --profile-dir ./mon-profil --configfile c.yaml
```

### `Invalid partition` / `Invalid qos specification`

Les valeurs par défaut viennent de GLiCID. Confrontez-les à `sinfo` et
`sacctmgr show qos`, puis corrigez votre copie du profil.

### `submit.sh: Permission denied`

Ne devrait plus arriver : le profil est reconstruit à chaque lancement avec un
appel `bash <chemin absolu>`, justement parce qu'une roue ne conserve pas le bit
d'exécution. Si vous invoquez snakemake à la main, faites de même.

### La session SSH tombe et le run meurt

Snakemake a besoin d'un processus contrôleur vivant. Lancez-le dans `tmux` ou
`nohup`.

## Résultats

### Deux runs identiques donnent des classements différents

Vérifiez `compute.deterministic: true`. Avec lui, l'exécution est bit-exacte à
graine fixée ; ce qui reste est la variabilité de **graine**, qui est réelle et
importante — voir [resultats.md](resultats.md#le-plancher-de-bruit-dabord).
Trois graines sont un minimum.

### Un gène a bougé de plusieurs dizaines de rangs entre deux runs

C'est attendu, y compris en tête de classement : le recouvrement du top-100
entre deux graines est de 29 à 49 gènes selon le régime de graphe. Raisonnez par
module.

### Tous les scores sont nuls ou le graphe est minuscule

Les caches OmniPath/HGNC sont introuvables, ou `data_root` ne pointe pas sur les
données. Sans eux, le pipeline tourne à vide : variables à zéro, aucune arête
supplémentaire. Le rapport QC le signale.

### L'ORA « aging » ne veut rien dire

Elle est spécifique à la sénescence. Sur une autre transition, ne la lisez pas —
voir [generalisation.md](generalisation.md).

## Signaler un problème

Joignez systématiquement :

```bash
stateshift doctor                      # environnement
stateshift run --dry-run --configfile … 2>&1 | tail -40   # résolution du DAG
```

plus la configuration utilisée et, sur cluster, la sortie du job SLURM fautif.
