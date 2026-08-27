# Démarrage — voir l'outil tourner en 5 minutes

Avant de réclamer des données ou de réserver du temps de calcul, faites tourner
la chaîne complète sur un jeu jouet auto-généré. Aucune donnée à obtenir, aucun
réseau, environ deux minutes sur un portable.

## Le jeu jouet

```bash
# 1. générer les données factices (nécessite un clone pour le générateur)
python tests/fixtures/make_tiny_dataset.py --out data_tiny

# 2. lancer la chaîne réelle dessus
stateshift run --backend local --configfile workflow/config/config.tiny.yaml
```

Vous obtenez un `cross_seed_gene_ranking.tsv` complet (43 colonnes), un rapport
QC, l'ORA et le SUMMARY — la chaîne réelle, sur ~65 gènes et 8 échantillons.

> **Ce n'est pas de la biologie.** Les valeurs d'expression sortent d'un
> générateur et l'écart entre les deux « états » est injecté à la main. Le
> classement produit ne veut rien dire. Ce jeu démontre une seule chose : que la
> mécanique est branchée. Il sert à vérifier qu'un refactor n'a rien cassé, et à
> voir l'outil bouger avant de réclamer les vraies données.

## Les trois configurations de vérification

| Config | Ce qu'elle prouve | Données requises | Durée |
|---|---|---|---|
| `config.tiny.yaml` | la chaîne **s'exécute** de bout en bout | aucune (générées) | ~2 min |
| `config.smoke.yaml` | le DAG **se résout** sur les vraies entrées | `data/` complet | secondes |
| `config.yaml` | production | `data/` complet | heures |

`config.tiny.yaml` est le seul exécutable par un tiers. Il a fait remonter
quatre défauts réels le jour de son écriture — `latent_dim` non relu depuis le
point de reprise, groupes de cellules figés sur HUVEC côté perturbation,
description du dataset couplée à `build.enabled`, et `--no-baselines` qui
faisait tomber l'agrégation via une figure. Aucun n'était atteignable sans
exécuter la chaîne.

Lister les configurations embarquées avec le paquet :

```bash
stateshift configs
```

## Sur vos propres données

L'assistant interactif écrit une configuration à partir de questions simples :
le contraste A vs B, les chemins, le backend, le nombre de graines, la portée de
la perturbation et les ablations.

```bash
stateshift init
```

Il écrit dans `workflow/config/` depuis un clone, dans `./stateshift-configs/`
depuis un paquet installé (`--out-dir` pour choisir). Ensuite, toujours vérifier
le DAG avant de consommer du calcul :

```bash
stateshift run --dry-run --configfile <votre-config>.yaml
stateshift run --backend local --configfile <votre-config>.yaml
```

## D'où lancer la commande

`stateshift run` s'exécute **dans votre répertoire de projet** : les chemins de
la configuration (`data_root`, `out_base`) sont résolus par rapport au
répertoire courant, et les sorties y sont écrites. Le paquet, lui, est retrouvé
tout seul — vous n'avez pas besoin d'être dans un clone.

```bash
mkdir -p ~/projets/mon-etude && cd ~/projets/mon-etude
stateshift init --out-dir .
stateshift run --configfile config.mon-etude.yaml
```

## Et après

- comprendre ce que vous venez de produire → [resultats.md](resultats.md)
- passer sur le cluster → [execution.md](execution.md)
- régler les paramètres → [configuration.md](configuration.md)
