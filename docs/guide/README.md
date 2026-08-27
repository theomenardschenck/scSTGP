# Guide scSTGP

Documentation utilisateur complète de **scSTGP** — priorisation des gènes qui
*pilotent* une transition entre deux états cellulaires.

> **Pourquoi deux noms.** L'outil s'appelle **scSTGP** (*single-cell State
> Transition Gene Prediction*). Il s'installe, s'importe et se lance sous le nom
> **`stateshift`** : `stgp` était déjà pris sur PyPI. Partout dans ce guide,
> `stateshift` désigne donc bien scSTGP — c'est le nom du paquet, pas un autre
> logiciel.

Ce répertoire est la seule documentation publiée. Le reste de `docs/` est un
cahier de laboratoire local (résultats non publiés, cibles, notes de
conception) et ne quitte pas la machine de l'auteur.

## Par où commencer

| Vous voulez… | Lisez |
|---|---|
| installer l'outil | [installation.md](installation.md) |
| le voir tourner en 5 minutes, sans données | [demarrage.md](demarrage.md) |
| lancer sur vos données, en local ou sur SLURM | [execution.md](execution.md) |
| **travailler depuis un clone git, et préparer les données amont** (pySCENIC, HuMess, co-expression) | [depuis-un-clone.md](depuis-un-clone.md) |
| comprendre un fichier de configuration | [configuration.md](configuration.md) |
| la liste des commandes et des scripts | [commandes.md](commandes.md) |
| lire et croire le classement produit | [resultats.md](resultats.md) |
| appliquer la méthode à une autre transition | [generalisation.md](generalisation.md) |
| débloquer une erreur | [depannage.md](depannage.md) |
| publier une nouvelle version | [publication.md](publication.md) |

## Le principe, en un écran

La difficulté n'est pas de trouver les gènes qui **changent** entre deux états —
une analyse différentielle le fait. Elle est de trouver ceux qui **causent** le
changement, sans se contenter de redécouvrir la liste dont on est parti.

L'architecture répond par une séparation stricte :

1. **L'encodeur ne voit jamais le contraste.** Un VGAE hétérogène apprend une
   représentation à partir de sources structurelles et condition-indépendantes
   (PPI STRING, Reactome, OmniPath signé, co-expression, régulons pySCENIC,
   métabolisme HuMess). Le logFC n'est jamais une variable d'entrée.
2. **La perturbation est *in silico*.** Chaque gène est éteint (KO), atténué
   (KD) ou surexprimé (OE) ; l'encodeur gelé propage l'effet dans le graphe.
3. **L'axe seul porte les deux états.** Le déplacement Δz est projeté sur un axe
   défini par le contraste A→B. Changer de transition = changer d'axe, pas de
   modèle.

C'est cette séparation qui rend le résultat non trivialement re-dérivable du
DE — et l'outil fournit de quoi le **vérifier** plutôt que de l'affirmer, ce que
détaille [resultats.md](resultats.md).

## Deux façons de s'en servir, et pourquoi les deux existent

| Mode | Commande | Pour qui |
|---|---|---|
| **paquet** | `pip install stateshift` puis `stateshift run …` | usage normal, poste de travail, cluster |
| **script** | `git clone …` puis `bash workflow/run.sh …` | développement, scripts SLURM historiques, environnement sans pip |

Les deux lancent **le même Snakefile avec les mêmes règles** : un run est
indistinguable de l'autre. Le mode script reste pris en charge et testé — il
n'est pas déprécié. Voir [execution.md](execution.md#les-deux-modes).

Le clone n'est pas qu'une commodité de développement : **toute la chaîne de
prétraitement en dépend** (co-expression, pySCENIC, HuMess), ainsi que les
grilles d'ablation SLURM. C'est l'objet de
[depuis-un-clone.md](depuis-un-clone.md).

## Statut et limites

Travail de stage M2 (Université de Nantes, équipe Petry). Le pipeline tourne de
bout en bout, en local comme sur SLURM. Ce qu'il faut savoir avant de citer un
résultat :

- **Le plancher de bruit domine beaucoup d'effets.** Deux agrégats 3-graines
  d'une même configuration corrèlent à r ≈ 0.79–0.94 selon le régime de graphe,
  et le recouvrement du top-100 entre deux graines n'est que de 29 à 49 gènes.
  Raisonnez par **module**, pas par gène isolé. Détail dans
  [resultats.md](resultats.md#le-plancher-de-bruit-dabord).
- **Le classement dépend du graphe.** Trois régimes de sources donnent trois
  têtes de classement différentes. Ce n'est pas un défaut à masquer, c'est une
  propriété à mesurer — et les ablations sont là pour ça.

## Citation

```bibtex
@unpublished{menard2026stateshift,
  author = {Ménard, Théo and Maillasson, Mike},
  title  = {stateshift: State Transition Gene Prediction by heterogeneous VGAE
            and in-silico perturbation},
  year   = {2026},
  note   = {Stage M2, Université de Nantes}
}
```

Licence MIT. Les dépendances gardent la leur : PyTorch Geometric (MIT),
OmniPath (GPL-3), pySCENIC (GPL-3) — redistribuer un travail dérivé qui les
inclut impose leurs conditions.
