# scSTGP — single-cell State Transition Gene Prediction

Prioriser les gènes qui **pilotent** le passage d'un état cellulaire à un autre,
à partir de données transcriptomiques (single-cell ou bulk) et de réseaux
biologiques curés.

> **Deux noms, une seule chose.** L'outil s'appelle **scSTGP** ; il s'installe et
> s'importe sous le nom **`stateshift`**, parce que `stgp` était déjà pris sur
> PyPI. Les commandes ci-dessous sont donc bien celles de scSTGP.

```bash
pip install "stateshift[run]"
stateshift doctor
stateshift init            # assistant : décrit votre contraste A vs B
stateshift run --backend local --configfile <votre-config>.yaml
```

📖 **[Documentation complète → `docs/guide/`](docs/guide/README.md)** —
installation, démarrage en 5 minutes, exécution locale et SLURM, configuration,
commandes, lecture des résultats, dépannage.

## Le principe, et pourquoi il tient

La difficulté n'est pas de trouver les gènes qui **changent** entre deux états —
une analyse différentielle le fait. Elle est de trouver ceux qui **causent** le
changement, et de le faire sans se contenter de redécouvrir la liste dont on est
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
— et le dépôt fournit les outils pour le **vérifier** plutôt que l'affirmer.

L'application de référence est la sénescence réplicative endothéliale (HUVEC
P4 → P16, scRNA-seq Drop-seq GSE102090), mais la méthode ne lui est pas propre :
une transition se définit au **readout**, pas dans le modèle. Voir
[docs/guide/generalisation.md](docs/guide/generalisation.md).

## Essayer sans les données

Le pipeline tourne de bout en bout sur un jeu jouet auto-généré — aucune donnée
à demander, environ deux minutes sur un portable :

```bash
python tests/fixtures/make_tiny_dataset.py --out data_tiny
stateshift run --backend local --configfile workflow/config/config.tiny.yaml
```

> **Ce n'est pas de la biologie.** Les valeurs sortent d'un générateur et l'écart
> entre les deux « états » est injecté à la main. Ce jeu démontre une seule
> chose : que la mécanique est branchée.

## Sortie principale

`cross_seed_gene_ranking.tsv`, un gène par ligne, trié par `driver_score` :
amplitude de l'effet × alignement sur l'axe, agrégé sur les graines, avec un
palier de preuve A–E. Détail dans
[docs/guide/resultats.md](docs/guide/resultats.md).

> **Le plancher de bruit avant tout.** Le recouvrement du top-100 entre deux
> graines n'est que de 29 à 49 gènes selon le régime de graphe. Trois graines
> sont un minimum, `compute.deterministic: true` est obligatoire, et le
> raisonnement doit porter sur des **modules**, pas sur des gènes isolés.

## Deux modes, par conception

| Mode | Usage | Qui l'utilise |
|---|---|---|
| paquet | `stateshift run …`, `import stateshift.data.loaders.bulk_rna` | usage normal, tests, code nouveau |
| script | `bash workflow/run.sh …`, `python src/gnn/gnn_vgae.py …` | scripts SLURM historiques, dev sans pip |

Les fichiers ne bougent pas : `src/` est monté comme paquet `stateshift` via
`package-dir`. Déplacer l'arborescence aurait cassé les vingt scripts cluster et
le Snakefile sans rien apporter. Les deux modes lancent le même DAG.

## Structure

```
src/gnn/            encodeur : construction du graphe, entraînement, scoring
src/perturbation/   KO / KD / OE et re-projection d'axes
src/validation/     tout ce qui met le résultat à l'épreuve
src/data/           loaders (bulk, protéomique, schéma DE) et préprocessing
src/optim/          recherche d'hyperparamètres
workflow/           Snakemake : DAG, configs, profil SLURM (embarqué au paquet)
scripts/            lanceurs cluster (grilles d'ablation, vagues, Optuna)
docs/guide/         documentation utilisateur
tests/              suite pytest + golden
```

## Statut

Travail de stage M2 (Université de Nantes, équipe Petry) — soutenance
16 septembre 2026. Le pipeline tourne de bout en bout, en local comme sur SLURM.

Ce dépôt contient **l'outil et son mode d'emploi** (`docs/guide/`). Le rapport
scientifique — méthodes détaillées, résultats par version, cahier de conception,
catalogue de cibles — n'est pas versionné, non plus que les scripts qui
fabriquent ses figures et ses annexes : ils lisent des résultats non publiés et
n'auraient rien à lire ici. Les demander à l'auteur.

## Citation

```bibtex
@unpublished{menard2026scstgp,
  author = {Ménard, Théo and Maillasson, Mike},
  title  = {scSTGP: single-cell State Transition Gene Prediction by
            heterogeneous VGAE and in-silico perturbation},
  year   = {2026},
  note   = {Stage M2, Université de Nantes. Distribué sous le nom `stateshift`}
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
