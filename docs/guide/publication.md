# Publier une version

Procédure de release du paquet `stateshift` sur PyPI. Elle est écrite pour être
suivie ligne à ligne, y compris dans six mois.

## Avant toute chose

Trois faits qui rendent une erreur coûteuse :

- **un numéro de version publié sur PyPI ne se réutilise jamais.** Même
  supprimé, `0.1.0` est brûlé définitivement. D'où la répétition sur TestPyPI.
- **un nom de projet non plus.** `stateshift` est libre aujourd'hui ; il sera
  réservé dès la première publication réussie.
- le nom d'import `stateshift` est distinct de `stgp`, **déjà occupé sur PyPI**
  par un paquet de génomique sans lien. C'est précisément ce qui a motivé le
  choix du nom.

## Pré-vol

```bash
cd <racine du dépôt>

# 1. la suite de tests passe
pytest
pytest -m slow                      # la chaîne réelle sur jeu jouet (~2 min)

# 2. le lint ne signale rien de bloquant
ruff check .

# 3. l'environnement est cohérent
stateshift doctor
```

Vérifications spécifiques à la mise en paquet, celles qui ont réellement
attrapé des défauts :

```bash
# le workflow voyage avec le paquet, et tous les scripts sont là
pytest tests/test_package_layout.py -v

# chaque point d'entrée répond encore à --help
pytest tests/test_cli_contract.py
```

## 1. Fixer la version

La version vit à **deux endroits** qui doivent concorder :

```
pyproject.toml     version = "0.1.0"
src/__init__.py    __version__ = "0.1.0"
```

Convention : `MAJEUR.MINEUR.CORRECTIF`. Tant que l'outil n'est pas stabilisé,
restez en `0.x` — cela signale explicitement que l'interface peut bouger.

Mettez à jour `CITATION.cff` (champ `version` et `date-released`) dans le même
mouvement.

## 2. Construire

```bash
python -m pip install --upgrade build twine
rm -rf dist/ build/ src/*.egg-info
python -m build
```

Vous obtenez `dist/stateshift-<version>.tar.gz` (source) et
`dist/stateshift-<version>-py3-none-any.whl` (roue).

## 3. Vérifier le contenu de la roue

Étape à ne pas sauter : c'est ici qu'on voit si le pipeline est bien embarqué.

```bash
python -m twine check dist/*

# le Snakefile, les configs et le profil SLURM sont-ils dans la roue ?
python -m zipfile -l dist/stateshift-*.whl | grep -E "Snakefile|config/.*yaml|submit.sh" | head

# aucun sous-paquet manquant ?
python -m zipfile -l dist/stateshift-*.whl | grep -c "\.py$"
```

Si `Snakefile` n'apparaît pas, `[tool.setuptools.package-data]` est en cause et
la roue est inutilisable — elle n'installerait qu'une bibliothèque.

### Les pièges déjà rencontrés

Trois défauts n'apparaissaient **que** dans un venv neuf, jamais depuis le
clone. Ils sont corrigés, mais la classe de problème reste :

| Symptôme | Cause | Garde en place |
|---|---|---|
| `memoire_figures.py` absent de l'installation | `validation/figures/` sans `__init__.py` ni entrée dans `packages` | `test_every_script_the_snakefile_calls_is_installed` |
| `pip install .[run]` échoue | `snakemake-minimal` est un paquet **conda**, absent de PyPI (côté pip : `snakemake`) | à revérifier à chaque changement d'extra |
| `snakemake --version` plante | pip résout `pulp` 3.x, or snakemake 7.32 appelle `pulp.list_solvers`, retiré en 2.8 | pin `pulp<2.8` |

Un dry-run ne les attrape pas : Snakemake n'inspecte jamais la commande shell
qu'il exécuterait. **Seule une installation propre les révèle.**

## 4. Répétition générale sur TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
```

Identifiants : un jeton d'API créé sur https://test.pypi.org/manage/account/token/
(nom d'utilisateur `__token__`, mot de passe = le jeton, préfixe `pypi-`).

Puis **installer depuis TestPyPI dans un environnement neuf**, ce qui est le
véritable test :

```bash
python -m venv /tmp/essai-stateshift
/tmp/essai-stateshift/bin/pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  stateshift

cd /tmp
/tmp/essai-stateshift/bin/stateshift doctor
/tmp/essai-stateshift/bin/stateshift configs
/tmp/essai-stateshift/bin/python -c "import stateshift; print(stateshift.snakefile())"
```

`--extra-index-url` est nécessaire : TestPyPI n'héberge pas les dépendances.

Le contrôle qui compte vraiment — la chaîne s'exécute-t-elle depuis un
répertoire quelconque ?

```bash
mkdir -p /tmp/essai-projet && cd /tmp/essai-projet
python <clone>/tests/fixtures/make_tiny_dataset.py --out data_tiny
cp <clone>/workflow/config/config.tiny.yaml .
/tmp/essai-stateshift/bin/stateshift run --backend local --cores 4 --configfile config.tiny.yaml
```

Attendu : 14 étapes, et un `cross_seed_gene_ranking.tsv` à 43 colonnes.

## 5. Publier sur PyPI

Uniquement après le succès de l'étape 4.

```bash
python -m twine upload dist/*
```

Jeton créé sur https://pypi.org/manage/account/token/.

Vérification finale, dans un environnement neuf :

```bash
python -m venv /tmp/verif && /tmp/verif/bin/pip install stateshift
/tmp/verif/bin/stateshift --version
```

## 6. Marquer la release

```bash
git tag -a v0.1.0 -m "stateshift 0.1.0"
git push origin v0.1.0
```

## Après publication

- Le nom est réservé : plus de retour en arrière possible sur `stateshift`.
- Pour corriger, publiez `0.1.1` — ne tentez pas de remplacer `0.1.0`.
- `pip install stateshift` installe le **cœur seul**. Rappelez dans l'annonce
  que `pip install "stateshift[run]"` est nécessaire pour exécuter le pipeline.

## Points laissés ouverts

Deux décisions n'appartiennent pas à cette procédure et doivent être tranchées
avant une publication publique :

1. **La licence MIT** est écrite dans `LICENSE` mais n'a pas été validée par le
   laboratoire. Les dépendances OmniPath et pySCENIC sont en GPL-3 : distribuer
   un dérivé qui les inclut impose leurs conditions.
2. **Le dépôt de destination** — compte personnel ou organisation du
   laboratoire — conditionne l'URL inscrite dans les métadonnées du paquet.
