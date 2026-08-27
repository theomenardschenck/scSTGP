"""Pipeline Snakemake, embarqué dans le paquet sous ``stateshift.workflow``.

Ce répertoire n'est pas une bibliothèque : il contient le Snakefile, les
configurations et le profil SLURM. Le rendre importable est un moyen, pas une
fin — c'est ce qui permet à setuptools de le faire VOYAGER avec la roue, donc à
``stateshift run`` de fonctionner hors d'un clone.

Le chemin du Snakefile installé se résout par ``stateshift.snakefile()``,
jamais par un chemin relatif au répertoire de travail.
"""
