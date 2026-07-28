# Généralisation — ce qui, dans (sc)STGP, est encore lié à la sénescence

(sc)STGP = *State Transition Gene Prediction*. La méthode prend deux états
cellulaires et prédit les gènes qui **pilotent** le passage de l'un à l'autre.
L'application de référence est la sénescence réplicative endothéliale (HUVEC
P4 → P16), mais rien dans la méthode n'y est propre.

Ce document dit où en est le découplage, mesuré sur le code (2026-07-27), pour
que quiconque appliquant l'outil à une autre transition sache exactement quoi
paramétrer et quoi modifier.

## Le principe qui rend la généralisation possible

L'encodeur ne voit **jamais** le contraste entre états. Il apprend une
représentation à partir de réseaux structurels et condition-indépendants
(PPI, Reactome, OmniPath, co-expression, régulons). Les deux états
n'interviennent qu'au **readout**, pour définir l'axe sur lequel on projette
l'effet d'une perturbation.

Conséquence : changer de transition = changer l'axe, pas le modèle. C'est
aussi ce qui protège de la circularité — le graphe ne peut pas « savoir » la
réponse qu'on lui demande.

## État du découplage

### Déjà générique

| Élément | Comment |
|---|---|
| Groupes de cellules | `GNN_CELL_GROUPS` (ex. `pro,sen`). Défaut HUVEC 5 groupes pour rétro-compatibilité. Lu par `_graph_build_body.py:449`. |
| Conditions HuMess | `GNN_HUMESS_CONDITIONS` |
| Matrice / métadonnées | `GNN_EXPR_MATRIX`, `GNN_GROUP_META` |
| Définition de l'axe | `perturbation.axis: phenotypic \| de \| effector` — l'axe DE-ancré ne demande qu'une table DE au schéma canonique |
| Schéma DE | `stgp.data.loaders.de_schema` : détection automatique des colonnes, `<A>_vs_<B>` arbitraire, sc et bulk indifféremment |
| Sources du graphe | chaque source est activable/désactivable (`--no-coexpr`, `--no-humess`, `--omnipath-edges`, …) |

### Encore lié à la sénescence

| Où | Quoi | Gravité |
|---|---|---|
| CLI de perturbation et de validation | Les pôles s'appellent `--quiescent-groups` / `--p16` (`perturb_top_genes.py`, `purity_source_attribution.py`, `readout_specificity.py`, `reproject_axes.py`) | **Cosmétique mais trompeur** — les valeurs sont libres, seul le nom est daté. À renommer `--pole-a` / `--pole-b` avec alias de compatibilité. |
| `compute_senescence_axes()` | Nom de la fonction qui construit *tout* axe, y compris multi-état | Cosmétique. Renommer `compute_state_axes()`. |
| `cosine_quiescent_like` | Colonne de sortie ; le nom est en outre déjà **faux** pour HUVEC (l'hypothèse « c0 = quiescent » a été invalidée : c0 est proliférant-persistant) | À renommer. Déjà signalé comme dette avant ce document. |
| Valeurs par défaut | `CELL_GROUPS = ("P4", "P16_cluster_0", …)` en dur (`perturb_top_genes.py:135`), `--quiescent-groups` défaut `"P4"` | Acceptable comme défaut rétro-compatible, à condition que ce soit dit. C'est fait ici. |
| Bases de vieillissement | L'ORA « aging » (CellAge, GenAge) et les listes Tier-1 sont spécifiques à la sénescence | **Structurel.** Pour une autre transition il faut une base de référence propre au phénotype, ou se limiter à l'ORA Reactome. |
| `build_supervised_labels.py` | Labels DEG codés `P4_vs_P16` + `cluster_0..3` | À paramétrer si la tête supervisée est utilisée. |
| `cluster_annotation.py` | Marqueurs biologiques d'annotation orientés sénescence/SASP | **Structurel** — dépend du système biologique. |

### Ce qui n'est pas transposable tel quel

L'**arbitrage de validité** repose sur des points d'appui propres à la
sénescence : rappel de drivers connus (CellAge), Tier-1 HUVEC, littérature
Zirkel/Ahn. Sur une nouvelle transition, ces points d'appui n'existent pas —
il faut les reconstituer, sinon il ne reste que les contrôles internes
(`pipeline_qc`, décoys, baselines head-to-head), qui disent si un résultat
est un artefact mais pas s'il est juste.

## Ordre recommandé pour appliquer l'outil à une autre transition

1. Table DE de la transition, au schéma canonique (`de_schema` la détecte).
2. `GNN_CELL_GROUPS` = les deux (ou N) états.
3. `perturbation.axis: de` avec `de_axis_file` pointant sur cette table.
4. Sources du graphe : garder celles qui sont condition-indépendantes ; la
   co-expression et HuMess sont data-dérivées, donc à recalculer sur le
   nouveau jeu (`build.enabled: true`).
5. `validation.qc.enabled: true` — le plancher de bruit est à re-mesurer par
   système, il n'est pas transférable.
6. Ne pas lire l'ORA « aging » : elle ne veut rien dire hors sénescence.
