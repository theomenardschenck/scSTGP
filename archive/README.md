# archive/ — code conservé pour référence, **hors pipeline**

Rien ici n'est appelé par le workflow, les scripts cluster ou les tests. Ces
fichiers sont gardés parce qu'ils documentent un état antérieur du code auquel
la documentation et les journaux de conception renvoient encore. Ne pas les
importer, ne pas les modifier : ils ne sont couverts par aucun test.

| Fichier | Ce que c'est | Remplacé par |
|---|---|---|
| `gnn_vgae_monolith_presplit.py` | `gnn_vgae.py` avant le découpage de juillet 2026 (4 827 lignes en un seul fichier) | `src/gnn/gnn_vgae.py` (orchestrateur mince) + `_graph_build.py` / `_train.py` / `_score.py` / `_config.py` / `_paths.py` / `_vgae_model.py` |
| `_supervised_standalone.py` | Second entraînement supervisé bout-en-bout, sans perte de reconstruction (`train_supervised` / `run_supervised` / `save_supervised_run`) | Rien : la tête supervisée est désormais **co-entraînée** dans la boucle VGAE (`_train_body.py`, `--supervised`). Cette voie avait zéro appelant. |

Le découpage a été validé bit-exact par `tests/golden/` au moment où il a été
fait. Ce filet n'est plus exécutable en l'état — il dépend d'un cache de graphe
(`output/gnn_vgae/_graph_cache_scrna.pkl`) qui n'existe plus ; voir
`tests/golden/run_golden.sh capture` pour le régénérer.
