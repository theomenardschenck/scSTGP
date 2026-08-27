# Lire les résultats — et savoir jusqu'où les croire

## La sortie principale

`cross_seed_gene_ranking.tsv`, un gène par ligne, trié par `driver_score`, 43
colonnes. Les essentielles :

| Colonne | Sens |
|---|---|
| `driver_score` | composite post-perturbation, agrégé sur les graines |
| `discovery_score` | + bonus non-DE (trouvailles portées par le graphe) |
| `validation_score` | + bonus DE-significatif et bases de vieillissement |
| `evidence_tier` A–E | A confirmé · B découverte · C effecteur · D hub · E bruit |
| `canon_diff` | amplitude de l'effet de la perturbation |
| `canon_cosine` | alignement de cet effet sur l'axe de transition |
| `mean_stability` | accord des graines sur le **signe** de l'effet |

Le score combine amplitude et alignement. Les deux ne se valent pas :
l'amplitude est très stable d'une graine à l'autre, l'alignement beaucoup
moins — or c'est l'alignement qui porte l'information de direction. C'est la
raison de fond des précautions ci-dessous.

Autres sorties, sous `<out_base>/<run>/analysis/` :

| Fichier | Contenu |
|---|---|
| `report/SUMMARY.md` | synthèse lisible du run |
| `qc/qc_report.md` | les cinq contrôles préalables |
| `interpret/driver_baselines.tsv` | le GNN contre des statistiques triviales |
| `interpret/ora/` | enrichissements Reactome / KEGG / Hallmark / vieillissement |
| `interpret/communities.tsv` | communautés Louvain de l'espace latent |

## Le plancher de bruit, d'abord

Un score de driver est facile à produire et difficile à croire. Avant toute
lecture, il faut savoir ce que vaut le bruit — sur ce pipeline il est **plus
grand que la plupart des effets qu'on cherche à mesurer**.

Ce qui a été mesuré, sur la version V6.1.3 :

- l'exécution est **bit-exacte** à configuration et graine fixées, `deterministic:
  true` activé : le bruit d'exécution est nul ;
- ce qui reste est la **variabilité de graine**. Entre deux agrégats à
  3 graines, la corrélation du `driver_score` est de **0.92–0.94** sur un graphe
  « pur », **0.79–0.80** sur un graphe riche ;
- **l'instabilité est en tête de classement, pas en queue.** Sur la moitié basse
  du classement r = 0.61–0.93, mais sur le **top-100 r = 0.32**, et sur la
  composante cosinus elle devient négative (−0.21) ;
- le **recouvrement du top-100** entre deux graines est de **49/100** sur graphe
  pur et **29/100** sur graphe riche.

Trois conséquences pratiques, non négociables :

1. **Trois graines minimum**, et `compute.deterministic: true`. En dessous, une
   ablation n'est pas interprétable — l'effet mesuré peut être entièrement du
   bruit de graine.
2. **Raisonnez par module, pas par gène.** Un gène qui passe du rang 4 au rang
   12 n'a rien fait de notable. Un groupe fonctionnel cohérent qui monte en bloc
   est un signal.
3. **Publiez le recouvrement, pas le rho global.** Un rho de 0.8 sur 13 000
   gènes masque un désaccord massif là où on regarde.

## La validation est de première classe

Les contrôles ne sont pas des extras : ce sont eux qui distinguent un résultat
d'une affirmation.

**Toujours actifs**

| Module | Question |
|---|---|
| `pipeline_qc` | les cinq contrôles préalables : plancher de bruit, multiplicité d'arêtes, recouvrement des sources, confusion degré↔readout, spécificité d'axe |
| `driver_baselines` | le GNN bat-il une statistique triviale, **à degré contrôlé** ? |

**Activables**

| Clé `validation.*` | Question |
|---|---|
| `purity_source` | quelle **source d'arête** porte réellement une cible ? |
| `head_to_head` | un outil plus simple (importance, betweenness) sort-il les mêmes cibles ? |
| `readout_specificity` | métriques de readout affranchies du degré |
| `decoy` | nulles de voisinage (rewire à degré préservé) et d'axe |
| `ora_de_baseline` | l'enrichissement dépasse-t-il un simple tri par DE ? |
| `signed_cascade` | rôle pro/anti par composition de signes multi-hop, sans axe |

Deux nulles à ne pas confondre : `decoy.enabled` teste la **structure**
(le voisinage porte-t-il l'information ?) ; `decoy.random_axis` teste la
**spécificité d'axe** (le résultat survit-il à un axe aléatoire ?). La seconde
est quasi gratuite si `cache_delta_z` est actif.

## Trois pièges d'interprétation

### Le degré

La corrélation entre `driver_score` et le degré total est forte (ρ ≈ +0.66), et
**le biais de degré change de signe selon la densité du graphe**. Un hub bien
classé n'est pas une découverte tant que le contrôle intra-degré n'a pas été
fait — c'est ce que produit `driver_baselines`.

### Le régime de graphe

Le haut du classement dépend des sources activées, et il en dépend beaucoup :

| Régime | Ce qui remonte |
|---|---|
| pur (bases curées seules) | hubs et machinerie cellulaire |
| riche (+ co-expression) | programme chromatinien |
| legacy (arêtes dupliquées) | un module métabolique qui **ne survit pas** au nettoyage |

Ce n'est pas un défaut à cacher : c'est la principale chose que les ablations
servent à mesurer. Une cible ne vaut d'être annoncée que si l'on sait **sur quel
régime** elle tient, et le nom du régime doit accompagner le résultat.

### La circularité

L'encodeur ne voit jamais le contraste — c'est l'invariant de conception. Deux
fuites ont malgré tout été identifiées et doivent rester surveillées :

- la **co-expression** est dérivée des données, donc partiellement porteuse du
  contraste. Le module chromatinien y est sensible ;
- le mode **supervisé joint** laisse remonter le gradient de la tête dans
  l'encodeur : les étiquettes DE façonnent alors le latent. Seul le mode
  `--supervised-detach` (sonde sur μ gelé) est anti-circulaire.

L'ORA est structurellement incapable de corroborer certains modules : les
histones du module chromatinien n'appartiennent à aucun ensemble Reactome ou
KEGG pertinent. Une absence d'enrichissement n'y est pas une réfutation.

## Ce que l'outil ne fait pas

- il ne démontre pas une causalité — il produit une **hypothèse ordonnée**, dont
  la certification est expérimentale ou, au minimum, externe (autre jeu de
  données) ;
- il ne remplace pas une analyse différentielle : il répond à une autre
  question, et son intérêt se mesure justement à **l'écart** avec le DE ;
- il ne choisit pas votre régime de graphe à votre place.
