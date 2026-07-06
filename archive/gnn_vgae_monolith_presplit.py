"""
GNN VGAE — Priorisation non supervisée de gènes dans la sénescence (HUVEC)
==========================================================================
Approche NON SUPERVISÉE par Variational Graph AutoEncoder (VGAE) sur un
graphe hétérogène. Le VGAE apprend des embeddings de gènes en reconstruisant
la topologie du graphe — sans aucun label DEG.

PROBLÈME RÉSOLU : le pipeline supervisé précédent (gnn.py) souffrait de
circularité (features = log2FC/padj → labels = DEG basés sur ces mêmes stats).
Ici, le score d'importance ÉMERGE de l'espace latent, pas d'une formule.

MODE V-sup (--supervised) : plafond CIRCULAIRE assumé. Réutilise le MÊME graphe
et le MÊME backbone HeteroEncoder, mais entraîne l'encodeur end-to-end sur les
labels DEG multi-label (+ features DE en nœud) et calcule l'importance par
cluster. Opposé au VGAE non supervisé (mesure ce que la topologie seule capte).
Cf. _supervised.py + build_supervised_labels.py + docs/technical/gnn_supervised.md.

DONNÉES : GSE102090 (HUVEC), n=1 par condition (P4, P16).

ARCHITECTURE :
  1. Graphe hétérogène :
     - Noeuds "gene"       : features topologiques (is_tf, variance, ppi_degree)
                              PAS de log2FC/padj (supprime la circularité)
     - Noeuds "cell_group" : 5 groupes (P4, P16_cluster_0..3)
     - Arêtes "expresses"  : mean_expr, pct, std, cv, q25, q75, tf_activity
     - Arêtes "ppi"        : STRING (combined_score, unsigned)
     - Arêtes "regulates"  : pySCENIC TF→cible (weight)
     - Arêtes "same_pathway" : REACTOME
     - Arêtes "coexpression" : GRNBoost2 P16 (V4.1) OU différentiel
                               P4∪P16 (V4.2, --coexpr-mode differential,
                               edge_dim=6 option A)
     - Arêtes "signaling"/"tf_curated" : OmniPath/SIGNOR/CollecTRI signé (V4)
     - Arêtes "reactome_fi"  : Reactome Functional Interactions signé
                               (V4.2, --use-reactome-fi)

  2. VGAE : encoder HeteroGNN (GATConv) → μ, σ → z ~ N(μ,σ²) → decoder (inner product)
     Loss = reconstruction des arêtes + KL divergence
     V4.2 : pondération γ_t par edge_type au niveau message
            (--edge-type-weights, _ScaledConv ; cf. §14bis.6octies rapport)

  CHANGELOG en-tête :
   - V4   : OmniPath signed (signaling + tf_curated), edge_dim=2
   - V4.1 : --include-omnipath-genes (endpoints OmniPath dans gene set)
   - V4.1.1 : is_tf = pySCENIC ∪ CollecTRI (section 5)
   - V4.2 : coexpr différentielle P4∪P16 (option A, edge_dim 1→6),
            γ_t par edge_type (_ScaledConv message-level),
            Reactome FI signé (edge_type 'reactome_fi')
   - V6   : généralisation bulk/scRNA via env (GNN_EXPR_MATRIX/GNN_GROUP_META/
            GNN_CELL_GROUPS/GNN_HUMESS_CONDITIONS, cf. docs/technical/
            gnn_vgae_paths.md) + cache du build §1-7 (--reuse-graph) :
            recharge le graphe si la signature de config est identique
            (sources/matrice+mtime/conditions/features/flags → invalide si
            le nb de gènes change), sinon rebuild. logFC jamais en feature.
            metadata gatée GROUP_META (bulk = samplesheet d'échantillons ;
            scRNA = merged_P4_P16_metadata.csv). Étapes optionnelles :
            --no-baselines (saute MLP §12 + DeepWalk §13bis ; garde Stat §13)
            et --no-validation (saute BDD aging §14) — hors-signature cache.

  3. Score d'importance émergent :
     - Centralité dans l'espace latent (norme de μ)
     - Reconstruction error par gène
     - Distance aux clusters fonctionnels

  4. Comparaison avec :
     - Baseline MLP (mêmes features, pas de graphe)
     - Baseline statistique (ranking par log2FC)

  5. Validation post-hoc : GenAge, CellAge, MSigDB, AgeAnno (PAS d'entraînement)
"""

import os
import re
import json
import pickle
import argparse
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric.utils import negative_sampling
from sklearn.decomposition import PCA as PCA_sk
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score, silhouette_score
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# OmniPath integration (V4) — chargé lazily : si les flags --use-omnipath-*
# sont OFF (défaut), on n'importe rien. Sinon on importe le module local.
# Cela évite de planter un run baseline si `omnipath` n'est pas installé.
_OPI = None  # peuplé dans la section 6g si MODULES["use_omnipath_*"]

# =============================================================================
# ARGUMENTS CLI — MODULARITÉ DU GRAPHE & DES FEATURES
# =============================================================================
# Permet d'activer/désactiver chaque source de données et chaque feature pour
# les études d'ablation (cf. §11 du rapport : « qu'apporte vraiment HuMess ?
# la coexpression ? »). Les drapeaux par défaut reproduisent la config V3.3
# (toutes sources actives). Un manifest run_config.json est exporté dans le
# run_dir pour rendre chaque ablation auditable.
#
# Exemples :
#   python gnn_vgae.py                          # baseline V3.3 complet
#   python gnn_vgae.py --no-humess --run-tag no_humess
#   python gnn_vgae.py --no-coexpr --no-humess  # ablation double
#   python gnn_vgae.py --exclude-features ppi_degree,reg_degree
#   python gnn_vgae.py --ppi-score-thresh 700 --run-tag ppi700
# Si --run-tag vaut "auto" (défaut), il est dérivé des modules désactivés.

def _parse_cli_args():
    p = argparse.ArgumentParser(add_help=True,
                                description="VGAE HUVEC — modulaire (sources & features).")
    # --- Sources de données (edge types) ---
    p.add_argument("--use-ppi", dest="use_ppi", action="store_true", default=True)
    p.add_argument("--no-ppi", dest="use_ppi", action="store_false")
    p.add_argument("--use-reactome", dest="use_reactome", action="store_true", default=True)
    p.add_argument("--no-reactome", dest="use_reactome", action="store_false")
    p.add_argument("--use-coexpr", dest="use_coexpr", action="store_true", default=True)
    p.add_argument("--no-coexpr", dest="use_coexpr", action="store_false")
    p.add_argument("--use-scenic-regulons", dest="use_scenic_regulons",
                   action="store_true", default=True)
    p.add_argument("--no-scenic-regulons", dest="use_scenic_regulons",
                   action="store_false")
    p.add_argument("--use-humess-edges", dest="use_humess_edges",
                   action="store_true", default=True,
                   help="arêtes metabolic_cocatalysis (GPR rules)")
    p.add_argument("--no-humess-edges", dest="use_humess_edges", action="store_false")
    p.add_argument("--use-humess-features", dest="use_humess_features",
                   action="store_true", default=True,
                   help="features de noeud imp_P4_z, imp_P16_z, imp_delta, has_humess")
    p.add_argument("--no-humess-features", dest="use_humess_features",
                   action="store_false")
    p.add_argument("--use-cell-group-edges", dest="use_cell_group_edges",
                   action="store_true", default=True,
                   help="arêtes bipartites cell_group ↔ gene (expresses/expressed_in)")
    p.add_argument("--no-cell-group-edges", dest="use_cell_group_edges",
                   action="store_false")
    # Convenience flag : --no-humess désactive arêtes ET features HuMess.
    p.add_argument("--no-humess", action="store_true", default=False,
                   help="raccourci : désactive --use-humess-edges ET --use-humess-features")

    # --- OmniPath (V4 — opt-in, OFF par défaut) ---
    # Sources pré-téléchargées via scripts/cache_omnipath.py — les compute
    # nodes Nautilus n'ayant pas Internet, on lit toujours depuis le cache TSV.
    p.add_argument("--use-omnipath-signaling", dest="use_omnipath_signaling",
                   action="store_true", default=False,
                   help="V4 : ajoute arêtes 'signaling' dirigées signées "
                        "(kinase-substrat OmniPath + SIGNOR causal). "
                        "edge_attr=[score, sign∈{−1,0,+1}].")
    p.add_argument("--no-omnipath-signaling", dest="use_omnipath_signaling",
                   action="store_false")
    p.add_argument("--use-omnipath-tf-curated", dest="use_omnipath_tf_curated",
                   action="store_true", default=False,
                   help="V4 : ajoute edge_type 'tf_curated' (CollecTRI, "
                        "fallback DoRothEA) à côté de pySCENIC 'regulates'. "
                        "~1186 TFs vs ~50 SCENIC, signe biologique préservé.")
    p.add_argument("--no-omnipath-tf-curated", dest="use_omnipath_tf_curated",
                   action="store_false")
    p.add_argument("--omnipath-cache-dir", default=None,
                   help="dossier des TSV OmniPath pré-téléchargés "
                        "(défaut : <DATA_DIR>/omnipath). Cf. "
                        "scripts/cache_omnipath.py")
    p.add_argument("--omnipath-download-if-missing", action="store_true",
                   default=False,
                   help="autorise le téléchargement à la volée si le cache "
                        "est absent (à n'utiliser que sur le frontal qui a "
                        "Internet ; OFF sur les compute nodes)")
    # V4.1 : include OmniPath endpoints in the selected gene set.
    p.add_argument("--include-omnipath-genes", dest="include_omnipath_genes",
                   action="store_true", default=False,
                   help="V4.1 : étend la sélection des gènes en section 3 avec "
                        "les endpoints d'OmniPath (CollecTRI + signaling) qui "
                        "sont aussi présents dans le scRNA-seq. Sans ce flag, "
                        "OmniPath ne fait qu'ajouter des arêtes entre gènes "
                        "déjà sélectionnés via PPI/SCENIC/coexpr/REACTOME ; "
                        "résultat : ~700 TFs CollecTRI sont perdus. "
                        "Requiert --use-omnipath-signaling et/ou "
                        "--use-omnipath-tf-curated.")

    # --- V4.2 : coexpression différentielle P4∪P16 (option A) ---
    p.add_argument("--coexpr-mode", choices=["p16_only", "differential"],
                   default="p16_only",
                   help="V4.2 : 'p16_only' = adjacencies.csv GRNBoost2 P16 "
                        "(comportement V4.1, edge_dim=1) ; 'differential' = "
                        "coexpr_diff.tsv (P4∪P16, option A, edge_dim=6 : "
                        "imp_p4, imp_p16, delta, cat_shared/p4/p16). "
                        "Cf. build_diff_coexpr.py + §V4.2 du rapport.")
    p.add_argument("--diff-coexpr-file",
                   default="data/pyscenic/diff_coexpr/coexpr_diff.tsv",
                   help="V4.2 : chemin du TSV différentiel produit par "
                        "build_diff_coexpr.py merge-adjacencies. Si laissé "
                        "à la valeur par défaut ET que --coexpr-method/"
                        "--coexpr-prune sont != défaut, le chemin sera "
                        "auto-résolu vers coexpr_diff.<method>.<prune>.tsv.")
    # --- V4.3 : grille comparaison méthodes GRN × élagage ---
    # Auto-résolution du chemin coexpr_diff selon (method, prune) si
    # --diff-coexpr-file n'a pas été surchargé. method='sklearn' +
    # prune='topk' = comportement V4.2 (fichier historique coexpr_diff.tsv).
    p.add_argument("--coexpr-method",
                   choices=["sklearn", "arboreto", "corr", "mi"],
                   default="sklearn",
                   help="V4.3 : méthode d'inférence GRN amont. sklearn = "
                        "grnboost2-local (défaut, comportement V4.2) ; "
                        "arboreto = grnboost2-diff canonique ; corr = "
                        "Pearson/Spearman ; mi = mutual_info_regression. "
                        "Persisté dans vgae_metrics.json.")
    p.add_argument("--coexpr-prune",
                   choices=["topk", "quantile", "mr", "zscore"],
                   default="topk",
                   help="V4.3 : méthode d'élagage des arêtes dans le "
                        "réseau coexpr. topk = per-target-topk (défaut) ; "
                        "quantile = global-quantile (baseline -) ; mr = "
                        "mutual-rank Obayashi 2018 ; zscore = z-score "
                        "per-target. Persisté dans vgae_metrics.json.")

    # --- V4.2 : Reactome FI (arêtes signées additionnelles) ---
    p.add_argument("--use-reactome-fi", dest="use_reactome_fi",
                   action="store_true", default=False,
                   help="V4.2 : ajoute edge_type 'reactome_fi' (Reactome "
                        "Functional Interactions, signed). ~45k arêtes "
                        "signées NOUVELLES (75% absentes de PPI/SIGNOR/"
                        "CollecTRI). edge_dim=2 [score, sign].")
    p.add_argument("--no-reactome-fi", dest="use_reactome_fi",
                   action="store_false")
    p.add_argument("--reactome-fi-file",
                   default="data/reactome_fi/FIsInGene_with_annotations.txt",
                   help="V4.2 : chemin du fichier Reactome FI décompressé "
                        "(Gene1, Gene2, Annotation, Direction, Score).")

    # --- V4.2 : pondération γ_t par edge_type (niveau message) ---
    # NB : la loss VGAE est poolée (PPI∪REACTOME∪reg∪coexpr dédupliqués),
    # donc on ne peut pas pondérer la *loss* par type proprement. Le
    # déséquilibre mesuré (‖h_PPI‖ ≈ 13× ‖h_signaling‖, §14bis.6bis) est
    # au niveau du MESSAGE-PASSING. γ_t scale donc la sortie de chaque
    # GATConv AVANT l'agrégation HeteroConv-sum. Toggleable pour A/B V4.2.
    p.add_argument("--edge-type-weights", default="",
                   help="V4.2 : pondération γ_t par edge_type, format "
                        "'ppi=0.1,coexpression=0.5,signaling=1.0'. Vide = "
                        "tous les γ_t=1.0 (comportement V4.1). Les types "
                        "non listés gardent γ=1.0. Appliqué au niveau du "
                        "message (pas de la loss). Cf. §14bis.6bis.")
    p.add_argument("--dedup-ppi-signed",
                   choices=["off", "remove", "annotate"], default="off",
                   help="V4.2 : si une paire (a,b) a une arête SIGNÉE "
                        "orientée (signaling/tf_curated/reactome_fi), "
                        "l'arête PPI non-signée est redondante. off "
                        "(DÉFAUT) = inchangé, diagnostic seul (compte "
                        "arêtes/gènes touchés). remove = supprime le "
                        "doublon PPI. annotate = garde l'arête PPI + "
                        "colonne flag has_signed_counterpart (edge_dim "
                        "1→2). Cf. §14bis.6quaterdecies. Bénéfice plein "
                        "avec V5 (message/decodeur signés).")

    # --- V5 : message-passing signé + décodeur bilinéaire signé (TIER 1c) ---
    # Tous opt-in, défaut OFF → backward-compat V4.x. Cf. §14bis.6septies
    # du rapport, prototypes dans src/gnn/_vgae_model.py:171-277.
    p.add_argument("--signed-message",
                   action="store_true", default=False,
                   help="V5 (TIER 1c.2) : remplace GATConv par SignedGATConv "
                        "pour les edge_types signés (signaling, tf_curated, "
                        "tf_curated_by, reactome_fi). Chaque message est "
                        "multiplié par son `sign` (colonne 1 de edge_attr) "
                        "⇒ une arête `sign=-1` propage `-W·h_j`. Ref : Derr "
                        "2018 ICDM SGCN §3.2. Défaut OFF (legacy).")
    p.add_argument("--signed-decoder",
                   action="store_true", default=False,
                   help="V5 (TIER 1c.3) : ajoute un BilinearSignedDecoder "
                        "(3 canaux W_+ / W_- / W_0) en parallèle du "
                        "décodeur cosinus. Active une loss auxiliaire "
                        "BCE(logit_bilin, sign∈{0,1}) sur les arêtes "
                        "positives des edge_types signés. La loss "
                        "principale (cosinus + KL) reste inchangée. Ref : "
                        "Liu 2024 NAR SGAT-bilinear, Yang 2015 ICLR "
                        "DistMult. Défaut OFF (legacy).")
    p.add_argument("--signed-loss-weight", type=float, default=1.0,
                   help="V5 (TIER 1c.4) : λ_signed multiplicateur de la "
                        "loss auxiliaire signée (ignoré si "
                        "--signed-decoder OFF). 1.0 = équivalent à la "
                        "recon_loss principale ; 0.5 = demi-poids. Défaut "
                        "1.0.")
    p.add_argument("--signed-decoder-dim", type=int, default=None,
                   help="V5.3 (TIER 1c.7) : dimension du sous-espace signed "
                        "(tête `signed_proj : R^latent → R^signed_dim` avant "
                        "le décodeur bilinéaire). Défaut None = LATENT_DIM "
                        "(pas de compression, signed_proj init=identité ⇒ "
                        "équivalent V5.2 au load checkpoint). Valeurs <"
                        "LATENT_DIM forcent la compression du sous-espace "
                        "signed (spécialisation). Réduit la concurrence "
                        "avec le décodeur cosinus (V5.2 : −0.022 AUC recon). "
                        "Cf. §14bis.6unvicesies.")

    # --- V5 phase 2 : VRAI hold-out signed pour gate 1c.5 rigoureux ---
    # Sans ces flags, la signed_aux_loss voit TOUTES les arêtes signées
    # à l'entraînement → gate 1c.5 reste in-sample. Avec --holdout-signed-tf-
    # fraction X > 0, on tire X% des « régulateurs » (= union des sym source
    # des edge_types signés) et on MASQUE leurs arêtes signées de la loss.
    # L'encodeur continue de voir ces arêtes via le message-passing
    # (SignedGATConv propage), mais leur signe n'est PAS appris.
    # test_signed_auc.py lit la liste persistée dans run_config.json et
    # évalue uniquement sur les arêtes hold-out → vrai test de
    # généralisation à des TFs jamais utilisés pour la loss.
    p.add_argument("--holdout-signed-tf-fraction", type=float, default=0.0,
                   help="V5 phase 2 : fraction des régulateurs (= union des "
                        "sym source des edge_types signés) à réserver au test "
                        "1c.5 rigoureux. Leurs arêtes signées sont retirées "
                        "de la signed_aux_loss (mais conservées dans le "
                        "graphe → encoder les voit toujours). 0.0 = pas de "
                        "hold-out (défaut, comportement V5.1). Valeur typique "
                        "défense : 0.2 (Liu 2024 NAR SGAT-bilinear §3).")
    p.add_argument("--holdout-signed-tf-seed", type=int, default=None,
                   help="V5 phase 2 : seed RNG pour le split TF hold-out. "
                        "Défaut : prend --seed (reproductibilité couplée au "
                        "training). Passer une valeur fixée (ex. 42) pour "
                        "comparer des configs A/B sur le MÊME set hold-out.")

    # --- Exclusion fine de features de noeud gene ---
    # Liste possible : is_tf, variance, ppi_degree, reg_degree,
    #                  imp_P4, imp_P16, imp_delta, has_humess
    p.add_argument("--exclude-features", default="",
                   help="liste séparée par des virgules de gene_features à exclure "
                        "(parmi : is_tf, variance, ppi_degree, reg_degree, "
                        "imp_P4, imp_P16, imp_delta, has_humess)")

    # --- Hyperparamètres de filtrage du graphe ---
    p.add_argument("--ppi-score-thresh", type=int, default=900,
                   help="seuil STRING combined_score (0-1000), defaut=900")
    p.add_argument("--coexpr-top-quantile", type=float, default=0.98,
                   help="quantile minimal pour garder une coexpression GRNBoost2 "
                        "(0.98 = top 2%%)")
    p.add_argument("--reactome-max-pathway", type=int, default=20,
                   help="taille max d'un pathway REACTOME pour rester informatif")

    # --- Étiquetage du run ---
    p.add_argument("--run-tag", default="auto",
                   help="suffixe ajouté à OUT_DIR pour ce run. 'auto' = construit "
                        "à partir des modules désactivés ; 'full' = baseline ; "
                        "ou nom libre.")
    p.add_argument("--seed", type=int, default=42,
                   help="graine aléatoire — appliquée à numpy + torch + cuda. "
                        "Inclus dans RUN_TAG (sauf si --run-tag <libre>) pour "
                        "ne pas écraser un run d'un autre seed.")

    p.add_argument("--n-epochs", dest="n_epochs", type=int, default=1200,
                   help="Nombre maximal d'epochs (early stopping arrête souvent "
                        "avant). Défaut 1000.")
    p.add_argument("--patience", type=int, default=150,
                   help="Patience de l'early stopping (epochs sans amélioration "
                        "AUC val avant arrêt). Défaut 100.")
    # --- Cache du graphe (itération rapide : saute la reconstruction §1-7) ---
    p.add_argument("--reuse-graph", dest="reuse_graph", action="store_true",
                   help="Réutilise le cache de build (§1-7) s'il est VALIDE "
                        "(même signature de config/sources/gènes). Sinon rebuild.")
    p.add_argument("--graph-cache", dest="graph_cache", default=None,
                   help="Chemin du cache de build (pickle). Défaut : "
                        "<OUT_DIR_BASE>/_graph_cache.pkl.")
    # --- Étapes optionnelles (généralisation : accélère les sweeps / portabilité) ---
    p.add_argument("--no-baselines", dest="no_baselines", action="store_true",
                   help="Saute les baselines ENTRAÎNÉES : MLP (§12) et DeepWalk/"
                        "Node2Vec (§13bis, lent). La baseline statistique |ΔExpr| "
                        "(§13, triviale) reste calculée. Sorties baseline absentes "
                        "du ranking (mlp_score/node2vec_score).")
    p.add_argument("--no-validation", dest="no_validation", action="store_true",
                   help="Saute la validation post-hoc sur BDD aging externes "
                        "(§14 : GenAge/CellAge/MSigDB/AgeAnno). Utile sans les "
                        "fichiers BDD ou sur un autre phénotype que la sénescence "
                        "(colonnes in_* / n_databases mises à 0).")
    # --- V5.4 (decoder-split, §14bis.6duovicies) ---
    p.add_argument("--decoder-split", dest="decoder_split", action="store_true",
                   help="V5.4 : route les arêtes SIGNÉES vers le décodeur "
                        "bilinéaire (existence via logsumexp) et libère le "
                        "cosinus des arêtes dirigées (recon cosinus = pool V5.1). "
                        "Nécessite --signed-decoder. Défaut OFF (backward-compat V5.3).")
    # --- V4.3-tune : hyperparamètres surchargeables (étaient hardcodés) ---
    p.add_argument("--kl-beta-max", dest="kl_beta_max", type=float, default=0.0005,
                   help="V4.3-tune : β final du KL annealing (0→kl_beta_max). "
                        "Défaut 0.0005. kl1 = 0.0001.")
    p.add_argument("--latent-dim", dest="latent_dim", type=int, default=64,
                   help="V4.3-tune : dimension de l'espace latent. Défaut 64.")
    # --- V-sup : VGAE circulaire (reconstruit) + features DE (+ tête option) ---
    # ARCHI : le VGAE reconstruit NORMALEMENT (recon + KL + décodeur → prêt pour
    # la perturbation Δμ standard) ; --de-features AJOUTE les stats DE aux nœuds
    # (circulaire) ; --supervised attache EN PLUS une tête de classification
    # entraînée CONJOINTEMENT (multi-tâche). Les deux activables indépendamment.
    p.add_argument("--de-features", dest="de_features", action="store_true",
                   default=False,
                   help="V-sup : injecte les features DE de nœud (log2FC global/par "
                        "cluster, -log10 padj, Delta-pct) — CIRCULAIRES. Le VGAE "
                        "reconstruit normalement AVEC ces features. Défaut OFF "
                        "(anti-circularité). Standalone = 'VGAE circulaire'.")
    p.add_argument("--no-de-features", dest="de_features", action="store_false",
                   help="Désactive explicitement les features DE (ablation).")
    p.add_argument("--supervised", dest="supervised", action="store_true",
                   help="V-sup : attache une TÊTE de classification multi-label "
                        "(P4_vs_P16 + cluster_0..3) entraînée CONJOINTEMENT à la "
                        "reconstruction (loss = recon + β·KL + λ·classif). Importance "
                        "par cluster (saliency). Typiquement avec --de-features.")
    p.add_argument("--supervised-loss-weight", dest="supervised_loss_weight",
                   type=float, default=1.0,
                   help="V-sup : λ du terme de classification dans la loss jointe. "
                        "Défaut 1.0.")
    p.add_argument("--supervised-recompute-labels",
                   dest="supervised_recompute_labels", action="store_true",
                   help="V-sup : force le recalcul de la DE par cluster (scanpy "
                        "Wilcoxon) au lieu de lire le cache DEGs_P16_cluster_*.csv.")

    args, _unknown = p.parse_known_args()

    # Application du raccourci --no-humess
    if args.no_humess:
        args.use_humess_edges = False
        args.use_humess_features = False

    # V4.3 : auto-résolution du chemin coexpr_diff si --diff-coexpr-file
    # n'a pas été surchargé (= reste sur la valeur par défaut historique).
    # method='sklearn' + prune='topk' → fichier V4.2 historique
    # `coexpr_diff.tsv` conservé (pas de cassure ascendante).
    _default_diff = "data/pyscenic/diff_coexpr/coexpr_diff.tsv"
    if args.diff_coexpr_file == _default_diff and (
            args.coexpr_method != "sklearn" or args.coexpr_prune != "topk"):
        args.diff_coexpr_file = (
            f"data/pyscenic/diff_coexpr/coexpr_diff."
            f"{args.coexpr_method}.{args.coexpr_prune}.tsv"
        )

    return args


CLI_ARGS = _parse_cli_args()

# Set des features à exclure (normalisées en minuscules, dépouillées d'espaces)
_EXCLUDED_FEATURES = {
    s.strip().lower() for s in CLI_ARGS.exclude_features.split(",") if s.strip()
}

# MODULES : dictionnaire compact qui résume la config — référé partout dans
# le pipeline pour gater les blocs (loading + edges + features).
MODULES = {
    "use_ppi":                   CLI_ARGS.use_ppi,
    "use_reactome":              CLI_ARGS.use_reactome,
    "use_coexpr":                CLI_ARGS.use_coexpr,
    "use_scenic_regulons":       CLI_ARGS.use_scenic_regulons,
    "use_humess_edges":          CLI_ARGS.use_humess_edges,
    "use_humess_features":       CLI_ARGS.use_humess_features,
    "use_cell_group_edges":      CLI_ARGS.use_cell_group_edges,
    "use_omnipath_signaling":    CLI_ARGS.use_omnipath_signaling,
    "use_omnipath_tf_curated":   CLI_ARGS.use_omnipath_tf_curated,
    "include_omnipath_genes":    CLI_ARGS.include_omnipath_genes,
    "use_reactome_fi":           CLI_ARGS.use_reactome_fi,
}

# V4.2 : mode coexpression (p16_only = V4.1 ; differential = option A).
COEXPR_MODE = CLI_ARGS.coexpr_mode
COEXPR_DIFFERENTIAL = (COEXPR_MODE == "differential")

# V4.2 : parsing de la pondération γ_t par edge_type (niveau message).
# Format "ppi=0.1,coexpression=0.5". Vide → {} → tous γ=1.0 (V4.1).
EDGE_TYPE_WEIGHTS: dict[str, float] = {}
if CLI_ARGS.edge_type_weights.strip():
    for _kv in CLI_ARGS.edge_type_weights.split(","):
        _k, _, _v = _kv.partition("=")
        _k, _v = _k.strip(), _v.strip()
        if _k and _v:
            try:
                EDGE_TYPE_WEIGHTS[_k] = float(_v)
            except ValueError:
                print(f"  [warn] --edge-type-weights : '{_kv}' ignoré "
                      f"(valeur non numérique)")
if EDGE_TYPE_WEIGHTS:
    print(f"  γ_t edge-type weights (message-level) : {EDGE_TYPE_WEIGHTS}")

# Matrice « feature → activée ? » — recoupe les modules et l'option
# --exclude-features. Une feature est INCLUSE ssi (a) sa source est active
# ET (b) elle n'est pas listée dans --exclude-features.
def _feature_enabled(name, source_ok=True):
    return source_ok and (name not in _EXCLUDED_FEATURES)


GENE_FEATURE_FLAGS = {
    "is_tf":       _feature_enabled("is_tf",      MODULES["use_scenic_regulons"]),
    "variance":    _feature_enabled("variance",   True),  # toujours dispo (group_stats)
    "ppi_degree":  _feature_enabled("ppi_degree", MODULES["use_ppi"]),
    "reg_degree":  _feature_enabled("reg_degree", MODULES["use_scenic_regulons"]),
    "imp_P4":      _feature_enabled("imp_p4",     MODULES["use_humess_features"]),
    "imp_P16":     _feature_enabled("imp_p16",    MODULES["use_humess_features"]),
    "imp_delta":   _feature_enabled("imp_delta",  MODULES["use_humess_features"]),
    "has_humess":  _feature_enabled("has_humess", MODULES["use_humess_features"]),
}


def _build_run_tag():
    """Construit un tag court à partir des modules désactivés.
    'full' si tout est actif. Ex : 'no-ppi.no-humess.s7' si ces deux sont
    off et seed=7. Si --run-tag est explicite, on respecte le tag libre tel
    quel (l'utilisateur est responsable de l'unicité par seed)."""
    tag = CLI_ARGS.run_tag
    if tag != "auto":
        return tag if tag else "full"
    parts = []
    if not MODULES["use_ppi"]:              parts.append("no-ppi")
    if not MODULES["use_reactome"]:         parts.append("no-reactome")
    if not MODULES["use_coexpr"]:           parts.append("no-coexpr")
    if not MODULES["use_scenic_regulons"]:  parts.append("no-scenic")
    if not MODULES["use_humess_edges"] and not MODULES["use_humess_features"]:
        parts.append("no-humess")
    else:
        if not MODULES["use_humess_edges"]:    parts.append("no-humess-edges")
        if not MODULES["use_humess_features"]: parts.append("no-humess-feats")
    if not MODULES["use_cell_group_edges"]: parts.append("no-cgrp")
    if MODULES["use_omnipath_signaling"]:   parts.append("op-sig")
    if MODULES["use_omnipath_tf_curated"]:  parts.append("op-tf")
    if MODULES["include_omnipath_genes"]:   parts.append("op-genes")
    if MODULES["use_reactome_fi"]:          parts.append("rfi")
    if COEXPR_DIFFERENTIAL:                 parts.append("coexdiff")
    # V4.3 : tag des choix méthode×prune (uniquement si != défaut).
    if CLI_ARGS.coexpr_method != "sklearn":
        parts.append(f"grn-{CLI_ARGS.coexpr_method}")
    if CLI_ARGS.coexpr_prune != "topk":
        parts.append(f"prune-{CLI_ARGS.coexpr_prune}")
    if EDGE_TYPE_WEIGHTS:                   parts.append("gw")
    if getattr(CLI_ARGS, "dedup_ppi_signed", "off") != "off":
        parts.append(f"dedup-{CLI_ARGS.dedup_ppi_signed}")
    if _EXCLUDED_FEATURES:                  parts.append("ex-" + "-".join(sorted(_EXCLUDED_FEATURES)))
    if CLI_ARGS.ppi_score_thresh != 900:    parts.append(f"ppi{CLI_ARGS.ppi_score_thresh}")
    if abs(CLI_ARGS.coexpr_top_quantile - 0.98) > 1e-9:
        parts.append(f"q{int(CLI_ARGS.coexpr_top_quantile*1000):04d}")
    if CLI_ARGS.reactome_max_pathway != 20:
        parts.append(f"rxmax{CLI_ARGS.reactome_max_pathway}")
    base = "full" if not parts else ".".join(parts)
    # Toujours suffixer par le seed pour différencier les runs multi-seed.
    return f"{base}.s{CLI_ARGS.seed}"


RUN_TAG = _build_run_tag()

# Application immédiate du seed à tous les RNG. Important de le faire AVANT
# tout import-side-effect ou allocation pour assurer la reproductibilité.
np.random.seed(CLI_ARGS.seed)
torch.manual_seed(CLI_ARGS.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CLI_ARGS.seed)

print("=" * 70)
print("Configuration modulaire :")
print("=" * 70)
for k, v in MODULES.items():
    flag = "ON " if v else "OFF"
    print(f"  [{flag}] {k}")
print(f"  Features actives  : "
      f"{[k for k, v in GENE_FEATURE_FLAGS.items() if v]}")
if _EXCLUDED_FEATURES:
    print(f"  Features exclues  : {sorted(_EXCLUDED_FEATURES)}")
print(f"  PPI threshold     : {CLI_ARGS.ppi_score_thresh}")
print(f"  Coexpr quantile   : {CLI_ARGS.coexpr_top_quantile}")
print(f"  REACTOME max size : {CLI_ARGS.reactome_max_pathway}")
print(f"  RUN_TAG           : {RUN_TAG}")
print()


# =============================================================================
# CONFIGURATION
# =============================================================================
# Cette section définit tous les chemins d'accès aux données et les
# hyperparamètres du pipeline. Les chemins pointent vers le cluster Nautilus
# (GLiCID) ; des chemins locaux commentés sont disponibles pour le debug.

# --- Chemins principaux sur le cluster Nautilus (GLiCID) ---
# LAB_DIR : racine de l'espace utilisateur sur le stockage partagé GLiCID.
# BASE_DIR : sous-dossier contenant le projet GNN (code + données).
# DATA_DIR : données d'entrée (scRNA-seq, PPI, pySCENIC, bases de données).
# SCENIC_DIR : sorties de pySCENIC (regulons, adjacencies, TF activity).
# OUT_DIR : dossier de sortie sur /scratch (écriture rapide, non sauvegardé).
# Chemins surchargeables par variable d'environnement (clone portable /
# orchestration Snakemake). Les défauts reproduisent le comportement
# historique sur GLiCID — exporter GNN_LAB_DIR / GNN_DATA_DIR /
# GNN_OUT_DIR_BASE (cf. workflow/Snakefile) pour pointer ailleurs.
# _REPO_ROOT : racine du dépôt (src/gnn/gnn_vgae.py → 3 niveaux au-dessus).
# Sert de défaut PORTABLE (local ET clone cluster), tout reste surchargeable env.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAB_DIR = os.environ.get(
    "GNN_LAB_DIR", "/LAB-DATA/GLiCID/users/USER@univ-nantes.fr/")
# BASE_DIR : défaut = racine du repo (data co-localisée avec le code). Sur GLiCID,
# le clone est sous LAB-DATA → _REPO_ROOT pointe déjà au bon endroit.
BASE_DIR = os.environ.get("GNN_BASE_DIR", _REPO_ROOT)
DATA_DIR = os.environ.get("GNN_DATA_DIR", os.path.join(BASE_DIR, "data"))
SCENIC_DIR = os.environ.get(
    "GNN_SCENIC_DIR", os.path.join(BASE_DIR, "output", "pyscenic"))
# OUT_DIR_BASE : racine de sortie. Le run_dir final = OUT_DIR_BASE / RUN_TAG.
# Défaut PORTABLE = <repo>/output/gnn_vgae (marche en local sans config).
# Sur GLiCID (bonne pratique) : exporter
#   GNN_OUT_DIR_BASE=/scratch/nautilus/users/<user>/gnn_vgae
# (écriture rapide sur scratch ; transférer ensuite les runs FIGÉS vers
#  /LAB-DATA/.../ pour archivage). cf. docs/technical/gnn_vgae_paths.md.
OUT_DIR_BASE = os.environ.get(
    "GNN_OUT_DIR_BASE", os.path.join(_REPO_ROOT, "output", "gnn_vgae"))
OUT_DIR = os.path.join(OUT_DIR_BASE, RUN_TAG)

# --- Chemins locaux (décommentez pour debug sur votre machine) ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(BASE_DIR, "..", "data")
# SCENIC_DIR = os.path.join(BASE_DIR, "..", "output", "pyscenic")
# OUT_DIR = os.path.join(BASE_DIR, "..", "output", "gnn_vgae")

# --- Sous-dossiers communs ---
# GNN_DATA_DIR : matrices normalisées + metadata (sortie du preprocessing Seurat).
# PPI_DIR : cache local de STRING (protein-protein interactions).
# DB_DIR : bases de données de validation (GenAge, CellAge, MSigDB, etc.).
# FIG_DIR : figures générées par le pipeline.
GNN_DATA_DIR = os.path.join(DATA_DIR, "gnn_data")
PPI_DIR = os.path.join(DATA_DIR, "PPI")
DB_DIR = os.path.join(DATA_DIR, "databases")
FIG_DIR = os.path.join(OUT_DIR, "figure")
# OMNIPATH_CACHE_DIR : TSV pré-téléchargés via scripts/cache_omnipath.py.
# Si non fourni en CLI, on dérive de DATA_DIR pour rester cohérent avec
# la convention de chemins (cache co-localisé avec les autres données).
OMNIPATH_CACHE_DIR = (CLI_ARGS.omnipath_cache_dir
                      if CLI_ARGS.omnipath_cache_dir is not None
                      else os.path.join(DATA_DIR, "omnipath"))

# --- HuMess (modélisation métabolique) ---
# HuMess utilise CarveMe (reconstruction de modèles métaboliques spécifiques
# au contexte) + Corner Sampling (échantillonnage de l'espace des flux) pour
# estimer l'importance métabolique de chaque gène dans les conditions P4 et P16.
# HUMESS_DIR pointe vers les sorties HuMess pour les HUVEC.
HUMESS_DIR = os.environ.get(
    "GNN_HUMESS_DIR", os.path.join(LAB_DIR, "humess", "output_huvec"))
# Local fallback si la structure diffère :
# HUMESS_DIR = "/home/USER/M2/S2/Stage/Projet_Colin/humess/output_huvec"
# Les deux conditions HuMess à comparer. Configurable (généralisation V6 : un
# dataset bulk a ses propres conditions, ex. "pro,sen") via env GNN_HUMESS_CONDITIONS.
# Défaut HUVEC = P4,P16 (rétro-compat). Doivent matcher HUMESS_DIR/models/<cond>/.
HUMESS_CONDITIONS = [c.strip() for c in
                     os.environ.get("GNN_HUMESS_CONDITIONS", "P4,P16").split(",")
                     if c.strip()]

# Matrice d'expression (cellules/échantillons × gènes) + assignation de groupe.
# Généralisation V6 : un dataset bulk fournit sa matrice + un meta sample→group.
#   GNN_EXPR_MATRIX : nom du fichier dans GNN_DATA_DIR (défaut HUVEC scRNA).
#   GNN_GROUP_META  : CSV optionnel (colonnes sample,group). Vide = logique
#                     HUVEC (passage/cluster_P16). Si fourni → groupe = colonne
#                     'group' directement (chemin bulk ; cf. section 4).
EXPR_MATRIX = os.environ.get("GNN_EXPR_MATRIX", "merged_P4_P16_normalized.csv")
GROUP_META = os.environ.get("GNN_GROUP_META", "")

# Création des dossiers de sortie s'ils n'existent pas
for d in [PPI_DIR, DB_DIR, OUT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# Manifest des modules activés/désactivés pour ce run — sert d'audit pour les
# études d'ablation (cf. perturb_report.py / cross_seed_report).
_MANIFEST_PATH = os.path.join(OUT_DIR, "run_config.json")
with open(_MANIFEST_PATH, "w") as _fh:
    json.dump({
        "run_tag": RUN_TAG,
        "seed": CLI_ARGS.seed,
        "modules": MODULES,
        "gene_feature_flags": GENE_FEATURE_FLAGS,
        "excluded_features": sorted(_EXCLUDED_FEATURES),
        "ppi_score_thresh": CLI_ARGS.ppi_score_thresh,
        "coexpr_top_quantile": CLI_ARGS.coexpr_top_quantile,
        "reactome_max_pathway": CLI_ARGS.reactome_max_pathway,
        "omnipath_cache_dir": OMNIPATH_CACHE_DIR,
        "omnipath_download_if_missing": CLI_ARGS.omnipath_download_if_missing,
        # V4.2
        "coexpr_mode": COEXPR_MODE,
        "diff_coexpr_file": CLI_ARGS.diff_coexpr_file,
        "use_reactome_fi": MODULES["use_reactome_fi"],
        "reactome_fi_file": CLI_ARGS.reactome_fi_file,
        "edge_type_weights": EDGE_TYPE_WEIGHTS,
        "dedup_ppi_signed": getattr(CLI_ARGS, "dedup_ppi_signed", "off"),
        # V4.3
        "coexpr_method": CLI_ARGS.coexpr_method,
        "coexpr_prune": CLI_ARGS.coexpr_prune,
        # V5 (TIER 1c) — wiring signed message + bilinear decoder
        "signed_message": CLI_ARGS.signed_message,
        "signed_decoder": CLI_ARGS.signed_decoder,
        "signed_loss_weight": CLI_ARGS.signed_loss_weight,
        # V5.4 (decoder-split) + V4.3-tune (kl_beta_max, latent_dim)
        "decoder_split": bool(getattr(CLI_ARGS, "decoder_split", False)),
        "kl_beta_max": CLI_ARGS.kl_beta_max,
        "latent_dim": CLI_ARGS.latent_dim,
        # V5.3 (TIER 1c.7) — tête sub-espace signed_proj
        "signed_decoder_dim": CLI_ARGS.signed_decoder_dim,
        # V5 phase 2 (TIER 1c.5 strict) — hold-out signed TF pour gate rigoureux
        # Les clés `holdout_signed_tf_set` et `holdout_signed_tf_seed_used`
        # sont enrichies plus tard (cf. section 10) une fois le pool construit.
        "holdout_signed_tf_fraction": CLI_ARGS.holdout_signed_tf_fraction,
        "holdout_signed_tf_seed": CLI_ARGS.holdout_signed_tf_seed,
    }, _fh, indent=2)
print(f"  Manifest écrit    : {_MANIFEST_PATH}")

# ── Paramètres du graphe ─────────────────────────────────────────────────────
# PPI_SCORE_THRESH : seuil de confiance STRING (0-1000). 900 = "highest
#   confidence". On ne garde que les interactions protéine-protéine très
#   fiables pour éviter le bruit dans le graphe.
PPI_SCORE_THRESH = CLI_ARGS.ppi_score_thresh
# REACTOME_MAX_PATHWAY : taille max d'un pathway REACTOME (en gènes).
#   Les très grands pathways (ex : "Metabolism") sont non informatifs —
#   ils connectent trop de gènes entre eux et diluent le signal.
#   Les très petits (< 2 gènes) ne créent pas d'arêtes utiles.
REACTOME_MAX_PATHWAY = CLI_ARGS.reactome_max_pathway
# COEXPR_TOP_QUANTILE : seules les co-expressions GRNBoost2 au-dessus de
#   ce quantile sont conservées. 0.98 = top 2% des poids → réseau épars
#   de haute confiance.
COEXPR_TOP_QUANTILE = CLI_ARGS.coexpr_top_quantile

# ── Paramètres du VGAE ──────────────────────────────────────────────────────
# HIDDEN_DIM : dimension des couches cachées du GNN (après projection des
#   features d'entrée). Chaque couche GATConv produit un vecteur de cette
#   taille pour chaque noeud. Avec N_HEADS=4, chaque tête travaille en
#   dimension HIDDEN_DIM/N_HEADS = 32, puis les résultats sont concaténés.
HIDDEN_DIM = 128
# LATENT_DIM : dimension de l'espace latent (μ et log(σ²) ont cette taille).
#   C'est l'espace dans lequel on calcule le score d'importance. 64 offre
#   un bon compromis entre expressivité et risque de KL collapse.
#   Surchargeable via --latent-dim (défaut 64).
LATENT_DIM = CLI_ARGS.latent_dim
# N_LAYERS : nombre de couches de message passing. Chaque couche permet
#   à un gène de "voir" un voisin de plus. Avec 3 couches, chaque gène
#   agrège l'information de ses voisins jusqu'à distance 3 dans le graphe.
N_LAYERS = 3
# N_HEADS : nombre de têtes d'attention dans GATConv. Le multi-head
#   attention permet au modèle d'apprendre différents types de relations
#   (ex : une tête pour la co-expression, une autre pour la régulation).
N_HEADS = 4
# DROPOUT : régularisation par extinction aléatoire de neurones pendant
#   l'entraînement. 0.2 = 20% des neurones sont désactivés à chaque
#   forward pass, ce qui réduit le surapprentissage.
DROPOUT = 0.2
# N_EPOCHS : nombre maximal d'epochs (itérations complètes sur les données).
#   L'early stopping arrêtera souvent avant (typiquement epoch 30-80).
#   Surchargeable via --n-epochs (défaut 1000).
N_EPOCHS = CLI_ARGS.n_epochs
# LR : learning rate de l'optimiseur Adam. 0.005 est relativement élevé
#   (typique des GNN qui convergent vite) mais compensé par le gradient
#   clipping et le weight decay.
LR = 0.005
# EDGE_SAMPLE_RATIO : fraction d'arêtes réservées pour le test. 10% des
#   arêtes sont masquées et le VGAE doit les prédire. Les 90% restantes
#   servent à l'entraînement.
EDGE_SAMPLE_RATIO = 0.1
# NEG_SAMPLE_RATIO : pour chaque arête positive (vraie connexion), on
#   échantillonne ce ratio d'arêtes négatives (paires non connectées).
#   1.0 = autant de négatifs que de positifs → classes équilibrées.
NEG_SAMPLE_RATIO = 1.0
# N_CLUSTERS : nombre de clusters K-means sur les embeddings finaux.
#   Sert à identifier des groupes fonctionnels de gènes dans l'espace latent.
N_CLUSTERS = 8

# ── Paramètres anti-KL-collapse ──────────────────────────────────────────────
# Le posterior collapse est un problème classique des VGAE : le modèle apprend
# à ignorer l'espace latent en produisant q(z|x) ≈ p(z) = N(0,I) pour tout x,
# ce qui donne des embeddings identiques et une AUC de 0.5.
# Solutions :
#   1. KL annealing : β commence à 0 et monte linéairement jusqu'à KL_BETA_MAX
#      pendant KL_WARMUP_EPOCHS. Le modèle apprend d'abord à bien reconstruire,
#      puis la régularisation KL entre progressivement.
#   2. Free bits : on impose un minimum de KL par dimension latente (FREE_BITS).
#      Si une dimension a KL < FREE_BITS, on ne la pénalise pas. Cela force le
#      modèle à utiliser au moins FREE_BITS nats d'information par dimension.
KL_BETA_MAX = CLI_ARGS.kl_beta_max   # β final ; --kl-beta-max (défaut 0.0005 ; kl1=0.0001)
KL_WARMUP_EPOCHS = 50     # Warmup court puis β stable — le cosinus n'a pas besoin de long warmup
FREE_BITS = 0.5           # Minimum KL par dimension latente (en nats)

# ── Paramètres du baseline MLP ───────────────────────────────────────────────
# Le MLP est une baseline "sans graphe" : il utilise les mêmes features que
# le VGAE mais ne fait PAS de message passing. Si le VGAE bat le MLP en AUC
# de reconstruction, cela prouve que la topologie du graphe apporte de
# l'information au-delà des features brutes.
MLP_HIDDEN = 64    # Dimension cachée du MLP (plus petit que le VGAE)
MLP_EPOCHS = 250   # Moins d'epochs car le MLP converge vite (pas de graphe)
MLP_LR = 0.001     # Learning rate plus faible que le VGAE (MLP est plus stable)

# ── Style des figures ────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 11})


# =============================================================================
# UTILITAIRES
# =============================================================================
def download_if_absent(url, local_path, label=""):
    """
    Ensure `local_path` exists, downloading from `url` on first use.

    Fetches external databases (STRING PPI, MSigDB REACTOME/Hallmarks, GenAge,
    CellAge, AgeAnno) and caches them locally. A custom User-Agent avoids
    rejections from some servers (e.g. Broad Institute).

    Offline-aware: on a node without internet (HPC compute nodes behind a proxy
    return 403) or when GNN_ALLOW_DOWNLOADS=0, a missing file raises a clear,
    actionable error instead of a cryptic urllib traceback — pre-stage the file
    on a node with internet (frontal) or rsync your local data/ cache.
    """
    if os.path.exists(local_path):
        print(f"    [cache] {label or os.path.basename(local_path)}")
        return
    _allow = os.environ.get("GNN_ALLOW_DOWNLOADS", "1").lower() not in ("0", "false", "no")
    if not _allow:
        raise FileNotFoundError(
            f"{local_path} absent et téléchargements désactivés "
            f"(GNN_ALLOW_DOWNLOADS=0). Pré-stagez ce fichier "
            f"({label or os.path.basename(local_path)}) : exécutez sur un nœud avec "
            f"internet (frontal) ou rsync depuis votre cache local data/.")
    print(f"    Téléchargement {label or os.path.basename(local_path)}...")
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (research)")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except Exception as _e:
        raise FileNotFoundError(
            f"Échec téléchargement {label or os.path.basename(local_path)} ({_e}). "
            f"Nœud probablement hors-ligne (HPC derrière proxy → 403). Pré-stagez "
            f"{local_path} : nœud avec internet (frontal) ou rsync du cache local "
            f"data/. URL : {url}") from _e
    with open(local_path, "wb") as f:
        f.write(raw)
    print(f"      OK ({len(raw) / 1e6:.1f} MB)")


# =============================================================================
# CACHE DU GRAPHE (§1-7) — réutilisation via --reuse-graph
# =============================================================================
# Le build du graphe (§1-7, ~40 min) peut être mis en cache (pickle). Avec
# --reuse-graph, on recharge le cache UNIQUEMENT si sa signature de config
# correspond : sources actives (MODULES), matrice/fichiers + mtime/taille,
# conditions HuMess, features exclues, flags CLI graphe. La matrice étant dans
# la signature (mtime+taille), tout changement de jeu de données — donc de
# NB DE GÈNES — invalide le cache. n_genes est stocké et ré-affiché au
# chargement pour vérification → jamais de graphe obsolète réutilisé.
_CACHE_VARS = ["CELL_GROUPS", "_COEXPR_DIM", "_dst", "_f", "_g", "_src", "b", "cell_group_features", "coexpr_dst", "coexpr_src", "coexpr_w_tensor", "col", "data", "edge_attr_cocat", "edge_attr_expresses", "edge_attr_ppi", "edge_attr_reactome_fi", "edge_attr_regulates", "edge_attr_signaling", "edge_attr_tf_curated", "edge_index_cocat", "edge_index_coexpr", "edge_index_expresses", "edge_index_pathway", "edge_index_ppi", "edge_index_reactome_fi", "edge_index_regulates", "edge_index_signaling", "edge_index_tf_curated", "f", "g", "gene_features", "gene_symbols", "gene_to_idx", "group_stats", "grp", "i", "idx", "j", "line", "mask", "mean_expr_per_group", "mu", "n_genes", "omnipath_endpoints", "op_sig_dst", "op_sig_src", "op_tf_dst", "op_tf_src", "pair", "parts", "ppi_dst", "ppi_src", "react_dst", "react_src", "reactome_fi_dst", "reactome_fi_src", "reg_dst", "reg_src", "score", "sign", "std", "target"]
import hashlib as _hashlib
def _mtime_sig(_p):
    try:
        _s = os.stat(_p); return (round(_s.st_mtime, 2), _s.st_size)
    except OSError:
        return None
def _resolve_expr_path():
    return EXPR_MATRIX if os.path.isabs(EXPR_MATRIX) else os.path.join(DATA_DIR, EXPR_MATRIX)
_TRAIN_ONLY = {"seed", "run_tag", "n_epochs", "patience", "reuse_graph",
               "graph_cache", "device", "lr", "kl_beta_max",
               "no_baselines", "no_validation",
               # V-sup : n'affectent que l'étape post-build (features DE ajoutées
               # APRÈS restauration du cache + tête jointe) → cache réutilisable.
               "supervised", "de_features", "supervised_loss_weight",
               "supervised_recompute_labels"}  # post-build only
_sig_obj = {
    "env": {_k: os.environ.get(_k, "") for _k in
            ("GNN_EXPR_MATRIX", "GNN_GROUP_META", "GNN_CELL_GROUPS",
             "GNN_HUMESS_CONDITIONS", "GNN_HUMESS_DIR", "GNN_DATA_DIR")},
    "modules": sorted((_k, bool(_v)) for _k, _v in MODULES.items()),
    "excluded": sorted(_EXCLUDED_FEATURES),
    "humess_conditions": list(HUMESS_CONDITIONS),
    "cli": {_k: str(_v) for _k, _v in sorted(vars(CLI_ARGS).items())
            if _k not in _TRAIN_ONLY},
    "mtimes": {_p: _mtime_sig(_p) for _p in
               (_resolve_expr_path(), GROUP_META or "", HUMESS_DIR,
                getattr(CLI_ARGS, "diff_coexpr_file", "") or "")},
}
_SIG = _hashlib.md5(repr(_sig_obj).encode()).hexdigest()
_GRAPH_CACHE = CLI_ARGS.graph_cache or os.path.join(OUT_DIR_BASE, "_graph_cache.pkl")
_REUSE_OK = False
if getattr(CLI_ARGS, "reuse_graph", False) and os.path.exists(_GRAPH_CACHE):
    try:
        with open(_GRAPH_CACHE, "rb") as _fh:
            _cache = pickle.load(_fh)
    except Exception as _e:
        _cache = None
        print(f"[reuse-graph] cache illisible ({_e}) -> rebuild complet")
    if _cache is not None and _cache.get("_sig") == _SIG:
        globals().update({_k: _v for _k, _v in _cache.items() if _k != "_sig"})
        _REUSE_OK = True
        print(f"[reuse-graph] OK cache VALIDE (signature identique) -> sections 1-7 sautees "
              f"(n_genes={_cache.get('n_genes')}, {len(_cache.get('gene_symbols', []))} symboles)")
    elif _cache is not None:
        print("[reuse-graph] cache OBSOLETE (config/sources/fichiers/nb-genes "
              "differents) -> rebuild complet du graphe")
elif getattr(CLI_ARGS, "reuse_graph", False):
    print(f"[reuse-graph] aucun cache a {_GRAPH_CACHE} -> build puis mise en cache")

if not _REUSE_OK:
    # =============================================================================
    # 1. CHARGEMENT DES DONNÉES scRNA-seq
    # =============================================================================
    # Les données proviennent de GSE102090 : des HUVEC (Human Umbilical Vein
    # Endothelial Cells) en scRNA-seq, deux passages :
    #   - P4 (passage 4) : cellules jeunes / prolifératives
    #   - P16 (passage 16) : cellules sénescentes (sénescence réplicative)
    # Le preprocessing a été fait dans Seurat (normalisation LogNormalize,
    # clustering des P16 en 4 sous-populations). On charge ici la metadata
    # (barcode, passage, cluster) et l'en-tête de la matrice normalisée
    # pour récupérer la liste de tous les gènes mesurés.
    print("=" * 70)
    print("1. Chargement des données scRNA-seq (P4 / P16)")
    print("=" * 70)

    # metadata : scRNA HUVEC = 1 ligne/cellule (barcode, passage, cluster_P16) ;
    #            bulk (GROUP_META) = 1 ligne/échantillon (sample, group). Sert au
    #            dénominateur n_cells_norm (§5) → DOIT refléter le bon univers
    #            (cellules en scRNA, échantillons en bulk), sinon feature faussée.
    if GROUP_META:
        # Bulk : pas de fichier HUVEC ; la "metadata" = la samplesheet sample→group.
        metadata = pd.read_csv(GROUP_META, sep=None, engine="python", header=None)
        metadata = metadata.rename(columns={0: "sample", 1: "group"})
        print(f"  Metadata (bulk) : {len(metadata)} échantillons ; "
              f"groupes : {sorted(metadata['group'].astype(str).unique())}")
        p16_meta = pd.DataFrame(); p16_clusters = []   # non utilisés en bulk
    else:
        _meta_path = os.path.join(GNN_DATA_DIR, "merged_P4_P16_metadata.csv")
        if not os.path.exists(_meta_path):
            raise FileNotFoundError(
                f"{_meta_path} introuvable → mode scRNA HUVEC (GNN_GROUP_META vide). "
                "Pour un dataset BULK/custom, définissez GNN_GROUP_META (samplesheet "
                "sample→group) ET GNN_EXPR_MATRIX. Sur cluster : vérifiez que les "
                "données sont bien transférées (data/ pas toujours rsync vers /scratch).")
        metadata = pd.read_csv(_meta_path)
        print(f"  Metadata : {len(metadata)} cellules")
        print(f"    P4  : {(metadata['passage'] == 'P4').sum()} cellules")
        print(f"    P16 : {(metadata['passage'] == 'P16').sum()} cellules")
        # Les P16 sont subdivisées en 4 clusters (clustering Seurat) = sous-états sén.
        p16_meta = metadata[metadata["passage"] == "P16"].copy()
        p16_clusters = sorted(p16_meta["cluster_P16"].dropna().unique())
        print(f"    Clusters P16 : {p16_clusters}")

    # POINT CLÉ : on prend TOUS les gènes de la matrice normalisée, sans
    # filtrer par DEG (Differentially Expressed Genes). C'est essentiel car :
    #   1. Le VGAE est non supervisé — il n'a pas besoin de labels DEG.
    #   2. Filtrer par DEG introduirait la circularité qu'on veut éviter
    #      (les DEG sont définis par log2FC/padj, qui étaient les features
    #       et les labels du pipeline supervisé précédent gnn.py).
    # On ne lit que l'en-tête (première ligne) pour économiser la mémoire —
    # la matrice complète sera chargée plus tard (section 4) avec usecols.
    with open(os.path.join(GNN_DATA_DIR, EXPR_MATRIX)) as f:
        header = f.readline().strip().split(",")
    # HUVEC scRNA : 4 colonnes meta (barcode, passage, cluster_P16, cell_state) puis
    # gènes. Bulk (GROUP_META) : 1 colonne meta (sample) puis gènes. Le reste = noms
    # de gènes (potentiellement entre guillemets).
    _n_meta_cols = 1 if GROUP_META else 4
    all_available_genes = [g.strip('"') for g in header[_n_meta_cols:]]

    # NOTE : cette liste sera filtrée en section 3 pour ne garder que les gènes
    # ayant au moins une connexion dans le graphe (PPI, SCENIC, pathway, etc.).
    # Un gène isolé (sans arête) ne reçoit aucun message pendant le GNN et
    # ne contribue pas à la reconstruction des arêtes → inutile pour le VGAE.
    print(f"  Gènes dans la matrice : {len(all_available_genes)}")

    # =============================================================================
    # 2. CHARGEMENT DES DONNÉES pySCENIC
    # =============================================================================
    # pySCENIC est un pipeline d'inférence de réseaux de régulation génique
    # (Gene Regulatory Networks, GRN) à partir de données scRNA-seq. Il produit :
    #   1. regulon_edges : liens TF → gène cible (avec un poids de régulation).
    #      Un "regulon" = un TF + l'ensemble de ses gènes cibles prédits.
    #   2. tf_activity (AUCell) : score d'activité de chaque TF dans chaque
    #      cluster. AUCell évalue si les cibles d'un TF sont enrichies parmi
    #      les gènes les plus exprimés d'une cellule.
    #   3. adjacencies (GRNBoost2) : réseau de co-expression brut inféré par
    #      GRNBoost2 (gradient boosting sur les paires de gènes). Les poids
    #      "importance" mesurent la force de la co-expression prédite.
    # Ces 3 sorties alimentent 3 types d'arêtes différents dans le graphe :
    #   - regulon_edges → arêtes "regulates" (section 6d)
    #   - tf_activity → feature d'arête sur "expresses" (section 6a)
    #   - adjacencies → arêtes "coexpression" (section 6e)
    print("\n" + "=" * 70)
    print("2. Chargement des données pySCENIC")
    print("=" * 70)

    # regulon_edges + tf_activity : lectures SCENIC, gatées par --no-scenic-regulons
    # (généralisation V6 : un dataset bulk sans pySCENIC les saute proprement).
    # Fallbacks VIDES tolérés downstream : regulon usage gaté (l.~1401) ; tf_activity
    # testé par `gene in tf_activity.columns` (section 6a) → 0 si vide.
    if MODULES["use_scenic_regulons"]:
        # regulon_edges : colonnes = TF, target_gene, weight ; TF suffixe "(+)" → TF_clean
        regulon_edges = pd.read_csv(os.path.join(SCENIC_DIR, "regulon_edges_TF_to_gene.csv"))
        regulon_edges["TF_clean"] = regulon_edges["TF"].str.replace(r"\(\+\)$", "", regex=True)
        print(f"  Regulon edges : {len(regulon_edges)} interactions TF→cible")
        # tf_activity : matrice (clusters × TFs) — score AUCell moyen par cluster,
        # feature d'arête cell_group→gene (section 6a) si le gène est un TF.
        tf_activity = pd.read_csv(
            os.path.join(SCENIC_DIR, "mean_TF_activity_per_cluster.csv"), index_col=0
        )
        tf_activity.columns = [c.replace("(+)", "") for c in tf_activity.columns]
        print(f"  TF activity : {tf_activity.shape[0]} clusters × {tf_activity.shape[1]} TFs")
    else:
        regulon_edges = pd.DataFrame(columns=["TF", "target_gene", "weight", "TF_clean"])
        tf_activity = pd.DataFrame()   # .columns / .index vides → 0 partout
        print("  SCENIC regulons/tf_activity : SKIP (--no-scenic-regulons)")

    # adjacencies (GRNBoost2) : réseau de co-expression brut.
    # Colonnes = TF, target, importance. On ne garde que le top 2% (COEXPR_TOP_QUANTILE)
    # pour avoir un réseau épars de haute confiance. Les poids faibles sont
    # vraisemblablement du bruit et ajouteraient des arêtes non informatives.
    # Modulaire : si --no-coexpr, on ne charge même pas le fichier (~700 Mo).
    if MODULES["use_coexpr"] and COEXPR_DIFFERENTIAL:
        # V4.2 option A : coexpr_diff.tsv produit par build_diff_coexpr.py.
        # Colonnes : TF, target, importance_p4_norm, importance_p16_norm,
        #            delta, cat_shared, cat_p4, cat_p16, ...
        _diff_path = CLI_ARGS.diff_coexpr_file
        if not os.path.isabs(_diff_path):
            _diff_path = os.path.join(BASE_DIR, _diff_path)
        if not os.path.exists(_diff_path):
            raise FileNotFoundError(
                f"--coexpr-mode differential mais {_diff_path} absent. "
                f"Lancer build_diff_coexpr.py (extract-matrices → GRNBoost2 "
                f"cluster → merge-adjacencies) d'abord."
            )
        adjacencies_filtered = pd.read_csv(_diff_path, sep="\t")
        # Le filtrage top-quantile est déjà fait PAR CONDITION dans
        # build_diff_coexpr.py (merge-adjacencies). On garde tout ici.
        print(f"  Adjacencies (V4.2 differential) : {len(adjacencies_filtered)} "
              f"arêtes P4∪P16, "
              f"catégories={adjacencies_filtered['category'].value_counts().to_dict()}")
    elif MODULES["use_coexpr"]:
        adjacencies = pd.read_csv(os.path.join(SCENIC_DIR, "adjacencies.csv"))
        importance_thresh = adjacencies["importance"].quantile(COEXPR_TOP_QUANTILE)
        adjacencies_filtered = adjacencies[adjacencies["importance"] >= importance_thresh].copy()
        print(f"  Adjacencies : {len(adjacencies)} → {len(adjacencies_filtered)} "
              f"(top {100*(1-COEXPR_TOP_QUANTILE):.0f}%, seuil={importance_thresh:.2f})")
    else:
        print("  Adjacencies : SKIP (--no-coexpr)")
        adjacencies_filtered = pd.DataFrame(columns=["TF", "target", "importance"])

    # =============================================================================
    # 2.5. PRÉ-CHARGEMENT OMNIPATH (V4.1) — pour étendre gene_to_idx
    # =============================================================================
    # Charge les caches OmniPath UNE FOIS, avant la section 3 (sélection des
    # gènes), pour pouvoir étendre `selected_genes` avec les endpoints d'OmniPath.
    # Sans ce pré-chargement, section 6g intervient TROP TARD : `gene_to_idx`
    # est déjà figé sur PPI/SCENIC/coexpr/REACTOME/HuMess, et `_project_to_graph`
    # filtre strictement → ~700 TFs CollecTRI étaient éliminés silencieusement.
    #
    # Note : le set retourné inclut TOUS les symboles présents dans les caches
    # OmniPath actifs. L'intersection avec `available_set` (gènes scRNA-mesurés)
    # se fait en section 3 — on n'invente pas de gènes hors mesure.
    omnipath_endpoints: set[str] = set()
    if (MODULES["include_omnipath_genes"]
            and (MODULES["use_omnipath_signaling"]
                 or MODULES["use_omnipath_tf_curated"])):
        print("\n" + "=" * 70)
        print("2.5. Pré-chargement OmniPath (V4.1) — pour expansion gene_to_idx")
        print("=" * 70)
        # Force le mode OFFLINE si l'utilisateur n'a pas autorisé le download :
        # évite que `import omnipath` déclenche les metadata pre-fetches HTTP
        # qui timeout 30+ min sur compute nodes Nautilus sans Internet.
        if not CLI_ARGS.omnipath_download_if_missing:
            os.environ["GNN_OMNIPATH_OFFLINE"] = "1"
        try:
            from omnipath_integration import (
                get_omnipath_endpoints as _opi_endpoints,
                silence_omnipath_logging as _silence_opi,
            )
            if not CLI_ARGS.omnipath_download_if_missing:
                _silence_opi()
            # Sources actives selon les flags
            _opi_sources = []
            if MODULES["use_omnipath_signaling"]:
                _opi_sources.extend(["signaling", "signor"])
            if MODULES["use_omnipath_tf_curated"]:
                _opi_sources.append("collectri")
            omnipath_endpoints = _opi_endpoints(
                cache_dir=OMNIPATH_CACHE_DIR,
                sources=_opi_sources,
                download_if_missing=CLI_ARGS.omnipath_download_if_missing,
            )
            print(f"  Endpoints OmniPath uniques (avant intersection scRNA) : "
                  f"{len(omnipath_endpoints)}")
        except ImportError as _e:
            print(f"  [warn] import omnipath_integration KO ({_e}) — "
                  f"--include-omnipath-genes inactif.")
    elif MODULES["include_omnipath_genes"]:
        print("\n[warn] --include-omnipath-genes activé mais aucune source "
              "OmniPath active (--use-omnipath-signaling / --use-omnipath-tf-curated). "
              "Le flag est ignoré.")

    # =============================================================================
    # 3. SÉLECTION DES GÈNES — BASÉE SUR LA CONNECTIVITÉ (pas les DEGs)
    # =============================================================================
    # JUSTIFICATION : dans un GNN, le message passing propage l'information le
    # long des arêtes du graphe. Un gène ISOLÉ (sans arête) :
    #   - Ne reçoit aucun message de ses voisins (pas d'agrégation)
    #   - Ne contribue à aucune arête à reconstruire (pas de signal de loss)
    #   - Son embedding sera uniquement basé sur ses features → pas d'utilité GNN
    # On ne garde donc que les gènes connectés dans au moins un réseau :
    # SCENIC (régulation), GRNBoost2 (co-expression), STRING (PPI), REACTOME
    # (pathway), ou HuMess (cocatalyse métabolique).
    # C'est un filtre TOPOLOGIQUE, pas statistique — aucun biais DEG.
    print("\n" + "=" * 70)
    print("3. Sélection des gènes par connectivité")
    print("=" * 70)

    # available_set : ensemble des gènes mesurés dans le scRNA-seq.
    # Sert de filtre pour ne garder que les gènes qu'on peut observer
    # (certains gènes des bases externes ne sont pas dans notre matrice).
    available_set = set(all_available_genes)

    # --- 3a. Gènes SCENIC (régulation transcriptionnelle) ---
    # Un gène est "SCENIC-connecté" s'il est soit un TF qui régule des cibles,
    # soit une cible régulée par un TF. On prend l'union des deux.
    # L'intersection avec available_set garantit que le gène est dans notre matrice.
    scenic_genes = set(regulon_edges["TF_clean"]) | set(regulon_edges["target_gene"])
    scenic_genes &= available_set

    # --- 3b. Gènes GRNBoost2 (co-expression filtrée au top 2%) ---
    # GRNBoost2 produit un réseau dirigé (TF → target) mais en pratique les liens
    # sont interprétés comme de la co-expression. On prend l'union des deux côtés.
    coexpr_genes = set(adjacencies_filtered["TF"].astype(str)) | set(adjacencies_filtered["target"].astype(str))
    coexpr_genes &= available_set

    # --- 3c. Gènes STRING PPI (protein-protein interactions) ---
    # STRING v12 : base de données d'interactions protéine-protéine (expérimentales,
    # co-expression, text mining, etc.). Le "combined_score" (0-1000) agrège
    # plusieurs canaux de preuve. On télécharge 2 fichiers :
    #   - links : les interactions (protein1, protein2, combined_score)
    #   - aliases : mapping identifiant STRING → symbole HGNC (ex : ENSP00000... → TP53)
    # Modulaire : si --no-ppi, on saute le téléchargement et la lecture (~1.6 Go),
    # ppi_hc / string2sym restent vides, aucune arête PPI ne sera créée.
    ppi_genes = set()
    ppi_hc = pd.DataFrame(columns=["protein1", "protein2", "combined_score"])
    sym2string = {}
    string2sym = {}
    if MODULES["use_ppi"]:
        PPI_FILE = os.path.join(PPI_DIR, "9606.protein.links.v12.0.txt.gz")
        PPI_ALIAS_FILE = os.path.join(PPI_DIR, "9606.protein.aliases.v12.0.txt.gz")
        download_if_absent(
            "https://stringdb-static.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
            PPI_FILE, "STRING links"
        )
        download_if_absent(
            "https://stringdb-static.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz",
            PPI_ALIAS_FILE, "STRING aliases"
        )

        # Construction du mapping symbole ↔ identifiant STRING.
        # On filtre par "Ensembl_HGNC" pour avoir des symboles de gènes standards
        # (ex : TP53, CDKN1A) et non des alias ambigus. Cela évite les collisions
        # où un alias pourrait correspondre à plusieurs protéines.
        aliases = pd.read_csv(PPI_ALIAS_FILE, sep="\t", compression="gzip")
        aliases_filt = aliases[
            (aliases["alias"].isin(available_set)) &
            (aliases["source"].str.contains("Ensembl_HGNC", na=False))
        ]
        # sym2string : "TP53" → "9606.ENSP00000269305"
        sym2string = dict(zip(aliases_filt["alias"], aliases_filt["#string_protein_id"]))
        # string2sym : inverse, pour reconvertir après filtrage
        string2sym = {v: k for k, v in sym2string.items()}
        string_ids = set(sym2string.values())

        # Chargement du réseau PPI complet puis filtrage :
        # 1. Les deux protéines doivent être dans notre ensemble de gènes
        # 2. Le combined_score doit dépasser PPI_SCORE_THRESH (900 = highest confidence)
        ppi_raw = pd.read_csv(PPI_FILE, sep=" ", compression="gzip")
        ppi_hc = ppi_raw[
            (ppi_raw["protein1"].isin(string_ids)) &
            (ppi_raw["protein2"].isin(string_ids)) &
            (ppi_raw["combined_score"] >= PPI_SCORE_THRESH)
        ]
        # Extraction des symboles de gènes impliqués dans au moins une PPI fiable
        for _, row in ppi_hc.iterrows():
            s1, s2 = string2sym.get(row["protein1"]), string2sym.get(row["protein2"])
            if s1 and s2:
                ppi_genes.update([s1, s2])
    else:
        print("  STRING PPI : SKIP (--no-ppi)")

    # --- 3d. Gènes REACTOME (pathways biologiques) ---
    # REACTOME via MSigDB : fichier GMT (Gene Matrix Transposed) où chaque ligne
    # est un pathway avec ses gènes membres. Format : NOM_PATHWAY <tab> URL <tab> GENE1 <tab> GENE2 ...
    # On filtre par taille : un pathway de 2-20 gènes est informatif. Au-delà de
    # REACTOME_MAX_PATHWAY gènes, le pathway est trop générique (ex : "Metabolism")
    # et connecterait des gènes sans rapport fonctionnel direct.
    # Modulaire : si --no-reactome, on saute le téléchargement et le parse.
    reactome_pathways = {}   # nom_pathway → set(gènes)
    reactome_genes = set()   # union de tous les gènes dans au moins un pathway
    if MODULES["use_reactome"]:
        MSIGDB_REACTOME = os.path.join(DB_DIR, "c2.cp.reactome.symbols.gmt")
        download_if_absent(
            "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/c2.cp.reactome.v2024.1.Hs.symbols.gmt",
            MSIGDB_REACTOME, "MSigDB REACTOME"
        )
        # Parse du fichier GMT REACTOME : on ne garde que les pathways de taille
        # raisonnable (2 à REACTOME_MAX_PATHWAY gènes) après intersection avec
        # les gènes disponibles dans notre matrice scRNA-seq.
        with open(MSIGDB_REACTOME) as f:
            for line in f:
                parts = line.strip().split("\t")
                # parts[0] = nom du pathway, parts[1] = URL, parts[2:] = gènes
                genes_in_pw = set(parts[2:]) & available_set
                if 2 <= len(genes_in_pw) <= REACTOME_MAX_PATHWAY:
                    reactome_pathways[parts[0]] = genes_in_pw
                    reactome_genes |= genes_in_pw
    else:
        print("  REACTOME : SKIP (--no-reactome)")

    # --- 3e. UNION FINALE : tous les gènes connectés par au moins un réseau ---
    # connected_genes = SCENIC ∪ GRNBoost2 ∪ PPI ∪ REACTOME (∪ OmniPath en V4.1)
    # (HuMess sera ajouté plus tard en section 6f, mais ses gènes sont typiquement
    # déjà dans l'un des réseaux ci-dessus.)
    # V4.1 : si --include-omnipath-genes, on ajoute les endpoints de signaling
    # (kinase-substrat + SIGNOR) et tf_curated (CollecTRI). L'intersection avec
    # `available_set` garantit qu'on n'invente pas de gènes hors scRNA-mesurés.
    # gene_symbols : array trié de tous les gènes retenus (l'index dans cet
    #   array = l'identifiant du noeud dans le graphe PyG).
    # gene_to_idx : dictionnaire inverse, symbole → index dans le graphe.
    connected_genes = scenic_genes | coexpr_genes | ppi_genes | reactome_genes
    _connected_before_opi = set(connected_genes & available_set)
    if omnipath_endpoints:
        connected_genes |= omnipath_endpoints
    gene_symbols = np.array(sorted(connected_genes & available_set))
    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}
    n_genes = len(gene_symbols)

    print(f"  Gènes connectés : {n_genes}")
    print(f"    SCENIC       : {len(scenic_genes & set(gene_symbols))}")
    print(f"    Co-expression: {len(coexpr_genes & set(gene_symbols))}")
    print(f"    PPI          : {len(ppi_genes & set(gene_symbols))}")
    print(f"    REACTOME     : {len(reactome_genes & set(gene_symbols))}")
    if omnipath_endpoints:
        _opi_in_graph = omnipath_endpoints & set(gene_symbols)
        _opi_new = _opi_in_graph - _connected_before_opi
        print(f"    OmniPath     : {len(_opi_in_graph)} "
              f"(dont {len(_opi_new)} nouveaux ∉ des 4 sources V3)")

    # =============================================================================
    # 4. CALCUL DE L'EXPRESSION PAR GROUPE — FEATURES D'ARÊTES
    # =============================================================================
    # ARCHITECTURE CLÉ : l'expression génique est placée sur les ARÊTES
    # (cell_group → gene) et NON sur les noeuds "gene". C'est le point central
    # de la suppression de la circularité :
    #   - Pipeline supervisé (gnn.py) : features de noeud = log2FC, padj → labels = DEG
    #     → circularité car les features CONTIENNENT l'information des labels.
    #   - Pipeline VGAE (ici) : features de noeud = topologiques (is_tf, variance, degree)
    #     → l'expression est sur les arêtes, le score d'importance ÉMERGE de
    #     l'espace latent sans jamais voir le log2FC/padj.
    #
    # On calcule 6 statistiques d'expression pour chaque combinaison (groupe, gène) :
    #   1. mean_expression : expression moyenne (LogNormalize) dans le groupe
    #   2. pct_expressing : fraction de cellules du groupe qui expriment le gène (> 0)
    #   3. std_expression : écart-type intra-groupe (mesure la variabilité cellulaire)
    #   4. cv_expression : coefficient de variation = std/mean (variabilité relative)
    #   5. q25, q75 : quartiles d'expression (caractérisent la distribution)
    # + 1 feature supplémentaire (tf_activity) ajoutée en section 6a.
    print("\n" + "=" * 70)
    print("4. Expression par groupe cellulaire (features d'arêtes)")
    print("=" * 70)

    # Les 5 groupes cellulaires du graphe :
    # P4 = cellules jeunes (1 seul groupe car pas de sous-clusters)
    # P16_cluster_0..3 = 4 sous-populations sénescentes (identifiées par Seurat)
    # Configurable (généralisation V6 : un dataset bulk a ses propres groupes, ex.
    # "pro,sen") via env GNN_CELL_GROUPS. Défaut HUVEC = 5 groupes (rétro-compat).
    CELL_GROUPS = [g.strip() for g in os.environ.get(
        "GNN_CELL_GROUPS",
        "P4,P16_cluster_0,P16_cluster_1,P16_cluster_2,P16_cluster_3").split(",")
        if g.strip()]

    # On charge la matrice normalisée (LogNormalize de Seurat) mais UNIQUEMENT
    # pour les gènes retenus en section 3 (gene_symbols). usecols évite de charger
    # les ~20000 gènes en mémoire quand on n'en utilise que ~5000.
    print("  Chargement de la matrice normalisée...")
    if GROUP_META:
        # ── Chemin BULK (généralisation V6) : matrice échantillons×gènes (1re col =
        #    sample) + meta sample→group ; le groupe est lu directement (pas de
        #    passage/cluster_P16). logFC JAMAIS chargé ici (idem HUVEC : on n'utilise
        #    que l'expression par groupe).
        _full = pd.read_csv(os.path.join(GNN_DATA_DIR, EXPR_MATRIX))
        _sample_col = _full.columns[0]
        # robuste : sniff séparateur (tsv/csv), sans en-tête (col0=sample, col1=group)
        _meta = pd.read_csv(GROUP_META, sep=None, engine="python", header=None)
        _g = dict(zip(_meta.iloc[:, 0].astype(str), _meta.iloc[:, 1].astype(str)))
        _full["group"] = _full[_sample_col].astype(str).map(_g)
        _keep = [g for g in gene_symbols.tolist() if g in _full.columns]
        normalized = _full.dropna(subset=["group"])[["group"] + _keep].copy()
        print(f"  [bulk] matrice : {normalized.shape} ; groupes : "
              f"{sorted(normalized['group'].unique())}")
    else:
        cols_to_read = ["barcode", "passage", "cluster_P16", "cell_state"] + gene_symbols.tolist()
        normalized = pd.read_csv(
            os.path.join(GNN_DATA_DIR, EXPR_MATRIX),
            usecols=cols_to_read,
        ).copy()  # .copy() défragmente le DataFrame (pandas alloue des blocs contigus)

        def assign_group(row):
            """Assigne chaque cellule à son groupe (P4 ou P16_cluster_X)."""
            if row["passage"] == "P4":
                return "P4"
            c = row["cluster_P16"]
            if pd.notna(c):
                return f"P16_cluster_{int(c)}"
            return None  # Cellules sans cluster assigné → seront supprimées

        normalized["group"] = normalized.apply(assign_group, axis=1)
        normalized = normalized.dropna(subset=["group"])  # Supprime les cellules sans groupe
    print(f"  Matrice : {normalized.shape}")

    # Calcul des statistiques par groupe cellulaire.
    # Pour chaque groupe, on calcule les 6 stats sur les colonnes de gènes.
    # Ces stats deviendront les features d'arêtes "expresses" en section 6a.
    print("  Calcul des statistiques (mean, pct, std, cv, q25, q75)...")
    group_stats = {}
    for grp in CELL_GROUPS:
        mask = normalized["group"] == grp
        sub = normalized.loc[mask, gene_symbols]  # Sous-matrice (cellules du groupe × gènes)
        n_cells = mask.sum()

        # mean_expression : expression moyenne normalisée du gène dans ce groupe
        mean_expr = sub.mean(axis=0).values.astype(np.float32)
        # pct_expressing : fraction de cellules exprimant ce gène (dropout rate = 1 - pct)
        pct_expr = (sub > 0).mean(axis=0).values.astype(np.float32)
        # std_expression : variabilité intra-groupe (forte = gène hétérogène)
        std_expr = sub.std(axis=0).values.astype(np.float32)
        # cv_expression : coefficient de variation (normalise la std par la mean)
        # Un CV élevé = bimodal ou très variable. +1e-8 évite la division par zéro.
        cv_expr = std_expr / (mean_expr + 1e-8)
        # q25, q75 : quartiles de la distribution d'expression
        q25 = sub.quantile(0.25, axis=0).values.astype(np.float32)
        q75 = sub.quantile(0.75, axis=0).values.astype(np.float32)

        group_stats[grp] = {
            "mean_expression": mean_expr, "pct_expressing": pct_expr,
            "std_expression": std_expr, "cv_expression": cv_expr,
            "q25": q25, "q75": q75, "n_cells": n_cells,
        }
        print(f"    {grp:20s} : {n_cells} cellules, mean={mean_expr.mean():.3f}")

    # Libération de la matrice normalisée (plusieurs Go) — les stats sont calculées.
    del normalized

    # =============================================================================
    # 5. FEATURES DES NOEUDS (topologiques uniquement — PAS de log2FC/padj)
    # =============================================================================
    # POINT ANTI-CIRCULARITÉ : les features de noeuds "gene" ne doivent JAMAIS
    # contenir les statistiques différentielles (log2FC, padj, delta_pct) car :
    #   - Le pipeline supervisé précédent (gnn.py) utilisait ces features ET ces
    #     mêmes stats comme labels DEG → circularité.
    #   - Ici, on utilise uniquement des propriétés TOPOLOGIQUES / INTRINSÈQUES :
    #     1. is_tf : le gène est-il un facteur de transcription ? (booléen, propriété
    #        intrinsèque du gène qui ne dépend pas du contexte expérimental)
    #     2. variance_across_groups : variabilité de l'expression ENTRE les 5 groupes
    #        cellulaires. C'est une mesure brute de variabilité (pas un test statistique
    #        comme le log2FC). Un gène avec haute variance = expression différente
    #        entre P4 et les clusters P16 → potentiellement intéressant.
    #     3. ppi_degree : nombre de voisins PPI (ajouté après section 6)
    #     4. reg_degree : nombre de liens de régulation (ajouté après section 6)
    #   + En V3 (avec HuMess) : 4 features métaboliques supplémentaires (section 6f)
    # Note : les features de degré (3-4) sont ajoutées APRÈS la construction des
    # arêtes en section 6, car on a besoin des arêtes pour les compter.
    print("\n" + "=" * 70)
    print("5. Features des noeuds (topologiques)")
    print("=" * 70)

    # Feature 1 : is_tf — binaire, 1.0 si le gène est un TF.
    # V3 : pySCENIC uniquement (~62 TFs après filtres motif + expression HUVEC).
    # V4.1+ : union pySCENIC ∪ CollecTRI sources lorsque
    # --use-omnipath-tf-curated est actif. CollecTRI (Müller-Dott 2023) couvre
    # ~1186 TFs curés depuis la littérature ; intersection avec les gènes
    # mesurés (gene_symbols) garantit qu'on n'invente aucun TF hors scRNA.
    # Permet à la logique TF-aware downstream (suffixe interpretation,
    # threshold B_discovery relaxé) de couvrir tous les TFs curés, pas
    # uniquement le sous-ensemble pySCENIC.
    scenic_tfs = set(regulon_edges["TF_clean"].unique())
    collectri_tfs: set[str] = set()
    if MODULES["use_omnipath_tf_curated"]:
        _collectri_cache = os.path.join(OMNIPATH_CACHE_DIR, "tf_collectri.tsv.gz")
        if os.path.exists(_collectri_cache):
            try:
                import gzip as _gz
                with _gz.open(_collectri_cache, "rt") as _f:
                    next(_f, None)  # header
                    for _line in _f:
                        _src = _line.split("\t", 1)[0]
                        # On exclut les hétérodimères type "NFKB1_REL"
                        if _src and "_" not in _src:
                            collectri_tfs.add(_src)
                print(f"  is_tf : CollecTRI source TFs lus = {len(collectri_tfs)} "
                      f"(union avec pySCENIC ci-dessous)")
            except Exception as _e:
                print(f"  is_tf : [warn] lecture CollecTRI échouée ({_e}) — "
                      f"on garde pySCENIC seul")
        else:
            print(f"  is_tf : [warn] cache CollecTRI absent à {_collectri_cache} — "
                  f"on garde pySCENIC seul")
    all_tfs = scenic_tfs | collectri_tfs
    is_tf = np.array([1.0 if g in all_tfs else 0.0 for g in gene_symbols],
                     dtype=np.float32)
    print(f"  is_tf : pySCENIC={len(scenic_tfs)} + CollecTRI={len(collectri_tfs)} "
          f"→ union ∩ available = {int(is_tf.sum())} TFs")

    # Feature 2 : variance_across_groups — variance de l'expression moyenne entre
    # les 5 groupes cellulaires. Calculée sur les mean_expression déjà calculées
    # en section 4. Normalisée par le max pour être dans [0, 1].
    # Intuition : un gène dont l'expression est stable entre P4 et P16 aura une
    # faible variance ; un gène fortement dérégulé aura une forte variance.
    mean_expr_per_group = np.array([
        group_stats[grp]["mean_expression"] for grp in CELL_GROUPS
    ])  # Shape : (5, n_genes) — 5 groupes × n_genes
    variance_across = mean_expr_per_group.var(axis=0).astype(np.float32)  # Variance sur l'axe des groupes
    variance_norm = variance_across / (variance_across.max() + 1e-8)

    # ── Cell group features ─────────────────────────────────────────────────────
    # Les noeuds "cell_group" ont aussi des features (3 dimensions) :
    #   1. is_senescent : 0 pour P4, 1 pour tous les P16 → encode la condition
    #   2. n_cells_norm : fraction du nombre total de cellules dans ce groupe
    #      → encode la taille relative du groupe
    #   3. cluster_idx : index normalisé du cluster (0 pour P4, 0.25/0.5/0.75/1.0
    #      pour les clusters P16) → encode l'identité du sous-cluster
    # Ces features permettent au modèle de distinguer les groupes cellulaires
    # et de moduler les messages "expresses" en conséquence.
    cell_group_features_list = []
    for grp_idx, grp in enumerate(CELL_GROUPS):
        # baseline = 1er groupe (HUVEC : P4 ; bulk : ex. pro). Générique au nommage.
        is_senescent = 0.0 if grp == CELL_GROUPS[0] else 1.0
        n_cells_norm = group_stats[grp]["n_cells"] / metadata.shape[0]
        # index normalisé de POSITION (agnostique au nommage). Pour les 5 groupes
        # HUVEC, grp_idx/(5-1) = 0, 0.25, 0.5, 0.75, 1.0 = identique à (c+1)/4.
        cluster_idx = grp_idx / max(1, len(CELL_GROUPS) - 1)
        cell_group_features_list.append([is_senescent, n_cells_norm, cluster_idx])
    cell_group_features = torch.tensor(cell_group_features_list, dtype=torch.float)

    # =============================================================================
    # 6. CONSTRUCTION DES ARÊTES
    # =============================================================================
    # Le graphe hétérogène contient 7 types d'arêtes (+ les reverses) :
    #   a. expresses     : cell_group → gene (expression, 7 features)
    #   b. ppi           : gene ↔ gene (STRING, 1 feature = score normalisé)
    #   c. same_pathway  : gene ↔ gene (REACTOME, pas de features)
    #   d. regulates     : TF → cible (pySCENIC, 1 feature = poids de régulation)
    #   e. coexpression  : gene ↔ gene (GRNBoost2 top 2%, 1 feature = importance)
    #   f. cocatalysis   : gene ↔ gene (HuMess, 2 features = [in_P4, in_P16])
    # Toutes les arêtes gene↔gene sont rendues BIDIRECTIONNELLES (i→j ET j→i)
    # car le message passing dans un GNN est directionnel : un noeud agrège les
    # messages de ses voisins entrants. Sans bidirectionnel, un gène ne verrait
    # que ses voisins dans un sens (ex : un TF verrait ses cibles mais pas l'inverse).
    print("\n" + "=" * 70)
    print("6. Construction des arêtes")
    print("=" * 70)

    # ── 6a. cell_group → gene (expression) ──────────────────────────────────────
    # Arêtes bipartites reliant chaque groupe cellulaire à chaque gène.
    # 7 features par arête = les stats d'expression calculées en section 4 +
    # le score d'activité TF (AUCell) pour ce gène dans ce cluster.
    # C'est un graphe COMPLET (chaque groupe est connecté à chaque gène),
    # donc n_arêtes = 5 groupes × n_genes.
    expr_src, expr_dst, expr_attrs = [], [], []
    if MODULES["use_cell_group_edges"]:
        for grp_idx, grp in enumerate(CELL_GROUPS):
            stats = group_stats[grp]
            # cluster_id sert à récupérer le score AUCell : None pour le groupe
            # baseline ; sinon l'index numérique du cluster (HUVEC P16_cluster_N) ou
            # le nom du groupe (bulk, ex. "sen" → tf_activity vide → lookup = 0).
            if grp == CELL_GROUPS[0]:
                cluster_id = None
            else:
                _suf = grp.split("_")[-1]
                cluster_id = int(_suf) if _suf.isdigit() else grp

            for gene_idx in range(n_genes):
                gene_name = gene_symbols[gene_idx]
                # tf_activity : score AUCell du TF dans ce cluster. Vaut 0 si le
                # gène n'est pas un TF ou si on est dans P4 (pas de clusters).
                tf_act = 0.0
                if cluster_id is not None and gene_name in tf_activity.columns:
                    if cluster_id in tf_activity.index:
                        tf_act = float(tf_activity.loc[cluster_id, gene_name])

                expr_src.append(grp_idx)     # Index du noeud cell_group source
                expr_dst.append(gene_idx)    # Index du noeud gene destination
                expr_attrs.append([
                    float(stats["mean_expression"][gene_idx]),   # Feature 1 : expression moyenne
                    float(stats["pct_expressing"][gene_idx]),     # Feature 2 : % de cellules exprimant
                    tf_act,                                       # Feature 3 : activité TF (AUCell)
                    float(stats["std_expression"][gene_idx]),     # Feature 4 : écart-type
                    float(stats["cv_expression"][gene_idx]),      # Feature 5 : coefficient de variation
                    float(stats["q25"][gene_idx]),                # Feature 6 : 1er quartile
                    float(stats["q75"][gene_idx]),                # Feature 7 : 3ème quartile
                ])

    # Conversion en tenseurs PyG (format COO : [2, n_edges])
    edge_index_expresses = (torch.tensor([expr_src, expr_dst], dtype=torch.long)
                            if expr_src else torch.zeros((2, 0), dtype=torch.long))
    edge_attr_expresses = (torch.tensor(expr_attrs, dtype=torch.float)
                           if expr_attrs else torch.zeros((0, 7), dtype=torch.float))
    # Z-score normalisation colonne par colonne pour que chaque feature ait
    # mean=0 et std=1. Important pour GATConv qui calcule des scores d'attention
    # sur les features d'arête — sans normalisation, les features à grande
    # échelle domineraient le calcul d'attention.
    if edge_attr_expresses.numel() > 0:
        for col in range(edge_attr_expresses.shape[1]):
            col_data = edge_attr_expresses[:, col]
            mu, std = col_data.mean(), col_data.std() + 1e-8
            edge_attr_expresses[:, col] = (col_data - mu) / std
    print(f"  expresses : {edge_index_expresses.shape[1]} arêtes, "
          f"7 features [mean, pct, tf_act, std, cv, q25, q75]"
          + ("" if MODULES["use_cell_group_edges"] else " [SKIP --no-cell-group-edges]"))

    # ── 6b. PPI STRING ──────────────────────────────────────────────────────────
    # Arêtes protéine-protéine de STRING (>= 900, highest confidence).
    # Chaque interaction est BIDIRECTIONNELLE (i→j ET j→i).
    # Feature = combined_score / 1000 (normalisé dans [0, 1]).
    # Rôle biologique : encode les interactions physiques entre protéines.
    # Les hubs PPI (gènes avec beaucoup de partenaires) reçoivent plus de
    # messages et tendent à avoir des embeddings plus informatifs.
    ppi_src, ppi_dst, ppi_w = [], [], []
    if MODULES["use_ppi"]:
        for _, row in ppi_hc.iterrows():
            s1, s2 = string2sym.get(row["protein1"]), string2sym.get(row["protein2"])
            if s1 and s2 and s1 in gene_to_idx and s2 in gene_to_idx:
                i, j = gene_to_idx[s1], gene_to_idx[s2]
                # Bidirectionnel : on ajoute les deux directions
                ppi_src.extend([i, j])
                ppi_dst.extend([j, i])
                ppi_w.extend([row["combined_score"] / 1000.0] * 2)  # Même poids dans les deux sens

    edge_index_ppi = (torch.tensor([ppi_src, ppi_dst], dtype=torch.long)
                      if ppi_src else torch.zeros((2, 0), dtype=torch.long))
    edge_attr_ppi = (torch.tensor(ppi_w, dtype=torch.float).unsqueeze(1)
                     if ppi_w else torch.zeros((0, 1), dtype=torch.float))
    print(f"  ppi : {len(ppi_src)//2} interactions ({len(ppi_src)} arêtes)"
          + ("" if MODULES["use_ppi"] else " [SKIP --no-ppi]"))

    # ── 6c. REACTOME (same_pathway) ─────────────────────────────────────────────
    # Pour chaque pathway REACTOME (2 à 20 gènes), on crée une arête entre
    # TOUTES les paires de gènes du pathway (graphe complet intra-pathway).
    # Pas de features d'arête — la simple existence de l'arête encode le fait
    # que les deux gènes partagent un pathway biologique.
    # react_pairs déduplique : si deux gènes partagent plusieurs pathways,
    # on ne crée qu'une seule arête (bidirectionnelle).
    # Rôle biologique : les gènes d'un même pathway coopèrent fonctionnellement.
    # Le message passing propagera l'information au sein des modules fonctionnels.
    react_src, react_dst = [], []
    react_pairs = set()  # Déduplique les paires (un gène peut être dans plusieurs pathways)
    if MODULES["use_reactome"]:
        for pw_genes in reactome_pathways.values():
            gene_list = sorted(pw_genes)
            # Double boucle : toutes les paires (i, j) avec i < j (évite les doublons a↔b / b↔a)
            for i_idx in range(len(gene_list)):
                for j_idx in range(i_idx + 1, len(gene_list)):
                    g1, g2 = gene_list[i_idx], gene_list[j_idx]
                    if g1 in gene_to_idx and g2 in gene_to_idx:
                        pair = (min(gene_to_idx[g1], gene_to_idx[g2]),
                                max(gene_to_idx[g1], gene_to_idx[g2]))
                        if pair not in react_pairs:
                            react_pairs.add(pair)
                            # Bidirectionnel
                            react_src.extend([pair[0], pair[1]])
                            react_dst.extend([pair[1], pair[0]])

    edge_index_pathway = (torch.tensor([react_src, react_dst], dtype=torch.long)
                         if react_src else torch.zeros((2, 0), dtype=torch.long))
    print(f"  pathway : {len(react_pairs)} paires ({len(react_src)} arêtes)"
          + ("" if MODULES["use_reactome"] else " [SKIP --no-reactome]"))

    # ── 6d. Regulon (pySCENIC) — arêtes "regulates" ────────────────────────────
    # Liens DIRIGÉS TF → gène cible (pas bidirectionnel à ce stade).
    # Feature = poids de régulation (confiance du lien dans le regulon).
    # Ces arêtes encodent la structure régulatrice : un TF qui active ou
    # réprime ses cibles. Le sens est important biologiquement.
    # MAIS on crée aussi l'arête inverse "regulated_by" (cible → TF) plus bas,
    # pour que les TFs reçoivent aussi de l'information de leurs cibles
    # pendant le message passing. Biologiquement, ça n'a pas de sens causal,
    # mais pour le GNN c'est nécessaire : sans ça, les cibles ne propagent
    # pas d'information vers les TFs qui les régulent.
    reg_src, reg_dst, reg_w = [], [], []
    reg_pairs = set()  # Déduplique si un TF régule la même cible dans plusieurs regulons
    if MODULES["use_scenic_regulons"]:
        for _, row in regulon_edges.iterrows():
            tf, target = row["TF_clean"], row["target_gene"]
            if tf in gene_to_idx and target in gene_to_idx:
                pair = (gene_to_idx[tf], gene_to_idx[target])
                if pair not in reg_pairs:
                    reg_pairs.add(pair)
                    reg_src.append(pair[0])      # TF (source)
                    reg_dst.append(pair[1])      # Cible (destination)
                    reg_w.append(float(row["weight"]))  # Poids de régulation

    edge_index_regulates = (torch.tensor([reg_src, reg_dst], dtype=torch.long)
                           if reg_src else torch.zeros((2, 0), dtype=torch.long))
    edge_attr_regulates = (torch.tensor(reg_w, dtype=torch.float).unsqueeze(1)
                          if reg_w else torch.zeros((0, 1), dtype=torch.float))
    # Normalisation min-max par le poids max (les poids SCENIC ne sont pas bornés)
    if edge_attr_regulates.numel() > 0:
        edge_attr_regulates = edge_attr_regulates / (edge_attr_regulates.max() + 1e-8)
    # Arête inverse "regulated_by" : cible → TF (même poids, sens inversé).
    # Permet aux TFs de recevoir de l'information de leurs cibles pendant
    # le message passing.
    edge_index_regulated_by = (torch.tensor([reg_dst, reg_src], dtype=torch.long)
                              if reg_src else torch.zeros((2, 0), dtype=torch.long))
    print(f"  regulates : {len(reg_pairs)} liens TF→cible"
          + ("" if MODULES["use_scenic_regulons"] else " [SKIP --no-scenic-regulons]"))

    # ── 6e. Co-expression (GRNBoost2) ──────────────────────────────────────────
    # Arêtes de co-expression inférées par GRNBoost2 (top 2% des poids).
    # GRNBoost2 utilise le gradient boosting pour prédire l'expression d'un
    # gène à partir de tous les autres. Les "TF" et "target" dans GRNBoost2
    # ne sont pas nécessairement des TF biologiques — c'est juste la nomenclature
    # du modèle (prédicteur → prédit). On traite ces arêtes comme non dirigées.
    # Feature = importance GRNBoost2 normalisée par le max.
    coexpr_src, coexpr_dst, coexpr_w = [], [], []
    coexpr_pairs = set()
    # V4.2 : en mode differential, edge_attr = 6 colonnes (option A).
    _COEXPR_DIM = 6 if COEXPR_DIFFERENTIAL else 1
    if MODULES["use_coexpr"]:
        for _, row in adjacencies_filtered.iterrows():
            g1, g2 = str(row["TF"]), str(row["target"])
            if g1 in gene_to_idx and g2 in gene_to_idx:
                i, j = gene_to_idx[g1], gene_to_idx[g2]
                pair = (min(i, j), max(i, j))
                if pair not in coexpr_pairs:
                    coexpr_pairs.add(pair)
                    coexpr_src.extend([i, j])
                    coexpr_dst.extend([j, i])
                    if COEXPR_DIFFERENTIAL:
                        # [imp_p4, imp_p16, delta, cat_shared, cat_p4, cat_p16]
                        feat = [
                            float(row["importance_p4_norm"]),
                            float(row["importance_p16_norm"]),
                            float(row["delta"]),
                            float(row["cat_shared"]),
                            float(row["cat_p4"]),
                            float(row["cat_p16"]),
                        ]
                        coexpr_w.append(feat)
                        coexpr_w.append(feat)  # bidirectionnel, même attr
                    else:
                        imp = float(row["importance"])
                        coexpr_w.extend([imp, imp])

    # Conversion en tenseurs PyG. Si aucune co-expression n'a été trouvée,
    # on crée un tenseur vide (0 arêtes) pour éviter les erreurs en aval.
    edge_index_coexpr = torch.tensor(
        [coexpr_src, coexpr_dst] if coexpr_src else [[], []], dtype=torch.long
    )
    if COEXPR_DIFFERENTIAL:
        # Pas de re-normalisation : imp_p4/imp_p16 déjà min-max normalisés
        # dans build_diff_coexpr.py ; delta ∈ [-1, 1] ; cat_* ∈ {0, 1}.
        coexpr_w_tensor = (torch.tensor(coexpr_w, dtype=torch.float)
                           if coexpr_w else torch.zeros((0, _COEXPR_DIM)))
    else:
        coexpr_w_tensor = (torch.tensor(coexpr_w, dtype=torch.float).unsqueeze(1)
                           if coexpr_w else torch.zeros((0, 1)))
        # Normalisation min-max par le max
        if coexpr_w_tensor.numel() > 0:
            coexpr_w_tensor = coexpr_w_tensor / (coexpr_w_tensor.max() + 1e-8)
    print(f"  coexpression : {len(coexpr_pairs)} paires ({len(coexpr_src)} arêtes)"
          + ("" if MODULES["use_coexpr"] else " [SKIP --no-coexpr]"))

    # ── 6f. HuMess : cocatalysis (A) + importance métabolique (B) ───────────────
    # HuMess (Métabolisme Humain par Échantillonnage de Solutions) est un pipeline
    # de modélisation métabolique qui produit deux informations complémentaires :
    #
    # PARTIE A — Arêtes "metabolic_cocatalysis" :
    #   Deux gènes sont reliés s'ils catalysent la même réaction métabolique
    #   (déduit des GPR rules = Gene-Protein-Reaction rules de CarveMe).
    #   Les GPR rules associent chaque réaction du modèle métabolique à une
    #   combinaison booléenne de gènes (ex : "(ALDOB or ALDOC) and GAPDH").
    #   Feature d'arête = [in_P4, in_P16] (binaire) : le lien peut exister
    #   dans P4 seul, P16 seul, ou les deux → le modèle apprend la
    #   CONDITION-SPÉCIFICITÉ du lien métabolique.
    #
    # PARTIE B — Features de noeud "gene" (ajoutées au vecteur de features) :
    #   Importance métabolique par gène = max(importance) sur les réactions
    #   catalysées par ce gène, calculée par Corner Sampling.
    #   On ajoute 4 features : imp_P4_z, imp_P16_z, imp_delta, has_humess.
    #   has_humess est un masque binaire (1 si le gène a des données HuMess)
    #   pour éviter que les 0 imputés pour les gènes absents du modèle
    #   métabolique soient interprétés comme "importance nulle" (ce serait
    #   un faux signal — l'absence de données ≠ absence d'importance).
    print("\n  HuMess (cocatalysis + importance)...")

    # Regex pour extraire les symboles de gène d'une règle GPR.
    # Exemple : "(ALDOB or ALDOC) and GAPDH" → {"ALDOB", "ALDOC", "GAPDH"}
    _gene_token_re = re.compile(r"[A-Za-z0-9_\-\.]+")
    _gpr_skip = {"and", "or", "AND", "OR", "And", "Or"}  # Mots-clés booléens à ignorer


    def _parse_gpr(gpr_str):
        """
        Extrait les symboles de gène d'une règle GPR (Gene-Protein-Reaction).
        Supprime les parenthèses, puis filtre les mots-clés booléens (and/or).
        Retourne un set de symboles de gènes.
        """
        cleaned = gpr_str.replace("(", " ").replace(")", " ")
        return {t for t in _gene_token_re.findall(cleaned) if t not in _gpr_skip}


    # --- PARTIE A : parse des GPR rules par condition (arêtes cocatalysis) ---
    # Pour chaque condition (P4, P16), on charge les GPR rules du modèle
    # métabolique CarveMe et on identifie quels gènes catalysent quelles réactions.
    # Format du fichier : réaction_id <TAB> GPR_rule
    # Exemple : "R_PFK → (PFKL or PFKM or PFKP)"
    # Modulaire : si --no-humess-edges, on n'ouvre même pas les fichiers GPR.
    reaction_to_genes = {}   # condition → {réaction : set(gènes)}
    gene_to_reactions = {c: {} for c in HUMESS_CONDITIONS}  # condition → {gène : set(réactions)}
    cocat_pair_flags = {}  # (index_gene_i, index_gene_j) → [in_P4, in_P16]
    cocat_src, cocat_dst, cocat_attr = [], [], []

    if MODULES["use_humess_edges"]:
        for cond in HUMESS_CONDITIONS:
            path = os.path.join(HUMESS_DIR, "models", cond, "stats", "carveme.gr-rules.tsv")
            cond_map = {}
            if not os.path.exists(path):
                print(f"    [warn] introuvable : {path}")
                reaction_to_genes[cond] = cond_map
                continue
            with open(path) as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    rxn, gpr = parts[0], parts[1]
                    genes = _parse_gpr(gpr)  # Extrait les gènes de la règle GPR
                    cond_map[rxn] = genes
                    # Index inverse : pour chaque gène, stocker les réactions qu'il catalyse
                    for g in genes:
                        gene_to_reactions[cond].setdefault(g, set()).add(rxn)
            reaction_to_genes[cond] = cond_map
            print(f"    {cond} : {len(cond_map)} réactions, "
                  f"{len(gene_to_reactions[cond])} gènes")

        # Construction des paires de cocatalyse (indices graphe).
        # Deux gènes sont co-catalytiques s'ils apparaissent dans la même GPR rule
        # (= catalysent la même réaction). Le flag [in_P4, in_P16] indique dans
        # quelle(s) condition(s) ce lien existe. Exemple :
        #   [1, 1] = la réaction existe dans les deux conditions
        #   [1, 0] = la réaction n'existe que dans P4 (perdue en sénescence)
        #   [0, 1] = la réaction apparaît en P16 (gain métabolique en sénescence)
        for cond_idx, cond in enumerate(HUMESS_CONDITIONS):
            for rxn, genes in reaction_to_genes[cond].items():
                # Convertir les symboles en indices du graphe (ignorer les gènes absents)
                idx_list = sorted({gene_to_idx[g] for g in genes if g in gene_to_idx})
                # Toutes les paires (graphe complet intra-réaction)
                for a in range(len(idx_list)):
                    for b in range(a + 1, len(idx_list)):
                        pair = (idx_list[a], idx_list[b])
                        flags = cocat_pair_flags.setdefault(pair, [0.0, 0.0])
                        flags[cond_idx] = 1.0  # Marquer la condition dans le flag

        # Conversion en arêtes bidirectionnelles avec features [in_P4, in_P16]
        for (i, j), flags in cocat_pair_flags.items():
            cocat_src.extend([i, j])       # Bidirectionnel
            cocat_dst.extend([j, i])
            cocat_attr.append(flags)       # Même flags dans les deux sens
            cocat_attr.append(flags)
    else:
        print("    cocatalysis : SKIP (--no-humess-edges)")

    edge_index_cocat = torch.tensor(
        [cocat_src, cocat_dst] if cocat_src else [[], []], dtype=torch.long
    )
    edge_attr_cocat = (torch.tensor(cocat_attr, dtype=torch.float)
                       if cocat_attr else torch.zeros((0, 2)))
    if MODULES["use_humess_edges"]:
        print(f"    cocatalysis : {len(cocat_pair_flags)} paires "
              f"({edge_index_cocat.shape[1]} arêtes, attr=[in_P4, in_P16])")

    # --- PARTIE B : importance métabolique par gène ---
    # Corner Sampling (CS) estime l'importance de chaque gène dans le modèle
    # métabolique en mesurant l'impact de sa suppression sur l'espace des flux.
    # Un gène peut catalyser plusieurs réactions → on prend le MAX d'importance
    # (le gène est "aussi important que sa réaction la plus importante").
    gene_importance = {c: np.zeros(n_genes, dtype=np.float32) for c in HUMESS_CONDITIONS}
    gene_in_model = {c: np.zeros(n_genes, dtype=np.float32) for c in HUMESS_CONDITIONS}

    if MODULES["use_humess_features"]:
        for cond in HUMESS_CONDITIONS:
            path = os.path.join(HUMESS_DIR, "models", cond, "cs",
                                f"cs_gene_to_importance_{cond}.tsv")
            if not os.path.exists(path):
                print(f"    [warn] introuvable : {path}")
                continue
            df_imp = pd.read_csv(path, sep="\t")
            # Format : symbol, bigg (réaction), importance (une ligne par (gène, réaction))
            df_imp = df_imp.dropna(subset=["symbol", "importance"])
            # Max importance par symbole : agrège les réactions → 1 valeur par gène
            agg = df_imp.groupby("symbol")["importance"].max()
            for sym, val in agg.items():
                if sym in gene_to_idx:
                    gene_importance[cond][gene_to_idx[sym]] = float(val)
                    gene_in_model[cond][gene_to_idx[sym]] = 1.0  # Marquer comme "a des données HuMess"
    else:
        print("    HuMess gene importance : SKIP (--no-humess-features)")

    # Transformation log1p + z-score par condition.
    # log1p : compresse les valeurs extrêmes (l'importance brute peut varier
    #   sur plusieurs ordres de grandeur).
    # z-score : centre et réduit, calculé UNIQUEMENT sur les gènes présents
    #   dans le modèle métabolique (mask). Les gènes absents restent à 0.
    # Cela évite que les gènes sans données HuMess (la majorité) polluent
    # la moyenne et l'écart-type du z-score.
    def _log1p_zscore(arr, mask):
        """log1p + z-score sur les éléments masqués ; les autres restent à 0."""
        out = np.zeros_like(arr)
        if mask.sum() < 2:
            return out
        vals = np.log1p(arr[mask.astype(bool)])
        mu, sd = vals.mean(), vals.std() + 1e-8
        out[mask.astype(bool)] = (vals - mu) / sd
        return out

    # imp_P4_z, imp_P16_z : importance métabolique z-scorée par condition. Clés
    # génériques = HUMESS_CONDITIONS (HUVEC : P4,P16 ; bulk : pro,sen). Les NOMS de
    # features (imp_P4/imp_P16/imp_delta) restent des labels internes (cf. GENE_
    # FEATURE_FLAGS) : imp_P4 = baseline (cond[0]), imp_P16 = état avancé (cond[-1]).
    _hc0, _hc1 = HUMESS_CONDITIONS[0], HUMESS_CONDITIONS[-1]
    imp_P4_z = _log1p_zscore(gene_importance[_hc0], gene_in_model[_hc0])
    imp_P16_z = _log1p_zscore(gene_importance[_hc1], gene_in_model[_hc1])
    # imp_delta : différence cond[-1] - cond[0], positif = gène plus important
    # métaboliquement à l'état avancé. Signal de REMODELAGE MÉTABOLIQUE.
    imp_delta = imp_P16_z - imp_P4_z
    # has_humess : masque binaire (1 si le gène apparaît dans le modèle métabolique
    # d'au moins une condition). Distingue "importance=0 car absent" vs "présent".
    has_humess = ((gene_in_model[_hc0] + gene_in_model[_hc1]) > 0).astype(np.float32)

    print(f"    importance  : {_hc0}={int(gene_in_model[_hc0].sum())} gènes, "
          f"{_hc1}={int(gene_in_model[_hc1].sum())} gènes, "
          f"has_humess={int(has_humess.sum())}/{n_genes}")

    # ── 6g. OmniPath (V4) : signaling dirigé signé + TF curé ────────────────────
    # DEUX nouveaux edge_types optionnels :
    #
    #   ("gene", "signaling", "gene")  — kinase-substrat OmniPath + SIGNOR causal
    #       directionnel, edge_attr = [score, sign∈{−1,0,+1}].
    #       Active une couche de message passing CAUSALE (asymétrique) où
    #       l'attention GAT peut apprendre que activation et inhibition portent
    #       des messages de polarité différente. Réf : Türei et al. Nat Commun
    #       2021 ; Lo Surdo et al. NAR 2023 (SIGNOR 3.0).
    #
    #   ("gene", "tf_curated", "gene") + reverse  — CollecTRI (fallback DoRothEA)
    #       TF→cible curé sur ~1186 TFs (vs ~50 pySCENIC), edge_attr = [score, sign].
    #       Co-existe avec "regulates" (pySCENIC, HUVEC-spécifique inféré du
    #       scRNA-seq) — laisse le GNN apprendre les deux sources distinctement
    #       (option (c) du design V4). Réf : Müller-Dott et al. Genome Biol 2023.
    #
    # Les deux sources sont chargées depuis un cache TSV pré-téléchargé
    # (`scripts/cache_omnipath.py` à lancer sur le frontal). En cas d'absence
    # de cache et de --omnipath-download-if-missing OFF, on saute proprement
    # l'arête sans crasher le run (modularité préservée).
    #
    # Default OFF : à activer explicitement via --use-omnipath-signaling /
    # --use-omnipath-tf-curated pour les ablations V4.
    op_sig_src = op_sig_dst = np.array([], dtype=np.int64)
    op_sig_attr = np.zeros((0, 2), dtype=np.float32)
    op_tf_src  = op_tf_dst  = np.array([], dtype=np.int64)
    op_tf_attr = np.zeros((0, 2), dtype=np.float32)

    if MODULES["use_omnipath_signaling"] or MODULES["use_omnipath_tf_curated"]:
        print("\n  OmniPath (V4)…")
        # Idem section 2.5 : force offline si download non autorisé (compute node).
        # Idempotent (déjà fait en 2.5 si --include-omnipath-genes, no-op sinon).
        if not CLI_ARGS.omnipath_download_if_missing:
            os.environ["GNN_OMNIPATH_OFFLINE"] = "1"
        try:
            from omnipath_integration import (
                load_signaling_directed,
                load_collectri_tf_target,
                load_signed_ppi_signor,
                merge_signed_directed,
                silence_omnipath_logging,
            )
            _OPI = True
            # Coupe le bruit quand on sait qu'on ne télécharge pas. Les compute
            # nodes Nautilus déclenchent des centaines de "WARNING:root:Failed
            # to download" sinon, à cause des metadata pre-fetches d'omnipath-py.
            if not CLI_ARGS.omnipath_download_if_missing:
                silence_omnipath_logging()
        except ImportError as _e:
            print(f"    [warn] import omnipath_integration KO ({_e}) — "
                  f"aucune arête OmniPath ne sera ajoutée.")
            _OPI = False

        if _OPI:
            _opi_kwargs = dict(
                cache_dir=OMNIPATH_CACHE_DIR,
                available_genes=available_set,
                gene_to_idx=gene_to_idx,
                download_if_missing=CLI_ARGS.omnipath_download_if_missing,
            )

            if MODULES["use_omnipath_signaling"]:
                # Fusion kinase-substrat OmniPath + PPI causal SIGNOR — même
                # sémantique sémantique (lien causal signé), même edge_type.
                sig_kin = load_signaling_directed(**_opi_kwargs)
                sig_sgn = load_signed_ppi_signor(**_opi_kwargs)
                op_sig_src, op_sig_dst, op_sig_attr = merge_signed_directed(
                    (sig_kin[0], sig_kin[1], sig_kin[2]),
                    (sig_sgn[0], sig_sgn[1], sig_sgn[2]),
                )
                n_pos = int((op_sig_attr[:, 1] > 0).sum())
                n_neg = int((op_sig_attr[:, 1] < 0).sum())
                n_neu = int((op_sig_attr[:, 1] == 0).sum())
                print(f"    signaling (kinase+SIGNOR) : {len(op_sig_src)} arêtes "
                      f"[+:{n_pos} −:{n_neg} 0:{n_neu}]")

            if MODULES["use_omnipath_tf_curated"]:
                tf_op = load_collectri_tf_target(**_opi_kwargs)
                op_tf_src, op_tf_dst, op_tf_attr = tf_op[0], tf_op[1], tf_op[2]
                n_pos = int((op_tf_attr[:, 1] > 0).sum())
                n_neg = int((op_tf_attr[:, 1] < 0).sum())
                n_neu = int((op_tf_attr[:, 1] == 0).sum())
                n_tfs = len(set(op_tf_src.tolist())) if op_tf_src.size > 0 else 0
                print(f"    tf_curated (CollecTRI)    : {len(op_tf_src)} arêtes, "
                      f"{n_tfs} TFs uniques [+:{n_pos} −:{n_neg} 0:{n_neu}]")

    # Conversion en tenseurs PyG (vides si désactivé ou cache absent)
    edge_index_signaling = (torch.tensor([op_sig_src.tolist(), op_sig_dst.tolist()],
                                         dtype=torch.long)
                            if op_sig_src.size > 0
                            else torch.zeros((2, 0), dtype=torch.long))
    edge_attr_signaling = (torch.tensor(op_sig_attr, dtype=torch.float)
                           if op_sig_attr.size > 0
                           else torch.zeros((0, 2), dtype=torch.float))
    edge_index_tf_curated = (torch.tensor([op_tf_src.tolist(), op_tf_dst.tolist()],
                                          dtype=torch.long)
                             if op_tf_src.size > 0
                             else torch.zeros((2, 0), dtype=torch.long))
    edge_attr_tf_curated = (torch.tensor(op_tf_attr, dtype=torch.float)
                            if op_tf_attr.size > 0
                            else torch.zeros((0, 2), dtype=torch.float))
    # Reverse pour message passing arrière (target → TF), même attr.
    edge_index_tf_curated_by = (torch.tensor([op_tf_dst.tolist(),
                                              op_tf_src.tolist()],
                                             dtype=torch.long)
                                if op_tf_src.size > 0
                                else torch.zeros((2, 0), dtype=torch.long))

    # ── V4.2 : Reactome FI signé ─────────────────────────────────────────────────
    # Reactome Functional Interactions (Wu 2010 Genome Biol). Fichier
    # FIsInGene_*_with_annotations.txt : Gene1, Gene2, Annotation, Direction,
    # Score. La colonne Direction encode le signe : '|' = inhibition, '>'/'<'
    # = activation/direction, '-'/'<->' = non signé. On exclut les FI
    # 'predicted' (computationnels non curés). edge_attr = [score, sign].
    # Apport mesuré (audit 2026-05-12) : ~45k arêtes signées NOUVELLES (75%
    # absentes de PPI/SIGNOR/CollecTRI). Cf. §14bis.6quater du rapport.
    reactome_fi_src, reactome_fi_dst, reactome_fi_attr = [], [], []
    if MODULES["use_reactome_fi"]:
        _fi_path = CLI_ARGS.reactome_fi_file
        if not os.path.isabs(_fi_path):
            # Reactome FI lives in the data tree → resolve under DATA_DIR so a single
            # GNN_DATA_DIR (e.g. LAB-DATA clone) covers it like PPI/databases. Strip a
            # leading "data/" so it works whether DATA_DIR ends in /data or not.
            _rel = _fi_path[5:] if _fi_path.startswith("data/") else _fi_path
            _fi_path = os.path.join(DATA_DIR, _rel)
        if not os.path.exists(_fi_path):
            print(f"  [warn] Reactome FI : fichier absent ({_fi_path}) — "
                  f"edge_type 'reactome_fi' vide. Télécharger via "
                  f"scripts/cache_reactome_fi.sh")
        else:
            _fi = pd.read_csv(_fi_path, sep="\t")
            _n_raw = len(_fi)
            # Exclure les FI purement prédites (non curées)
            _fi = _fi[~_fi["Annotation"].astype(str).str.contains(
                "predicted", case=False, na=False)]

            def _fi_sign(d: str) -> float:
                d = str(d)
                if "|" in d:                    # notation inhibition Reactome FI
                    return -1.0
                if ">" in d or "<" in d:        # direction d'activation
                    return 1.0
                return 0.0

            _seen_fi = set()
            for _, r in _fi.iterrows():
                g1, g2 = str(r["Gene1"]), str(r["Gene2"])
                if g1 in gene_to_idx and g2 in gene_to_idx:
                    i, j = gene_to_idx[g1], gene_to_idx[g2]
                    pair = (min(i, j), max(i, j))
                    if pair in _seen_fi:
                        continue
                    _seen_fi.add(pair)
                    sign = _fi_sign(r["Direction"])
                    score = float(r["Score"]) if not pd.isna(r["Score"]) else 1.0
                    # Bidirectionnel (FI = interaction fonctionnelle, on
                    # propage dans les deux sens comme PPI/coexpr)
                    reactome_fi_src.extend([i, j])
                    reactome_fi_dst.extend([j, i])
                    reactome_fi_attr.extend([[score, sign], [score, sign]])
            n_fi_signed = sum(1 for a in reactome_fi_attr if a[1] != 0)
            print(f"  Reactome FI : {_n_raw} brut → {len(_seen_fi)} paires "
                  f"curées dans le graphe ({n_fi_signed//2} signées)")

    edge_index_reactome_fi = (torch.tensor([reactome_fi_src, reactome_fi_dst],
                                           dtype=torch.long)
                              if reactome_fi_src
                              else torch.zeros((2, 0), dtype=torch.long))
    edge_attr_reactome_fi = (torch.tensor(reactome_fi_attr, dtype=torch.float)
                             if reactome_fi_attr
                             else torch.zeros((0, 2), dtype=torch.float))

    # ── V4.2 : déduplication PPI vs arêtes signées+orientées ─────────────────────
    # Pour une paire (a,b), si une arête SIGNÉE ORIENTÉE existe (signaling /
    # tf_curated / reactome_fi avec sign≠0), l'arête PPI non-signée
    # non-orientée est redondante : elle gonfle le compte et son message
    # symétrique s'ajoute (agrégation HeteroConv-sum) au message signé →
    # dilution partielle (cf. §14bis.6bis : PPI domine déjà ‖h‖).
    #
    # `--dedup-ppi-signed {off,remove,annotate}` (DÉFAUT off — comportement
    # inchangé). Le DIAGNOSTIC (combien d'arêtes/gènes seraient touchés)
    # est TOUJOURS calculé et loggé, même en mode off, pour décider sur
    # chiffres. Cf. §14bis.6quaterdecies. NB : bénéfice « bien orienter le
    # message » seulement PARTIEL en V4.2 (le signe n'entre qu'en
    # attention, design A) ; bénéfice PLEIN avec V5 (SignedGATConv message
    # + BilinearSignedDecoder). Les deux sont décorrélés mais
    # complémentaires.
    def _signed_pair_set(*eis_attrs) -> set[tuple[int, int]]:
        """Paires (min,max) couvertes par une arête signée (sign≠0)."""
        s: set[tuple[int, int]] = set()
        for ei, attr in eis_attrs:
            if ei.numel() == 0:
                continue
            if attr is not None and attr.numel() > 0 and attr.dim() == 2 \
                    and attr.shape[1] >= 2:
                sign = attr[:, 1]
                for k in range(ei.shape[1]):
                    if float(sign[k]) != 0.0:
                        a, b = int(ei[0, k]), int(ei[1, k])
                        s.add((min(a, b), max(a, b)))
            else:
                for a, b in zip(ei[0].tolist(), ei[1].tolist()):
                    s.add((min(a, b), max(a, b)))
        return s


    _signed_pairs = _signed_pair_set(
        (edge_index_signaling, edge_attr_signaling),
        (edge_index_tf_curated, edge_attr_tf_curated),
        (edge_index_reactome_fi, edge_attr_reactome_fi),
    )
    # Diagnostic : combien d'arêtes PPI redondantes ? combien de gènes
    # perdraient TOUTES leurs arêtes PPI si on dédupliquait ?
    _n_ppi_before = int(edge_index_ppi.shape[1]) if edge_index_ppi.numel() else 0
    _ppi_redundant_mask = None
    if _n_ppi_before > 0 and _signed_pairs:
        import numpy as _np
        _src = edge_index_ppi[0].numpy()
        _dst = edge_index_ppi[1].numpy()
        _ppi_redundant_mask = _np.array([
            (min(int(a), int(b)), max(int(a), int(b))) in _signed_pairs
            for a, b in zip(_src, _dst)
        ])
        _n_redundant = int(_ppi_redundant_mask.sum())
        # Gènes qui n'ont QUE des arêtes PPI redondantes (perdraient tout PPI)
        _deg_all = _np.zeros(n_genes, dtype=_np.int64)
        _deg_keep = _np.zeros(n_genes, dtype=_np.int64)
        for i, (a, b) in enumerate(zip(_src, _dst)):
            _deg_all[int(a)] += 1
            if not _ppi_redundant_mask[i]:
                _deg_keep[int(a)] += 1
        _n_genes_lose_ppi = int(((_deg_all > 0) & (_deg_keep == 0)).sum())
        _n_genes_with_ppi = int((_deg_all > 0).sum())
        print(f"\n  [dedup-ppi diag] PPI redondant (paire signée existante) : "
              f"{_n_redundant}/{_n_ppi_before} arêtes "
              f"({100*_n_redundant/_n_ppi_before:.1f}%)")
        print(f"  [dedup-ppi diag] gènes perdant TOUTES leurs arêtes PPI si "
              f"dédup : {_n_genes_lose_ppi}/{_n_genes_with_ppi} "
              f"({100*_n_genes_lose_ppi/max(1,_n_genes_with_ppi):.1f}%)")
        _dedup_mode = getattr(CLI_ARGS, "dedup_ppi_signed", "off")
        if _dedup_mode == "remove":
            _keep = ~_ppi_redundant_mask
            edge_index_ppi = edge_index_ppi[:, _keep.tolist()]
            if edge_attr_ppi.numel() > 0:
                edge_attr_ppi = edge_attr_ppi[_keep.tolist()]
            print(f"  [dedup-ppi] mode=remove : PPI {_n_ppi_before} → "
                  f"{edge_index_ppi.shape[1]} arêtes")
        elif _dedup_mode == "annotate":
            # edge_dim 1→2 : ajoute une colonne flag (1 si paire signée).
            _flag = torch.tensor(_ppi_redundant_mask.astype("float32")).unsqueeze(1)
            if edge_attr_ppi.numel() > 0:
                edge_attr_ppi = torch.cat([edge_attr_ppi, _flag], dim=1)
            else:
                edge_attr_ppi = _flag
            print(f"  [dedup-ppi] mode=annotate : edge_attr_ppi → dim "
                  f"{edge_attr_ppi.shape[1]} (col 1 = has_signed_counterpart)")
        else:
            print(f"  [dedup-ppi] mode=off : aucune modif (diagnostic seul)")

    # ── Finaliser les features de noeuds gene avec les degrés ────────────────────
    # Les features de degré (nombre de voisins) sont calculées APRÈS la
    # construction des arêtes. Elles capturent la "centralité" de chaque gène
    # dans les différents réseaux. Un gène hub PPI avec beaucoup d'interactions
    # protéine-protéine aura un ppi_degree élevé.
    # Feature 3 : ppi_degree — nombre de voisins PPI (normalisé par le max)
    ppi_degree = np.zeros(n_genes, dtype=np.float32)
    for idx in ppi_src:
        ppi_degree[idx] += 1
    ppi_degree_norm = ppi_degree / (ppi_degree.max() + 1e-8)

    # Feature 4 : reg_degree — nombre de liens de régulation (TF→cible + cible→TF)
    # On compte les deux directions : un TF qui régule 100 cibles aura un haut
    # degré, mais une cible régulée par 5 TFs aussi.
    reg_degree = np.zeros(n_genes, dtype=np.float32)
    for idx in reg_src + reg_dst:
        reg_degree[idx] += 1
    reg_degree_norm = reg_degree / (reg_degree.max() + 1e-8)

    # ASSEMBLAGE FINAL DES FEATURES DE NOEUDS "gene" (modulaire — V3.5+) :
    # Chaque colonne n'est incluse que si GENE_FEATURE_FLAGS[name] est True
    # (dépend (a) de la source activée, (b) de --exclude-features).
    # Ordre canonique conservé : is_tf, variance, ppi_degree, reg_degree,
    # imp_P4, imp_P16, imp_delta, has_humess. AUCUNE feature ne contient
    # log2FC, padj, ou delta_pct (anti-circularité).
    _FEATURE_VECTORS = [
        ("is_tf",      is_tf),
        ("variance",   variance_norm),
        ("ppi_degree", ppi_degree_norm),
        ("reg_degree", reg_degree_norm),
        ("imp_P4",     imp_P4_z),
        ("imp_P16",    imp_P16_z),
        ("imp_delta",  imp_delta),
        ("has_humess", has_humess),
    ]
    _active_feature_names = [n for n, _ in _FEATURE_VECTORS if GENE_FEATURE_FLAGS[n]]
    _active_feature_arrays = [v for n, v in _FEATURE_VECTORS if GENE_FEATURE_FLAGS[n]]

    # Garantie d'au moins une feature : si tout est exclu, on retombe sur un
    # vecteur de biais constant pour ne pas casser nn.Linear(0, hidden).
    if not _active_feature_arrays:
        print("    [warn] toutes les features exclues — fallback bias constant")
        _active_feature_names = ["bias"]
        _active_feature_arrays = [np.ones(n_genes, dtype=np.float32)]

    gene_features = torch.tensor(
        np.column_stack(_active_feature_arrays),
        dtype=torch.float,
    )
    print(f"\n  Gene features : {gene_features.shape}")
    print(f"    actives : {_active_feature_names}")
    print(f"    PAS de log2FC, padj, delta_pct (circularité supprimée)")

    # =============================================================================
    # 7. ASSEMBLAGE DU GRAPHE HÉTÉROGÈNE
    # =============================================================================
    # On assemble tous les noeuds et arêtes dans un objet HeteroData de PyG.
    # HeteroData est le format standard de PyTorch Geometric pour les graphes
    # hétérogènes (plusieurs types de noeuds et d'arêtes). Chaque type d'arête
    # est identifié par un triplet (source_type, relation, dest_type).
    #
    # Structure finale du graphe :
    #   Noeuds :
    #     "gene"       : n_genes noeuds, 8 features topologiques
    #     "cell_group" : 5 noeuds (P4 + 4 clusters P16), 3 features
    #   Arêtes (jusqu'à 8 types) :
    #     cell_group → gene (expresses) : 5 × n_genes, 7 features
    #     gene → cell_group (expressed_in) : reverse, mêmes features
    #     gene ↔ gene (ppi) : STRING high-confidence, 1 feature
    #     gene ↔ gene (same_pathway) : REACTOME, pas de features
    #     gene → gene (regulates) : pySCENIC TF→cible, 1 feature
    #     gene → gene (regulated_by) : reverse de regulates, mêmes features
    #     gene ↔ gene (coexpression) : GRNBoost2 top 2%, 1 feature
    #     gene ↔ gene (metabolic_cocatalysis) : HuMess GPR, 2 features
    print("\n" + "=" * 70)
    print("7. Assemblage du graphe hétérogène")
    print("=" * 70)

    # HeteroData stocke les features et arêtes indexées par type
    data = HeteroData()

    # --- Noeuds ---
    data["gene"].x = gene_features            # (n_genes, 8)
    data["gene"].num_nodes = n_genes
    data["cell_group"].x = cell_group_features  # (5, 3)
    data["cell_group"].num_nodes = len(CELL_GROUPS)

    # --- Arêtes bipartites cell_group ↔ gene (conditionnelles V3.5+) ---
    # "expresses" : cell_group → gene (le groupe cellulaire exprime le gène)
    # "expressed_in" : gene → cell_group (reverse, mêmes features)
    # Skip total si --no-cell-group-edges.
    if edge_index_expresses.numel() > 0:
        data["cell_group", "expresses", "gene"].edge_index = edge_index_expresses
        data["cell_group", "expresses", "gene"].edge_attr = edge_attr_expresses
        data["gene", "expressed_in", "cell_group"].edge_index = torch.stack([
            edge_index_expresses[1], edge_index_expresses[0]  # Inverse src/dst
        ])
        data["gene", "expressed_in", "cell_group"].edge_attr = edge_attr_expresses

    # --- Arêtes gene ↔ gene (toutes conditionnelles V3.5+) ---
    if edge_index_ppi.numel() > 0:
        data["gene", "ppi", "gene"].edge_index = edge_index_ppi
        data["gene", "ppi", "gene"].edge_attr = edge_attr_ppi
    if edge_index_pathway.numel() > 0:
        data["gene", "same_pathway", "gene"].edge_index = edge_index_pathway
        # same_pathway n'a pas d'edge_attr (existence binaire suffit)
    if edge_index_regulates.numel() > 0:
        data["gene", "regulates", "gene"].edge_index = edge_index_regulates
        data["gene", "regulates", "gene"].edge_attr = edge_attr_regulates
        data["gene", "regulated_by", "gene"].edge_index = edge_index_regulated_by
        data["gene", "regulated_by", "gene"].edge_attr = edge_attr_regulates  # Mêmes poids
    if edge_index_coexpr.numel() > 0:
        data["gene", "coexpression", "gene"].edge_index = edge_index_coexpr
        data["gene", "coexpression", "gene"].edge_attr = coexpr_w_tensor
    if edge_index_cocat.numel() > 0:
        data["gene", "metabolic_cocatalysis", "gene"].edge_index = edge_index_cocat
        data["gene", "metabolic_cocatalysis", "gene"].edge_attr = edge_attr_cocat
    # OmniPath V4 — signaling dirigé signé (kinase-substrat + SIGNOR causal)
    if edge_index_signaling.numel() > 0:
        data["gene", "signaling", "gene"].edge_index = edge_index_signaling
        data["gene", "signaling", "gene"].edge_attr = edge_attr_signaling
    # OmniPath V4 — TF→cible curé (CollecTRI), edge_type SÉPARÉ de "regulates"
    # pour que le GNN apprenne distinctement les deux sources (option (c) du
    # design : pySCENIC HUVEC-spécifique vs CollecTRI méta-curation).
    if edge_index_tf_curated.numel() > 0:
        data["gene", "tf_curated", "gene"].edge_index = edge_index_tf_curated
        data["gene", "tf_curated", "gene"].edge_attr = edge_attr_tf_curated
        data["gene", "tf_curated_by", "gene"].edge_index = edge_index_tf_curated_by
        data["gene", "tf_curated_by", "gene"].edge_attr = edge_attr_tf_curated
    # V4.2 — Reactome FI signé (edge_type séparé, ablation-able via --no-reactome-fi)
    if edge_index_reactome_fi.numel() > 0:
        data["gene", "reactome_fi", "gene"].edge_index = edge_index_reactome_fi
        data["gene", "reactome_fi", "gene"].edge_attr = edge_attr_reactome_fi

    print(f"  Noeuds gene       : {n_genes} (features={gene_features.shape[1]})")
    print(f"  Noeuds cell_group : {len(CELL_GROUPS)}")
    print(f"  Arêtes ppi        : {edge_index_ppi.shape[1]}")
    print(f"  Arêtes pathway    : {edge_index_pathway.shape[1]}")
    print(f"  Arêtes regulates  : {edge_index_regulates.shape[1]}")
    print(f"  Arêtes coexpr     : {edge_index_coexpr.shape[1] if edge_index_coexpr.numel() > 0 else 0}")
    print(f"  Arêtes cocat      : {edge_index_cocat.shape[1] if edge_index_cocat.numel() > 0 else 0}")
    print(f"  Arêtes signaling  : {edge_index_signaling.shape[1]}"
          + ("" if MODULES["use_omnipath_signaling"] else " [SKIP --no-omnipath-signaling]"))
    print(f"  Arêtes tf_curated : {edge_index_tf_curated.shape[1]}"
          + ("" if MODULES["use_omnipath_tf_curated"] else " [SKIP --no-omnipath-tf-curated]"))
    print(f"  Arêtes reactome_fi: {edge_index_reactome_fi.shape[1]}"
          + ("" if MODULES["use_reactome_fi"] else " [SKIP --no-reactome-fi]"))
    print(f"  Coexpr mode       : {COEXPR_MODE}"
          + (f" (edge_dim={_COEXPR_DIM})" if MODULES["use_coexpr"] else ""))

    # Sauvegarde du graphe complet pour réutilisation (perturbations, etc.)
    torch.save(data, os.path.join(OUT_DIR, "hetero_graph_vgae.pt"))

    # ----- mise en cache du build (sections 1-7) pour --reuse-graph -----
    # Robustesse : on saute les variables non-picklables (ex. handle de fichier
    # temporaire `f`/`fh` réutilisé en aval — non requis au reload) en testant
    # chaque valeur ; et on écrit dans un .tmp puis rename atomique pour ne
    # JAMAIS laisser un cache partiel/corrompu en cas d'échec.
    try:
        os.makedirs(os.path.dirname(_GRAPH_CACHE) or ".", exist_ok=True)
        _bundle = {"_sig": _SIG}
        _skipped = []
        for _k in _CACHE_VARS:
            if _k not in globals():
                continue
            _v = globals()[_k]
            try:
                pickle.dumps(_v, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                _skipped.append(_k)  # non-picklable & non requis au reload
                continue
            _bundle[_k] = _v
        _tmp = _GRAPH_CACHE + ".tmp"
        with open(_tmp, "wb") as _fh:
            pickle.dump(_bundle, _fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(_tmp, _GRAPH_CACHE)  # rename atomique
        print(f"[reuse-graph] cache ecrit -> {_GRAPH_CACHE} "
              f"(n_genes={globals().get('n_genes')}, {len(_bundle)-1} vars, sig={_SIG[:8]})"
              + (f" [skip non-picklables: {_skipped}]" if _skipped else ""))
    except Exception as _e:
        print(f"[reuse-graph] echec ecriture cache ({_e}) -- non bloquant")
        try:
            os.path.exists(_GRAPH_CACHE + ".tmp") and os.remove(_GRAPH_CACHE + ".tmp")
        except OSError:
            pass

# =============================================================================
# 8. MODÈLE VGAE
# =============================================================================
# Le VGAE (Variational Graph AutoEncoder) est un modèle génératif qui apprend
# des représentations (embeddings) de gènes en reconstruisant la structure
# du graphe. Il a deux composants :
#
#   1. ENCODER (HeteroGNN → μ, log(σ²)) :
#      Empile N couches de GATConv (Graph Attention Network) sur le graphe
#      hétérogène. Chaque couche agrège les messages des voisins avec un
#      mécanisme d'attention appris. Après N couches, chaque gène a un
#      vecteur h ∈ R^hidden qui encode son contexte local dans le graphe.
#      Deux têtes linéaires projettent h vers μ ∈ R^latent et log(σ²) ∈ R^latent.
#
#   2. DECODER (cosine similarity → probabilité d'arête) :
#      Prédit P(arête i↔j) = σ(τ · cos(z_i, z_j)), où z est échantillonné
#      du posterior q(z|x) = N(μ, diag(σ²)). Le décodeur est NON paramétrique
#      (pas de poids appris, seulement le scalaire τ).
#
#   LOSS = reconstruction_loss + β · KL_divergence
#      - Reconstruction : BCE entre les logits du décodeur et les labels
#        (1 pour les arêtes réelles, 0 pour les paires non connectées)
#      - KL : D_KL(q(z|x) || N(0,I)) régularise l'espace latent
#      - β : coefficient d'annealing (0 → KL_BETA_MAX)
# =============================================================================
print("\n" + "=" * 70)
print("8. Modèle VGAE")
print("=" * 70)


# V5 (TIER 1c) — edge_types pour lesquels le `sign` ∈ {-1, 0, +1} est
# présent en colonne 1 de edge_attr (convention V4 : edge_attr=[score, sign]).
# Routés vers SignedGATConv si --signed-message, et vers le canal bilinéaire
# signé si --signed-decoder. Les autres edge_types restent unsigned
# (PPI/coexpr/REACTOME : décodeur cosinus, GATConv standard).
SIGNED_EDGE_TYPES: set = {
    ("gene", "signaling", "gene"),
    ("gene", "tf_curated", "gene"),
    ("gene", "tf_curated_by", "gene"),
    ("gene", "reactome_fi", "gene"),
}


class SignedGATConv(GATConv):
    """V5 (TIER 1c.2) — GATConv qui multiplie chaque message par son sign d'arête.

    Extension du design A (V4 edge_attr=[score, sign] consommé par
    l'attention via edge_dim=2) : force le `sign` à influencer aussi
    le MESSAGE. Pour une arête `sign=-1`, propage `-W·h_j` au lieu de
    `+W·h_j` — sémantique « inhibition » explicitement codée dans la
    mise à jour d'embedding.

    Ref : Derr et al. 2018 *ICDM* SGCN §3.2 (balance theory pour
    message-passing signé), adapté au cadre GAT (PyG).

    Sign lu depuis `edge_attr[:, sign_col]` (convention V4 : colonne 1).
    Backward-compat : si edge_attr None ou sign_col hors borne, retombe
    sur GATConv standard. Source de vérité : src/gnn/_vgae_model.py:171.
    """

    def __init__(self, *args, sign_col: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.sign_col = sign_col
        # Buffer transient (forward → message). PyG ne passe pas edge_attr
        # à message() — on doit le « stash » nous-mêmes.
        self._current_edge_sign = None

    def forward(self, x, edge_index, edge_attr=None, size=None,
                return_attention_weights=None):
        if edge_attr is not None and edge_attr.ndim >= 2 \
                and edge_attr.shape[1] > self.sign_col:
            self._current_edge_sign = edge_attr[:, self.sign_col]
        else:
            self._current_edge_sign = None
        return super().forward(x, edge_index, edge_attr=edge_attr,
                               size=size,
                               return_attention_weights=return_attention_weights)

    def message(self, x_j, alpha):
        out = super().message(x_j, alpha)
        sign = self._current_edge_sign
        if sign is None:
            return out
        scale = torch.where(sign == 0, torch.ones_like(sign), sign)
        scale = scale.view(-1, *([1] * (out.ndim - 1)))
        return out * scale


class BilinearSignedDecoder(nn.Module):
    """V5.3 (TIER 1c.3 + sub-espace) — décodeur bilinéaire 3 canaux avec tête signed.

    Architecture :
        z_signed = signed_proj(z)            # V5.3 — projection sub-espace
        logit_pos  = z_signed_i · W_+ · z_signed_j
        logit_neg  = z_signed_i · W_- · z_signed_j
        logit_zero = z_signed_i · W_0 · z_signed_j

    **V5.3 (§14bis.6unvicesies)** : la projection `signed_proj` filtre
    les gradients du `signed_aux_loss` avant qu'ils n'atteignent
    l'encodeur. Elle absorbe une partie de la contrainte signed, ce qui
    réduit la concurrence avec le décodeur cosinus principal (V5.2
    mesurait −0.022 AUC recon dûe à cette concurrence).

    Backward-compat V5.1/V5.2 : `signed_proj` init = identité + 0 bias
    ⇒ comportement initial identique à V5.2 (équivalence numérique exacte
    si signed_dim == latent_dim). Les checkpoints V5.2 se chargent avec
    `strict=False` + post-init signed_proj=identity.

    Source de vérité : src/gnn/_vgae_model.py:251.

    Ref : Liu et al. 2024 *NAR* SGAT-bilinear ; Yang et al. 2015 *ICLR*
    DistMult ; tête sub-espace = inspiration multi-task task heads.
    """

    def __init__(self, latent_dim: int, signed_dim: int | None = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.signed_dim = signed_dim if signed_dim is not None else latent_dim

        # V5.3 — projection latent → sous-espace signed.
        self.signed_proj = nn.Linear(latent_dim, self.signed_dim, bias=True)
        with torch.no_grad():
            if self.signed_dim == latent_dim:
                self.signed_proj.weight.copy_(torch.eye(latent_dim))
            else:
                init = torch.zeros(self.signed_dim, latent_dim)
                init[:self.signed_dim, :self.signed_dim] = torch.eye(self.signed_dim)
                self.signed_proj.weight.copy_(init)
            self.signed_proj.bias.zero_()

        # 3 matrices bilinéaires dans le sous-espace signed.
        eye = torch.eye(self.signed_dim)
        self.W_pos = nn.Parameter(eye + 0.01 * torch.randn(self.signed_dim, self.signed_dim))
        self.W_neg = nn.Parameter(eye + 0.01 * torch.randn(self.signed_dim, self.signed_dim))
        self.W_zero = nn.Parameter(eye + 0.01 * torch.randn(self.signed_dim, self.signed_dim))

    def _project(self, z: torch.Tensor) -> torch.Tensor:
        """V5.3 — projette `z` dans le sous-espace signed via signed_proj."""
        return self.signed_proj(z)

    def forward_signed(self, z: torch.Tensor, edge_index: torch.Tensor,
                       edge_sign: torch.Tensor) -> torch.Tensor:
        """Logits bilinéaires — TRAINING uniquement (V5.0 hérité).

        ⚠ NE PAS utiliser pour l'évaluation `AUC(activate vs inhibit)` :
        la sélection par sign donne déjà la réponse. La `signed_aux_loss`
        V5.1+ utilise `predict_sign_score()` (BCE contrastive).
        """
        z_signed = self._project(z)
        z_src = z_signed[edge_index[0]]
        z_dst = z_signed[edge_index[1]]
        logit_pos = (z_src @ self.W_pos * z_dst).sum(dim=-1)
        logit_neg = (z_src @ self.W_neg * z_dst).sum(dim=-1)
        logit_zero = (z_src @ self.W_zero * z_dst).sum(dim=-1)
        mask_pos = (edge_sign > 0).float()
        mask_neg = (edge_sign < 0).float()
        mask_zero = (edge_sign == 0).float()
        return mask_pos * logit_pos + mask_neg * logit_neg + mask_zero * logit_zero

    def predict_sign_score(self, z: torch.Tensor,
                           edge_index: torch.Tensor) -> torch.Tensor:
        """Score sign-agnostique pour évaluation AUC(activate vs inhibit).

        Retourne `logit_pos − logit_neg` SANS utiliser le sign cible.
        Score correct pour gate 1c.5 (Liu 2024 *NAR* SGAT-bilinear §3) et
        utilisé par la `signed_aux_loss` V5.1+.
        """
        z_signed = self._project(z)
        z_src = z_signed[edge_index[0]]
        z_dst = z_signed[edge_index[1]]
        logit_pos = (z_src @ self.W_pos * z_dst).sum(dim=-1)
        logit_neg = (z_src @ self.W_neg * z_dst).sum(dim=-1)
        return logit_pos - logit_neg

    def predict_signed_existence(self, z: torch.Tensor,
                                 edge_index: torch.Tensor) -> torch.Tensor:
        """V5.4 (decoder-split, §14bis.6duovicies) — score d'EXISTENCE bilinéaire.

        `logsumexp(logit_pos, logit_neg, logit_zero)` = enveloppe
        différentiable de `max` (Bishop 2006 §4.5) : reproduit `max`
        asymptotiquement quand un canal domine, mais diffuse le gradient
        sur les 3 canaux quand ils sont proches. Utilisé par
        `recon_loss_signed` quand `--decoder-split` : le bilinéaire prend
        en charge la reconstruction d'EXISTENCE des arêtes signées (le
        cosinus ne les voit plus → libéré des arêtes dirigées). Distinct
        de `predict_sign_score` (= logit_pos − logit_neg), qui porte le SIGNE.
        """
        z_signed = self._project(z)
        z_src = z_signed[edge_index[0]]
        z_dst = z_signed[edge_index[1]]
        logit_pos = (z_src @ self.W_pos * z_dst).sum(dim=-1)
        logit_neg = (z_src @ self.W_neg * z_dst).sum(dim=-1)
        logit_zero = (z_src @ self.W_zero * z_dst).sum(dim=-1)
        return torch.logsumexp(
            torch.stack([logit_pos, logit_neg, logit_zero], dim=-1), dim=-1)


class _ScaledConv(nn.Module):
    """Wrapper V4.2 : multiplie la sortie d'un conv par γ_t (scalaire fixe).

    Permet d'appliquer une pondération par edge_type AU NIVEAU DU MESSAGE
    (avant l'agrégation HeteroConv-sum), pour rééquilibrer le déséquilibre
    ‖h_PPI‖ ≫ ‖h_signaling‖ mesuré en §14bis.6bis du rapport. γ_t est un
    hyperparamètre fixe (non appris) — toggleable via --edge-type-weights.

    Transparent pour HeteroConv : on relaie *args/**kwargs vers le conv
    encapsulé (signature GATConv : (x, edge_index, edge_attr=...)).
    """

    def __init__(self, conv: nn.Module, gamma: float):
        super().__init__()
        self.conv = conv
        self.gamma = float(gamma)

    def forward(self, *args, **kwargs):
        out = self.conv(*args, **kwargs)
        return out * self.gamma


class HeteroEncoder(nn.Module):
    """
    Encoder hétérogène multi-couches pour le VGAE.

    Architecture :
      1. Projection linéaire : features brutes (8 dim gene, 3 dim cell_group)
         → espace caché commun (hidden dim). Nécessaire car les deux types de
         noeuds ont des dimensions d'entrée différentes.
      2. N couches de message passing hétérogène (HeteroConv + GATConv) :
         - HeteroConv : wrapper qui applique un GATConv INDÉPENDANT par type
           d'arête, puis SOMME les messages pour chaque noeud destination.
           Ainsi, un gène reçoit des messages séparés de ses voisins PPI,
           de ses co-membres de pathway, de son TF régulateur, etc., et
           ces messages sont combinés par sommation.
         - GATConv : Graph Attention Network avec multi-head attention.
           Pour chaque arête (i, j), le modèle apprend un score d'attention
           α_ij qui pondère le message de j vers i. Un gène "important"
           dans le voisinage aura un α élevé.
         - BatchNorm + ReLU + Dropout + Résiduel : stabilisent l'entraînement.
           La connexion résiduelle (x_new = x_new + x_prev) empêche la
           disparition du signal dans les couches profondes.
      3. Deux têtes linéaires : hidden → latent pour μ et log(σ²).

    Les noeuds cell_group participent au message passing (ils envoient et
    reçoivent des messages via les arêtes "expresses"/"expressed_in") mais
    SEULS les gènes sont projetés dans l'espace latent (μ, log(σ²)).
    """

    # Catalogue de tous les edge_types possibles dans le graphe, avec leur
    # dimension d'edge_attr. Filtré dynamiquement à l'init selon les
    # types réellement présents dans `data.edge_types` (ablations modulaires).
    # None = pas d'edge_attr → GATConv sans edge_dim (attention purement topologique).
    EDGE_TYPE_CATALOG = [
        (("gene", "ppi", "gene"), 1),
        (("gene", "same_pathway", "gene"), None),
        (("gene", "regulates", "gene"), 1),
        (("gene", "regulated_by", "gene"), 1),
        (("cell_group", "expresses", "gene"), 7),
        (("gene", "expressed_in", "cell_group"), 7),
        (("gene", "coexpression", "gene"), 1),
        (("gene", "metabolic_cocatalysis", "gene"), 2),
        # OmniPath V4 — edge_attr = [score, sign∈{−1,0,+1}]
        (("gene", "signaling", "gene"), 2),
        (("gene", "tf_curated", "gene"), 2),
        (("gene", "tf_curated_by", "gene"), 2),
        # V4.2 — Reactome FI signed (edge_attr = [score, sign])
        (("gene", "reactome_fi", "gene"), 2),
    ]

    def __init__(self, gene_in, cell_in, hidden, latent, n_layers,
                 n_heads=4, dropout=0.2, available_edge_types=None,
                 edge_dim_overrides=None, edge_type_weights=None,
                 signed_message=False, signed_edge_types=None):
        """
        Args:
            available_edge_types: itérable de tuples (src, rel, dst) — typiquement
                `list(data.edge_types)`. Si fourni, seuls les types listés sont
                instanciés. None = utilise tout le catalogue (rétro-compatible).
            edge_dim_overrides: dict {(src,rel,dst): edge_dim} — V4.2, écrase
                la dimension d'edge_attr du catalogue (ex. coexpression 1→6
                en mode differential). None = catalogue inchangé.
            edge_type_weights: dict {(src,rel,dst) ou 'rel': γ_t} — V4.2,
                facteur multiplicatif appliqué à la sortie du GATConv de ce
                type AVANT l'agrégation HeteroConv-sum. None/{} = tous γ=1.0
                (comportement V4.1). Cf. §14bis.6bis du rapport.
            signed_message: V5 (TIER 1c.2). Si True, utilise SignedGATConv
                pour les edge_types listés dans `signed_edge_types` ET dont
                edge_dim>=2 (sign disponible). Défaut False (V4.x legacy).
            signed_edge_types: set de tuples (src, rel, dst). Défaut =
                SIGNED_EDGE_TYPES (signaling, tf_curated, tf_curated_by,
                reactome_fi). Ignoré si signed_message=False.
        """
        super().__init__()
        self._edge_dim_overrides = edge_dim_overrides or {}
        self._edge_type_weights = edge_type_weights or {}
        self._signed_message = bool(signed_message)
        self._signed_edge_types = set(
            tuple(et) for et in (signed_edge_types or SIGNED_EDGE_TYPES)
        )
        self.n_layers = n_layers
        # Chaque tête d'attention travaille en dimension head_dim = hidden/n_heads.
        # Les résultats des n_heads têtes sont concaténés → sortie = hidden.
        head_dim = hidden // n_heads

        # Couches de projection : amènent les features d'entrée (dimensions
        # différentes pour gene et cell_group) dans un espace commun de
        # dimension hidden. Sans ça, le message passing ne pourrait pas
        # sommer les représentations de types de noeuds différents.
        self.gene_proj = nn.Linear(gene_in, hidden)   # 8 → 128
        self.cell_proj = nn.Linear(cell_in, hidden)   # 3 → 128

        self.convs = nn.ModuleList()   # N couches de convolution hétérogène
        self.norms = nn.ModuleList()   # N couches de BatchNorm par type de noeud

        # Filtrage dynamique du catalogue selon les edge_types réellement
        # présents dans le graphe (modularité V3.5+ : si --no-coexpr,
        # le type ("gene", "coexpression", "gene") n'est pas dans data.edge_types
        # et on n'instancie pas son GATConv).
        if available_edge_types is None:
            edge_types_dims = list(self.EDGE_TYPE_CATALOG)
        else:
            available = set(tuple(et) for et in available_edge_types)
            edge_types_dims = [(et, dim) for et, dim in self.EDGE_TYPE_CATALOG
                               if et in available]
            # Avertit si des edge_types existent dans le graphe mais pas dans
            # le catalogue — il faudrait les ajouter à EDGE_TYPE_CATALOG.
            unknown = available - {et for et, _ in self.EDGE_TYPE_CATALOG}
            if unknown:
                print(f"  [warn] edge_types inconnus dans le catalogue : {unknown}")
        if not edge_types_dims:
            raise ValueError(
                "HeteroEncoder : aucun edge_type actif. Vérifie --no-* / la "
                "présence de data.edge_types."
            )
        # V4.2 : appliquer les overrides de dimension d'edge_attr (ex.
        # coexpression 1→6 en mode differential).
        if self._edge_dim_overrides:
            edge_types_dims = [
                (et, self._edge_dim_overrides.get(et, dim))
                for et, dim in edge_types_dims
            ]
        self.edge_dims = {et: dim for et, dim in edge_types_dims}

        # V4.2 : résolution des γ_t par edge_type. La clé peut être le
        # tuple complet (src,rel,dst) ou juste le 'rel' (str) pour
        # simplifier la CLI (--edge-type-weights ppi=0.1,...).
        def _resolve_gamma(et):
            if et in self._edge_type_weights:
                return float(self._edge_type_weights[et])
            rel = et[1]
            if rel in self._edge_type_weights:
                return float(self._edge_type_weights[rel])
            return 1.0
        self.edge_gammas = {et: _resolve_gamma(et) for et, _ in edge_types_dims}
        _non_unit = {k[1]: v for k, v in self.edge_gammas.items() if v != 1.0}
        if _non_unit:
            print(f"  HeteroEncoder γ_t (message-level) : {_non_unit}")

        _signed_used = []  # debug : edge_types effectivement routés vers SignedGATConv
        for _ in range(n_layers):
            conv_dict = {}
            for et, ed in edge_types_dims:
                # Chaque type d'arête a son propre GATConv indépendant.
                # GATConv(in_channels, out_channels, heads, ...) :
                #   - in = hidden (dimension commune après projection)
                #   - out = head_dim (par tête)
                #   - concat=True : les n_heads têtes sont concaténées → hidden
                #   - add_self_loops=False : pas de self-loop car c'est géré
                #     par la connexion résiduelle
                #   - edge_dim=ed : active l'attention sur les features d'arête
                conv_kwargs = dict(heads=n_heads, concat=True,
                                   dropout=dropout, add_self_loops=False)
                if ed is not None:
                    conv_kwargs["edge_dim"] = ed
                # V5 (TIER 1c.2) : SignedGATConv pour les edge_types signés
                # uniquement si edge_dim >= 2 (sign en colonne 1). Pour les
                # autres types ou si --signed-message OFF : GATConv standard.
                _use_signed = (
                    self._signed_message
                    and et in self._signed_edge_types
                    and ed is not None and ed >= 2
                )
                if _use_signed:
                    _gat = SignedGATConv(hidden, head_dim, sign_col=1, **conv_kwargs)
                    if et[1] not in _signed_used:
                        _signed_used.append(et[1])
                else:
                    _gat = GATConv(hidden, head_dim, **conv_kwargs)
                # V4.2 : si γ_t ≠ 1.0, on enveloppe le GATConv dans un
                # scaler qui multiplie sa sortie par γ_t AVANT que
                # HeteroConv(aggr="sum") ne somme les canaux. Ainsi
                # h_i = Σ_t γ_t · h_i^t (rééquilibrage §14bis.6bis).
                _g = self.edge_gammas.get(et, 1.0)
                conv_dict[et] = (_ScaledConv(_gat, _g) if _g != 1.0 else _gat)
            # HeteroConv : applique chaque GATConv sur son type d'arête,
            # puis agrège (aggr="sum") les résultats pour chaque noeud.
            # Un gène qui a des voisins PPI ET des voisins pathway recevra
            # la somme des messages PPI + la somme des messages pathway.
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            # BatchNorm par type de noeud : normalise les activations pour
            # stabiliser l'entraînement. Chaque type a sa propre statistique
            # car gene et cell_group ont des distributions différentes.
            self.norms.append(nn.ModuleDict({
                "gene": nn.BatchNorm1d(hidden),
                "cell_group": nn.BatchNorm1d(hidden),
            }))

        if self._signed_message:
            if _signed_used:
                print(f"  HeteroEncoder V5 SignedGATConv actif sur : {_signed_used}")
            else:
                print(f"  [warn] --signed-message ON mais aucun edge_type "
                      f"signé éligible présent (signed_edge_types ∩ "
                      f"available avec edge_dim≥2 vide).")

        self.dropout = nn.Dropout(dropout)

        # Deux têtes linéaires pour la partie variationnelle :
        #   mu_head : hidden → latent (moyenne du posterior gaussien)
        #   logvar_head : hidden → latent (log-variance du posterior)
        # On prédit log(σ²) plutôt que σ² directement car :
        #   1. log(σ²) peut être négatif (σ² < 1) ou positif (σ² > 1)
        #   2. Pas besoin de contrainte de positivité
        #   3. Numériquement plus stable
        self.mu_head = nn.Linear(hidden, latent)
        self.logvar_head = nn.Linear(hidden, latent)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """
        Forward pass de l'encoder.

        Étapes :
          1. Projection des features brutes → espace caché commun (hidden)
          2. N couches de message passing hétérogène avec résiduel
          3. Extraction de μ et log(σ²) pour les gènes uniquement

        Args:
            x_dict : {"gene": (n_genes, gene_in), "cell_group": (5, cell_in)}
            edge_index_dict : {(src_type, rel, dst_type): (2, n_edges), ...}
            edge_attr_dict : optionnel, {(src,rel,dst): (n_edges, edge_dim), ...}
                             Passé à chaque GATConv pour l'attention pondérée
                             par les features d'arête.

        Returns:
            mu : (n_genes, latent) — moyenne du posterior
            logvar : (n_genes, latent) — log-variance du posterior
        """
        # Étape 1 : projection + ReLU → les deux types de noeuds sont
        # maintenant dans le même espace de dimension hidden.
        x_dict = {
            "gene": F.relu(self.gene_proj(x_dict["gene"])),
            "cell_group": F.relu(self.cell_proj(x_dict["cell_group"])),
        }

        # Étape 2 : N couches de message passing
        for i in range(self.n_layers):
            # Sauvegarder l'état précédent pour la connexion résiduelle.
            # .clone() est nécessaire car la convolution modifie x_dict in-place.
            x_prev = {k: v.clone() for k, v in x_dict.items()}
            # Filtrer les types d'arêtes vides (ex : coexpression peut être vide
            # si COEXPR_TOP_QUANTILE est trop restrictif)
            active_edges = {k: v for k, v in edge_index_dict.items() if v.numel() > 0}
            # Message passing hétérogène : chaque type d'arête a son GATConv,
            # les résultats sont sommés par noeud destination.
            if edge_attr_dict is not None:
                # HeteroConv accepte un edge_attr_dict keyé par edge_type ;
                # il passe edge_attr_dict[et] comme kwarg edge_attr à chaque
                # GATConv. On ne garde que les types réellement présents ET
                # dont le GATConv attend un edge_dim (skip same_pathway).
                active_attrs = {
                    k: edge_attr_dict[k]
                    for k in active_edges
                    if k in edge_attr_dict and self.edge_dims.get(k) is not None
                }
                x_dict = self.convs[i](x_dict, active_edges,
                                       edge_attr_dict=active_attrs)
            else:
                x_dict = self.convs[i](x_dict, active_edges)

            # HeteroConv supprime du dict les types de noeuds sans arête
            # entrante (ex : --no-cell-group-edges → cell_group n'a aucune
            # arête vers lui → drop). On les restaure depuis x_prev en
            # identité (pas de BN/ReLU pour ne pas pourrir les stats).
            for key, prev in x_prev.items():
                if key not in x_dict:
                    x_dict[key] = prev

            for key in list(x_dict.keys()):
                if key not in x_prev:
                    continue
                if x_dict[key] is x_prev[key]:
                    continue  # identité restaurée → on saute BN/ReLU/dropout
                # BatchNorm : normalise les activations (mean=0, var=1)
                x_dict[key] = self.norms[i][key](x_dict[key])
                # ReLU : non-linéarité (permet au réseau d'apprendre des
                # fonctions complexes, pas juste des combinaisons linéaires)
                x_dict[key] = F.relu(x_dict[key])
                # Dropout : régularisation (éteint 20% des neurones aléatoirement)
                x_dict[key] = self.dropout(x_dict[key])
                # CONNEXION RÉSIDUELLE : x_new = x_new + x_prev
                # Empêche la disparition du gradient dans les couches profondes.
                # Aussi, garantit que l'information des features originales n'est
                # pas complètement perdue après 3 couches de message passing.
                x_dict[key] = x_dict[key] + x_prev[key]

        # Étape 3 : seuls les gènes sont projetés dans l'espace latent.
        # Les cell_groups ont servi de "relais" d'information mais ne nous
        # intéressent pas pour le ranking final.
        gene_h = x_dict["gene"]  # (n_genes, hidden)
        mu = self.mu_head(gene_h)          # (n_genes, latent)
        logvar = self.logvar_head(gene_h)  # (n_genes, latent)

        return mu, logvar


class VGAE(nn.Module):
    """
    Variational Graph AutoEncoder — le modèle complet.

    Combine l'encoder (HeteroGNN), le reparametrization trick, et le
    décodeur cosinus pour reconstruire le graphe.

    ARCHITECTURE :
      Encoder : HeteroGNN → μ, log(σ²)      [paramétrique, appris]
      Sampling : z = μ + ε·σ, ε ~ N(0,I)    [reparametrization trick]
      Decoder : logit = τ · cos(z_i, z_j)   [quasi-paramétrique, seul τ est appris]

    POURQUOI LE DÉCODEUR COSINUS (et pas le produit scalaire classique) ?
      Dans un VGAE classique, le décodeur est σ(z_i^T z_j) = produit scalaire.
      Le problème : la KL divergence pousse μ → 0 (vers la prior N(0,I)).
      Quand ||μ|| diminue, les produits scalaires z_i^T z_j → 0 aussi,
      et les probabilités d'arête → σ(0) = 0.5 → AUC = 0.5 (collapse).

      Avec le cosinus : cos(z_i, z_j) = (z_i^T z_j) / (||z_i|| · ||z_j||)
      → seule la DIRECTION compte, pas la norme. La KL peut réduire ||μ||
      sans affecter la reconstruction. Le modèle n'a plus à "lutter" entre
      reconstruire (besoin de grandes normes) et régulariser (besoin de petites normes).

      τ (température apprise) contrôle la "confiance" des prédictions :
      - τ petit → logits proches de 0 → prédictions molles (incertain)
      - τ grand → logits extrêmes → prédictions tranchées (sur-confiant)
      On clampe τ ≤ tau_max pour éviter le surapprentissage.
    """

    def __init__(self, encoder, tau_init=2.0, tau_max=3.0,
                 bilinear_decoder=None):
        super().__init__()
        self.encoder = encoder
        # τ est stocké comme log(τ) pour garantir τ > 0 (exp est toujours positif).
        # Initialisé à τ = 2.0 (logits modérés).
        # tau_max = 3.0 empêche la sur-confiance : au-delà, les logits sont
        # trop grands et le modèle surapprent les arêtes d'entraînement.
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(tau_init))))
        self.log_tau_max = np.log(tau_max)
        # V5 (TIER 1c.3) : décodeur bilinéaire signé optionnel. Si présent,
        # il est utilisé en AUXILIAIRE de `decode()` (cosinus) pour la loss
        # signée sur les arêtes positives des edge_types signés. Le décodeur
        # principal (cosinus) reste actif pour la loss de reconstruction
        # principale — backward-compat V4.x.
        self.bilinear_decoder = bilinear_decoder

    def decode_signed(self, z, edge_index, edge_sign):
        """V5 (TIER 1c.3) — logits bilinéaires sur arêtes signées.

        Raise AttributeError si `bilinear_decoder=None` (= --signed-decoder OFF).
        """
        if self.bilinear_decoder is None:
            raise AttributeError(
                "VGAE.decode_signed appelé mais bilinear_decoder=None. "
                "Active --signed-decoder à l'init du modèle."
            )
        return self.bilinear_decoder.forward_signed(z, edge_index, edge_sign)

    def reparametrize(self, mu, logvar):
        """
        Reparametrization trick : z = μ + ε·σ, avec ε ~ N(0,I).

        POURQUOI ? On veut échantillonner z ~ N(μ, σ²I) mais le sampling
        n'est pas différentiable (on ne peut pas backpropager à travers
        un échantillonnage aléatoire). Le trick : on écrit z = μ + ε·σ
        avec ε fixe (tiré une fois). Le gradient passe par μ et σ (déterministes),
        pas par ε (aléatoire). C'est la base de tous les VAE.

        En ÉVALUATION (self.training=False) : on utilise directement μ
        (la moyenne du posterior), sans bruit. C'est le "MAP estimate" —
        le point le plus probable de la distribution apprise.
        """
        # Clamp logvar pour éviter les explosions numériques :
        # logvar < -10 → σ ≈ 0.007 (trop petit, gradient vanish)
        # logvar > 10  → σ ≈ 148 (trop grand, instabilité)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        if self.training:
            std = torch.exp(0.5 * logvar)  # σ = exp(log(σ²)/2) = exp(logvar/2)
            eps = torch.randn_like(std)     # ε ~ N(0, I), même shape que std
            return mu + eps * std            # z ~ N(μ, σ²I)
        # En évaluation : pas de bruit, on utilise μ directement
        return mu

    def encode(self, x_dict, edge_index_dict, edge_attr_dict=None):
        mu, logvar = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

    def decode(self, z, edge_index):
        """
        Decoder cosinus : τ · cos(z_i, z_j) → logits.

        cos(z_i, z_j) ∈ [-1, 1] ne dépend que de la direction des embeddings,
        pas de leur norme. La KL peut donc réduire ||μ|| sans affecter la
        reconstruction. τ (appris) contrôle la confiance des prédictions.
        """
        src, dst = edge_index
        z_src = F.normalize(z[src], dim=1)  # L2-normalisation → vecteurs unitaires
        z_dst = F.normalize(z[dst], dim=1)
        cos_sim = (z_src * z_dst).sum(dim=1)
        # Clamp τ pour éviter le surapprentissage par sur-confiance
        log_tau_clamped = torch.clamp(self.log_tau, max=self.log_tau_max)
        tau = torch.exp(log_tau_clamped)
        return tau * cos_sim

    def kl_loss(self, mu, logvar, free_bits=0.0):
        """
        KL divergence avec free bits : D_KL(q(z|x) || p(z)), p(z) = N(0,I).

        Formule analytique de la KL pour deux gaussiennes :
          D_KL = -0.5 × Σ_d (1 + log(σ²_d) - μ²_d - σ²_d)
        où d indexe les dimensions latentes.

        FREE BITS — mécanisme anti-KL-collapse :
          Sans free bits, la KL peut pousser TOUTES les dimensions vers la
          prior N(0,1), ce qui donne des embeddings identiques (collapse).
          Avec free bits = λ nats, on impose un MINIMUM de λ nats de KL
          par dimension. Si une dimension a KL < λ, elle n'est PAS pénalisée
          (le gradient de la KL est 0 pour cette dimension).
          Résultat : le modèle est FORCÉ d'utiliser au moins λ nats
          d'information par dimension latente, ce qui garantit que l'espace
          latent encode de l'information utile.

        Args:
            mu : (n_genes, latent) — moyenne du posterior
            logvar : (n_genes, latent) — log-variance du posterior
            free_bits : minimum de KL par dimension (en nats)

        Returns:
            KL totale (scalaire) sommée sur les dimensions latentes
        """
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        # KL par gène par dimension : shape (n_genes, latent_dim)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        # Moyenne sur les gènes → une valeur par dimension latente : shape (latent_dim,)
        kl_per_dim_mean = kl_per_dim.mean(dim=0)
        # Free bits : si KL d'une dimension < free_bits, on la remonte à free_bits
        # (pas de pénalisation → le modèle peut librement encoder de l'info)
        kl_clamped = torch.clamp(kl_per_dim_mean, min=free_bits)
        # Somme sur toutes les dimensions latentes → scalaire unique
        return kl_clamped.sum()


# =============================================================================
# V-sup : préparation du mode supervisé circulaire (labels + features DE)
# =============================================================================
# S'exécute dans les deux chemins (build frais OU cache restauré) car
# gene_features/gene_symbols/data sont des globals après le bloc `if not
# _REUSE_OK`. Les labels DEG (P4_vs_P16 + cluster_0..3) sont construits quel que
# soit --de-features ; les FEATURES DE (circulaires) ne sont concaténées que si
# --de-features (défaut ON en mode supervisé). Le VGAE non supervisé n'est JAMAIS
# affecté (SUP_LABELS reste None, gene_features inchangées).
SUP_LABELS = None
_VSUP_ON = getattr(CLI_ARGS, "de_features", False) or getattr(CLI_ARGS, "supervised", False)
if _VSUP_ON:
    import sys as _sys_sup
    # Robuste aux 2 layouts : local (src/gnn/, src/data/preprocess/) ET cluster
    # à plat (tous les .py sous src/). On ajoute les candidats au sys.path.
    _HERE_DIR = os.path.dirname(os.path.abspath(__file__))
    for _cand in (os.path.join(_HERE_DIR, "..", "data", "preprocess"), _HERE_DIR):
        _cand = os.path.abspath(_cand)
        if os.path.isdir(_cand) and _cand not in _sys_sup.path:
            _sys_sup.path.insert(0, _cand)
    from build_supervised_labels import build_supervised_labels as _build_sup_labels
    print("\n" + "=" * 70)
    print("MODE V-sup — VGAE CIRCULAIRE (reconstruit) "
          f"{'+ features DE ' if CLI_ARGS.de_features else ''}"
          f"{'+ tête classif jointe' if CLI_ARGS.supervised else ''}")
    print("=" * 70)
    # Labels requis pour la tête (--supervised) ET/OU pour dériver les features DE.
    SUP_LABELS = _build_sup_labels(
        gene_symbols, GNN_DATA_DIR,
        recompute=getattr(CLI_ARGS, "supervised_recompute_labels", False))
    if getattr(CLI_ARGS, "de_features", False):
        # Le VGAE RECONSTRUIT avec ces features DE circulaires en plus (l'invariant
        # anti-circularité du VGAE non supervisé n'est levé QUE si --de-features).
        _de_mat = SUP_LABELS.de_feature_matrix()
        gene_features = torch.cat(
            [gene_features, torch.tensor(_de_mat, dtype=torch.float)], dim=1)
        data["gene"].x = gene_features
        print(f"  [V-sup] +{_de_mat.shape[1]} features DE CIRCULAIRES "
              f"{SUP_LABELS.de_feature_names()} → gene_features "
              f"{tuple(gene_features.shape)} (le VGAE reconstruit AVEC)")
    else:
        print("  [V-sup] --supervised sans --de-features : tête sur topologie seule")

# --- Instanciation du modèle ---
# L'encoder prend les features brutes (8 dim gene, 3 dim cell_group) et
# produit μ, log(σ²) en dimension latent=64 pour chaque gène.
# Le VGAE enveloppe l'encoder + le décodeur cosinus avec τ.
encoder = HeteroEncoder(
    gene_in=gene_features.shape[1],       # n features actives (modulaire)
    cell_in=cell_group_features.shape[1],  # 3 features de groupe
    hidden=HIDDEN_DIM,                     # 128 (dimension cachée)
    latent=LATENT_DIM,                     # 64 (dimension de l'espace latent)
    n_layers=N_LAYERS,                     # 3 couches de message passing
    n_heads=N_HEADS,                       # 4 têtes d'attention
    dropout=DROPOUT,                       # 0.2
    available_edge_types=list(data.edge_types),  # filtre selon ablations
    # V4.2 : edge_dim overrides — coexpr 1→6 (differential) ;
    # ppi 1→2 si --dedup-ppi-signed annotate (colonne flag ajoutée).
    edge_dim_overrides=(lambda _o: _o or None)({
        **({("gene", "coexpression", "gene"): _COEXPR_DIM}
           if COEXPR_DIFFERENTIAL else {}),
        **({("gene", "ppi", "gene"): int(edge_attr_ppi.shape[1])}
           if getattr(CLI_ARGS, "dedup_ppi_signed", "off") == "annotate"
           and edge_attr_ppi.numel() > 0 else {}),
    }),
    # V4.2 : pondération γ_t par edge_type (message-level), toggleable.
    edge_type_weights=EDGE_TYPE_WEIGHTS or None,
    # V5 (TIER 1c.2) : SignedGATConv pour les edge_types signés si flag actif.
    signed_message=CLI_ARGS.signed_message,
    signed_edge_types=SIGNED_EDGE_TYPES,
)
# V5 (TIER 1c.3) : décodeur bilinéaire signé optionnel, instancié uniquement
# si --signed-decoder. Sinon, VGAE utilise uniquement le décodeur cosinus
# historique (backward-compat V4.x).
# V5.3 (TIER 1c.7) : tête sub-espace `signed_proj` paramétrée par
# --signed-decoder-dim (défaut LATENT_DIM ⇒ équivalent V5.2 numérique au
# load checkpoint).
SIGNED_DECODER_DIM = (CLI_ARGS.signed_decoder_dim
                      if CLI_ARGS.signed_decoder_dim is not None
                      else LATENT_DIM)
_bilinear_decoder = (BilinearSignedDecoder(LATENT_DIM, signed_dim=SIGNED_DECODER_DIM)
                     if CLI_ARGS.signed_decoder else None)
model = VGAE(encoder, bilinear_decoder=_bilinear_decoder)
if CLI_ARGS.signed_decoder:
    print(f"  VGAE V5 BilinearSignedDecoder actif (λ_signed="
          f"{CLI_ARGS.signed_loss_weight}, signed_dim={SIGNED_DECODER_DIM} "
          f"{'(= latent_dim, équiv V5.2 init)' if SIGNED_DECODER_DIM == LATENT_DIM else '(compression vs latent)'}).")

total_params = sum(p.numel() for p in model.parameters())
print(f"  VGAE : {N_LAYERS} couches GATConv, hidden={HIDDEN_DIM}, latent={LATENT_DIM}")
print(f"  Paramètres : {total_params:,}")

# =============================================================================
# 9. PRÉPARATION DES ARÊTES POUR L'ENTRAÎNEMENT
# =============================================================================
# Le VGAE apprend à reconstruire le graphe : étant donné des embeddings z,
# le décodeur prédit quelles paires (i, j) sont connectées. Pour évaluer
# la qualité de cette reconstruction, on a besoin d'un split train/test :
#   - TRAIN : arêtes utilisées pour calculer la loss (le modèle voit ces arêtes)
#   - TEST : arêtes masquées pendant l'entraînement (le modèle doit les prédire)
# L'AUC sur le test mesure la capacité de généralisation du modèle.
#
# IMPORTANT : on combine TOUS les types d'arêtes gene↔gene (PPI, pathway,
# régulation, coexpression, cocatalyse) en un seul pool. Le décodeur ne
# distingue pas les types d'arêtes — il prédit simplement "arête ou pas".
# Les types d'arêtes restent SÉPARÉS dans le graphe d'entrée de l'encoder
# (chaque type a son GATConv), mais la supervision est sur le pool combiné.
print("\n" + "=" * 70)
print("9. Préparation des arêtes d'entraînement")
print("=" * 70)

# Collecter toutes les arêtes gene↔gene (non dirigées, dédupliquées).
# On stocke chaque arête comme (min(i,j), max(i,j)) pour dédupliquer
# les arêtes bidirectionnelles (i→j et j→i comptent comme une seule paire).
#
# V5.2 (2026-05-29) : les edge_types SIGNÉS (signaling, tf_curated,
# reactome_fi) sont MAINTENANT inclus dans le pool de reconstruction.
# Justification (cf. §14bis.6vicies du rapport) :
#   - Avant V5.2 : signed edges étaient injectées dans le graphe pour
#     l'encoder mais ABSENTES de all_gene_edges → décodeur cosinus
#     aveugle au sous-graphe signed + fuite par negative sampling
#     (paires signed tirées comme "négatifs" alors qu'elles existent).
#   - À partir de V5.2 : signed edges contribuent à la loss recon
#     comme arêtes positives (sans distinction de signe — la sémantique
#     signed reste l'affaire du BilinearSignedDecoder via la
#     signed_aux_loss). Plus de fuite ; cosinus apprend l'existence
#     de toutes les arêtes du graphe.
#   - tf_curated_by est la copie inverse de tf_curated (mêmes paires
#     (TF, target) avec src/dst swappés) → dédup automatique via
#     (min, max), pas de double comptage.
#   - n_edges passe typiquement de ~100k à ~150k → AUC recon **non
#     comparable** aux versions V3/V4/V4.1/V5.1.
# V5.4 (decoder-split, §14bis.6duovicies) : si --decoder-split (+ --signed-decoder),
# les arêtes SIGNÉES quittent le pool cosinus (all_gene_edges) → elles sont
# reconstruites par le bilinéaire (existence, voir signed_exist_pairs plus bas).
# Le cosinus retrouve alors le pool V5.1 (non-signé) → AUC comparable V5.1 (~0.97).
# Sinon (V5.2/V5.3), les signées restent dans le pool cosinus (AUC ~0.94).
DECODER_SPLIT = (bool(getattr(CLI_ARGS, "decoder_split", False))
                 and CLI_ARGS.signed_decoder)
_unsigned_sources = [
    (ppi_src, ppi_dst),
    (react_src, react_dst),
    (reg_src, reg_dst), (reg_dst, reg_src),  # Regulates + regulated_by
    (coexpr_src, coexpr_dst),
]
_signed_sources = [
    (op_sig_src, op_sig_dst),       # signaling (OmniPath kinase + SIGNOR)
    (op_tf_src, op_tf_dst),         # tf_curated (CollecTRI) — tf_curated_by inverse
    (reactome_fi_src, reactome_fi_dst),  # Reactome FI signé
]
all_gene_edges = set()
for src_list, dst_list in (_unsigned_sources if DECODER_SPLIT
                           else _unsigned_sources + _signed_sources):
    for s, d in zip(src_list, dst_list):
        pair = (min(s, d), max(s, d))
        all_gene_edges.add(pair)

# V5.4 : pool d'EXISTENCE signé (décodé par le bilinéaire via
# predict_signed_existence). Paires signées dédupliquées, PRIVÉES de celles
# déjà présentes dans le pool cosinus (évite double label). Vide si pas de split.
signed_exist_pairs: set = set()
if DECODER_SPLIT:
    for src_list, dst_list in _signed_sources:
        for s, d in zip(src_list, dst_list):
            signed_exist_pairs.add((min(s, d), max(s, d)))
    signed_exist_pairs -= all_gene_edges
    print(f"  V5.4 decoder-split ON : pool cosinus {len(all_gene_edges)} "
          f"(non-signé) + pool existence bilinéaire {len(signed_exist_pairs)} (signé)")

all_edges = np.array(list(all_gene_edges))
n_edges = len(all_edges)
# Décomposition pour audit (V5.2) : combien d'arêtes ont été ajoutées par
# l'inclusion des signed ? Aide à comparer V5.1 (sans) vs V5.2 (avec).
_signed_pair_count = 0
for src_list, dst_list in [
    (op_sig_src, op_sig_dst),
    (op_tf_src, op_tf_dst),
    (reactome_fi_src, reactome_fi_dst),
]:
    for s, d in zip(src_list, dst_list):
        _signed_pair_count += 1
print(f"  Arêtes gene↔gene uniques : {n_edges} "
      f"(dont ~{_signed_pair_count} arêtes signées brutes contribuant au pool, "
      f"après dédup avec PPI/REACTOME)")

# Split train/test aléatoire — utilise CLI_ARGS.seed (défaut 42) pour
# reproductibilité ET multi-seed inter-runs (V3.6+).
# EDGE_SAMPLE_RATIO = 10% des arêtes sont réservées pour le test.
# Ce split est fait au niveau des paires non dirigées (pas des arêtes bidirectionnelles).
np.random.seed(CLI_ARGS.seed)
perm = np.random.permutation(n_edges)
n_test = max(1, int(n_edges * EDGE_SAMPLE_RATIO))
test_edge_idx = perm[:n_test]      # Premiers 10% → test
train_edge_idx = perm[n_test:]     # 90% restants → train

test_edges = all_edges[test_edge_idx]
train_edges = all_edges[train_edge_idx]

# Conversion en format PyG : tenseur [2, n_edges] avec src en ligne 0, dst en ligne 1.
pos_train = torch.tensor(train_edges.T, dtype=torch.long)
# Rendre bidirectionnel : pour chaque paire (i, j) du train, on ajoute (j, i).
# .flip(0) inverse les lignes src/dst → transforme (i→j) en (j→i).
# Concaténation → le tenseur contient (i→j) ET (j→i) pour chaque paire.
pos_train_bidir = torch.cat([pos_train, pos_train.flip(0)], dim=1)

pos_test = torch.tensor(test_edges.T, dtype=torch.long)
pos_test_bidir = torch.cat([pos_test, pos_test.flip(0)], dim=1)

print(f"  Train : {len(train_edges)} arêtes positives")
print(f"  Test  : {len(test_edges)} arêtes positives")

# V5.4 (decoder-split) : split 90/10 du pool d'existence signé, bidirectionnel,
# avec le MÊME seed que le pool cosinus (cohérence train/test). Ces tenseurs
# alimentent recon_loss_signed (existence bilinéaire) + l'AUC combinée.
pos_train_signed_bidir = None
pos_test_signed_bidir = None
if DECODER_SPLIT and signed_exist_pairs:
    _se = np.array(list(signed_exist_pairs))
    _perm_s = np.random.permutation(len(_se))
    _n_test_s = max(1, int(len(_se) * EDGE_SAMPLE_RATIO))
    _se_test = _se[_perm_s[:_n_test_s]]
    _se_train = _se[_perm_s[_n_test_s:]]
    _pt_s = torch.tensor(_se_train.T, dtype=torch.long)
    pos_train_signed_bidir = torch.cat([_pt_s, _pt_s.flip(0)], dim=1)
    _pte_s = torch.tensor(_se_test.T, dtype=torch.long)
    pos_test_signed_bidir = torch.cat([_pte_s, _pte_s.flip(0)], dim=1)
    print(f"  V5.4 signed-existence : {len(_se_train)} train / "
          f"{len(_se_test)} test (bilinéaire)")

# =============================================================================
# 10. ENTRAÎNEMENT DU VGAE
# =============================================================================
# Boucle d'entraînement standard PyTorch avec quelques spécificités VGAE :
#   1. Negative sampling : à chaque epoch, on échantillonne des paires NON
#      connectées (négatifs) pour équilibrer la loss. Sans ça, le modèle
#      apprendrait juste à prédire "arête partout" (AUC = 0.5).
#   2. KL annealing : β augmente linéairement de 0 à KL_BETA_MAX. Au début
#      de l'entraînement (β ≈ 0), le modèle apprend UNIQUEMENT à reconstruire.
#      La KL entre progressivement pour régulariser l'espace latent.
#   3. Free bits : minimum de KL par dimension (empêche le collapse).
#   4. Early stopping : on arrête si l'AUC test ne s'améliore plus.
#   5. Gradient clipping : empêche les explosions de gradient.
print("\n" + "=" * 70)
print("10. Entraînement du VGAE")
print("=" * 70)

# =============================================================================
# V-sup : TÊTE de classification JOINTE (multi-tâche) — cf. _supervised.py
# =============================================================================
# Si --supervised : on N'INTERROMPT PAS la reconstruction. On attache une tête
# de classification sur μ, entraînée CONJOINTEMENT dans la boucle VGAE
# (loss = recon + β·KL + λ·classif). L'encodeur reste entraîné par
# reconstruction → le run reste PERTURBATION-READY (Δμ standard). Défini AVANT
# l'optimizer (ses params y sont ajoutés).
_SUP_HEAD = None
_SUP_LABELS_T = _SUP_CONF_T = _SUP_TRAIN_MASK = _SUP_TEST_MASK = None
if getattr(CLI_ARGS, "supervised", False):
    import sys as _sys_sup2
    _HERE_DIR2 = os.path.dirname(os.path.abspath(__file__))
    if _HERE_DIR2 not in _sys_sup2.path:
        _sys_sup2.path.insert(0, _HERE_DIR2)
    from _supervised import SupervisedHead, _node_split as _sup_node_split
    _SUP_HEAD = SupervisedHead(LATENT_DIM, n_labels=len(SUP_LABELS.label_names))
    _SUP_LABELS_T = torch.tensor(SUP_LABELS.labels)
    _SUP_CONF_T = torch.tensor(SUP_LABELS.confidence)
    _SUP_TRAIN_MASK, _SUP_TEST_MASK = _sup_node_split(
        len(gene_symbols), 0.2, CLI_ARGS.seed)
    print(f"  [V-sup] tête classif JOINTE : latent={LATENT_DIM} → "
          f"{len(SUP_LABELS.label_names)} labels, λ={CLI_ARGS.supervised_loss_weight} "
          f"(recon + KL conservés → perturbation-ready)")

# Adam avec weight_decay = 1e-4 (L2 regularization sur les poids).
# Le weight_decay empêche les poids de devenir trop grands, ce qui
# complémente la KL (qui régularise l'espace latent, pas les poids).
_opt_params = list(model.parameters())
if _SUP_HEAD is not None:                 # V-sup : la tête est co-entraînée
    _opt_params += list(_SUP_HEAD.parameters())
optimizer = torch.optim.Adam(_opt_params, lr=LR, weight_decay=1e-4)

# Préparer les dictionnaires de features et d'arêtes pour l'encoder.
# x_dict : features des noeuds par type
# edge_index_dict : arêtes par type (pour le message passing de l'encoder)
# NOTE : l'encoder voit TOUTES les arêtes du graphe (y compris les arêtes test).
# Seul le DÉCODEUR est évalué sur les arêtes test. C'est la convention standard
# des VGAE : l'encoder a accès au graphe complet pour le message passing,
# mais la supervision (loss) est calculée uniquement sur les arêtes train/test.
x_dict = {"gene": data["gene"].x, "cell_group": data["cell_group"].x}
edge_index_dict = {}
edge_attr_dict = {}   # features d'arête passées aux GATConv via edge_dim
# Boucle dynamique sur tous les edge_types présents dans le graphe (modularité
# V3.5+) — auparavant hard-codée, donc cassait sur les ablations partielles.
for et_key in data.edge_types:
    store = data[et_key]
    edge_index_dict[et_key] = store.edge_index
    if "edge_attr" in store and store.edge_attr is not None:
        edge_attr_dict[et_key] = store.edge_attr

# V5 (TIER 1c.4) : pool des arêtes signées POSITIVES pour la loss auxiliaire.
# {edge_type: (edge_index_dirigé, sign∈{-1,0,+1})}. Construit une seule fois,
# indépendamment du split train/test du pool de reconstruction (smoke test :
# on entraîne sur l'ensemble du pool signé).
signed_pos_pool: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
if CLI_ARGS.signed_decoder:
    for et_key in data.edge_types:
        if tuple(et_key) not in SIGNED_EDGE_TYPES:
            continue
        ea = edge_attr_dict.get(et_key)
        ei = edge_index_dict.get(et_key)
        if ea is None or ea.ndim < 2 or ea.shape[1] < 2 or ei is None \
                or ei.numel() == 0:
            continue
        # Convention V4 : edge_attr=[score, sign]. Seules les arêtes signées
        # (sign != 0) contribuent à la loss auxiliaire — sign=0 = info inconnue.
        sign = ea[:, 1].float()
        mask = sign != 0
        if not mask.any():
            continue
        signed_pos_pool[tuple(et_key)] = (
            ei[:, mask].clone(),
            sign[mask].clone(),
        )
    if signed_pos_pool:
        _summary = {et[1]: int(ei.shape[1])
                    for et, (ei, _) in signed_pos_pool.items()}
        print(f"  V5 signed_pos_pool (auxiliaire) : {_summary}")
    else:
        print("  [warn] --signed-decoder ON mais signed_pos_pool vide "
              "(pas d'edge_type signé avec sign≠0 dans le graphe).")

# V5 phase 2 (TIER 1c.5 strict) : split TF hold-out pour gate rigoureux.
# On retire de la signed_aux_loss les arêtes dont src OU dst est un
# régulateur hold-out. L'encoder voit toujours ces arêtes via le
# message-passing → seul le SIGNE est masqué de la loss aux.
# Liste hold-out persistée dans run_config.json + signed_holdout_edges.tsv
# pour `test_signed_auc.py --mode holdout`.
signed_holdout_pool: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
HOLDOUT_TF_SET: list[str] = []
HOLDOUT_TF_SEED = (CLI_ARGS.holdout_signed_tf_seed
                   if CLI_ARGS.holdout_signed_tf_seed is not None
                   else CLI_ARGS.seed)
if (CLI_ARGS.signed_decoder and signed_pos_pool
        and CLI_ARGS.holdout_signed_tf_fraction > 0.0):
    # 1. Set des régulateurs = union des sym source de TOUTES les arêtes signées
    #    présentes (ie. les nœuds qui apparaissent en src d'au moins un
    #    edge_type signé). Pour tf_curated_by les sym src sont les targets ;
    #    mais comme on filtre ensuite par (src OU dst) ∈ hold-out, tout
    #    régulateur de tf_curated sera attrapé en tf_curated_by aussi.
    _regulator_idx = set()
    for et_key, (ei, _) in signed_pos_pool.items():
        _regulator_idx.update(ei[0].tolist())
    _regulator_syms = sorted(str(gene_symbols[i]) for i in _regulator_idx
                             if i < len(gene_symbols))
    if len(_regulator_syms) < 5:
        print(f"  [warn] --holdout-signed-tf-fraction ON mais seulement "
              f"{len(_regulator_syms)} régulateurs trouvés — hold-out désactivé.")
    else:
        _rng = np.random.default_rng(HOLDOUT_TF_SEED)
        _n_holdout = max(1, int(len(_regulator_syms)
                                * CLI_ARGS.holdout_signed_tf_fraction))
        HOLDOUT_TF_SET = sorted(_rng.choice(_regulator_syms,
                                            size=_n_holdout,
                                            replace=False).tolist())
        _holdout_idx_set = {gene_to_idx[s] for s in HOLDOUT_TF_SET
                            if s in gene_to_idx}

        # 2. Split par edge_type : edge en hold-out si src OR dst ∈ set.
        for et_key in list(signed_pos_pool.keys()):
            ei, sign = signed_pos_pool[et_key]
            _src = ei[0].numpy()
            _dst = ei[1].numpy()
            # vectorisé : True si src ou dst dans set
            _src_in = np.isin(_src, list(_holdout_idx_set))
            _dst_in = np.isin(_dst, list(_holdout_idx_set))
            _is_holdout = _src_in | _dst_in
            _is_train = ~_is_holdout

            if _is_holdout.sum() > 0:
                signed_holdout_pool[et_key] = (
                    ei[:, _is_holdout].clone(),
                    sign[_is_holdout].clone(),
                )
            if _is_train.sum() > 0:
                signed_pos_pool[et_key] = (
                    ei[:, _is_train].clone(),
                    sign[_is_train].clone(),
                )
            else:
                # Tout est en hold-out (rare avec X≤0.5) → on retire l'edge_type
                signed_pos_pool.pop(et_key, None)

        # 3. Log + persistance TSV des arêtes hold-out (pour test_signed_auc.py).
        _ho_summary = {et[1]: int(ei.shape[1])
                       for et, (ei, _) in signed_holdout_pool.items()}
        _tr_summary = {et[1]: int(ei.shape[1])
                       for et, (ei, _) in signed_pos_pool.items()}
        print(f"  V5 phase 2 hold-out signed TF (seed={HOLDOUT_TF_SEED}, "
              f"frac={CLI_ARGS.holdout_signed_tf_fraction:.0%}) :")
        print(f"    régulateurs : {len(HOLDOUT_TF_SET)}/{len(_regulator_syms)} "
              f"({100*len(HOLDOUT_TF_SET)/len(_regulator_syms):.1f}%)")
        print(f"    arêtes train (loss aux) : {_tr_summary}")
        print(f"    arêtes hold-out (eval pure) : {_ho_summary}")

        # TSV : 1 ligne par (edge_type, src_sym, dst_sym, sign).
        _ho_rows = []
        for et_key, (ei, sign) in signed_holdout_pool.items():
            for j in range(ei.shape[1]):
                _ho_rows.append({
                    "edge_type": et_key[1],
                    "src_sym": str(gene_symbols[int(ei[0, j])]),
                    "dst_sym": str(gene_symbols[int(ei[1, j])]),
                    "src_idx": int(ei[0, j]),
                    "dst_idx": int(ei[1, j]),
                    "sign": float(sign[j]),
                })
        if _ho_rows:
            pd.DataFrame(_ho_rows).to_csv(
                os.path.join(OUT_DIR, "signed_holdout_edges.tsv"),
                sep="\t", index=False,
            )
            print(f"    persisté : signed_holdout_edges.tsv "
                  f"({len(_ho_rows)} arêtes)")

        # 4. Réécriture du manifest avec le set hold-out résolu (= reproductible).
        try:
            with open(_MANIFEST_PATH) as _fh:
                _manifest = json.load(_fh)
            _manifest.update({
                "holdout_signed_tf_set": HOLDOUT_TF_SET,
                "holdout_signed_tf_seed_used": HOLDOUT_TF_SEED,
                "holdout_signed_edges_per_edge_type": _ho_summary,
                "signed_train_edges_per_edge_type": _tr_summary,
            })
            with open(_MANIFEST_PATH, "w") as _fh:
                json.dump(_manifest, _fh, indent=2)
            print(f"    manifest enrichi : {_MANIFEST_PATH}")
        except Exception as _e:
            print(f"    [warn] échec enrichissement manifest : {_e}")
elif CLI_ARGS.holdout_signed_tf_fraction > 0.0 and not CLI_ARGS.signed_decoder:
    print(f"  [warn] --holdout-signed-tf-fraction={CLI_ARGS.holdout_signed_tf_fraction} "
          f"ignoré (nécessite --signed-decoder).")

# Historique pour les plots
train_losses = []      # Loss totale à chaque epoch
recon_losses = []      # Loss de reconstruction à chaque epoch
kl_losses = []         # KL loss (non pondérée) à chaque epoch
kl_betas = []          # β du KL annealing à chaque epoch
test_aucs = []         # AUC test (évaluée toutes les 10 epochs)
test_aps = []          # AP test (évaluée toutes les 10 epochs)
eval_epochs = []       # Numéros d'epoch où on a évalué
best_test_auc = 0.0    # Meilleure AUC test observée
best_test_ap = 0.0     # AP au meilleur epoch (par AUC)
best_epoch = 0         # Epoch correspondante
# PATIENCE : si l'AUC val ne s'améliore pas pendant PATIENCE epochs, on arrête.
# Surchargeable via --patience (défaut 100).
PATIENCE = CLI_ARGS.patience
# GRAD_CLIP_NORM = 1.0 : on clamp la norme du gradient total à 1.0.
# Empêche les mises à jour explosives (ex : quand τ change brusquement).
GRAD_CLIP_NORM = 1.0
patience_counter = 0

model.train()  # Mode entraînement (active dropout + BatchNorm en mode train)
if _SUP_HEAD is not None:
    _SUP_HEAD.train()
for epoch in range(N_EPOCHS):
    optimizer.zero_grad()  # Remet les gradients à zéro

    # --- Forward pass de l'encoder ---
    # z = échantillon du posterior q(z|x) via reparametrization trick
    # mu, logvar = paramètres du posterior (pour la KL loss)
    z, mu, logvar = model.encode(x_dict, edge_index_dict, edge_attr_dict)

    # --- Negative sampling ---
    # Pour chaque arête positive (vraie connexion), on tire une arête
    # négative (paire non connectée). Le ratio 1:1 donne des classes
    # équilibrées. Les négatifs sont RE-ÉCHANTILLONNÉS à chaque epoch
    # pour que le modèle voie des négatifs variés (pas toujours les mêmes).
    neg_train = negative_sampling(
        pos_train_bidir, num_nodes=n_genes,
        num_neg_samples=pos_train_bidir.shape[1],
    )

    # --- Loss de reconstruction (BCE) ---
    # Le décodeur produit des logits (τ · cos) pour chaque paire.
    # binary_cross_entropy_with_logits applique sigmoid + BCE en une seule
    # opération (numériquement plus stable que sigmoid séparé).
    pos_scores = model.decode(z, pos_train_bidir)   # Logits pour les vrais voisins
    neg_scores = model.decode(z, neg_train)          # Logits pour les non-voisins

    # pos_loss : on veut que les logits des vrais voisins soient HAUTS (label=1)
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_scores, torch.ones_like(pos_scores)
    )
    # neg_loss : on veut que les logits des non-voisins soient BAS (label=0)
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores, torch.zeros_like(neg_scores)
    )
    recon_loss = pos_loss + neg_loss

    # --- V5.4 (decoder-split) : recon_loss_signed (EXISTENCE bilinéaire) ---
    # Le cosinus ne voit plus les arêtes signées ; le bilinéaire reconstruit
    # leur existence via predict_signed_existence = logsumexp(W+,W-,W0).
    # BCE(existence(pos signées), 1) + BCE(existence(neg), 0). Le SIGNE reste
    # porté par signed_aux_loss (predict_sign_score). Cf. §14bis.6duovicies.
    if DECODER_SPLIT and pos_train_signed_bidir is not None:
        _ps = pos_train_signed_bidir.to(z.device)
        _neg_s = negative_sampling(
            _ps, num_nodes=n_genes, num_neg_samples=_ps.shape[1],
        )
        _exist_pos = model.bilinear_decoder.predict_signed_existence(z, _ps)
        _exist_neg = model.bilinear_decoder.predict_signed_existence(z, _neg_s)
        recon_loss_signed = (
            F.binary_cross_entropy_with_logits(_exist_pos, torch.ones_like(_exist_pos))
            + F.binary_cross_entropy_with_logits(_exist_neg, torch.zeros_like(_exist_neg))
        )
        recon_loss = recon_loss + recon_loss_signed

    # --- V5.1 (TIER 1c.4) : loss auxiliaire signée CONTRASTIVE -----------
    # Cible binaire : sign>0 → 1 (activation), sign<0 → 0 (inhibition).
    # BCE sur le score SIGN-AGNOSTIQUE `predict_sign_score = logit_pos −
    # logit_neg`. Optimise directement la métrique du gate 1c.5.
    #
    # ⚠ Correction V5.1 (2026-05-29) — la formulation V5.0 utilisait
    # `forward_signed(z, ei, sign)` qui SÉLECTIONNAIT le canal selon le
    # sign cible. Conséquence : `logit_pos` était entraîné haut pour les
    # activations, `logit_neg` bas pour les inhibitions, mais les deux
    # canaux n'étaient PAS contraints l'un par rapport à l'autre. Le
    # score différentiel `logit_pos − logit_neg` était donc indépendant
    # du sign cible → AUC ≈ 0.5 sur le gate 1c.5 (mesuré 2026-05-29 sur
    # v5-full.s{1,2,3}, cf. §14bis.6septies du rapport).
    #
    # V5.1 fix : on entraîne directement le score différentiel, ce qui
    # force `logit_pos > logit_neg` pour activations et inverse pour
    # inhibitions. Cf. SGAT-bilinear Liu 2024 *NAR* §3.
    signed_aux_loss = torch.tensor(0.0, device=z.device)
    if CLI_ARGS.signed_decoder and signed_pos_pool:
        _aux_terms = []
        for et_key, (ei, sign) in signed_pos_pool.items():
            ei = ei.to(z.device)
            sign = sign.to(z.device)
            # Score différentiel : (z_src·W_+·z_dst) − (z_src·W_-·z_dst)
            score = model.bilinear_decoder.predict_sign_score(z, ei)
            target = (sign > 0).float()  # +1 → 1 ; -1 → 0
            _aux_terms.append(F.binary_cross_entropy_with_logits(
                score, target
            ))
        if _aux_terms:
            signed_aux_loss = torch.stack(_aux_terms).mean()

    # --- KL divergence avec annealing + free bits ---
    # KL ANNEALING : β augmente linéairement de 0 à KL_BETA_MAX pendant
    # les KL_WARMUP_EPOCHS premières epochs. Calendrier :
    #   epoch 0   → β = 0        (pas de KL, reconstruction pure)
    #   epoch 25  → β = KL_BETA_MAX/2  (KL progressive)
    #   epoch 50+ → β = KL_BETA_MAX    (régularisation complète)
    # Cela permet au modèle de d'abord apprendre à bien reconstruire les
    # arêtes (encoder des embeddings informatifs), PUIS d'introduire la
    # régularisation KL qui va lisser et structurer l'espace latent.
    kl_beta = min(KL_BETA_MAX, KL_BETA_MAX * epoch / max(1, KL_WARMUP_EPOCHS))
    kl_loss = model.kl_loss(mu, logvar, free_bits=FREE_BITS)

    # Loss totale = reconstruction + β × KL + λ_signed × signed_aux (V5)
    loss = recon_loss + kl_beta * kl_loss \
        + CLI_ARGS.signed_loss_weight * signed_aux_loss
    # V-sup : + λ × classification JOINTE (multi-tâche) sur μ. L'encodeur reste
    # entraîné par reconstruction ; la tête ajoute le signal DEG par cluster.
    if _SUP_HEAD is not None:
        _clf_logits = _SUP_HEAD(mu)
        _clf_bce = F.binary_cross_entropy_with_logits(
            _clf_logits[_SUP_TRAIN_MASK], _SUP_LABELS_T[_SUP_TRAIN_MASK],
            reduction="none")
        _clf_w = _SUP_CONF_T[_SUP_TRAIN_MASK].unsqueeze(1)
        _clf_loss = (_clf_bce * _clf_w).sum() / (
            _clf_w.sum() * _clf_logits.shape[1] + 1e-8)
        loss = loss + CLI_ARGS.supervised_loss_weight * _clf_loss
    loss.backward()
    # Gradient clipping : empêche les mises à jour explosives
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    train_losses.append(loss.item())
    recon_losses.append(recon_loss.item())
    kl_losses.append(kl_loss.item())
    kl_betas.append(kl_beta)

    # --- Évaluation sur les arêtes test (toutes les 10 epochs) ---
    # On évalue sur des arêtes que le modèle n'a PAS vues dans la loss.
    # L'AUC mesure la capacité du modèle à distinguer les vraies arêtes
    # (positifs) des paires non connectées (négatifs).
    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()  # Mode évaluation (désactive dropout, BatchNorm en mode eval)
        with torch.no_grad():  # Pas de calcul de gradient (économie mémoire + vitesse)
            z_eval, mu_eval, _ = model.encode(x_dict, edge_index_dict, edge_attr_dict)

            # Scores pour les arêtes test positives (vraies)
            pos_test_scores = torch.sigmoid(model.decode(z_eval, pos_test_bidir))
            # Négatifs test : nouvelles paires non connectées
            neg_test = negative_sampling(
                pos_test_bidir, num_nodes=n_genes,
                num_neg_samples=pos_test_bidir.shape[1],
            )
            neg_test_scores = torch.sigmoid(model.decode(z_eval, neg_test))

            # Labels : 1 pour positifs, 0 pour négatifs
            test_labels = torch.cat([
                torch.ones(pos_test_scores.shape[0]),
                torch.zeros(neg_test_scores.shape[0]),
            ])
            test_scores = torch.cat([pos_test_scores, neg_test_scores])

            # V5.4 (decoder-split) : AUC COMBINÉE sur tout le graphe — on ajoute
            # les scores d'EXISTENCE bilinéaire sur le test pool signé (que le
            # cosinus ne voit plus). La métrique rapportée couvre cosinus
            # (non-signé) + bilinéaire (signé). Cf. §14bis.6duovicies propriété 3.
            if DECODER_SPLIT and pos_test_signed_bidir is not None:
                _pts = pos_test_signed_bidir.to(z_eval.device)
                _neg_s = negative_sampling(
                    _pts, num_nodes=n_genes, num_neg_samples=_pts.shape[1],
                )
                _ex_pos = torch.sigmoid(
                    model.bilinear_decoder.predict_signed_existence(z_eval, _pts))
                _ex_neg = torch.sigmoid(
                    model.bilinear_decoder.predict_signed_existence(z_eval, _neg_s))
                test_labels = torch.cat([
                    test_labels,
                    torch.ones(_ex_pos.shape[0]), torch.zeros(_ex_neg.shape[0]),
                ])
                test_scores = torch.cat([test_scores, _ex_pos, _ex_neg])

            # AUC-ROC et Average Precision (AP)
            # AUC = 0.5 → modèle aléatoire (collapse)
            # AUC > 0.9 → le modèle reconstruit bien le graphe
            try:
                auc = roc_auc_score(test_labels.numpy(), test_scores.numpy())
                ap = average_precision_score(test_labels.numpy(), test_scores.numpy())
            except ValueError:
                auc, ap = 0.5, 0.5  # Fallback si toutes les prédictions sont identiques

            test_aucs.append(auc)
            test_aps.append(ap)
            eval_epochs.append(epoch)

            # Sauvegarder le meilleur modèle (par AUC test)
            if auc > best_test_auc:
                best_test_auc = auc
                best_test_ap = ap
                best_epoch = epoch
                patience_counter = 0  # Reset du compteur de patience
                torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_vgae.pt"))
                if _SUP_HEAD is not None:   # V-sup : tête au même best epoch
                    torch.save(_SUP_HEAD.state_dict(),
                               os.path.join(OUT_DIR, "best_sup_head.pt"))
            else:
                patience_counter += 10  # +10 car on évalue toutes les 10 epochs

        model.train()  # Retour en mode entraînement

        # Log détaillé toutes les 50 epochs
        if (epoch + 1) % 50 == 0:
            tau_val = np.exp(model.log_tau.item())  # τ actuel
            _signed_log = (f" sgn={signed_aux_loss.item():.4f}"
                           if CLI_ARGS.signed_decoder else "")
            print(f"    Epoch {epoch+1:3d}/{N_EPOCHS} — "
                  f"Loss: {loss.item():.4f} (recon={recon_loss.item():.4f} "
                  f"kl={kl_loss.item():.4f} β={kl_beta:.5f} τ={tau_val:.2f}"
                  f"{_signed_log}) — "
                  f"Test AUC: {auc:.4f}, AP: {ap:.4f}")

    # Early stopping — si l'AUC ne s'améliore plus, on arrête avant
    # de gaspiller des epochs et de risquer le collapse
    if patience_counter >= PATIENCE:
        print(f"\n  Early stopping à epoch {epoch+1} (pas d'amélioration "
              f"depuis {PATIENCE} epochs)")
        break

print(f"\n  Meilleur modèle : epoch {best_epoch+1}, AUC = {best_test_auc:.4f}")

# Charger le meilleur modèle (celui avec la meilleure AUC test).
# weights_only=True : sécurité PyTorch — ne charge que les tenseurs, pas
# d'objets Python arbitraires (évite les attaques de sérialisation).
model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_vgae.pt"), weights_only=True))

# =============================================================================
# V-sup : finalisation (graphe augmenté + tête classif) après reconstruction
# =============================================================================
# Re-sauve hetero_graph_vgae.pt (le save du build est topologie-seule et absent
# en --reuse-graph) → les 2 perturbations chargent le graphe EXACT (gene.x
# augmenté DE si --de-features), cohérent avec vgae_weights.pt.
if _VSUP_ON:
    torch.save(data, os.path.join(OUT_DIR, "hetero_graph_vgae.pt"))
    print(f"  [V-sup] hetero_graph_vgae.pt re-sauvé "
          f"(gene.x={tuple(data['gene'].x.shape)}) → perturbation OK")
if _SUP_HEAD is not None:
    from _supervised import finalize_supervised
    _hp_best = os.path.join(OUT_DIR, "best_sup_head.pt")
    if os.path.exists(_hp_best):
        _SUP_HEAD.load_state_dict(torch.load(_hp_best, weights_only=True))
    finalize_supervised(
        model, _SUP_HEAD, x_dict, edge_index_dict, edge_attr_dict,
        SUP_LABELS, gene_symbols, OUT_DIR,
        hyperparams={"hidden": HIDDEN_DIM, "latent": LATENT_DIM,
                     "n_layers": N_LAYERS, "n_heads": N_HEADS,
                     "signed_message": CLI_ARGS.signed_message,
                     "signed_decoder": CLI_ARGS.signed_decoder},
        train_mask=_SUP_TRAIN_MASK, test_mask=_SUP_TEST_MASK, data=data)

# =============================================================================
# 11. EXTRACTION DES EMBEDDINGS ET SCORE D'IMPORTANCE ÉMERGENT
# =============================================================================
# C'est la section la plus importante du pipeline : on extrait les embeddings
# du VGAE entraîné et on calcule un SCORE D'IMPORTANCE COMPOSITE à 5
# composantes. Ce score est "émergent" car il n'est PAS défini a priori
# (contrairement au log2FC qui est un test statistique). Il ÉMERGE de
# l'espace latent appris par le VGAE à partir de la topologie du graphe.
#
# Les 5 composantes capturent 5 aspects différents de l'"importance" :
#   1. emb_norm (norme de μ) : remarquabilité topologique
#   2. density (k-NN cosinus) : centralité dans l'espace latent
#   3. recon_fidelity (1 - erreur) : fiabilité de la reconstruction
#   4. certainty (1 - σ) : stabilité/confiance du modèle
#   5. specificity (entropie de Shannon) : spécificité d'expression
#
# Score final = moyenne des 5 composantes (pas de poids arbitraires).
print("\n" + "=" * 70)
print("11. Score d'importance émergent")
print("=" * 70)

# Extraire les embeddings du meilleur modèle (mode évaluation → pas de bruit).
# On utilise μ (la moyenne du posterior) comme embedding final, pas z
# (échantillon bruité). μ est déterministe et plus stable pour le ranking.
model.eval()
with torch.no_grad():
    z_final, mu_final, logvar_final = model.encode(x_dict, edge_index_dict, edge_attr_dict)
    gene_emb = mu_final.numpy()  # (n_genes, latent_dim=64)

# ── Score 1 : Excentricité dans l'espace latent ────────────────────────────────
# La norme L2 de μ mesure à quel point un gène est "excentrique" par rapport
# au centre de l'espace latent. Avec le décodeur cosinus, la norme est libre
# (non contrainte par la reconstruction) — une grande norme signifie que la
# KL n'a pas réussi à ramener ce gène vers la prior, ce qui indique un
# profil topologique distinctif que le modèle "refuse" de simplifier.
emb_norm = np.linalg.norm(gene_emb, axis=1).astype(np.float32)
emb_norm_score = emb_norm / (emb_norm.max() + 1e-8)

# ── Score 2 : Reconstruction error par gène ──────────────────────────────────
# Pour chaque gène, on mesure à quel point le VGAE reconstruit correctement
# ses arêtes. Un gène mal reconstruit a un comportement atypique dans le réseau.
with torch.no_grad():
    z_eval = mu_final
    recon_error = np.zeros(n_genes, dtype=np.float32)
    edge_count = np.zeros(n_genes, dtype=np.float32)

    # Pour chaque arête positive, calculer l'erreur via le décodeur cosinus
    # (cohérent avec le décodeur utilisé pendant l'entraînement)
    edge_list = list(all_gene_edges)
    edge_src = torch.tensor([pair[0] for pair in edge_list], dtype=torch.long)
    edge_dst = torch.tensor([pair[1] for pair in edge_list], dtype=torch.long)
    edge_idx = torch.stack([edge_src, edge_dst])
    scores = torch.sigmoid(model.decode(z_eval, edge_idx)).numpy()
    for i, (s, d) in enumerate(edge_list):
        error = 1.0 - scores[i]
        recon_error[s] += error
        recon_error[d] += error
        edge_count[s] += 1
        edge_count[d] += 1

    # Moyenne par gène (éviter division par zéro pour les gènes sans arête)
    safe_count = np.where(edge_count > 0, edge_count, 1.0)
    recon_error_mean = np.where(edge_count > 0, recon_error / safe_count, 0.0)
    recon_error_score = recon_error_mean / (recon_error_mean.max() + 1e-8)

# ── Score 3 : Incertitude (σ) ────────────────────────────────────────────────
# Un gène avec une grande σ est un gène dont le VGAE est "incertain" —
# le réseau autour de ce gène ne contraint pas bien son embedding.
sigma = torch.exp(0.5 * logvar_final).numpy()
sigma_mean = sigma.mean(axis=1).astype(np.float32)
uncertainty_score = sigma_mean / (sigma_mean.max() + 1e-8)

# ── Score 4 : Densité locale (k-NN dans l'espace latent) ────────────────────
# Un gène entouré de beaucoup de voisins proches est un hub fonctionnel.
#
# CORRECTION : avec le décodeur cosinus, on utilise maintenant la similarité
# cosinus pour la k-NN (cohérent avec l'entraînement). De plus, on normalise
# le score par RANG (pas par valeur) pour éliminer l'artefact des "gènes
# fantômes" qui formaient un paquet isolé avec densité=1.0 dégénérée.
# Le rang est invariant aux échelles : un paquet isolé n'aura plus
# automatiquement le score max.
from sklearn.neighbors import NearestNeighbors
k_neighbors = min(20, n_genes - 1)

# Normaliser les embeddings pour utiliser la similarité cosinus
# (cohérent avec le décodeur)
gene_emb_normed = gene_emb / (np.linalg.norm(gene_emb, axis=1, keepdims=True) + 1e-8)
knn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="cosine")
knn.fit(gene_emb_normed)
distances, neighbors = knn.kneighbors(gene_emb_normed)
# Exclure le gène lui-même (premier voisin = lui-même avec distance 0)
distances = distances[:, 1:]
neighbors = neighbors[:, 1:]

# Densité brute = inverse de la distance moyenne (cosine ∈ [0, 2])
mean_dist = distances.mean(axis=1)
local_density_raw = 1.0 / (mean_dist + 1e-8)

# Normalisation par RANG plutôt que par valeur — élimine les dégénérescences
# où plusieurs gènes ont la même distance dégénérée (paquets isolés)
from scipy.stats import rankdata
density_ranks = rankdata(local_density_raw, method="average")
density_score = (density_ranks / density_ranks.max()).astype(np.float32)

# Pénaliser les "paquets isolés" : si tous les voisins d'un gène sont entre eux
# à des distances quasi identiques (signe d'un cluster dégénéré), on réduit
# son score. Mesure : variance des distances aux k voisins.
dist_variance = distances.var(axis=1)
# Plus la variance est petite (= cluster homogène isolé), plus on pénalise
# Normaliser par rang également
variance_ranks = rankdata(dist_variance, method="average") / len(dist_variance)
# Score final : densité × diversité du voisinage
density_score = (density_score * variance_ranks).astype(np.float32)
density_score = density_score / (density_score.max() + 1e-8)

# ── Score 5 : Spécificité d'expression (entropie de Shannon) ────────────────
# Mesure à quel point l'expression d'un gène est SPÉCIFIQUE à un sous-ensemble
# de groupes plutôt qu'ubiquitaire. Un gène marqueur d'un cluster P16 a une
# faible entropie (signal concentré), un gène housekeeping a une haute entropie.
# Pour la sénescence, les gènes spécifiques aux clusters P16 sont les plus
# intéressants. La spécificité = 1 - entropie_normalisée.
#
# IMPORTANT : on filtre aussi les gènes silencieux (mean_expression total < seuil)
# car ils créent des artefacts (gènes "fantômes" dans des paquets isolés).
print("  Calcul de la spécificité d'expression (entropie de Shannon)...")

# mean_expr_per_group a déjà été calculé en section 5 sur les gènes filtrés.
# Shape : (n_groups, n_genes) → on transpose pour avoir (n_genes, n_groups)
expr_matrix = mean_expr_per_group.T  # (n_genes, n_groups)

# Total d'expression par gène (somme sur les groupes)
total_expr = expr_matrix.sum(axis=1)
# Probabilités p_g = expression dans groupe g / total
# Avec epsilon pour éviter log(0)
eps = 1e-8
prob = expr_matrix / (total_expr[:, None] + eps)
# Entropie de Shannon : H = -Σ p_g log(p_g)
entropy = -np.sum(prob * np.log(prob + eps), axis=1)
# Normaliser par log(n_groups) pour avoir entropie ∈ [0, 1]
max_entropy = np.log(len(CELL_GROUPS))
entropy_norm = entropy / max_entropy
# Spécificité = 1 - entropie : haute pour gènes spécifiques d'un groupe
specificity = 1.0 - entropy_norm

# Filtre anti-fantômes : un gène silencieux (total_expr trop bas) a une
# spécificité non significative (artefact numérique). On la met à 0.
# Seuil : 5e percentile de l'expression totale
silent_threshold = np.percentile(total_expr, 5)
specificity[total_expr < silent_threshold] = 0.0
specificity_score = specificity.astype(np.float32)
n_silenced = int((total_expr < silent_threshold).sum())
print(f"    Gènes silencieux filtrés : {n_silenced} (expression totale < {silent_threshold:.4f})")

# ── Score composite (combinaison non pondérée, 5 composantes) ───────────────
# POURQUOI PAS DE POIDS ? Chaque composante capture un aspect ORTHOGONAL
# de l'importance. Pondérer reviendrait à décider a priori que la norme est
# "plus importante" que la densité, ce qui serait arbitraire. La moyenne
# simple laisse le score émerger de l'espace latent sans biais humain.
#
# Ce que chaque composante capture :
#   emb_norm_score        : remarquabilité topologique (le gène est "loin du centre")
#   density_score         : centralité latente (hub dans l'espace des embeddings)
#   1 - recon_error_score : fiabilité (le VGAE reconstruit bien ses arêtes)
#   1 - uncertainty_score : certitude (σ faible = le modèle est confiant)
#   specificity_score     : spécificité d'expression (concentré sur certains groupes)
#
# Les composantes "inversées" (1 - x) transforment des scores "plus haut = pire"
# en scores "plus haut = mieux" :
#   - recon_error haute = mauvaise reconstruction → on veut l'inverse
#   - uncertainty haute = grande σ = modèle incertain → on veut l'inverse
importance_score = (
    emb_norm_score
    + density_score
    + (1 - recon_error_score)    # Fiabilité = 1 - erreur
    + (1 - uncertainty_score)    # Certitude = 1 - incertitude
    + specificity_score
) / 5.0
# Normalisation finale dans [0, 1]
importance_score = importance_score / (importance_score.max() + 1e-8)

print(f"  Composantes du score :")
print(f"    Norme μ (remarquabilité) : mean={emb_norm_score.mean():.3f}")
print(f"    Densité locale (corrigée): mean={density_score.mean():.3f}")
print(f"    1-Recon error (fiabilité): mean={(1-recon_error_score).mean():.3f}")
print(f"    1-Incertitude (certitude): mean={(1-uncertainty_score).mean():.3f}")
print(f"    Spécificité (entropie)   : mean={specificity_score.mean():.3f}")
print(f"    Score composite          : mean={importance_score.mean():.3f}")

top_idx = np.argsort(importance_score)[::-1][:15]
print(f"\n  Top 15 gènes par score d'importance émergent :")
print(f"    {'Gène':15s} {'Score':>7s} {'Norme':>7s} {'Densit':>7s} {'Recon':>7s} {'Certit':>7s} {'Spéc':>7s}")
for idx in top_idx:
    print(f"    {gene_symbols[idx]:15s} {importance_score[idx]:.4f} "
          f"{emb_norm_score[idx]:.4f} {density_score[idx]:.4f} "
          f"{1-recon_error_score[idx]:.4f} {1-uncertainty_score[idx]:.4f} "
          f"{specificity_score[idx]:.4f}")

# ── Clustering des embeddings ────────────────────────────────────────────────
# K-means sur les embeddings μ pour identifier des groupes fonctionnels
# de gènes. Les gènes d'un même cluster ont des embeddings proches →
# ils occupent un rôle similaire dans le graphe biologique.
# Le silhouette score mesure la qualité du clustering :
#   > 0.3 = structure claire, < 0.1 = pas de structure nette.
# n_init=10 : K-means est lancé 10 fois avec des initialisations différentes
# et le meilleur résultat (inertie minimale) est retenu.
print(f"\n  Clustering K-means (k={N_CLUSTERS})...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
gene_clusters = kmeans.fit_predict(gene_emb)
sil_score = silhouette_score(gene_emb, gene_clusters)
print(f"    Silhouette score : {sil_score:.4f}")

for c in range(N_CLUSTERS):
    n_in = (gene_clusters == c).sum()
    mean_imp = importance_score[gene_clusters == c].mean()
    print(f"    Cluster {c} : {n_in:5d} gènes, importance moyenne = {mean_imp:.4f}")

# ── Étapes optionnelles : toggles --no-baselines / --no-validation ───────────
RUN_BASELINES = not getattr(CLI_ARGS, "no_baselines", False)
RUN_VALIDATION = not getattr(CLI_ARGS, "no_validation", False)
# Sentinelles : si une section est sautée, ses variables aval restent définies.
mlp_auc = float("nan"); mlp_ap = float("nan")
mlp_gene_score = None; node2vec_score = None
genage_symbols = cellage_symbols = msigdb_aging_genes = ageanno_genes = aging_local_symbols = set()
databases = []
if not RUN_BASELINES:
    print("[skip] baselines entraînées (MLP §12 + DeepWalk §13bis) — --no-baselines")
if not RUN_VALIDATION:
    print("[skip] validation post-hoc BDD aging (§14) — --no-validation")

if RUN_BASELINES:
    # =============================================================================
    # 12. BASELINE MLP (même features, pas de graphe)
    # =============================================================================
    # OBJECTIF DE CETTE BASELINE : mesurer l'apport de la TOPOLOGIE du graphe.
    # Le MLP utilise les MÊMES features que le VGAE (is_tf, variance, degree, etc.)
    # mais NE fait PAS de message passing. Il prédit directement si une arête
    # existe entre deux gènes en concaténant leurs features.
    #
    # Si VGAE AUC >> MLP AUC → la structure du graphe (qui est voisin de qui)
    # apporte de l'information au-delà des features brutes.
    # Si VGAE AUC ≈ MLP AUC → les features seules suffisent, le graphe n'aide pas.
    #
    # C'est un test d'ABLATION : on retire le graphe et on regarde si le modèle
    # perd en performance. Si oui, le message passing est utile.
    print("\n" + "=" * 70)
    print("12. Baseline MLP (sans graphe)")
    print("=" * 70)


    class MLPBaseline(nn.Module):
        """
        MLP baseline pour la prédiction de liens (sans graphe).

        Pour prédire si une arête (i, j) existe :
          1. On concatène les features de i et j : h = [x_i ; x_j]  (dim = 2×gene_in)
          2. On passe h dans un MLP à 3 couches : 2×gene_in → hidden → hidden/2 → 1
          3. La sortie est un logit (pas de sigmoid, appliquée dans la loss)

        Le MLP n'a AUCUNE notion de voisinage ou de graphe : il ne voit que
        les features des deux gènes. Pas de message passing.
        """

        def __init__(self, in_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim * 2, hidden_dim),   # Concaténation → hidden
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),  # hidden → hidden/2
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),       # hidden/2 → 1 logit
            )

        def forward(self, x, edge_index):
            src, dst = edge_index
            # Concaténation des features des deux gènes de chaque paire
            h = torch.cat([x[src], x[dst]], dim=1)  # (n_pairs, 2×in_dim)
            return self.net(h).squeeze(-1)  # (n_pairs,) logits


    mlp = MLPBaseline(gene_features.shape[1], MLP_HIDDEN)
    mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=MLP_LR)

    # Entraînement du MLP — même loss et même split que le VGAE
    mlp.train()
    for epoch in range(MLP_EPOCHS):
        mlp_optimizer.zero_grad()

        # Mêmes arêtes positives et négatifs que le VGAE
        pos_scores = mlp(gene_features, pos_train_bidir)
        neg_edges = negative_sampling(pos_train_bidir, num_nodes=n_genes,
                                      num_neg_samples=pos_train_bidir.shape[1])
        neg_scores = mlp(gene_features, neg_edges)

        loss = (F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
                + F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores)))
        loss.backward()
        mlp_optimizer.step()

    # Évaluation MLP sur les mêmes arêtes test que le VGAE
    mlp.eval()
    with torch.no_grad():
        pos_s = torch.sigmoid(mlp(gene_features, pos_test_bidir))
        neg_test_mlp = negative_sampling(pos_test_bidir, num_nodes=n_genes,
                                         num_neg_samples=pos_test_bidir.shape[1])
        neg_s = torch.sigmoid(mlp(gene_features, neg_test_mlp))

        mlp_labels = torch.cat([torch.ones_like(pos_s), torch.zeros_like(neg_s)])
        mlp_scores = torch.cat([pos_s, neg_s])
        mlp_auc = roc_auc_score(mlp_labels.numpy(), mlp_scores.numpy())
        mlp_ap = average_precision_score(mlp_labels.numpy(), mlp_scores.numpy())

    print(f"  MLP — Test AUC: {mlp_auc:.4f}, AP: {mlp_ap:.4f}")
    print(f"  VGAE — Test AUC: {best_test_auc:.4f}")
    print(f"  Δ AUC (VGAE - MLP) = {best_test_auc - mlp_auc:+.4f}")
    if best_test_auc > mlp_auc:
        print(f"  → La topologie du graphe apporte +{(best_test_auc - mlp_auc)*100:.1f}% d'AUC")
    else:
        print(f"  → Le MLP fait aussi bien/mieux — la topologie n'aide pas ici")

    # --- Score MLP par gène (analogue de vgae_recon_fidelity) -------------------
    # Pour chaque gène, on prend la PROBABILITÉ MOYENNE que ses vraies arêtes
    # soient correctement prédites par le MLP. C'est la version sans-graphe
    # de `recon_fidelity` : un gène est "important MLP" si ses interactions
    # sont prédictibles uniquement à partir des features concaténées des deux
    # gènes, sans message passing. Utile comme 2e baseline pour voir ce que
    # le message passing apporte vraiment au scoring (pas juste à l'AUC).
    mlp.eval()
    with torch.no_grad():
        all_pos = torch.cat([pos_train_bidir, pos_test_bidir], dim=1)
        pos_scores_all = torch.sigmoid(mlp(gene_features, all_pos)).numpy()
    src_np = all_pos[0].numpy()
    dst_np = all_pos[1].numpy()
    mlp_score_sum = np.zeros(n_genes, dtype=np.float32)
    mlp_edge_cnt = np.zeros(n_genes, dtype=np.float32)
    np.add.at(mlp_score_sum, src_np, pos_scores_all)
    np.add.at(mlp_score_sum, dst_np, pos_scores_all)
    np.add.at(mlp_edge_cnt, src_np, 1.0)
    np.add.at(mlp_edge_cnt, dst_np, 1.0)
    mlp_gene_score = np.where(mlp_edge_cnt > 0,
                              mlp_score_sum / np.maximum(mlp_edge_cnt, 1.0),
                              0.0).astype(np.float32)
    # Normalisation [0, 1] pour comparer aux autres scores.
    mlp_gene_score = mlp_gene_score / (mlp_gene_score.max() + 1e-8)
    print(f"  Score MLP par gène : mean={mlp_gene_score.mean():.4f}, "
          f"gènes couverts={int((mlp_edge_cnt > 0).sum())}/{n_genes}")
    top_mlp = np.argsort(mlp_gene_score)[::-1][:10]
    print(f"  Top 10 gènes (baseline MLP) :")
    for idx in top_mlp:
        print(f"    {gene_symbols[idx]:15s} score={mlp_gene_score[idx]:.4f}")

# =============================================================================
# 13. BASELINE STATISTIQUE (ranking par expression différentielle)
# =============================================================================
# OBJECTIF : comparer le VGAE avec la méthode la plus simple possible —
# ranger les gènes par leur différence d'expression absolue entre P4 et P16.
# C'est un proxy de |log2FC| (sans test statistique formel car n=1 par condition).
#
# Si le VGAE identifie des gènes à haut score qui ont un BAS score statistique
# → ce sont des "candidats de découverte" : des gènes qui ne changeraient pas
# d'expression de manière dramatique mais qui ont un rôle important dans le
# réseau biologique (hubs, régulateurs, etc.). C'est la plus-value du VGAE.
print("\n" + "=" * 70)
print("13. Baseline statistique (ranking expression)")
print("=" * 70)

# Pseudo log2FC : |mean(avancé) - mean(baseline)| sur les données LogNormalize.
# Baseline = CELL_GROUPS[0] (HUVEC : P4 ; bulk : pro). "Avancé global" = moyenne
# des autres groupes (HUVEC : 4 clusters P16 ; bulk : sen). Générique au nommage
# et au nombre de groupes (byte-identique HUVEC).
p4_mean = group_stats[CELL_GROUPS[0]]["mean_expression"]
p16_means = np.array([
    group_stats[g]["mean_expression"] for g in CELL_GROUPS[1:]
])
p16_global_mean = p16_means.mean(axis=0)  # Moyenne sur les groupes "avancés"

# Valeur absolue car on s'intéresse aux gènes dérégulés (up OU down)
pseudo_lfc = np.abs(p16_global_mean - p4_mean).astype(np.float32)
stat_score = pseudo_lfc / (pseudo_lfc.max() + 1e-8)  # Normalisation [0, 1]

print(f"  Score statistique (|ΔExpr P16-P4|) : mean={stat_score.mean():.4f}")
top_stat = np.argsort(stat_score)[::-1][:10]
print(f"  Top 10 gènes (baseline stat) :")
for idx in top_stat:
    print(f"    {gene_symbols[idx]:15s} score={stat_score[idx]:.4f}")

if RUN_BASELINES:
    # =============================================================================
    # 13bis. BASELINE GRAPHE SANS VAE — DeepWalk (random walks + skip-gram)
    # =============================================================================
    # OBJECTIF : isoler la contribution du VAE par rapport à la pure topologie
    # du graphe. On apprend des embeddings de gènes via des random walks
    # uniformes + skip-gram (word2vec) — on utilise la STRUCTURE du graphe mais :
    #   - pas de features de noeud (ignore gene_features)
    #   - pas de reconstruction probabiliste (pas de μ, pas de σ)
    #   - pas d'hétérogénéité de types d'arêtes (on aplatit tout)
    #
    # Si le ranking VGAE ≈ ranking DeepWalk → le VGAE n'apporte que la topologie,
    # pas le VAE (et pas les features). Si VGAE ≠ DeepWalk → le VAE + features
    # apportent quelque chose qu'on peut défendre en soutenance.
    #
    # NOTE : implémentation self-contained en pur PyTorch (pas de dépendance
    # torch-cluster, qui fournit normalement le C++ des random walks dans
    # PyG.Node2Vec). Walks uniformes = DeepWalk = Node2Vec(p=1, q=1).
    print("\n" + "=" * 70)
    print("13bis. Baseline DeepWalk (graphe sans VAE, sans features)")
    print("=" * 70)

    # Concaténer toutes les arêtes gène↔gène en un graphe homogène non orienté.
    _gene_edges_list = []
    for et in data.edge_types:
        if et[0] == "gene" and et[2] == "gene":
            ei = data[et].edge_index
            if ei.numel() > 0:
                _gene_edges_list.append(ei)
    _all_gene_ei = (torch.cat(_gene_edges_list, dim=1) if _gene_edges_list
                    else torch.zeros((2, 0), dtype=torch.long))
    # Symétrisation (random walks sur graphe non orienté)
    _all_gene_ei = torch.cat([_all_gene_ei, _all_gene_ei.flip(0)], dim=1)
    # Dédoublonnage des arêtes
    _u = torch.unique(_all_gene_ei.t(), dim=0).t()

    # --- Adjacence en format CSR pour échantillonner des voisins en vectoriel ---
    # adj_flat   : (E,) IDs des voisins, concaténés par noeud source
    # adj_offset : (n+1,) décalages → les voisins du noeud i sont
    #              adj_flat[adj_offset[i] : adj_offset[i+1]]
    # degrees    : (n,) nombre de voisins par noeud
    _src = _u[0].cpu().numpy()
    _dst = _u[1].cpu().numpy()
    # Tri par source pour construire CSR
    _sort_idx = np.argsort(_src, kind="stable")
    _src = _src[_sort_idx]
    _dst = _dst[_sort_idx]
    degrees = np.bincount(_src, minlength=n_genes).astype(np.int64)
    adj_offset = np.concatenate([[0], np.cumsum(degrees)]).astype(np.int64)
    adj_flat = torch.as_tensor(_dst, dtype=torch.long)
    adj_offset_t = torch.as_tensor(adj_offset, dtype=torch.long)
    degrees_t = torch.as_tensor(degrees, dtype=torch.long)
    n_isolated = int((degrees_t == 0).sum())
    print(f"  Graphe gène↔gène homogène : {n_genes} noeuds, "
          f"{_u.shape[1]} arêtes (dédupliquées), {n_isolated} isolés")


    def _random_walk_step(current, adj_flat, adj_offset_t, degrees_t):
        """Un pas de marche aléatoire uniforme, vectorisé.

        Pour chaque noeud courant :
          - si deg > 0 : tirer un voisin uniformément parmi adj
          - si deg = 0 : rester sur place (pas de voisin disponible)
        """
        deg = degrees_t[current]                           # (B,)
        rand = torch.rand(current.shape[0])                # (B,) ∈ [0,1)
        # Index d'un voisin dans adj_flat = offset[node] + ⌊rand × deg⌋
        safe_deg = deg.clamp(min=1)                         # évite div par 0
        idx = (rand * safe_deg.float()).long().clamp(max=safe_deg - 1)
        neighbor = adj_flat[adj_offset_t[current] + idx]   # (B,)
        return torch.where(deg > 0, neighbor, current)


    def _random_walks(start_nodes, walk_length):
        walks = torch.zeros(start_nodes.shape[0], walk_length, dtype=torch.long)
        walks[:, 0] = start_nodes
        for step in range(1, walk_length):
            walks[:, step] = _random_walk_step(
                walks[:, step - 1], adj_flat, adj_offset_t, degrees_t)
        return walks


    # --- Hyperparamètres du skip-gram ---
    WALKS_PER_NODE = 10
    WALK_LENGTH = 20
    WINDOW = 5           # context_size=10 → fenêtre 5 de chaque côté
    NUM_NEG = 5          # négatifs tirés par paire positive
    N2V_EPOCHS = 5
    N2V_BATCH = 128
    N2V_LR = 0.01

    # --- Table des embeddings (une seule matrice, partagée center/context) ---
    n2v_emb_param = nn.Embedding(n_genes, LATENT_DIM)
    nn.init.uniform_(n2v_emb_param.weight,
                     a=-0.5 / LATENT_DIM, b=0.5 / LATENT_DIM)
    n2v_optim = torch.optim.Adam(n2v_emb_param.parameters(), lr=N2V_LR)

    # Pré-calcul des offsets pour extraire les paires (center, context) d'un walk.
    # Pour chaque décalage j ∈ [-WINDOW, WINDOW] \ {0}, on récupère
    # (walks[:, i], walks[:, i+j]) pour tout i valide.
    _pair_offsets = [j for j in range(-WINDOW, WINDOW + 1) if j != 0]

    n2v_emb_param.train()
    for epoch in range(N2V_EPOCHS):
        # Tous les noeuds de départ (répétés WALKS_PER_NODE fois), mélangés.
        all_starts = torch.arange(n_genes).repeat_interleave(WALKS_PER_NODE)
        all_starts = all_starts[torch.randperm(all_starts.shape[0])]

        loss_sum = 0.0
        n_batches = 0
        for b in range(0, all_starts.shape[0], N2V_BATCH):
            batch_starts = all_starts[b:b + N2V_BATCH]
            walks = _random_walks(batch_starts, WALK_LENGTH)     # (B, L)

            centers, contexts = [], []
            for j in _pair_offsets:
                i_start = max(0, -j)
                i_end = WALK_LENGTH - max(0, j)
                centers.append(walks[:, i_start:i_end].flatten())
                contexts.append(walks[:, i_start + j:i_end + j].flatten())
            centers = torch.cat(centers)                          # (P,)
            contexts = torch.cat(contexts)
            n_pairs = centers.shape[0]
            if n_pairs == 0:
                continue

            # Négatifs : tirage uniforme sur {0,…,n_genes-1}.
            neg = torch.randint(0, n_genes, (n_pairs, NUM_NEG))

            u = n2v_emb_param(centers)                            # (P, D)
            v_pos = n2v_emb_param(contexts)                       # (P, D)
            v_neg = n2v_emb_param(neg)                            # (P, K, D)

            pos_score = (u * v_pos).sum(dim=1)                    # (P,)
            neg_score = torch.bmm(v_neg, u.unsqueeze(2)).squeeze(2)  # (P, K)
            pos_loss = -F.logsigmoid(pos_score).mean()
            neg_loss = -F.logsigmoid(-neg_score).mean()
            loss = pos_loss + neg_loss

            n2v_optim.zero_grad()
            loss.backward()
            n2v_optim.step()
            loss_sum += float(loss)
            n_batches += 1

        avg = loss_sum / max(1, n_batches)
        print(f"  DeepWalk epoch {epoch+1}/{N2V_EPOCHS} — "
              f"loss_mean={avg:.4f} ({n_batches} batches)")

    n2v_emb_param.eval()
    n2v_emb = n2v_emb_param.weight.detach().cpu().numpy()         # (n_genes, LATENT_DIM)

    # Score par gène : norme + densité k-NN, même structure que vgae_importance
    # mais sans recon_fidelity / certainty / specificity (ne s'appliquent pas à
    # un modèle non probabiliste sans features).
    n2v_norm_raw = np.linalg.norm(n2v_emb, axis=1)
    n2v_norm_score = (n2v_norm_raw / (n2v_norm_raw.max() + 1e-8)).astype(np.float32)

    n2v_emb_normed = n2v_emb / (np.linalg.norm(n2v_emb, axis=1, keepdims=True) + 1e-8)
    _k = min(20, n_genes - 1)
    _knn_n2v = NearestNeighbors(n_neighbors=_k + 1, metric="cosine")
    _knn_n2v.fit(n2v_emb_normed)
    _d_n2v, _ = _knn_n2v.kneighbors(n2v_emb_normed)
    _d_n2v = _d_n2v[:, 1:]
    n2v_density_raw = 1.0 / (_d_n2v.mean(axis=1) + 1e-8)
    n2v_density_score = (n2v_density_raw / (n2v_density_raw.max() + 1e-8)).astype(np.float32)

    node2vec_score = ((n2v_norm_score + n2v_density_score) / 2.0).astype(np.float32)
    node2vec_score = node2vec_score / (node2vec_score.max() + 1e-8)

    print(f"  Score Node2Vec : mean={node2vec_score.mean():.4f}")
    top_n2v = np.argsort(node2vec_score)[::-1][:10]
    print(f"  Top 10 gènes (baseline Node2Vec) :")
    for idx in top_n2v:
        print(f"    {gene_symbols[idx]:15s} score={node2vec_score[idx]:.4f}")

if RUN_VALIDATION:
    # =============================================================================
    # 14. VALIDATION POST-HOC (BDD externes)
    # =============================================================================
    # OBJECTIF : vérifier que les gènes à haut score VGAE sont enrichis dans des
    # bases de données INDÉPENDANTES de vieillissement/sénescence.
    #
    # POINT CRUCIAL : ces bases de données ne sont JAMAIS utilisées dans
    # l'entraînement du VGAE. Elles ne servent qu'à ÉVALUER après coup (post-hoc).
    # Si les gènes à haut score VGAE sont surreprésentés dans GenAge/CellAge/etc.,
    # cela valide que le score capture quelque chose de biologiquement pertinent
    # pour la sénescence — sans circularité.
    #
    # 5 bases de données de validation :
    #   1. GenAge : gènes humains associés au vieillissement (organisme entier)
    #   2. CellAge : gènes associés à la sénescence cellulaire spécifiquement
    #   3. MSigDB aging : gene sets Hallmark liés au vieillissement (MSigDB)
    #   4. AgeAnno : DEGs du vieillissement en scRNA-seq (multi-tissus)
    #   5. Aging local : base locale de gènes liés à l'âge (custom)
    #
    # Test statistique : Mann-Whitney U unilatéral.
    #   H0 : les gènes de la BDD n'ont pas un score plus élevé que les autres
    #   H1 : les gènes de la BDD ont un score significativement plus élevé
    #   p < 0.05 → enrichissement significatif → le score capture le signal
    print("\n" + "=" * 70)
    print("14. Validation post-hoc (GenAge, CellAge, MSigDB, AgeAnno)")
    print("   → PAS utilisées dans l'entraînement, uniquement pour évaluer")
    print("=" * 70)

    # ── Téléchargement des BDD ───────────────────────────────────────────────────
    GENAGE_ZIP = os.path.join(DB_DIR, "genage_human.zip")
    GENAGE_FILE = os.path.join(DB_DIR, "genage_human.csv")
    download_if_absent(
        "https://genomics.senescence.info/genes/human_genes.zip",
        GENAGE_ZIP, "GenAge"
    )
    if not os.path.exists(GENAGE_FILE):
        with zipfile.ZipFile(GENAGE_ZIP, "r") as z_file:
            csv_names = [n for n in z_file.namelist() if n.endswith(".csv")]
            if csv_names:
                with z_file.open(csv_names[0]) as f:
                    with open(GENAGE_FILE, "wb") as out:
                        out.write(f.read())

    genage = pd.read_csv(GENAGE_FILE)
    genage_symbols = set(genage["symbol"].dropna()) if "symbol" in genage.columns else set()

    MSIGDB_HALLMARK = os.path.join(DB_DIR, "h.all.symbols.gmt")
    download_if_absent(
        "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt",
        MSIGDB_HALLMARK, "MSigDB Hallmarks"
    )
    msigdb_sets = {}
    with open(MSIGDB_HALLMARK) as f:
        for line in f:
            parts = line.strip().split("\t")
            msigdb_sets[parts[0]] = set(parts[2:])

    AGING_KEYWORDS = ["SENESCENCE", "P53", "APOPTOSIS", "INFLAMMATORY", "TNFA",
                      "IL6", "KRAS", "MTORC", "REACTIVE_OXYGEN", "DNA_REPAIR",
                      "OXIDATIVE", "AGING"]
    msigdb_aging_genes = set()
    for name, genes_set in msigdb_sets.items():
        if any(kw in name.upper() for kw in AGING_KEYWORDS):
            msigdb_aging_genes |= genes_set

    CELLAGE_ZIP = os.path.join(DB_DIR, "cellAge.zip")
    CELLAGE_FILE = os.path.join(DB_DIR, "cellage3.tsv")
    download_if_absent(
        "https://genomics.senescence.info/cells/cellAge.zip",
        CELLAGE_ZIP, "CellAge"
    )
    if not os.path.exists(CELLAGE_FILE):
        with zipfile.ZipFile(CELLAGE_ZIP, "r") as z_file:
            tsv_names = [n for n in z_file.namelist() if n.lower().endswith(('.tsv', '.csv'))]
            if tsv_names:
                with z_file.open(tsv_names[0]) as f:
                    with open(CELLAGE_FILE, "wb") as out:
                        out.write(f.read())

    cellage = pd.read_csv(CELLAGE_FILE, sep='\t', engine='python',
                           on_bad_lines='skip', quoting=3, dtype=str)
    cellage_symbol_col = None
    for col in cellage.columns:
        if "symbol" in col.lower() or "gene" in col.lower() or "name" in col.lower():
            cellage_symbol_col = col
            break
    if cellage_symbol_col is None:
        cellage_symbol_col = cellage.columns[0]
    cellage_symbols = set(cellage[cellage_symbol_col].dropna().str.strip())

    AGEANNO_DIR = os.path.join(DB_DIR, "ageanno")
    os.makedirs(AGEANNO_DIR, exist_ok=True)
    AGEANNO_DEGS = os.path.join(AGEANNO_DIR, "aging_DEGs.txt")
    download_if_absent(
        "https://raw.githubusercontent.com/vikkihuangkexin/AgeAnno/main/scRNA/Aging-related%20DEGs.txt",
        AGEANNO_DEGS, "AgeAnno DEGs"
    )
    ageanno_degs = pd.read_csv(AGEANNO_DEGS, sep=",", encoding="latin-1")
    ageanno_genes = set(ageanno_degs["gene"].dropna().unique())

    aging_local = pd.read_csv(os.path.join(DATA_DIR, "human_age_related_gene.csv"))
    aging_local_symbols = set(aging_local["Symbol"].dropna())

    # ── Évaluation : les gènes à haut score sont-ils enrichis dans les BDD ? ────
    # Pour CHAQUE approche (VGAE, MLP-based, stat), on mesure l'enrichissement.
    # On utilise le Mann-Whitney U test : les gènes dans la BDD ont-ils un
    # score significativement plus élevé que les autres ?

    def evaluate_ranking(scores, score_name, gene_syms, databases):
        """
        Évalue un ranking par enrichissement dans les BDD externes.

        Pour chaque base de données, on compare les scores des gènes IN (dans la BDD)
        vs OUT (pas dans la BDD) avec un test de Mann-Whitney U unilatéral.
        Si p < 0.05, les gènes de la BDD ont un score significativement plus élevé
        → le score capture un signal pertinent pour cette BDD.
        """
        print(f"\n  [{score_name}]")
        print(f"    {'Base':20s} {'In_graph':>9s} {'MeanScore_in':>13s} "
              f"{'MeanScore_out':>14s} {'U_pvalue':>10s} {'Enrichi':>8s}")
        print("    " + "-" * 75)

        results = {}
        for db_name, db_genes in databases:
            in_graph = np.array([g in db_genes for g in gene_syms])
            n_in = in_graph.sum()
            if n_in < 5:
                print(f"    {db_name:20s} {n_in:9d}   (trop peu de gènes)")
                continue

            scores_in = scores[in_graph]
            scores_out = scores[~in_graph]
            mean_in = scores_in.mean()
            mean_out = scores_out.mean()

            # Test unilatéral : les gènes de la BDD ont-ils un score plus élevé ?
            _, p_val = mannwhitneyu(scores_in, scores_out, alternative="greater")
            enriched = "OUI" if p_val < 0.05 else "non"

            print(f"    {db_name:20s} {n_in:9d} {mean_in:13.4f} {mean_out:14.4f} "
                  f"{p_val:10.2e} {enriched:>8s}")
            results[db_name] = {"n": n_in, "mean_in": mean_in, "mean_out": mean_out,
                                "p_value": p_val, "enriched": p_val < 0.05}
        return results

    databases = [
        ("GenAge", genage_symbols),
        ("CellAge", cellage_symbols),
        ("MSigDB aging", msigdb_aging_genes),
        ("AgeAnno", ageanno_genes),
        ("Aging local", aging_local_symbols),
    ]

    vgae_results = evaluate_ranking(importance_score, "VGAE (non supervisé)",
                                     gene_symbols, databases)
    stat_results = evaluate_ranking(stat_score, "Baseline statistique (|ΔExpr|)",
                                     gene_symbols, databases)

# =============================================================================
# 15. VISUALISATIONS
# =============================================================================
# 3 figures principales :
#   15a. Courbes d'entraînement (loss + AUC au fil des epochs)
#        → vérifie la convergence et l'absence de collapse
#   15b. PCA des embeddings (3 vues : score, clusters, BDD sénescence)
#        → visualise la structure de l'espace latent en 2D
#   15c. VGAE vs baseline statistique (scatter + violin)
#        → identifie les candidats de découverte (haut VGAE, bas stat)
print("\n" + "=" * 70)
print("15. Visualisations")
print("=" * 70)

# ── 15a. Loss et AUC d'entraînement ─────────────────────────────────────────
# Panneau gauche : loss totale (reconstruction + KL) au fil des epochs.
#   La loss devrait diminuer puis se stabiliser.
# Panneau droit : AUC test au fil des epochs.
#   AUC > 0.9 = bonne reconstruction, ~0.5 = collapse.
#   La ligne verte (MLP baseline) montre le niveau atteignable sans graphe.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, color="#3498DB", alpha=0.8)
axes[0].axvline(best_epoch, ls="--", color="grey", lw=0.8,
                label=f"Best (epoch {best_epoch+1})")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("VGAE — Loss (reconstruction + KL)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(eval_epochs, test_aucs, color="#E74C3C", alpha=0.8, marker="o", ms=3)
if RUN_BASELINES:
    axes[1].axhline(mlp_auc, color="#2ECC71", ls="--", lw=1.5, label=f"MLP baseline ({mlp_auc:.3f})")
axes[1].axhline(0.5, color="grey", ls=":", lw=1)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("AUC")
axes[1].set_title("VGAE vs MLP — Reconstruction d'arêtes (test)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "vgae_training.png"))
plt.close(fig)
print("  → vgae_training.png")

# ── 15b. PCA des embeddings ─────────────────────────────────────────────────
# PCA 2D sur les embeddings 64D pour visualisation.
# Les embeddings sont dans un espace de haute dimension (latent_dim=64) —
# la PCA projette sur les 2 axes de plus grande variance.
# 3 sous-figures :
#   Gauche : coloré par score d'importance → les gènes rouges sont les plus importants
#   Centre : coloré par cluster K-means → structure fonctionnelle
#   Droite : gènes des BDD de sénescence en rouge → vérification visuelle
pca = PCA_sk(n_components=2)
gene_pca = pca.fit_transform(gene_emb)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Coloré par score d'importance
sc0 = axes[0].scatter(gene_pca[:, 0], gene_pca[:, 1], c=importance_score,
                       cmap="YlOrRd", s=8, alpha=0.7, rasterized=True)
plt.colorbar(sc0, ax=axes[0], label="Score d'importance")
axes[0].set_title("Embeddings VGAE — Score d'importance émergent")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

# Coloré par cluster K-means
cluster_colors = plt.cm.tab10(np.linspace(0, 1, N_CLUSTERS))
for c in range(N_CLUSTERS):
    mask_c = gene_clusters == c
    axes[1].scatter(gene_pca[mask_c, 0], gene_pca[mask_c, 1],
                    c=[cluster_colors[c]], s=8, alpha=0.7, label=f"C{c}",
                    rasterized=True)
axes[1].legend(fontsize=7, ncol=2)
axes[1].set_title(f"Embeddings VGAE — K-means (k={N_CLUSTERS}, sil={sil_score:.3f})")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

# Coloré par présence dans les BDD de sénescence
any_db = np.array([
    any(g in db for _, db in databases) for g in gene_symbols
])
colors_db = np.where(any_db, "#E74C3C", "#CCCCCC")
axes[2].scatter(gene_pca[~any_db, 0], gene_pca[~any_db, 1],
                c="#CCCCCC", s=5, alpha=0.3, label="Autres", rasterized=True)
axes[2].scatter(gene_pca[any_db, 0], gene_pca[any_db, 1],
                c="#E74C3C", s=15, alpha=0.8, label="BDD sénescence", rasterized=True)
axes[2].legend()
axes[2].set_title("Embeddings VGAE — Gènes des BDD de sénescence")
axes[2].set_xlabel("PC1")
axes[2].set_ylabel("PC2")

fig.suptitle("Espace latent du VGAE (non supervisé)", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "vgae_embeddings.png"))
plt.close(fig)
print("  → vgae_embeddings.png")

# ── 15c. Comparaison des scores VGAE vs stat ────────────────────────────────
# Figure clé pour identifier les "candidats de découverte".
# Panneau gauche : scatter VGAE (y) vs stat (x).
#   - Diagonale : gènes identifiés par les deux approches (concordance)
#   - QUADRANT HAUT-GAUCHE : haut VGAE + bas stat → candidats de découverte
#     Ce sont des gènes que l'approche statistique simple (|ΔExpr|) ne
#     trouverait pas, mais que le VGAE identifie grâce à leur rôle dans
#     le réseau biologique. C'est la plus-value principale du VGAE.
#   - Quadrant bas-droite : haut stat + bas VGAE → gènes très dérégulés
#     mais pas centraux dans le réseau (housekeeping dérégulés, etc.)
# Panneau droit : violin plot du score VGAE par BDD de validation.
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter VGAE vs stat (tous les gènes en bleu)
axes[0].scatter(stat_score, importance_score, s=5, alpha=0.3,
                c="#3498DB", rasterized=True)
# Surligner les gènes des BDD de sénescence en rouge
for g_idx in range(n_genes):
    if any(gene_symbols[g_idx] in db for _, db in databases):
        axes[0].scatter(stat_score[g_idx], importance_score[g_idx],
                        s=20, c="#E74C3C", alpha=0.7, zorder=5)
axes[0].set_xlabel("Score statistique (|ΔExpr|)")
axes[0].set_ylabel("Score VGAE (émergent)")
axes[0].set_title("VGAE vs Baseline statistique")
axes[0].grid(True, alpha=0.3)

# Définition des candidats de découverte :
#   - haut VGAE = top 10% (percentile 90)
#   - bas stat = bottom 50% (médiane)
# Les gènes dans cette zone sont détectés par le VGAE MAIS PAS par la
# méthode statistique simple → découvertes potentielles.
high_vgae = importance_score > np.percentile(importance_score, 90)
low_stat = stat_score < np.percentile(stat_score, 50)
discovery_candidates = high_vgae & low_stat
n_disc = discovery_candidates.sum()
# Lignes pointillées délimitant le quadrant de découverte
axes[0].axhline(np.percentile(importance_score, 90), color="red", ls=":", alpha=0.5)
axes[0].axvline(np.percentile(stat_score, 50), color="red", ls=":", alpha=0.5)
axes[0].annotate(f"Candidats de\ndécouverte\n(n={n_disc})",
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, ha='left', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Distribution des scores par BDD
all_score_data = []
for db_name, db_genes in databases:
    in_mask = np.array([g in db_genes for g in gene_symbols])
    if in_mask.sum() > 0:
        all_score_data.extend([
            {"gene": g, "score": importance_score[i], "source": f"{db_name} (in)",
             "method": "VGAE"}
            for i, g in enumerate(gene_symbols) if in_mask[i]
        ])

if databases:
    axes[1].violinplot(
        [importance_score[np.array([g in db for g in gene_symbols])]
         for _, db in databases if np.array([g in db for g in gene_symbols]).sum() > 5],
        showmeans=True, showmedians=True,
    )
axes[1].set_xticks(range(1, len(databases) + 1))
axes[1].set_xticklabels([name for name, _ in databases], rotation=45, ha="right", fontsize=9)
axes[1].set_ylabel("Score VGAE")
axes[1].set_title("Score VGAE par base de données")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "vgae_vs_baseline.png"))
plt.close(fig)
print("  → vgae_vs_baseline.png")

# =============================================================================
# 16. EXPORT DES RÉSULTATS
# =============================================================================
# 3 fichiers de sortie principaux :
#   1. gene_ranking_vgae.csv : tableau complet (1 ligne par gène) avec tous
#      les scores, rangs, clusters, et annotations BDD. C'est le fichier
#      principal utilisé par perturb_top_genes.py pour sélectionner les
#      gènes à perturber in silico.
#   2. gene_embeddings_vgae.csv : embeddings bruts μ (n_genes × latent_dim).
#      Utilisés par gnn_perturbation.py pour charger le modèle et calculer
#      les effets des perturbations.
#   3. vgae_weights.pt : poids du modèle PyTorch (encoder + décodeur τ).
#      Utilisés par gnn_perturbation.py pour reconstruire le modèle et
#      simuler des perturbations (knockdown, knockout, overexpress).
print("\n" + "=" * 70)
print("16. Export des résultats")
print("=" * 70)

results = pd.DataFrame({"gene": gene_symbols})

# --- Scores (les 5 composantes + le composite + la baseline statistique) ---
results["vgae_importance"] = importance_score       # Score composite final
results["vgae_emb_norm"] = emb_norm_score           # Composante 1 : norme μ
results["vgae_density"] = density_score             # Composante 2 : densité k-NN
results["vgae_recon_fidelity"] = 1 - recon_error_score  # Composante 3 : 1-erreur
results["vgae_certainty"] = 1 - uncertainty_score   # Composante 4 : 1-σ
results["vgae_specificity"] = specificity_score     # Composante 5 : spécificité
results["stat_score"] = stat_score                  # Baseline 1 : |ΔExpr P16-P4| (toujours)
if RUN_BASELINES:
    results["mlp_score"] = mlp_gene_score           # Baseline 2 : MLP sans graphe (features only)
    results["node2vec_score"] = node2vec_score      # Baseline 3 : Node2Vec (graphe sans VAE, sans features)
results["cluster"] = gene_clusters                  # Cluster K-means (0 à N_CLUSTERS-1)

# --- Rangs (pour faciliter les comparaisons entre runs) ---
results["rank_vgae"] = results["vgae_importance"].rank(ascending=False).astype(int)
results["rank_stat"] = results["stat_score"].rank(ascending=False).astype(int)
if RUN_BASELINES:
    results["rank_mlp"] = results["mlp_score"].rank(ascending=False).astype(int)
    results["rank_node2vec"] = results["node2vec_score"].rank(ascending=False).astype(int)

# --- Candidats de découverte (haut VGAE, bas statistique) ---
# Flag binaire : 1 si le gène est dans le quadrant haut-gauche du scatter
results["discovery_candidate"] = discovery_candidates.astype(int)

# --- Annotations BDD de validation (binaire, 0 ou 1) ---
# Permet de vérifier rapidement si un gène du top ranking est déjà connu
# dans les bases de sénescence/vieillissement.
results["in_genage"] = [1 if g in genage_symbols else 0 for g in gene_symbols]
results["in_cellage"] = [1 if g in cellage_symbols else 0 for g in gene_symbols]
results["in_msigdb_aging"] = [1 if g in msigdb_aging_genes else 0 for g in gene_symbols]
results["in_ageanno"] = [1 if g in ageanno_genes else 0 for g in gene_symbols]
results["in_aging_local"] = [1 if g in aging_local_symbols else 0 for g in gene_symbols]
# n_databases : nombre de BDD dans lesquelles le gène apparaît (0-5)
# Un gène dans 5/5 BDD est un gène de sénescence "classique" bien validé.
# Un gène dans 0/5 BDD mais avec un haut score VGAE est une découverte potentielle.
results["n_databases"] = results[["in_genage", "in_cellage", "in_msigdb_aging",
                                   "in_ageanno", "in_aging_local"]].sum(axis=1)

# Tri par score d'importance décroissant (les gènes les plus importants en premier)
results = results.sort_values("vgae_importance", ascending=False)
results.to_csv(os.path.join(OUT_DIR, "gene_ranking_vgae.csv"), index=False)
print(f"  → gene_ranking_vgae.csv ({len(results)} gènes)")

# Embeddings bruts (matrice n_genes × latent_dim) — index = symboles de gènes
emb_df = pd.DataFrame(gene_emb, index=gene_symbols)
emb_df.to_csv(os.path.join(OUT_DIR, "gene_embeddings_vgae.csv"))
print(f"  → gene_embeddings_vgae.csv ({gene_emb.shape})")

# Poids du modèle PyTorch (pour réutilisation dans les perturbations)
torch.save(model.state_dict(), os.path.join(OUT_DIR, "vgae_weights.pt"))
print("  → vgae_weights.pt")

# Sidecar JSON : historique d'entraînement + métriques finales — permet
# aux scripts d'analyse cross-seed de reconstruire courbes loss/AUC/AP
# sans devoir reparser log.txt. Format human-readable, indépendant de torch.
import json as _json

# --- Stats d'arêtes (V4.1) : breakdown par type + signe + couverture vs PPI ---
# Pour chaque edge_type construit en section 6/7, on rapporte :
#   - n_edges (taille du edge_index)
#   - n_pos / n_neg / n_unsigned (si edge_attr[:,1] = sign)
# Et aux niveaux agrégés :
#   - n_signed_total = somme des arêtes des types signés (signaling, tf_curated)
#   - frac_signed   = n_signed / total (mesure le poids causal du graphe)
#   - signed_non_ppi : arêtes signées dont la paire (a,b) n'a PAS d'arête PPI
#       → quantifie ce qu'OmniPath apporte au-delà de STRING (apport causal pur).
def _edge_stats(ei, attr=None):
    """Compte arêtes d'un type, et breakdown +/−/0 si attr[:,1]=sign."""
    n = int(ei.shape[1]) if ei.numel() > 0 else 0
    out = {"n_edges": n}
    if (attr is not None and attr.numel() > 0
            and attr.dim() == 2 and attr.shape[1] >= 2):
        # convention V4 : colonne 1 = sign ∈ {−1, 0, +1}
        sign_col = attr[:, 1]
        out["n_pos"]      = int((sign_col > 0).sum())
        out["n_neg"]      = int((sign_col < 0).sum())
        out["n_unsigned"] = int((sign_col == 0).sum())
    return out


# Catalogue { edge_type_name : (edge_index, edge_attr_ou_None) }. None pour
# les types sans edge_attr (same_pathway) ou non instanciés.
_edge_catalog = {
    "ppi":                   (edge_index_ppi,         edge_attr_ppi),
    "same_pathway":          (edge_index_pathway,     None),
    "regulates":             (edge_index_regulates,   edge_attr_regulates),
    "coexpression":          (edge_index_coexpr,      coexpr_w_tensor),
    "metabolic_cocatalysis": (edge_index_cocat,       edge_attr_cocat),
    "signaling":             (edge_index_signaling,   edge_attr_signaling),
    "tf_curated":            (edge_index_tf_curated,  edge_attr_tf_curated),
    "reactome_fi":           (edge_index_reactome_fi, edge_attr_reactome_fi),
    "expresses":             (edge_index_expresses,   edge_attr_expresses),
}
_per_type = {name: _edge_stats(ei, attr)
             for name, (ei, attr) in _edge_catalog.items()}

# Agrégats : types qui portent un signe biologique (causal/régulatoire).
# reactome_fi (V4.2) est signé (edge_attr=[score, sign]) → inclus.
_signed_types = ("signaling", "tf_curated", "regulates", "reactome_fi")
_signed_total_edges = sum(_per_type[t]["n_edges"] for t in _signed_types)
_total_gene_gene = sum(_per_type[t]["n_edges"] for t in
                       ("ppi", "same_pathway", "regulates", "coexpression",
                        "metabolic_cocatalysis", "signaling", "tf_curated",
                        "reactome_fi"))

# Couverture vs PPI : combien d'arêtes signées (signaling / tf_curated) NE
# sont PAS doublées par une arête PPI (bidirectionnelle) ? Mesure l'apport
# causal pur d'OmniPath au-delà de STRING.
_ppi_pairs: set[tuple[int, int]] = set()
if edge_index_ppi.numel() > 0:
    _ppi_pairs = {(int(s), int(d)) for s, d in zip(
        edge_index_ppi[0].tolist(), edge_index_ppi[1].tolist())}


def _count_non_ppi(ei) -> int:
    """Nb d'arêtes signées dont aucune des deux orientations n'est dans PPI."""
    if ei.numel() == 0 or not _ppi_pairs:
        return int(ei.shape[1]) if ei.numel() > 0 else 0
    cnt = 0
    for s, d in zip(ei[0].tolist(), ei[1].tolist()):
        if (int(s), int(d)) not in _ppi_pairs and (int(d), int(s)) not in _ppi_pairs:
            cnt += 1
    return cnt


_signaling_non_ppi  = _count_non_ppi(edge_index_signaling)
_tf_curated_non_ppi = _count_non_ppi(edge_index_tf_curated)
_reactome_fi_non_ppi = _count_non_ppi(edge_index_reactome_fi)

# Logs console — utile pour diag interactif sans ouvrir le JSON
print("\n  Stats arêtes (V4.1) :")
for name, st in _per_type.items():
    extra = ""
    if "n_pos" in st:
        extra = (f"  [+:{st['n_pos']} −:{st['n_neg']} "
                 f"0:{st['n_unsigned']}]")
    print(f"    {name:<22} : {st['n_edges']:>8}{extra}")
if (_per_type["signaling"]["n_edges"] > 0
        or _per_type["tf_curated"]["n_edges"] > 0
        or _per_type["reactome_fi"]["n_edges"] > 0):
    print(f"  Couverture causale hors PPI :")
    print(f"    signaling non-PPI   : {_signaling_non_ppi} "
          f"/ {_per_type['signaling']['n_edges']}")
    print(f"    tf_curated non-PPI  : {_tf_curated_non_ppi} "
          f"/ {_per_type['tf_curated']['n_edges']}")
    if _per_type["reactome_fi"]["n_edges"] > 0:
        print(f"    reactome_fi non-PPI : {_reactome_fi_non_ppi} "
              f"/ {_per_type['reactome_fi']['n_edges']}")
if _total_gene_gene > 0:
    print(f"  frac signed / total gene-gene : "
          f"{_signed_total_edges / _total_gene_gene:.3f}")

_edge_stats_block = {
    "per_type": _per_type,
    "aggregates": {
        "n_signed_total":   int(_signed_total_edges),
        "n_gene_gene_total": int(_total_gene_gene),
        "frac_signed_of_gene_gene": (
            float(_signed_total_edges / _total_gene_gene)
            if _total_gene_gene > 0 else 0.0),
    },
    "omnipath_vs_ppi": {
        "signaling_total":      _per_type["signaling"]["n_edges"],
        "signaling_non_ppi":    int(_signaling_non_ppi),
        "tf_curated_total":     _per_type["tf_curated"]["n_edges"],
        "tf_curated_non_ppi":   int(_tf_curated_non_ppi),
        "reactome_fi_total":    _per_type["reactome_fi"]["n_edges"],
        "reactome_fi_non_ppi":  int(_reactome_fi_non_ppi),
    },
    "include_omnipath_genes":   bool(MODULES["include_omnipath_genes"]),
    "n_omnipath_endpoints_in_graph": int(
        len(set(omnipath_endpoints) & set(gene_symbols))
        if omnipath_endpoints else 0),
}

_metrics = {
    "version": 1,
    "seed": int(getattr(CLI_ARGS, "seed", 42)),
    "run_tag": str(globals().get("RUN_TAG", "")),
    "best_epoch": int(best_epoch + 1),  # 1-indexé pour cohérence avec les logs
    "best_auc": float(best_test_auc),
    "best_ap": float(best_test_ap),
    "mlp_auc": float(mlp_auc) if RUN_BASELINES else None,
    "mlp_ap": float(mlp_ap) if RUN_BASELINES else None,
    "delta_auc_vgae_minus_mlp": float(best_test_auc - mlp_auc) if RUN_BASELINES else None,
    "n_epochs_planned": int(N_EPOCHS),
    "n_epochs_run": int(len(train_losses)),
    "early_stopped": bool(len(train_losses) < N_EPOCHS),
    "patience": int(PATIENCE),
    "n_genes": int(n_genes),
    "n_train_edges": int(len(train_edges)),
    "n_test_edges": int(len(test_edges)),
    "history": {
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": [float(x) for x in train_losses],
        "recon_loss": [float(x) for x in recon_losses],
        "kl_loss": [float(x) for x in kl_losses],
        "kl_beta": [float(x) for x in kl_betas],
    },
    "eval_history": {
        "epoch": [int(e + 1) for e in eval_epochs],
        "test_auc": [float(x) for x in test_aucs],
        "test_ap": [float(x) for x in test_aps],
    },
    "hyperparams": {
        "lr": float(LR),
        "kl_beta_max": float(KL_BETA_MAX),
        "kl_warmup_epochs": int(KL_WARMUP_EPOCHS),
        "free_bits": float(FREE_BITS),
        "latent_dim": int(LATENT_DIM),
        "edge_sample_ratio": float(EDGE_SAMPLE_RATIO),
        "grad_clip_norm": float(GRAD_CLIP_NORM),
        # V5 flags — requis par gnn_perturbation.load_run pour reconstruire
        # le bon encoder (SignedGATConv) et accepter le state_dict
        # (clés bilinear_decoder.*). Sans ces flags persistés, le load
        # silencieusement ignore les signes ou refuse le state_dict.
        "signed_message": bool(getattr(CLI_ARGS, "signed_message", False)),
        "signed_decoder": bool(getattr(CLI_ARGS, "signed_decoder", False)),
        "signed_loss_weight": float(getattr(CLI_ARGS, "signed_loss_weight", 0.0)),
        # V5.4 (decoder-split) — requis pour identifier les runs V5.4 et
        # reproduire le régime (cosinus non-signé + bilinéaire existence).
        "decoder_split": bool(getattr(CLI_ARGS, "decoder_split", False)),
        # V4.3 — choix méthode×prune amont (consommé par cross-method report).
        "coexpr_method": CLI_ARGS.coexpr_method,
        "coexpr_prune": CLI_ARGS.coexpr_prune,
    },
    "edge_stats": _edge_stats_block,
}
with open(os.path.join(OUT_DIR, "vgae_metrics.json"), "w") as _f:
    _json.dump(_metrics, _f, indent=2)
print("  → vgae_metrics.json")

# Expression par groupe cellulaire — requis par gnn_perturbation.py pour le
# shift pondéré au niveau gène. On exporte mean_expression et pct_expressing
# sous leur forme BRUTE (avant Z-scoring), car gnn_vgae.py applique un Z-score
# in-place à edge_attr_expresses ce qui détruit les valeurs originales dans le
# HeteroData sauvé.
_group_expr_rows = {"gene": gene_symbols}
for grp in CELL_GROUPS:
    _group_expr_rows[f"mean_{grp}"] = group_stats[grp]["mean_expression"]
    _group_expr_rows[f"pct_{grp}"] = group_stats[grp]["pct_expressing"]
pd.DataFrame(_group_expr_rows).to_csv(
    os.path.join(OUT_DIR, "group_expression.tsv"), sep="\t", index=False)
print(f"  → group_expression.tsv (mean + pct × {len(CELL_GROUPS)} groupes)")

# =============================================================================
# RÉSUMÉ
# =============================================================================
print("\n" + "=" * 70)
print("RÉSUMÉ — VGAE NON SUPERVISÉ + BASELINES")
print("=" * 70)

# Compter les candidats de découverte dans les BDD
disc_genes = gene_symbols[discovery_candidates]
disc_in_db = sum(1 for g in disc_genes
                 if any(g in db for _, db in databases))

_mlp_auc_str = f"{mlp_auc:.4f}" if RUN_BASELINES else "sautée (--no-baselines)"
_delta_str = ((f"{best_test_auc - mlp_auc:+.4f} "
               f"({'topologie utile' if best_test_auc > mlp_auc else 'topologie non utile'})")
              if RUN_BASELINES else "—")
print(f"""
Graphe hétérogène :
  Noeuds gene       : {n_genes} (features: is_tf, variance, ppi_degree, reg_degree)
  Noeuds cell_group : {len(CELL_GROUPS)}
  Arêtes expresses  : {edge_index_expresses.shape[1]} (7 features)
  Arêtes PPI        : {edge_index_ppi.shape[1]}
  Arêtes pathway    : {edge_index_pathway.shape[1]}
  Arêtes regulates  : {edge_index_regulates.shape[1]}
  Arêtes coexpr     : {edge_index_coexpr.shape[1] if edge_index_coexpr.numel() > 0 else 0}
  TOTAL gene↔gene   : {n_edges} arêtes uniques

VGAE :
  Architecture : {N_LAYERS} couches GATConv × {N_HEADS} têtes, latent={LATENT_DIM}
  Paramètres   : {total_params:,}
  Best epoch   : {best_epoch + 1}

Reconstruction d'arêtes (test) :
  VGAE  AUC : {best_test_auc:.4f}
  MLP   AUC : {_mlp_auc_str}
  Δ AUC     : {_delta_str}

Candidats de découverte :
  (haut score VGAE + bas score statistique)
  {n_disc} gènes, dont {disc_in_db} dans les BDD de sénescence

Fichiers : {OUT_DIR}/
Figures  : {FIG_DIR}/
""")
