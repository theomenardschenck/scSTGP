################################################################################
#
#  PIPELINE pySCENIC — Analyse de réseau de régulation transcriptionnelle
#  Sénescence cellulaire réplicative (HUVECs P16)
#
#  Prérequis : avoir exécuté le pipeline R (senescence_pipeline.R)
#              et sauvegardé l'objet Seurat P16
#
#  Ce script :
#    1. Exporte la count matrix depuis R
#    2. Exécute les 3 étapes SCENIC (GRNBoost2 → RcisTarget → AUCell)
#    3. Analyse et visualise les regulons / TFs
#
################################################################################
"""
================================================================================
PIPELINE pySCENIC — HUVECs P16 Sénescence
================================================================================

Installation :
    pip install pyscenic loompy scanpy pandas numpy seaborn matplotlib umap-learn

Fichiers de référence à télécharger (une seule fois) :
    1. Base de données de TFs humains :
       https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt

    2. Base de données de motifs (feather) — CHOISIR UNE des deux :
       https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg38/
       Fichier : hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather
       OU
       hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather

    3. Annotations motifs → TFs :
       https://resources.aertslab.org/cistarget/motif2tf/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl

    Stocke ces fichiers dans un dossier "scenic_refs/"

================================================================================
"""

import os
import glob
import pickle
import pandas as pd
import numpy as np
import scanpy as sc
import loompy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def main():

    from dask.distributed import Client, LocalCluster
    import os
    
    # Option recommandée : limiter le nombre de workers pour éviter les problèmes de mémoire
    n_workers = min(6, os.cpu_count() or 4)   # adapte selon ta machine
    
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,          # 1 thread : évite race condition dans ctxcore
        dashboard_address=None,
        silence_logs=True,             # réduit le bruit
        memory_limit="8GB"             # adapte selon ta RAM
    )
    client = Client(cluster)

    print(f"Dask cluster démarré avec {n_workers} workers")

    # =============================================================================
    # 1. CONFIGURATION DES CHEMINS
    # =============================================================================

    # --- Adapte ces chemins à ton système ---
    WORK_DIR = os.path.expanduser("/home/USER/M2/S2/Stage/Projet_Colin/huvec_gnn/data/pyscenic")
    RESULTS_DIR = "/home/USER/M2/S2/Stage/Projet_Colin/huvec_gnn/output/pyscenic"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Fichier d'expression exporté depuis R
    EXPR_MATRIX = os.path.join(WORK_DIR, "expr_matrix_P16.csv")
    CELL_METADATA = os.path.join(WORK_DIR, "cell_metadata_P16.csv")

    # Fichiers de référence SCENIC (à télécharger, voir ci-dessus)
    REFS_DIR = os.path.join(WORK_DIR, "scenic_refs")

    TF_LIST = os.path.join(REFS_DIR, "allTFs_hg38.txt")
    RANKING_DB = glob.glob(os.path.join(REFS_DIR, "*.feather"))
    MOTIF_ANNOTATIONS = os.path.join(
        REFS_DIR, "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"
    )

    # Fichiers intermédiaires
    ADJACENCIES_FILE = os.path.join(RESULTS_DIR, "adjacencies.csv")
    REGULONS_FILE = os.path.join(RESULTS_DIR, "regulons.pkl")
    AUCELL_LOOM = os.path.join(RESULTS_DIR, "scenic_output.loom")
    AUCELL_MTX = os.path.join(RESULTS_DIR, "auc_matrix.csv")

    # =============================================================================
    # 2. CHARGER LES DONNÉES
    # =============================================================================

    print("=" * 60)
    print("Chargement des données...")
    print("=" * 60)

    # Charger la matrice d'expression (cellules × gènes)
    expr_df = pd.read_csv(EXPR_MATRIX, index_col=0)
    print(f"Matrice d'expression : {expr_df.shape[0]} cellules × {expr_df.shape[1]} gènes")

    # Charger les métadonnées
    metadata = pd.read_csv(CELL_METADATA, index_col=0)
    print(f"Métadonnées : {metadata.shape[0]} cellules, {metadata.shape[1]} colonnes")

    # Charger la liste des TFs
    tf_list = pd.read_csv(TF_LIST, header=None)[0].tolist()
    print(f"Nombre de TFs de référence : {len(tf_list)}")

    # Filtrer les TFs présents dans notre matrice
    # Note : les underscores ont été remplacés par des tirets dans R
    tf_in_data = [tf for tf in tf_list if tf in expr_df.columns]
    tf_in_data_dash = [
        tf for tf in tf_list
        if tf.replace("_", "-") in expr_df.columns
    ]
    # Combiner les deux
    tf_in_data = list(set(tf_in_data + [
        tf.replace("_", "-") for tf in tf_list
        if tf.replace("_", "-") in expr_df.columns
    ]))
    print(f"TFs trouvés dans les données : {len(tf_in_data)}")

    # =============================================================================
    # 3. ÉTAPE 1 — GRNBoost2 : Inférence du réseau de co-expression
    # =============================================================================
    # GRNBoost2 identifie les paires TF-gène cible par gradient boosting.
    # C'est l'étape la plus longue (~10-30 min selon le nombre de cellules/gènes).

    print("\n" + "=" * 60)
    print("ÉTAPE 1 : GRNBoost2 — Réseau de co-expression")
    print("=" * 60)

    if os.path.exists(ADJACENCIES_FILE):
        print(f"Fichier d'adjacences déjà existant : {ADJACENCIES_FILE}")
        print("Chargement des résultats précédents...")
        adjacencies = pd.read_csv(ADJACENCIES_FILE)
    else:
        from arboreto.algo import grnboost2

        # Exécuter GRNBoost2
        print("Exécution de GRNBoost2 (peut prendre 10-30 minutes)...")
        adjacencies = grnboost2(
            expression_data=expr_df,
            tf_names=tf_in_data,
            client_or_address=client,
            seed=42,
            verbose=True
        )

        # Sauvegarder les adjacences
        adjacencies.to_csv(ADJACENCIES_FILE, index=False)
        print(f"Adjacences sauvegardées : {ADJACENCIES_FILE}")

    print(f"Nombre de paires TF-cible : {len(adjacencies)}")
    print(f"Nombre de TFs uniques : {adjacencies['TF'].nunique()}")
    print("\nTop 10 interactions par importance :")
    print(adjacencies.sort_values("importance", ascending=False).head(10))

    # =============================================================================
    # 4. ÉTAPE 2 — RcisTarget : Filtrage par motifs cis-régulateurs
    # =============================================================================
    # RcisTarget ne garde que les paires TF-cible ayant un motif de liaison enrichi
    # dans la région promotrice du gène cible. Ça élimine les corrélations spurieuses.

    print("\n" + "=" * 60)
    print("ÉTAPE 2 : RcisTarget — Filtrage par motifs")
    print("=" * 60)

    if os.path.exists(REGULONS_FILE):
        print(f"Regulons déjà calculés : {REGULONS_FILE}")
        with open(REGULONS_FILE, "rb") as f:
            regulons = pickle.load(f)
    else:
        from pyscenic.prune import prune2df, df2regulons
        from pyscenic.utils import modules_from_adjacencies
        from ctxcore.rnkdb import FeatherRankingDatabase

        # Charger les bases de données de ranking
        print("Chargement des bases de données de motifs...")
        dbs = [FeatherRankingDatabase(db_path, name=os.path.basename(db_path)) for db_path in RANKING_DB]
        print(f"Bases de données chargées : {len(dbs)}")

        # Convertir les adjacences en modules (Sequence[Regulon])
        print("Conversion des adjacences en modules...")
        modules = modules_from_adjacencies(adjacencies, expr_df)
        print(f"Modules créés : {len(modules)}")

        # Pruning : ne garder que les modules enrichis en motifs
        print("Pruning des modules (peut prendre quelques minutes)...")
        df_motifs = prune2df(
            rnkdbs=dbs,
            modules=modules,
            motif_annotations_fname=MOTIF_ANNOTATIONS,
            client_or_address="custom_multiprocessing",
            num_workers=n_workers,
            module_chunksize=100,
        )

        # Convertir en regulons
        regulons = df2regulons(df_motifs)

        # Sauvegarder
        with open(REGULONS_FILE, "wb") as f:
            pickle.dump(regulons, f)
        print(f"Regulons sauvegardés : {REGULONS_FILE}")

    print(f"Nombre de regulons (TFs) identifiés : {len(regulons)}")

    # Afficher les regulons et leur taille
    regulon_sizes = {r.name: len(r.genes) for r in regulons}
    regulon_sizes_df = pd.DataFrame(
        list(regulon_sizes.items()),
        columns=["Regulon", "Nb_genes_cibles"]
    ).sort_values("Nb_genes_cibles", ascending=False)

    print("\nTop 20 regulons par nombre de gènes cibles :")
    print(regulon_sizes_df.head(20).to_string(index=False))

    # =============================================================================
    # 5. ÉTAPE 3 — AUCell : Scoring de l'activité des regulons par cellule
    # =============================================================================
    # AUCell calcule un score (AUC) pour chaque regulon dans chaque cellule.
    # Score élevé = le regulon est actif dans cette cellule.

    print("\n" + "=" * 60)
    print("ÉTAPE 3 : AUCell — Activité des regulons par cellule")
    print("=" * 60)

    if os.path.exists(AUCELL_MTX):
        print(f"Matrice AUCell déjà calculée : {AUCELL_MTX}")
        auc_matrix = pd.read_csv(AUCELL_MTX, index_col=0)
    else:
        from pyscenic.aucell import aucell

        # Calculer les scores AUCell
        print("Calcul des scores AUCell...")
        auc_matrix = aucell(expr_df, regulons)

        # Sauvegarder
        auc_matrix.to_csv(AUCELL_MTX)
        print(f"Matrice AUCell sauvegardée : {AUCELL_MTX}")

    print(f"Matrice AUCell : {auc_matrix.shape[0]} cellules × {auc_matrix.shape[1]} regulons")

    # =============================================================================
    # 6. ANALYSE DES RÉSULTATS
    # =============================================================================

    print("\n" + "=" * 60)
    print("ANALYSE DES RÉSULTATS")
    print("=" * 60)

    # Ajouter les clusters aux données AUCell
    auc_with_clusters = auc_matrix.copy()
    auc_with_clusters["cluster"] = metadata.loc[auc_matrix.index, "seurat_clusters"]

    # --- 6a. Top 25 TFs les plus variables (comme dans l'article) ---
    tf_variance = auc_matrix.var(axis=0).sort_values(ascending=False)
    top25_tfs = tf_variance.head(25).index.tolist()

    print(f"\nTop 25 TFs les plus variables :")
    for i, tf in enumerate(top25_tfs, 1):
        print(f"  {i:2d}. {tf} (variance: {tf_variance[tf]:.6f})")

    # --- 6b. Activité moyenne des TFs par cluster ---
    mean_activity = auc_with_clusters.groupby("cluster")[top25_tfs].mean()
    print("\nActivité moyenne des top 25 TFs par cluster :")
    print(mean_activity.round(4).to_string())

    # =============================================================================
    # 7. VISUALISATIONS
    # =============================================================================

    print("\n" + "=" * 60)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("=" * 60)

    # --- 7a. Heatmap des top 25 TFs par cluster ---
    fig, ax = plt.subplots(figsize=(14, 8))

    # Z-score par TF pour mieux voir les différences entre clusters
    mean_zscore = mean_activity.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    sns.heatmap(
        mean_zscore.T,
        cmap="RdBu_r",
        center=0,
        annot=mean_activity.T.round(3),
        fmt="",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Z-score d'activité"}
    )
    ax.set_title("Activité des Top 25 TFs par cluster (Z-score)", fontsize=14)
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Facteur de transcription", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "heatmap_top25_TFs_par_cluster.pdf"),
        dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(RESULTS_DIR, "heatmap_top25_TFs_par_cluster.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.show()
    print("Heatmap sauvegardée.")

    # --- 7b. Heatmap AUCell complète (cellules individuelles) ---
    fig, ax = plt.subplots(figsize=(16, 10))

    # Trier les cellules par cluster
    sorted_cells = auc_with_clusters.sort_values("cluster").index
    auc_sorted = auc_matrix.loc[sorted_cells, top25_tfs]

    # Créer les annotations de cluster pour les lignes
    cluster_labels = auc_with_clusters.loc[sorted_cells, "cluster"]
    cluster_colors_map = dict(
        zip(
            sorted(cluster_labels.unique()),
            sns.color_palette("tab10", n_colors=cluster_labels.nunique())
        )
    )
    row_colors = cluster_labels.map(cluster_colors_map)

    g = sns.clustermap(
        auc_sorted,
        col_cluster=True,
        row_cluster=False,
        row_colors=row_colors,
        cmap="YlOrRd",
        figsize=(16, 10),
        xticklabels=True,
        yticklabels=False,
        cbar_kws={"label": "Score AUCell"}
    )
    g.fig.suptitle("Activité AUCell — Top 25 TFs (cellules triées par cluster)", y=1.02)
    plt.savefig(
        os.path.join(RESULTS_DIR, "clustermap_AUCell_top25.pdf"),
        dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(RESULTS_DIR, "clustermap_AUCell_top25.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.show()
    print("Clustermap sauvegardée.")

    # --- 7c. Violin plots des TFs clés par cluster ---
    # Sélectionner les 6 TFs les plus variables pour le violin plot
    top6_tfs = top25_tfs[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, tf in enumerate(top6_tfs):
        data_plot = pd.DataFrame({
            "AUCell Score": auc_matrix[tf],
            "Cluster": metadata.loc[auc_matrix.index, "seurat_clusters"].astype(str)
        })

        sns.violinplot(
            data=data_plot,
            x="Cluster",
            y="AUCell Score",
            ax=axes[i],
            palette="Set2",
            inner="box",
            cut=0
        )
        axes[i].set_title(tf, fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Cluster")
        axes[i].set_ylabel("AUCell Score")

    plt.suptitle(
        "Distribution de l'activité des TFs clés par cluster",
        fontsize=15, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "violin_top6_TFs.pdf"),
        dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(RESULTS_DIR, "violin_top6_TFs.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.show()
    print("Violin plots sauvegardés.")

    # --- 7d. UMAP colorée par activité des TFs (via Scanpy) ---
    print("\nGénération des UMAP colorées par activité TF...")

    # Créer un objet AnnData pour Scanpy
    adata = sc.AnnData(X=expr_df.values, obs=metadata.loc[expr_df.index])
    adata.var_names = expr_df.columns.tolist()
    adata.obs_names = expr_df.index.tolist()

    # Ajouter les scores AUCell comme observations
    for tf in top6_tfs:
        adata.obs[f"AUCell_{tf}"] = auc_matrix.loc[adata.obs_names, tf].values

    # PCA + UMAP via Scanpy
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=10)
    sc.pp.neighbors(adata, n_pcs=10)
    sc.tl.umap(adata)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, tf in enumerate(top6_tfs):
        sc.pl.umap(
            adata,
            color=f"AUCell_{tf}",
            ax=axes[i],
            show=False,
            title=tf,
            color_map="RdYlBu_r"
        )

    plt.suptitle("UMAP — Activité AUCell des TFs clés", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "umap_AUCell_top6_TFs.pdf"),
        dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(RESULTS_DIR, "umap_AUCell_top6_TFs.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.show()
    print("UMAP AUCell sauvegardées.")

    # =============================================================================
    # 8. EXPORT DES RÉSULTATS POUR LE GNN
    # =============================================================================

    print("\n" + "=" * 60)
    print("EXPORT DES DONNÉES POUR LE GNN HÉTÉROGÈNE")
    print("=" * 60)

    # --- 8a. Table des regulons (TF → gènes cibles) ---
    regulon_table = []
    for reg in regulons:
        tf_name = reg.name
        for gene, weight in zip(reg.genes, reg.weights):
            regulon_table.append({
                "TF": tf_name,
                "target_gene": gene,
                "weight": weight
            })

    regulon_df = pd.DataFrame(regulon_table)
    regulon_df.to_csv(
        os.path.join(RESULTS_DIR, "regulon_edges_TF_to_gene.csv"),
        index=False
    )
    print(f"Arêtes TF → gène cible : {len(regulon_df)} interactions")

    # --- 8b. Matrice AUCell (activité TF par cellule) ---
    # Déjà sauvegardée en AUCELL_MTX

    # --- 8c. Activité TF par cluster (résumé) ---
    mean_activity.to_csv(os.path.join(RESULTS_DIR, "mean_TF_activity_per_cluster.csv"))
    print("Activité moyenne TF par cluster sauvegardée.")

    # --- 8d. Adjacences GRNBoost2 (réseau de co-expression complet) ---
    # Déjà sauvegardé en ADJACENCIES_FILE

    # --- 8e. Résumé pour le GNN ---
    summary = {
        "Nombre de regulons (TFs)": len(regulons),
        "Nombre total d'interactions TF-cible": len(regulon_df),
        "Nombre de cellules": auc_matrix.shape[0],
        "Top 25 TFs variables": ", ".join(top25_tfs),
    }

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["Métrique", "Valeur"]
    )
    summary_df.to_csv(
        os.path.join(RESULTS_DIR, "scenic_summary.csv"),
        index=False
    )

    print("\n" + "=" * 60)
    print("PIPELINE pySCENIC TERMINÉ")
    print("=" * 60)
    print(f"\nRésultats dans : {RESULTS_DIR}")
    print(f"  - heatmap_top25_TFs_par_cluster.pdf/png")
    print(f"  - clustermap_AUCell_top25.pdf/png")
    print(f"  - violin_top6_TFs.pdf/png")
    print(f"  - umap_AUCell_top6_TFs.pdf/png")
    print(f"  - regulon_edges_TF_to_gene.csv     ← arêtes pour le GNN")
    print(f"  - mean_TF_activity_per_cluster.csv  ← features nœuds pour le GNN")
    print(f"  - auc_matrix.csv                    ← activité cellule-level")
    print(f"  - adjacencies.csv                   ← réseau co-expression complet")
    print(f"  - scenic_summary.csv")
    print()
    print("POUR LE GNN HÉTÉROGÈNE :")
    print("  Nœuds 'TF'       : les regulons identifiés")
    print("  Nœuds 'gène'     : les gènes cibles de chaque regulon")
    print("  Nœuds 'cluster'  : les 4 clusters de sénescence")
    print("  Arêtes TF→gène   : regulon_edges_TF_to_gene.csv (pondérées)")
    print("  Arêtes TF→cluster: mean_TF_activity_per_cluster.csv (activité)")
    print("  Arêtes gène→gène : adjacencies.csv (co-expression GRNBoost2)")

    client.close()
    cluster.close()

if __name__ == '__main__':
    main()

################################################################################
# NOTES
################################################################################
#
# 1. FICHIERS DE RÉFÉRENCE :
#    Les bases de données feather sont volumineuses (~1-2 Go chacune).
#    Télécharge-les une seule fois et réutilise-les pour d'autres analyses.
#    URL : https://resources.aertslab.org/cistarget/databases/
#
# 2. TEMPS D'EXÉCUTION :
#    - GRNBoost2 : 10-30 min (dépend du nombre de cellules et gènes)
#    - RcisTarget : 5-15 min
#    - AUCell : 1-5 min
#    Total : environ 30-60 min pour ~5000 cellules × 5000 gènes
#
# 3. MÉMOIRE :
#    Les bases feather sont memory-mapped, mais prévois ~8-16 Go de RAM.
#
# 4. ALTERNATIVE LIGNE DE COMMANDE :
#    Si tu préfères la CLI pySCENIC :
#
#    # Étape 1 : GRNBoost2
#    pyscenic grn expr_matrix.loom allTFs_hg38.txt -o adjacencies.csv
#
#    # Étape 2 : RcisTarget (pruning)
#    pyscenic ctx adjacencies.csv \
#        hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather \
#        --annotations_fname motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl \
#        --expression_mtx_fname expr_matrix.loom \
#        --output regulons.csv
#
#    # Étape 3 : AUCell
#    pyscenic aucell expr_matrix.loom regulons.csv \
#        --output scenic_output.loom
#
# 5. TIRETS vs UNDERSCORES :
#    Seurat remplace les "_" par "-" dans les noms de gènes. pySCENIC et les
#    bases de motifs utilisent les noms originaux avec "_". Le script gère
#    cette correspondance, mais vérifie que les TFs sont bien trouvés.
#    Si peu de TFs sont détectés, essaie de reconvertir :
#    expr_df.columns = expr_df.columns.str.replace("-", "_")
#
################################################################################