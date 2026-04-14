"""
Analyse pySCENIC : Inférence de réseaux de régulation — HUVEC (GSE98440)
========================================================================
Pipeline SCENIC (Single-Cell rEgulatory Network Inference and Clustering)
appliqué aux données bulk RNA-seq HUVEC prolifératives vs sénescentes.

  Étape 1 — GRNBoost2/GENIE3 : inférence du réseau de co-expression TF → cibles
  Étape 2 — cisTarget : validation par enrichissement de motifs (pruning)
  Étape 3 — AUCell    : scoring d'activité des régulons par échantillon

Note : SCENIC est conçu pour le single-cell, mais fonctionne en bulk
       RNA-seq. Avec 6 échantillons les résultats sont exploratoires ;
       ils complètent les régulons AgeAnno déjà utilisés dans le GNN.

Prérequis :
  pip install pyscenic

Bases cisTarget (à placer dans data/databases/scenic/) :
  Télécharger depuis https://resources.aertslab.org/cistarget/ :
  - hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather
  - hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather
  - motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl
"""

import os
import sys
import json
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu


# ── Vérification des dépendances pySCENIC ───────────────────────────────────
try:
    from arboreto.algo import grnboost2
    from arboreto.algo import genie3
    from pyscenic.utils import modules_from_adjacencies
    from pyscenic.prune import prune2df, df2regulons
    from pyscenic.aucell import aucell
    from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
except ImportError as e:
    print("=" * 60)
    print("ERREUR : dépendances pySCENIC manquantes")
    print("=" * 60)
    print(f"  {e}")
    print("\n  Installation :")
    print("    pip install pyscenic")
    print("\n  Ou via conda :")
    print("    conda install -c bioconda pyscenic")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "RNAseq")
DB_DIR = os.path.join(BASE_DIR, "..", "data", "databases", "scenic")
AGEANNO_DIR = os.path.join(BASE_DIR, "..", "data", "databases", "ageanno")
OUT_DIR = os.path.join(BASE_DIR, "..", "output", "scenic")
FIG_DIR = os.path.join(OUT_DIR, "figure")
for d in [DB_DIR, OUT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Fichiers RNA-seq ─────────────────────────────────────────────────────────
COUNTS_FILE = os.path.join(DATA_DIR, "GSE98440_norm_counts_HUVECpro_sen.csv")
DE_FILE = os.path.join(DATA_DIR, "GSE98440_diff_expr_analysis_afterNorm_HUVEC_2reps.txt")

# ── Fichiers cisTarget ───────────────────────────────────────────────────────
RANKING_DBS = [
    os.path.join(DB_DIR, "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"),
    os.path.join(DB_DIR, "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"),
]
MOTIF_ANNOTATIONS = os.path.join(DB_DIR, "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl")
TF_LIST_FILE = os.path.join(DB_DIR, "allTFs_hg38.txt")

# ── AgeAnno (comparaison) ───────────────────────────────────────────────────
AGEANNO_TF_FILE = os.path.join(AGEANNO_DIR, "TF_regulon.txt")

# ── Paramètres ───────────────────────────────────────────────────────────────
PADJ_THRESH = 0.05
LFC_THRESH = 1.0
N_WORKERS = 4                   # Threads pour GRNBoost2/GENIE3
ADJACENCY_THRESH = 1.0          # Seuil importance GRNBoost2/GENIE3 pour garder un lien
MIN_GENES_REGULON = 5           # Taille min d'un régulon après pruning
TOP_N_REGULONS_PLOT = 30        # Nombre de régulons à afficher dans les figures

# ── Fichiers de cache (résultats intermédiaires) ────────────────────────────
CACHE_ADJACENCIES = os.path.join(OUT_DIR, "adjacencies.csv")
CACHE_REGULONS = os.path.join(OUT_DIR, "regulons.json")
CACHE_AUC = os.path.join(OUT_DIR, "auc_matrix.csv")

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


# =============================================================================
# UTILITAIRES
# =============================================================================
def download_if_absent(url, local_path, label=""):
    """Télécharge un fichier si absent."""
    if os.path.exists(local_path):
        print(f"    [cache] {label or os.path.basename(local_path)}")
        return True
    print(f"    Téléchargement {label or os.path.basename(local_path)}...")
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0 (research)")
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        with open(local_path, "wb") as f:
            f.write(data)
        print(f"      OK ({len(data) / 1e6:.1f} MB)")
        return True
    except Exception as exc:
        print(f"      ÉCHEC : {exc}")
        return False


def main():
    """Fonction principale contenant tout le code d'analyse SCENIC"""

    # Exemple typique avec pySCENIC + Dask :
    
    from dask.distributed import Client, LocalCluster
    import os
    
    # Option recommandée : limiter le nombre de workers pour éviter les problèmes de mémoire
    n_workers = min(6, os.cpu_count() or 4)   # adapte selon ta machine
    
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=2,          # 1 ou 2 selon ta machine
        dashboard_address=None,
        silence_logs=True,             # réduit le bruit
        memory_limit="8GB"             # adapte selon ta RAM
    )
    client = Client(cluster)
    
    print(f"Dask cluster démarré avec {n_workers} workers")
        
    # =============================================================================
    # 1. CHARGEMENT DES DONNÉES RNA-SEQ
    # =============================================================================
    print("=" * 70)
    print("1. Chargement des données RNA-seq")
    print("=" * 70)

    # Counts normalisés (Ensembl IDs × 6 échantillons)
    counts_raw = pd.read_csv(COUNTS_FILE, sep="\t", index_col=0)
    print(f"  Counts : {counts_raw.shape[0]} gènes × {counts_raw.shape[1]} échantillons")

    # Table DE pour le mapping Ensembl → HGNC
    de = pd.read_csv(DE_FILE, sep="\t")
    de = de[de["ensembl_gene_id"].str.startswith("ENSG", na=False)].copy()
    de = de.dropna(subset=["hgnc_symbol"])
    de = de[de["hgnc_symbol"] != "NA"]

    # Mapping Ensembl → HGNC (un symbole par Ensembl ID, garder le premier)
    ensembl_to_hgnc = dict(zip(de["ensembl_gene_id"], de["hgnc_symbol"]))

    # Convertir la matrice de counts en symboles HGNC
    counts_hgnc = counts_raw.copy()
    counts_hgnc.index = counts_hgnc.index.map(lambda x: ensembl_to_hgnc.get(x, None))
    counts_hgnc = counts_hgnc[counts_hgnc.index.notna()]
    # En cas de doublons, garder celui avec l'expression la plus élevée
    counts_hgnc = counts_hgnc.groupby(counts_hgnc.index).max()

    # Filtrer les gènes faiblement exprimés (mean > 10 counts)
    gene_mean = counts_hgnc.mean(axis=1)
    counts_filtered = counts_hgnc[gene_mean > 10]
    print(f"  Après mapping HGNC et filtrage : {counts_filtered.shape[0]} gènes")

    # Log-transformation pour GRNBoost2/GENIE3
    expression = np.log2(counts_filtered + 1)

    # Transposer : pySCENIC attend (échantillons × gènes)
    expression_matrix = expression.T
    print(f"  Matrice d'expression : {expression_matrix.shape[0]} échantillons × "
        f"{expression_matrix.shape[1]} gènes")

    # Identifier les conditions
    pro_cols = [c for c in expression_matrix.index if "pro" in c]
    sen_cols = [c for c in expression_matrix.index if "sen" in c]
    print(f"  Prolifératifs : {pro_cols}")
    print(f"  Sénescents    : {sen_cols}")


    # =============================================================================
    # 2. LISTE DES FACTEURS DE TRANSCRIPTION
    # =============================================================================
    print("\n" + "=" * 70)
    print("2. Liste des facteurs de transcription humains")
    print("=" * 70)

    download_if_absent(
        "https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt",
        TF_LIST_FILE,
        "Liste TFs humains (hg38)",
    )

    if os.path.exists(TF_LIST_FILE):
        tf_list = [line.strip() for line in open(TF_LIST_FILE) if line.strip()]
    else:
        print("  ATTENTION : liste TF non disponible, utilisation des TFs AgeAnno")
        tf_list = []
        if os.path.exists(AGEANNO_TF_FILE):
            ageanno_tf = pd.read_csv(AGEANNO_TF_FILE, sep=",", encoding="latin-1")
            tf_list = list(ageanno_tf["TF"].dropna().unique())

    # Filtrer les TFs présents dans notre matrice d'expression
    tf_names = sorted(set(tf_list) & set(expression_matrix.columns))
    print(f"  TFs dans la liste       : {len(tf_list)}")
    print(f"  TFs dans nos données    : {len(tf_names)}")

    # Garder seulement les gènes les plus variables (fortement recommandé)
    # Cela réduit le bruit et évite que l'algorithme ne trouve presque rien
    n_top_genes = min(8000, expression_matrix.shape[1])   # ou 5000 si trop lent
    gene_var = expression_matrix.var(axis=0)
    top_genes = gene_var.nlargest(n_top_genes).index
    expr_reduced = expression_matrix[top_genes]

    # =============================================================================
    # 3. ÉTAPE 1 — GRNBoost2/GENIE3 : Inférence du réseau de régulation
    # =============================================================================
    print("\n" + "=" * 70)
    print("3. GRNBoost2/GENIE3 — Inférence du réseau TF → cibles")
    print("=" * 70)
    print("  (Note : 6 échantillons bulk — résultats exploratoires)")
    
    print(f"  Réduction à {len(top_genes)} gènes les plus variables pour GRN")

    if os.path.exists(CACHE_ADJACENCIES):
        print(f"  [cache] Chargement des adjacences depuis {os.path.basename(CACHE_ADJACENCIES)}")
        adjacencies = pd.read_csv(CACHE_ADJACENCIES)
    # else:
    #     print(f"  Lancement GRNBoost2/GENIE3 ({len(tf_names)} TFs × "
    #         f"{expression_matrix.shape[1]} gènes, {N_WORKERS} threads)...")
    #     adjacencies = GRNBoost2/GENIE3(
    #         expression_data=expression_matrix,
    #         tf_names=tf_names,
    #         verbose=True,
    #         seed=42,
    #     )
    #     adjacencies.to_csv(CACHE_ADJACENCIES, index=False)
    #     print(f"  → Sauvé : {os.path.basename(CACHE_ADJACENCIES)}")
    else:
        print(f"  Lancement GENIE3 ({len(tf_names)} TFs × "
              f"{expr_reduced.shape[1]} gènes)...")
        
        # Utilise GENIE3 au lieu de GRNBoost2/GENIE3
        from arboreto.algo import genie3
        
        adjacencies = genie3(
            expression_data=expr_reduced,
            tf_names=tf_names,
            verbose=True,
            seed=42,
            # client_or_address=client,   # GENIE3 peut fonctionner sans client dask dans certaines versions
        )
        
        adjacencies.to_csv(CACHE_ADJACENCIES, index=False)
        print(f"  → Sauvé : {os.path.basename(CACHE_ADJACENCIES)}")

    print(f"\n  Résultat :")
    print(f"    {len(adjacencies)} liens TF → cible")
    print(f"    {adjacencies['TF'].nunique()} TFs actifs")
    print(f"    Importance : min={adjacencies['importance'].min():.3f}, "
        f"max={adjacencies['importance'].max():.3f}, "
        f"median={adjacencies['importance'].median():.3f}")

    # Filtrer par seuil d'importance
    adj_filtered = adjacencies[adjacencies["importance"] > ADJACENCY_THRESH].copy()
    print(f"    Après filtrage (importance > {ADJACENCY_THRESH}) : {len(adj_filtered)} liens")


    # =============================================================================
    # 4. VISUALISATIONS 
    # =============================================================================
    print("\n" + "=" * 70)
    print("4. Visualisations")
    print("=" * 70)

    # ── 4a. Distribution des scores d'importance ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(adjacencies["importance"], bins=100, color="#3498DB",
                edgecolor="white", lw=0.3, log=True)
    axes[0].axvline(ADJACENCY_THRESH, ls="--", color="#E74C3C", lw=1.2,
                    label=f"Seuil = {ADJACENCY_THRESH}")
    axes[0].set_xlabel("Importance (GRNBoost2/GENIE3)")
    axes[0].set_ylabel("Nombre de liens (log)")
    axes[0].set_title("Distribution des scores d'importance")
    axes[0].legend()

    # Top 30 TFs par nombre de cibles (filtrées)
    tf_counts = adj_filtered.groupby("TF").size().nlargest(TOP_N_REGULONS_PLOT)
    axes[1].barh(range(len(tf_counts)), tf_counts.values, color="#E74C3C")
    axes[1].set_yticks(range(len(tf_counts)))
    axes[1].set_yticklabels(tf_counts.index, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Nombre de gènes cibles")
    axes[1].set_title(f"Top {TOP_N_REGULONS_PLOT} TFs (importance > {ADJACENCY_THRESH})")

    fig.suptitle("GRNBoost2/GENIE3 — Réseau de régulation HUVEC", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "GRNBoost2/GENIE3_overview.png"), bbox_inches="tight")
    plt.close(fig)
    print("  → Sauvé : GRNBoost2/GENIE3_overview.png")

    # ── 4b. Top TFs par importance moyenne ───────────────────────────────────────
    tf_mean_imp = adj_filtered.groupby("TF")["importance"].agg(["mean", "count"])
    tf_mean_imp = tf_mean_imp[tf_mean_imp["count"] >= MIN_GENES_REGULON]
    tf_mean_imp = tf_mean_imp.sort_values("mean", ascending=False).head(TOP_N_REGULONS_PLOT)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(tf_mean_imp)), tf_mean_imp["mean"], color="#9B59B6")
    ax.set_yticks(range(len(tf_mean_imp)))
    ax.set_yticklabels(tf_mean_imp.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance moyenne (GRNBoost2/GENIE3)")
    ax.set_title(f"Top TFs par importance moyenne (≥{MIN_GENES_REGULON} cibles)")
    # Annoter avec le nombre de cibles
    for i, (_, row) in enumerate(tf_mean_imp.iterrows()):
        ax.text(row["mean"] + 0.05, i, f"n={int(row['count'])}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "top_tf_importance.png"))
    plt.close(fig)
    print("  → Sauvé : top_tf_importance.png")


    # =============================================================================
    # 5. ÉTAPE 2 — cisTarget : Enrichissement de motifs (pruning)
    # =============================================================================
    print("\n" + "=" * 70)
    print("5. cisTarget — Enrichissement de motifs")
    print("=" * 70)

    # Vérifier la disponibilité des bases cisTarget
    dbs_available = all(os.path.exists(f) for f in RANKING_DBS)
    motifs_available = os.path.exists(MOTIF_ANNOTATIONS)

    regulons = None

    if os.path.exists(CACHE_REGULONS):
        print(f"  [cache] Chargement des régulons depuis {os.path.basename(CACHE_REGULONS)}")
        with open(CACHE_REGULONS) as f:
            regulons_dict = json.load(f)
        print(f"    {len(regulons_dict)} régulons chargés")

        # Reconstruire les modules pour AUCell (format frozenset attendu)
        # On utilise df2regulons via une reconstruction simplifiée
        from ctxcore.genesig import GeneSignature
        regulons = [
            GeneSignature(name=name, gene2weight=targets)
            for name, targets in regulons_dict.items()
            if len(targets) >= MIN_GENES_REGULON
        ]
        print(f"    {len(regulons)} régulons avec ≥{MIN_GENES_REGULON} gènes")

    elif dbs_available and motifs_available:
        print("  Bases cisTarget trouvées — lancement du pruning...")

        # Créer les modules à partir des adjacences
        modules = modules_from_adjacencies(adjacencies, expression_matrix)
        print(f"    {len(modules)} modules créés à partir des adjacences")

        # Charger les bases de ranking
        dbs = [RankingDatabase(fname=f) for f in RANKING_DBS]
        print(f"    {len(dbs)} bases de ranking chargées")

        # Pruning par enrichissement de motifs
        print("    Pruning en cours (peut prendre plusieurs minutes)...")
        df_motifs = prune2df(dbs, modules, MOTIF_ANNOTATIONS, num_workers=N_WORKERS)

        # Convertir en régulons
        regulons = df2regulons(df_motifs)
        regulons = [r for r in regulons if len(r) >= MIN_GENES_REGULON]
        print(f"    {len(regulons)} régulons après pruning (≥{MIN_GENES_REGULON} gènes)")

        # Sauvegarder en JSON (TF → {target: weight})
        regulons_dict = {}
        for reg in regulons:
            regulons_dict[reg.name] = dict(reg.gene2weight) if hasattr(reg, 'gene2weight') else {g: 1.0 for g in reg.genes}
        with open(CACHE_REGULONS, "w") as f:
            json.dump(regulons_dict, f, indent=2)
        print(f"  → Sauvé : {os.path.basename(CACHE_REGULONS)}")

    else:
        print("  ATTENTION : Bases cisTarget non trouvées.")
        print("  L'étape de pruning par motifs est ignorée.")
        print(f"  Fichiers manquants :")
        for f in RANKING_DBS:
            if not os.path.exists(f):
                print(f"    - {os.path.basename(f)}")
        if not motifs_available:
            print(f"    - {os.path.basename(MOTIF_ANNOTATIONS)}")
        print("\n  Téléchargement (dans data/databases/scenic/) :")
        print("    https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg38/refseq_r80/mc_v10_clust/gene_based/")
        print("    https://resources.aertslab.org/cistarget/motif2tf/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl")

        # Fallback : créer des régulons non-prunés à partir des adjacences filtrées
        print("\n  Utilisation des adjacences GRNBoost2/GENIE3 brutes (sans pruning motif)...")
        from ctxcore.genesig import GeneSignature
        regulons_dict = {}
        for tf, group in adj_filtered.groupby("TF"):
            targets = dict(zip(group["target"], group["importance"]))
            if len(targets) >= MIN_GENES_REGULON:
                regulons_dict[f"{tf}(+)"] = targets

        regulons = [
            GeneSignature(name=name, gene2weight=targets)
            for name, targets in regulons_dict.items()
        ]
        print(f"    {len(regulons)} régulons (non prunés, importance > {ADJACENCY_THRESH})")

        with open(CACHE_REGULONS, "w") as f:
            json.dump(regulons_dict, f, indent=2)
        print(f"  → Sauvé : {os.path.basename(CACHE_REGULONS)}")


    # =============================================================================
    # 6. ÉTAPE 3 — AUCell : Scoring d'activité des régulons
    # =============================================================================
    print("\n" + "=" * 70)
    print("6. AUCell — Activité des régulons par échantillon")
    print("=" * 70)

    if regulons is None or len(regulons) == 0:
        print("  Aucun régulon disponible — étape ignorée.")
        auc_mtx = None
    else:
        if os.path.exists(CACHE_AUC):
            print(f"  [cache] Chargement depuis {os.path.basename(CACHE_AUC)}")
            auc_mtx = pd.read_csv(CACHE_AUC, index_col=0)
        else:
            print(f"  Calcul AUCell pour {len(regulons)} régulons × "
                f"{expression_matrix.shape[0]} échantillons...")
            auc_mtx = aucell(expression_matrix, regulons, num_workers=N_WORKERS)
            auc_mtx.to_csv(CACHE_AUC)
            print(f"  → Sauvé : {os.path.basename(CACHE_AUC)}")

        print(f"\n  Matrice AUCell : {auc_mtx.shape[0]} échantillons × "
            f"{auc_mtx.shape[1]} régulons")
        print(f"  AUC : min={auc_mtx.values.min():.4f}, "
            f"max={auc_mtx.values.max():.4f}, "
            f"mean={auc_mtx.values.mean():.4f}")


    # =============================================================================
    # 7. ANALYSE DIFFÉRENTIELLE DES RÉGULONS (Prolif vs Sénescent)
    # =============================================================================
    print("\n" + "=" * 70)
    print("7. Activité différentielle des régulons")
    print("=" * 70)

    diff_results = None

    if auc_mtx is not None and len(auc_mtx.columns) > 0:
        results = []
        for regulon_name in auc_mtx.columns:
            vals_pro = auc_mtx.loc[pro_cols, regulon_name].values
            vals_sen = auc_mtx.loc[sen_cols, regulon_name].values

            mean_pro = vals_pro.mean()
            mean_sen = vals_sen.mean()
            diff = mean_sen - mean_pro

            # Mann-Whitney U (non-paramétrique, adapté aux petits échantillons)
            try:
                stat, pval = mannwhitneyu(vals_sen, vals_pro, alternative="two-sided")
            except ValueError:
                pval = 1.0

            results.append({
                "regulon": regulon_name,
                "mean_prolif": mean_pro,
                "mean_senescent": mean_sen,
                "diff_sen_pro": diff,
                "pvalue": pval,
            })

        diff_results = pd.DataFrame(results).sort_values("pvalue")
        diff_results.to_csv(os.path.join(OUT_DIR, "regulon_activity_diff.csv"), index=False)
        print(f"  → Sauvé : regulon_activity_diff.csv")

        n_sig = (diff_results["pvalue"] < 0.05).sum()
        print(f"  Régulons significatifs (p < 0.05) : {n_sig} / {len(diff_results)}")
        print(f"  (Note : n=3 par groupe — puissance statistique limitée)")

        # Top régulons
        if n_sig > 0:
            print(f"\n  Top régulons différentiels :")
            for _, row in diff_results.head(10).iterrows():
                direction = "↑ Sén." if row["diff_sen_pro"] > 0 else "↓ Sén."
                print(f"    {row['regulon']:20s} {direction} "
                    f"Δ={row['diff_sen_pro']:+.4f}  p={row['pvalue']:.4f}")


    # =============================================================================
    # 8. VISUALISATIONS SCENIC
    # =============================================================================
    print("\n" + "=" * 70)
    print("8. Visualisations SCENIC")
    print("=" * 70)

    if auc_mtx is not None and len(auc_mtx.columns) > 0:

        # ── 8a. Heatmap d'activité des régulons ──────────────────────────────────
        # Sélectionner les régulons les plus variables ou significatifs
        if diff_results is not None:
            top_regulons = diff_results.head(TOP_N_REGULONS_PLOT)["regulon"].tolist()
        else:
            auc_var = auc_mtx.var()
            top_regulons = auc_var.nlargest(TOP_N_REGULONS_PLOT).index.tolist()

        if len(top_regulons) > 0:
            plot_data = auc_mtx[top_regulons].T

            # Z-score par régulon pour la visualisation
            plot_z = plot_data.subtract(plot_data.mean(axis=1), axis=0).divide(
                plot_data.std(axis=1).clip(lower=1e-8), axis=0
            )

            # Annoter les colonnes avec la condition
            col_colors = pd.Series(
                ["#2ECC71" if "pro" in c else "#E74C3C" for c in plot_z.columns],
                index=plot_z.columns,
                name="Condition",
            )

            cmap = LinearSegmentedColormap.from_list("bwr", ["#3498DB", "white", "#E74C3C"])
            g = sns.clustermap(
                plot_z,
                cmap=cmap,
                center=0,
                vmin=-2,
                vmax=2,
                row_cluster=True,
                col_cluster=False,
                yticklabels=True,
                xticklabels=True,
                figsize=(8, max(6, len(top_regulons) * 0.35)),
                col_colors=col_colors,
                dendrogram_ratio=(0.1, 0.05),
                cbar_kws={"label": "Z-score (AUCell)"},
                linewidths=0.3,
            )
            g.fig.suptitle("Activité des régulons — HUVEC Prolif. vs Sénescent",
                            y=1.02, fontsize=13)
            g.savefig(os.path.join(FIG_DIR, "regulon_activity_heatmap.png"),
                    bbox_inches="tight")
            plt.close("all")
            print("  → Sauvé : regulon_activity_heatmap.png")

        # ── 8b. Barplot des régulons différentiels ───────────────────────────────
        if diff_results is not None and len(diff_results) > 0:
            top_diff = diff_results.head(TOP_N_REGULONS_PLOT).copy()
            top_diff = top_diff.sort_values("diff_sen_pro")

            fig, ax = plt.subplots(figsize=(10, max(6, len(top_diff) * 0.35)))
            colors_bar = ["#E74C3C" if d > 0 else "#3498DB"
                        for d in top_diff["diff_sen_pro"]]
            ax.barh(range(len(top_diff)), top_diff["diff_sen_pro"], color=colors_bar)
            ax.set_yticks(range(len(top_diff)))
            ax.set_yticklabels(top_diff["regulon"], fontsize=8)
            ax.axvline(0, color="black", lw=0.5)
            ax.set_xlabel("Δ AUCell (Sénescent − Prolifératif)")
            ax.set_title("Régulons différentiels — HUVEC")

            # Marquer les significatifs
            for i, (_, row) in enumerate(top_diff.iterrows()):
                marker = " *" if row["pvalue"] < 0.05 else ""
                ax.text(
                    row["diff_sen_pro"] + (0.001 if row["diff_sen_pro"] >= 0 else -0.001),
                    i,
                    f"p={row['pvalue']:.3f}{marker}",
                    va="center",
                    ha="left" if row["diff_sen_pro"] >= 0 else "right",
                    fontsize=7,
                )

            fig.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, "differential_regulons.png"),
                        bbox_inches="tight")
            plt.close(fig)
            print("  → Sauvé : differential_regulons.png")

        # ── 8c. PCA sur les scores AUCell ────────────────────────────────────────
        if auc_mtx.shape[1] >= 2:
            from sklearn.decomposition import PCA as PCAmodel
            pca = PCAmodel(n_components=min(2, auc_mtx.shape[0] - 1))
            pcs = pca.fit_transform(auc_mtx.values)

            fig, ax = plt.subplots(figsize=(8, 7))
            conditions = ["Prolifératif" if "pro" in s else "Sénescent"
                        for s in auc_mtx.index]
            cond_colors = {"Prolifératif": "#2ECC71", "Sénescent": "#E74C3C"}

            for cond, color in cond_colors.items():
                mask = [c == cond for c in conditions]
                ax.scatter(pcs[mask, 0], pcs[mask, 1] if pcs.shape[1] > 1 else np.zeros(sum(mask)),
                        c=color, s=150, label=cond, edgecolors="black", zorder=3)
                for idx, m in enumerate(mask):
                    if m:
                        label_text = auc_mtx.index[idx]
                        y_val = pcs[idx, 1] if pcs.shape[1] > 1 else 0
                        ax.annotate(label_text, (pcs[idx, 0], y_val),
                                    fontsize=8, ha="left", va="bottom",
                                    xytext=(5, 5), textcoords="offset points")

            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            if pcs.shape[1] > 1:
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            ax.set_title("PCA — Activité des régulons (AUCell)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, "pca_regulon_activity.png"))
            plt.close(fig)
            print("  → Sauvé : pca_regulon_activity.png")

    else:
        print("  Pas de matrice AUCell — visualisations ignorées.")


    # =============================================================================
    # 9. COMPARAISON AVEC LES RÉGULONS AgeAnno
    # =============================================================================
    print("\n" + "=" * 70)
    print("9. Comparaison avec les régulons AgeAnno")
    print("=" * 70)

    if os.path.exists(AGEANNO_TF_FILE) and regulons is not None:
        ageanno_tf = pd.read_csv(AGEANNO_TF_FILE, sep=",", encoding="latin-1")

        # Construire le dict AgeAnno TF → targets (même logique que gnn.py)
        ageanno_tf_targets = {}
        for _, row in ageanno_tf.iterrows():
            tf = str(row["TF"]).strip()
            targets_str = str(row.get("Targets", ""))
            if targets_str and targets_str != "nan":
                targets = {t.strip() for t in targets_str.split(";")}
                ageanno_tf_targets.setdefault(tf, set()).update(targets)

        # Extraire les TFs pySCENIC (nettoyer le nom "TF(+)" → "TF")
        scenic_tf_targets = {}
        for reg in regulons:
            tf_name = reg.name.rstrip("(+)").rstrip("(-)").strip()
            scenic_tf_targets[tf_name] = set(reg.genes) if hasattr(reg, 'genes') else set(reg.gene2weight.keys())

        # Overlap des TFs
        ageanno_tfs = set(ageanno_tf_targets.keys())
        scenic_tfs = set(scenic_tf_targets.keys())
        common_tfs = ageanno_tfs & scenic_tfs

        print(f"  AgeAnno TFs   : {len(ageanno_tfs)}")
        print(f"  pySCENIC TFs  : {len(scenic_tfs)}")
        print(f"  TFs communs   : {len(common_tfs)}")
        print(f"  Nouveaux TFs (pySCENIC uniquement) : {len(scenic_tfs - ageanno_tfs)}")

        # Pour les TFs communs, calculer le Jaccard index des targets
        jaccard_scores = {}
        for tf in common_tfs:
            targets_a = ageanno_tf_targets[tf]
            targets_s = scenic_tf_targets[tf]
            intersection = targets_a & targets_s
            union = targets_a | targets_s
            jaccard_scores[tf] = len(intersection) / len(union) if union else 0

        if jaccard_scores:
            jaccard_df = pd.DataFrame([
                {"TF": tf, "jaccard": j,
                "n_ageanno": len(ageanno_tf_targets[tf]),
                "n_scenic": len(scenic_tf_targets[tf]),
                "n_overlap": len(ageanno_tf_targets[tf] & scenic_tf_targets[tf])}
                for tf, j in jaccard_scores.items()
            ]).sort_values("jaccard", ascending=False)

            jaccard_df.to_csv(os.path.join(OUT_DIR, "ageanno_vs_scenic_overlap.csv"), index=False)
            print(f"\n  Jaccard index (overlap des cibles) :")
            print(f"    mean={jaccard_df['jaccard'].mean():.3f}, "
                f"median={jaccard_df['jaccard'].median():.3f}")
            print(f"  Top 5 TFs avec meilleur overlap :")
            for _, row in jaccard_df.head(5).iterrows():
                print(f"    {row['TF']:12s}  Jaccard={row['jaccard']:.3f}  "
                    f"(AgeAnno={row['n_ageanno']}, SCENIC={row['n_scenic']}, "
                    f"overlap={row['n_overlap']})")
            print(f"  → Sauvé : ageanno_vs_scenic_overlap.csv")

            # ── Figure : Jaccard distribution ────────────────────────────────────
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].hist(jaccard_df["jaccard"], bins=20, color="#2ECC71",
                        edgecolor="white", lw=0.5)
            axes[0].set_xlabel("Jaccard Index")
            axes[0].set_ylabel("Nombre de TFs")
            axes[0].set_title("Overlap des cibles : pySCENIC vs AgeAnno")

            # Scatter plot : nombre de cibles AgeAnno vs SCENIC
            axes[1].scatter(jaccard_df["n_ageanno"], jaccard_df["n_scenic"],
                            c=jaccard_df["jaccard"], cmap="RdYlGn", s=50,
                            edgecolors="black", lw=0.3)
            for _, row in jaccard_df.nlargest(5, "jaccard").iterrows():
                axes[1].annotate(row["TF"], (row["n_ageanno"], row["n_scenic"]),
                                fontsize=7, ha="left", va="bottom",
                                xytext=(3, 3), textcoords="offset points")
            axes[1].set_xlabel("Nb cibles AgeAnno")
            axes[1].set_ylabel("Nb cibles pySCENIC")
            axes[1].set_title("Taille des régulons")
            cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
            cbar.set_label("Jaccard")

            fig.suptitle("Comparaison régulons pySCENIC HUVEC vs AgeAnno (multi-tissus)",
                        fontsize=13, y=1.02)
            fig.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, "scenic_vs_ageanno.png"),
                        bbox_inches="tight")
            plt.close(fig)
            print("  → Sauvé : scenic_vs_ageanno.png")
    else:
        print("  Comparaison ignorée (fichier AgeAnno ou régulons manquants)")


    # =============================================================================
    # 10. EXPORT POUR INTÉGRATION GNN
    # =============================================================================
    print("\n" + "=" * 70)
    print("10. Export pour intégration GNN")
    print("=" * 70)

    # ── 10a. Arêtes TF → cible (format compatible gnn.py) ───────────────────────
    # Export un CSV simple TF,target,weight utilisable dans le GNN
    edges_for_gnn = []
    if regulons is not None:
        for reg in regulons:
            tf_name = reg.name.rstrip("(+)").rstrip("(-)").strip()
            if hasattr(reg, 'gene2weight'):
                for target, weight in reg.gene2weight.items():
                    if target != tf_name:
                        edges_for_gnn.append({"TF": tf_name, "target": target,
                                            "weight": weight})
            else:
                for gene in reg.genes:
                    if gene != tf_name:
                        edges_for_gnn.append({"TF": tf_name, "target": gene,
                                            "weight": 1.0})

    edges_df = pd.DataFrame(edges_for_gnn)
    edges_df.to_csv(os.path.join(OUT_DIR, "tf_target_edges_huvec.csv"), index=False)
    print(f"  Arêtes TF→cible : {len(edges_df)} liens "
        f"({edges_df['TF'].nunique()} TFs)")
    print(f"  → Sauvé : tf_target_edges_huvec.csv")

    # ── 10b. Activité différentielle par régulon (feature potentielle pour GNN) ──
    if diff_results is not None:
        # Créer un mapping gene → regulon_activity_diff
        # Pour chaque TF, on attribue la différence d'activité à tous ses gènes cibles
        gene_regulon_features = {}
        for _, row in diff_results.iterrows():
            tf_name = row["regulon"].rstrip("(+)").rstrip("(-)").strip()
            gene_regulon_features[tf_name] = {
                "regulon_activity_diff": row["diff_sen_pro"],
                "regulon_pvalue": row["pvalue"],
                "is_scenic_tf": 1.0,
            }

        feat_df = pd.DataFrame.from_dict(gene_regulon_features, orient="index")
        feat_df.index.name = "gene"
        feat_df.to_csv(os.path.join(OUT_DIR, "scenic_gene_features.csv"))
        print(f"  Features TF pour GNN : {len(feat_df)} gènes")
        print(f"  → Sauvé : scenic_gene_features.csv")

    # ── 10c. Matrice AUCell brute ────────────────────────────────────────────────
    if auc_mtx is not None:
        print(f"  Matrice AUCell déjà sauvée : auc_matrix.csv")


    # =============================================================================
    # 11. RÉSUMÉ FINAL
    # =============================================================================
    print("\n" + "=" * 70)
    print("RÉSUMÉ DE L'ANALYSE pySCENIC")
    print("=" * 70)

    n_regulons = len(regulons) if regulons else 0
    n_edges = len(edges_for_gnn)
    n_sig_reg = (diff_results["pvalue"] < 0.05).sum() if diff_results is not None else 0

    print(f"""
    Dataset     : GSE98440 — HUVEC prolifératives vs sénescentes (bulk RNA-seq)
    Échantillons: {len(pro_cols)} prolifératifs + {len(sen_cols)} sénescents
    Gènes       : {expression_matrix.shape[1]} (après filtrage)
    TFs testés  : {len(tf_names)}

    GRNBoost2/GENIE3 :
    Liens TF→cible bruts       : {len(adjacencies)}
    Liens filtrés (imp > {ADJACENCY_THRESH})  : {len(adj_filtered)}

    Régulons :
    Nombre de régulons          : {n_regulons}
    Régulons différentiels (p<0.05) : {n_sig_reg}

    Export GNN :
    Arêtes TF→cible HUVEC      : {n_edges}

    Fichiers de sortie : {OUT_DIR}/
    Figures            : {FIG_DIR}/
    """)

    client.close()
    cluster.close()

if __name__ == '__main__':
    main()