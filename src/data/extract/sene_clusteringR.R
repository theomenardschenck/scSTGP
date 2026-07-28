################################################################################
#
#  PIPELINE scRNA-seq — Sénescence cellulaire réplicative (HUVECs)
#  Reproduction de : "Single-cell RNA-seq analysis reveals the multi-step
#                     process of cellular senescence"
#
#  Données : GSE102090 / GSE98448
#    - GSM2723761_D2P4  = Passage 4  (cellules jeunes, contrôle)
#    - GSM2723762_D2P16 = Passage 16 (cellules sénescentes)
#
#  Objectif : obtenir les 4 clusters de sénescence (P16)
#
################################################################################

# ==============================================================================
# 0. INSTALLATION DES PACKAGES (à exécuter une seule fois)
# ==============================================================================

# STGP_ROOT : racine du depot. Ces scripts pointaient en dur vers un chemin
# WSL absolu (Projet_Colin/huvec_gnn) qui n'existe plus.
STGP_ROOT <- Sys.getenv("STGP_ROOT", unset = getwd())

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

packages_cran <- c("Seurat", "ggplot2", "dplyr",
                   "patchwork", "pheatmap", "viridis", "Matrix")
packages_bioc <- c("fgsea", "msigdbr", "slingshot", "SingleCellExperiment",
                   "clusterProfiler", "org.Hs.eg.db", "MAST")

for (pkg in packages_cran) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}
for (pkg in packages_bioc) {
  if (!requireNamespace(pkg, quietly = TRUE)) BiocManager::install(pkg)
}
install.packages("vctrs")
install.packages("purrr")
BiocManager::install("slingshot")
# ==============================================================================
# 1. CHARGEMENT DES LIBRAIRIES
# ==============================================================================

library(slingshot)
library(SingleCellExperiment)
library(Seurat)
library(ggplot2)
library(dplyr)
library(tibble)      
library(patchwork)
library(Matrix)
library(fgsea)
library(msigdbr)

# ==============================================================================
# 2. CHARGEMENT DES DONNÉES
# ==============================================================================

# --- Définir le répertoire contenant tes fichiers ---
# Adapte ce chemin à ton propre système
data_dir <- file.path(STGP_ROOT, "data/DROPseq")

# --- Option A : Chargement depuis les fichiers MTX séparés (recommandé) ---

# Fichier de gènes commun aux deux échantillons
genes <- read.table(
  file.path(data_dir, "GSE102090_genes.tsv"),
  header = FALSE,
  stringsAsFactors = FALSE
)

# CORRECTION : Dédupliquer les noms de gènes (sinon Seurat refuse les rownames en double)
# make.unique() ajoute un suffixe .1, .2, etc. aux doublons
gene_names <- make.unique(genes$V2)

# ---- Passage 4 (cellules jeunes) ----
barcodes_p4 <- read.table(
  file.path(data_dir, "GSM2723761_D2P4_barcodes.tsv"),
  header = FALSE,
  stringsAsFactors = FALSE
)
matrix_p4 <- readMM(file.path(data_dir, "GSM2723761_D2P4_matrix.mtx"))

# Convertir en dgCMatrix (format sparse compressé attendu par Seurat)
matrix_p4 <- as(matrix_p4, "dgCMatrix")

rownames(matrix_p4) <- gene_names         # Noms de gènes dédupliqués
colnames(matrix_p4) <- barcodes_p4$V1     # Barcodes cellulaires
# Préfixer les barcodes pour éviter les collisions
colnames(matrix_p4) <- paste0("P4_", colnames(matrix_p4))

# Remplacer les underscores dans les noms de gènes (Seurat les refuse)
rownames(matrix_p4) <- gsub("_", "-", rownames(matrix_p4))

# ---- Passage 16 (cellules sénescentes) ----
barcodes_p16 <- read.table(
  file.path(data_dir, "GSM2723762_D2P16_barcodes.tsv"),
  header = FALSE,
  stringsAsFactors = FALSE
)
matrix_p16 <- readMM(file.path(data_dir, "GSM2723762_D2P16_matrix.mtx"))

# Convertir en dgCMatrix
matrix_p16 <- as(matrix_p16, "dgCMatrix")

rownames(matrix_p16) <- gene_names
colnames(matrix_p16) <- barcodes_p16$V1
colnames(matrix_p16) <- paste0("P16_", colnames(matrix_p16))

# Remplacer les underscores
rownames(matrix_p16) <- gsub("_", "-", rownames(matrix_p16))

# --- Option B : Chargement depuis le HDF5 (alternative) ---
# Si tu préfères utiliser le fichier HDF5, décommente le bloc ci-dessous :
#
# library(hdf5r)
# hdf5_path <- file.path(data_dir, "GSE102090_filtered_gene_bc_matrices_h5.hdf5")
# data_h5 <- Read10X_h5(hdf5_path)
# # Puis sépare les cellules par passage selon les barcodes

# ==============================================================================
# 3. CRÉATION DES OBJETS SEURAT
# ==============================================================================

# Objet pour P4
seurat_p4 <- CreateSeuratObject(
  counts = matrix_p4,
  project = "P4_young",
  min.cells = 3,
  min.features = 200
)
seurat_p4$passage <- "P4"

# Objet pour P16
seurat_p16 <- CreateSeuratObject(
  counts = matrix_p16,
  project = "P16_senescent",
  min.cells = 3,
  min.features = 200
)
seurat_p16$passage <- "P16"

# ==============================================================================
# 4. CONTRÔLE QUALITÉ (QC)
# ==============================================================================

# Calcul du pourcentage de gènes mitochondriaux
seurat_p4[["percent.mt"]]  <- PercentageFeatureSet(seurat_p4,  pattern = "^MT-")
seurat_p16[["percent.mt"]] <- PercentageFeatureSet(seurat_p16, pattern = "^MT-")

# --- Visualisation QC avant filtrage ---
p_qc_p4 <- VlnPlot(seurat_p4,
                   features = c("nFeature_RNA", "nCount_RNA", "percent.mt"),
                   ncol = 3, pt.size = 0.1) +
  plot_annotation(title = "QC - Passage 4")

p_qc_p16 <- VlnPlot(seurat_p16,
                    features = c("nFeature_RNA", "nCount_RNA", "percent.mt"),
                    ncol = 3, pt.size = 0.1) +
  plot_annotation(title = "QC - Passage 16")

print(p_qc_p4)
print(p_qc_p16)

# --- Filtrage selon les critères de l'article ---
# nFeature_RNA : entre 1500 et 5000
# nCount_RNA   : entre 300 et 30000
# percent.mt   : < 6%

seurat_p4 <- subset(seurat_p4,
                    subset = nFeature_RNA > 1500 &
                      nFeature_RNA < 5000 &
                      nCount_RNA > 300 &
                      nCount_RNA < 30000 &
                      percent.mt < 6)

seurat_p16 <- subset(seurat_p16,
                     subset = nFeature_RNA > 1500 &
                       nFeature_RNA < 5000 &
                       nCount_RNA > 300 &
                       nCount_RNA < 30000 &
                       percent.mt < 6)

cat("Cellules après QC — P4 :", ncol(seurat_p4), "\n")
cat("Cellules après QC — P16:", ncol(seurat_p16), "\n")

# ==============================================================================
# 5. ANALYSE PRINCIPALE SUR P16 (les 4 clusters de sénescence)
# ==============================================================================
# L'article se concentre sur P16 pour identifier les stades de sénescence.
# On garde P4 pour comparaison ultérieure.

seu <- seurat_p16

# --- 5a. Normalisation ---
seu <- NormalizeData(seu, normalization.method = "LogNormalize", scale.factor = 10000)

# --- 5b. Sélection des gènes hautement variables ---
seu <- FindVariableFeatures(seu, selection.method = "vst", nfeatures = 5000)

# Visualisation des gènes les plus variables
top20 <- head(VariableFeatures(seu), 20)
p_var <- VariableFeaturePlot(seu)
p_var <- LabelPoints(plot = p_var, points = top20, repel = TRUE)
print(p_var)

# --- 5c. Scaling ---
seu <- ScaleData(seu, features = rownames(seu))

# --- 5d. PCA ---
seu <- RunPCA(seu, features = VariableFeatures(seu), npcs = 50)

# ElbowPlot pour déterminer le nombre de PC à retenir
p_elbow <- ElbowPlot(seu, ndims = 50) +
  geom_vline(xintercept = 10, linetype = "dashed", color = "red") +
  ggtitle("ElbowPlot — ligne rouge = 10 PC retenues")
print(p_elbow)

# --- 5e. Clustering ---
# L'article utilise 10 PC et résolution 0.3, mais selon la version de Seurat
# et le nombre exact de cellules après QC, il faut parfois ajuster.
# On teste plusieurs résolutions pour trouver celle qui donne 4 clusters.

set.seed(42)  # Reproductibilité
seu <- FindNeighbors(seu, dims = 1:10)

# Balayage de résolutions pour trouver 4 clusters
resolutions <- c(0.2, 0.25, 0.3, 0.35)
for (res in resolutions) {
  seu <- FindClusters(seu, resolution = res, verbose = FALSE)
  n_cl <- length(levels(Idents(seu)))
  cat("Résolution", res, "→", n_cl, "clusters\n")
}

# Choisir automatiquement la résolution donnant 4 clusters
best_res <- NULL
for (res in resolutions) {
  seu <- FindClusters(seu, resolution = res, verbose = FALSE)
  if (length(levels(Idents(seu))) == 4) {
    best_res <- res
    break
  }
}

if (is.null(best_res)) {
  cat("ATTENTION : aucune résolution testée ne donne exactement 4 clusters.\n")
  cat("On utilise 0.25 par défaut — ajuste manuellement si nécessaire.\n")
  best_res <- 0.25
  seu <- FindClusters(seu, resolution = best_res, verbose = FALSE)
}

cat("Résolution retenue :", best_res, "\n")
cat("Nombre de clusters :", length(levels(Idents(seu))), "\n")
cat("Taille des clusters :\n")
print(table(Idents(seu)))

# --- 5f. UMAP ---
set.seed(42)
seu <- RunUMAP(seu, dims = 1:10)

# NOTE : L'orientation du UMAP est arbitraire (pas de "haut"/"bas" absolu).
# Si tu veux inverser les axes pour matcher la figure de l'article, décommente :
# seu@reductions$umap@cell.embeddings[, 1] <- -seu@reductions$umap@cell.embeddings[, 1]
# seu@reductions$umap@cell.embeddings[, 2] <- -seu@reductions$umap@cell.embeddings[, 2]

p_umap <- DimPlot(seu, reduction = "umap", label = TRUE, pt.size = 0.5) +
  ggtitle(paste0("UMAP — Clusters P16 (résolution ", best_res, ")"))
print(p_umap)

# ==============================================================================
# 6. GÈNES DIFFÉRENTIELLEMENT EXPRIMÉS (DEGs)
# ==============================================================================

# DEGs de chaque cluster vs tous les autres
markers_all <- FindAllMarkers(
  seu,
  only.pos = FALSE,
  min.pct = 0.25,
  logfc.threshold = 0.25,
  test.use = "wilcox"
)

# Filtrer les résultats significatifs
markers_sig <- markers_all %>% filter(p_val_adj < 0.01)

cat("Nombre total de DEGs significatifs :", nrow(markers_sig), "\n")
cat("DEGs par cluster :\n")
print(table(markers_sig$cluster))

# --- Top 10 DEGs par cluster ---
top10_markers <- markers_sig %>%
  group_by(cluster) %>%
  arrange(desc(abs(avg_log2FC))) %>%
  slice_head(n = 10) %>%
  ungroup()

print(top10_markers %>% select(cluster, gene, avg_log2FC, p_val_adj))

# --- Heatmap des top 10 DEGs ---
top10_genes <- top10_markers %>% pull(gene) %>% unique()
p_heatmap <- DoHeatmap(seu, features = top10_genes, size = 3) +
  ggtitle("Top 10 DEGs par cluster")
print(p_heatmap)

# ==============================================================================
# 7. GENE SET ENRICHMENT ANALYSIS (GSEA) — Hallmarks MSigDB
# ==============================================================================

# Récupérer les Hallmarks depuis MSigDB
# Construction de la liste nommée sans deframe() pour éviter les problèmes de dépendance
hallmarks_df <- msigdbr(species = "Homo sapiens", category = "H") %>%
  select(gs_name, gene_symbol)

hallmarks <- split(hallmarks_df$gene_symbol, hallmarks_df$gs_name)

# GSEA par cluster
gsea_results <- list()

for (cl in levels(Idents(seu))) {
  # Extraire les DEGs pour ce cluster
  cluster_markers <- markers_all %>% filter(cluster == cl)
  
  # Créer le vecteur de rangs (classé par log2FC)
  ranks <- setNames(cluster_markers$avg_log2FC, cluster_markers$gene)
  ranks <- sort(ranks, decreasing = TRUE)
  
  # Exécuter fGSEA
  fgsea_res <- fgsea(
    pathways = hallmarks,
    stats    = ranks,
    minSize  = 15,
    maxSize  = 500
  )
  
  fgsea_res <- fgsea_res %>%
    arrange(pval) %>%
    mutate(cluster = cl)
  
  gsea_results[[cl]] <- fgsea_res
}

# Combiner tous les résultats
gsea_combined <- bind_rows(gsea_results)

# Afficher les top pathways par cluster
cat("\n=== Top 5 pathways enrichis par cluster ===\n")
for (cl in levels(Idents(seu))) {
  cat("\n--- Cluster", cl, "---\n")
  top_paths <- gsea_combined %>%
    filter(cluster == cl, padj < 0.05) %>%
    arrange(pval) %>%
    slice_head(n = 5)
  print(top_paths %>% select(pathway, NES, padj))
}

# ==============================================================================
# 8. MODULE SCORES — Prolifération & Sénescence
# ==============================================================================

# Gènes de prolifération (Hallmark E2F Targets / G2M Checkpoint)
prolif_genes <- msigdbr(species = "Homo sapiens", category = "H") %>%
  filter(gs_name %in% c("HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT")) %>%
  pull(gene_symbol) %>%
  unique()

# Gènes associés à la sénescence (Hallmark P53 Pathway + custom SASP)
senescence_genes <- msigdbr(species = "Homo sapiens", category = "H") %>%
  filter(gs_name %in% c("HALLMARK_P53_PATHWAY",
                        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
                        "HALLMARK_INFLAMMATORY_RESPONSE")) %>%
  pull(gene_symbol) %>%
  unique()

# Calcul des module scores
seu <- AddModuleScore(seu, features = list(prolif_genes), name = "Proliferation_Score")
seu <- AddModuleScore(seu, features = list(senescence_genes), name = "Senescence_Score")

# Visualisation
p_prolif <- FeaturePlot(seu, features = "Proliferation_Score1", pt.size = 0.5) +
  scale_color_viridis_c() +
  ggtitle("Score de Prolifération")

p_senes <- FeaturePlot(seu, features = "Senescence_Score1", pt.size = 0.5) +
  scale_color_viridis_c() +
  ggtitle("Score de Sénescence")

print(p_prolif | p_senes)

# Violin plots par cluster
p_vln_scores <- VlnPlot(seu,
                        features = c("Proliferation_Score1", "Senescence_Score1"),
                        ncol = 2, pt.size = 0.1) +
  plot_annotation(title = "Scores par cluster")
print(p_vln_scores)

# ==============================================================================
# 9. ANALYSE DE TRAJECTOIRE — Slingshot
# ==============================================================================
# Slingshot remplace Monocle 3 (qui a des problèmes d'installation sur R 4.5+).
# Il produit les mêmes résultats : pseudotime et trajectoire entre clusters.


# Conversion Seurat → SingleCellExperiment
sce <- as.SingleCellExperiment(seu)

# Récupérer les coordonnées UMAP et les clusters
umap_coords <- Embeddings(seu, "umap")
clusters <- Idents(seu)

# Exécuter Slingshot sur les coordonnées PCA (plus stable que UMAP)
pca_coords <- Embeddings(seu, "pca")[, 1:10]
sce <- slingshot(
  sce,
  clusterLabels = clusters,
  reducedDim = "PCA"
  # start.clus = "X"  # Décommente et remplace X par le cluster prolifératif
  #                    # (celui avec le Proliferation_Score le plus élevé)
)

# Extraire le pseudotime (lineage principale)
pseudotime_values <- slingPseudotime(sce)

# Ajouter le pseudotime au Seurat object
# On prend la première lineage (ou la moyenne si plusieurs)
if (is.matrix(pseudotime_values)) {
  # Moyenne des lineages (en ignorant les NA)
  seu$pseudotime <- rowMeans(pseudotime_values, na.rm = TRUE)
  # Pseudotime par lineage individuelle
  for (i in seq_len(ncol(pseudotime_values))) {
    seu[[paste0("pseudotime_lineage", i)]] <- pseudotime_values[, i]
  }
} else {
  seu$pseudotime <- pseudotime_values
}

# --- Visualisation du pseudotime sur UMAP ---
p_pseudo <- FeaturePlot(seu, features = "pseudotime", pt.size = 0.5) +
  scale_color_viridis_c(option = "inferno", na.value = "grey90") +
  ggtitle("Pseudotime — Progression de la sénescence (Slingshot)")
print(p_pseudo)

# --- Visualisation des courbes de trajectoire sur UMAP ---
# Tracer les courbes Slingshot sur l'espace UMAP
# (Slingshot calcule sur PCA, on projette sur UMAP pour la visualisation)
p_traj <- DimPlot(seu, reduction = "umap", label = TRUE, pt.size = 0.5) +
  ggtitle("Clusters avec trajectoire Slingshot")
print(p_traj)

# --- Pseudotime par cluster (boxplot) ---
df_pseudo <- data.frame(
  cluster = Idents(seu),
  pseudotime = seu$pseudotime
)

p_pseudo_box <- ggplot(df_pseudo, aes(x = cluster, y = pseudotime, fill = cluster)) +
  geom_boxplot(outlier.size = 0.5) +
  theme_minimal() +
  ggtitle("Distribution du pseudotime par cluster") +
  xlab("Cluster") +
  ylab("Pseudotime")
print(p_pseudo_box)

#--- Gènes variant avec le pseudotime (optionnel, plus long) ---
#Pour identifier les gènes dont l'expression change le long du pseudotime :
library(tradeSeq)
BiocManager::install("tradeSeq")
sce <- fitGAM(sce)
ATres <- associationTest(sce)


# ==============================================================================
# 10. ANALYSE COMBINÉE P4 vs P16 (optionnel, pour contexte)
# ==============================================================================

# Normaliser P4 de la même façon
seurat_p4 <- NormalizeData(seurat_p4)
seurat_p4 <- FindVariableFeatures(seurat_p4, nfeatures = 5000)

# Merger les deux objets
seu_merged <- merge(seurat_p4, seurat_p16, add.cell.ids = c("P4", "P16"))
seu_merged <- NormalizeData(seu_merged)
seu_merged <- FindVariableFeatures(seu_merged, nfeatures = 5000)
seu_merged <- ScaleData(seu_merged)
seu_merged <- RunPCA(seu_merged, npcs = 30)
seu_merged <- FindNeighbors(seu_merged, dims = 1:10)
seu_merged <- FindClusters(seu_merged, resolution = 0.3)
seu_merged <- RunUMAP(seu_merged, dims = 1:10)

p_merged <- DimPlot(seu_merged, reduction = "umap",
                    group.by = "passage", pt.size = 0.5) +
  ggtitle("UMAP — P4 vs P16")
print(p_merged)

# ==============================================================================
# 11. SAUVEGARDE DES RÉSULTATS
# ==============================================================================

# Créer le répertoire de sortie
output_dir <- file.path(STGP_ROOT, "output/r_clustering")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)


# # Sauvegarder l'objet Seurat P16 avec les clusters
saveRDS(seu, file = file.path(output_dir, "seurat_P16_clustered.rds"))
saveRDS(seu_merged, file = file.path(output_dir, "seurat_P16_P4_merged.rds"))
# Sauvegarder les DEGs
write.csv(markers_all, file = file.path(output_dir, "DEGs_all_clusters.csv"),
          row.names = FALSE)
write.csv(markers_sig, file = file.path(output_dir, "DEGs_significant.csv"),
          row.names = FALSE)

# Sauvegarder les résultats GSEA
write.csv(
  gsea_combined %>% select(-leadingEdge),
  file = file.path(output_dir, "GSEA_hallmarks_results.csv"),
  row.names = FALSE
)

# Sauvegarder les métadonnées des cellules (clusters, scores, etc.)
write.csv(
  seu@meta.data,
  file = file.path(output_dir, "cell_metadata_P16.csv"),
  row.names = TRUE
)

# # Sauvegarder les plots
ggsave(file.path(output_dir, "merge_UMAP_P4_P16.png"), p_merged,
       width = 8, height = 6)
ggsave(file.path(output_dir, "UMAP_clusters_P16.pdf"), p_umap,
       width = 8, height = 6)
ggsave(file.path(output_dir, "heatmap_top10_DEGs.pdf"), p_heatmap,
       width = 12, height = 8)

cat("\n========================================\n")
cat("Pipeline terminé avec succès !\n")
cat("Résultats sauvegardés dans :", output_dir, "\n")
cat("========================================\n")

# ==============================================================================
# 12. EXPORT P4 + P16 FUSIONNÉS POUR LE GNN
# ==============================================================================
# Le GNN a besoin de cellules proliférantes (P4) ET sénescentes (P16) pour
# apprendre à discriminer les états cellulaires.
#
# Labels produits pour chaque cellule :
#   - passage     : "P4" ou "P16"
#   - cluster_P16 : numéro du cluster P16 (NA pour les cellules P4)
#   - cell_state  : label combiné lisible :
#       "P4_proliferative"
#       "P16_cluster_0", "P16_cluster_1", ...
#
# Structure de sortie :
#   gnn_data/
#     ├── merged_P4_P16_counts.csv         ← matrice complète labellisée
#     ├── merged_P4_P16_metadata.csv       ← métadonnées enrichies
#     ├── merged_P4_P16_normalized.csv     ← expression normalisée (log)
#     ├── P4_counts.csv                    ← cellules P4 seules
#     ├── P16_cluster_X_counts.csv         ← par cluster P16
#     ├── DEGs_cluster_X.csv               ← DEGs par cluster P16
#     ├── DEGs_P4_vs_P16.csv               ← DEGs entre passages
#     └── cluster_summary.csv
data_dir <- file.path(STGP_ROOT, "data")
gnn_dir <- file.path(data_dir, "gnn_data")
dir.create(gnn_dir, recursive = TRUE, showWarnings = FALSE)

cat("\n=== Export P4 + P16 fusionnés pour le GNN ===\n")

# --- 12a. Préparer P4 avec les mêmes gènes que P16 ---

# S'assurer que P4 est normalisé
seurat_p4 <- NormalizeData(seurat_p4)
seurat_p4 <- FindVariableFeatures(seurat_p4, nfeatures = 5000)
seurat_p4 <- ScaleData(seurat_p4, features = rownames(seurat_p4))

# Ajouter les labels à P4 : toutes les cellules P4 sont "proliferative"
seurat_p4$cluster_P16 <- NA
seurat_p4$cell_state  <- "P4_proliferative"

# Ajouter les labels à P16 : par cluster de sénescence
seu$cluster_P16 <- as.character(Idents(seu))
seu$cell_state  <- paste0("P16_cluster_", Idents(seu))

# --- 12b. Fusionner P4 + P16 ---

# Gènes communs entre les deux datasets
common_genes <- intersect(rownames(seurat_p4), rownames(seu))
cat("Gènes communs P4/P16 :", length(common_genes), "\n")

# Merger les objets Seurat
seu_full <- merge(seurat_p4, seu)
seu_full <- JoinLayers(seu_full)

cat("Dataset fusionné :", ncol(seu_full), "cellules ×", length(common_genes), "gènes\n")
cat("  P4  :", sum(seu_full$passage == "P4"), "cellules\n")
cat("  P16 :", sum(seu_full$passage == "P16"), "cellules\n")
cat("  États cellulaires :\n")
print(table(seu_full$cell_state))

# --- 12c. Export de la count matrix brute (cellules × gènes) ---

counts_merged <- GetAssayData(seu_full, layer = "counts")[common_genes, ]
counts_merged_df <- as.data.frame(as.matrix(t(counts_merged)))

# Ajouter les labels en premières colonnes
counts_merged_df <- cbind(
  barcode    = rownames(counts_merged_df),
  passage    = seu_full$passage,
  cluster_P16 = seu_full$cluster_P16,
  cell_state = seu_full$cell_state,
  counts_merged_df
)

write.csv(
  counts_merged_df,
  file.path(gnn_dir, "merged_P4_P16_counts.csv"),
  row.names = FALSE
)
cat("\nmatrice brute : merged_P4_P16_counts.csv\n")

# --- 12d. Export de la matrice normalisée (log-normalized) ---

# Normaliser le dataset fusionné
seu_full <- NormalizeData(seu_full)
norm_merged <- GetAssayData(seu_full, layer = "data")[common_genes, ]
norm_merged_df <- as.data.frame(as.matrix(t(norm_merged)))

norm_merged_df <- cbind(
  barcode    = rownames(norm_merged_df),
  passage    = seu_full$passage,
  cluster_P16 = seu_full$cluster_P16,
  cell_state = seu_full$cell_state,
  norm_merged_df
)

write.csv(
  norm_merged_df,
  file.path(gnn_dir, "merged_P4_P16_normalized.csv"),
  row.names = FALSE
)
cat("matrice normalisée : merged_P4_P16_normalized.csv\n")

# --- 12e. Export des métadonnées enrichies ---

meta_full <- seu_full@meta.data
meta_full$barcode <- rownames(meta_full)

write.csv(
  meta_full,
  file.path(gnn_dir, "merged_P4_P16_metadata.csv"),
  row.names = FALSE
)
cat("métadonnées : merged_P4_P16_metadata.csv\n")

# --- 12f. Export par sous-groupe (P4 seul + chaque cluster P16) ---

# P4 seul
counts_p4 <- GetAssayData(seurat_p4, layer = "counts")[common_genes, ]
counts_p4_df <- as.data.frame(as.matrix(t(counts_p4)))
counts_p4_df <- cbind(
  barcode    = rownames(counts_p4_df),
  passage    = "P4",
  cell_state = "P4_proliferative",
  counts_p4_df
)
write.csv(counts_p4_df, file.path(gnn_dir, "P4_counts.csv"), row.names = FALSE)
cat("\nP4 :", nrow(counts_p4_df), "cellules → P4_counts.csv\n")

# Chaque cluster P16
cluster_ids <- levels(Idents(seu))
for (cl in cluster_ids) {
  seu_cl <- subset(seu, idents = cl)
  counts_cl <- GetAssayData(seu_cl, layer = "counts")[common_genes, ]
  counts_cl_df <- as.data.frame(as.matrix(t(counts_cl)))
  counts_cl_df <- cbind(
    barcode    = rownames(counts_cl_df),
    passage    = "P16",
    cluster_P16 = cl,
    cell_state = paste0("P16_cluster_", cl),
    counts_cl_df
  )
  write.csv(
    counts_cl_df,
    file.path(gnn_dir, paste0("P16_cluster_", cl, "_counts.csv")),
    row.names = FALSE
  )
  cat("P16 cluster", cl, ":", nrow(counts_cl_df), "cellules →",
      paste0("P16_cluster_", cl, "_counts.csv\n"))
}

# --- 12g. DEGs entre P4 et P16 ---

cat("\nCalcul des DEGs P4 vs P16...\n")
seu_full <- FindVariableFeatures(seu_full, nfeatures = 5000)
seu_full <- ScaleData(seu_full, features = VariableFeatures(seu_full))

Idents(seu_full) <- "passage"
degs_passage <- FindMarkers(
  seu_full,
  ident.1 = "P16",
  ident.2 = "P4",
  test.use = "wilcox",
  min.pct = 0.25,
  logfc.threshold = 0.25
)
degs_passage$gene <- rownames(degs_passage)
write.csv(degs_passage, file.path(gnn_dir, "DEGs_P4_vs_P16.csv"), row.names = FALSE)
cat("DEGs P4 vs P16 :", nrow(degs_passage), "gènes → DEGs_P4_vs_P16.csv\n")

# --- 12h. DEGs par cluster P16 (déjà calculés dans la section 6) ---

for (cl in cluster_ids) {
  degs_cl <- markers_sig %>% filter(cluster == cl)
  write.csv(
    degs_cl,
    file.path(gnn_dir, paste0("DEGs_P16_cluster_", cl, ".csv")),
    row.names = FALSE
  )
}

# --- 12i. Résumé complet ---

summary_df <- data.frame(
  group      = c("P4_proliferative", paste0("P16_cluster_", cluster_ids)),
  passage    = c("P4", rep("P16", length(cluster_ids))),
  n_cells    = c(
    ncol(seurat_p4),
    sapply(cluster_ids, function(cl) sum(Idents(seu) == cl))
  ),
  n_DEGs     = c(
    nrow(degs_passage),
    sapply(cluster_ids, function(cl) nrow(markers_sig %>% filter(cluster == cl)))
  )
)
write.csv(summary_df, file.path(gnn_dir, "cluster_summary.csv"), row.names = FALSE)

cat("\n=== Résumé du dataset pour le GNN ===\n")
print(summary_df)

# Sauvegarder l'objet Seurat fusionné
saveRDS(seu_full, file.path(gnn_dir, "seurat_P4_P16_merged.rds"))

cat("\n=== Fichiers exportés dans", gnn_dir, "===\n")
cat("Fichiers globaux :\n")
cat("  merged_P4_P16_counts.csv       — count matrix brute, toutes cellules\n")
cat("  merged_P4_P16_normalized.csv   — expression log-normalisée\n")
cat("  merged_P4_P16_metadata.csv     — métadonnées enrichies\n")
cat("  seurat_P4_P16_merged.rds       — objet Seurat complet\n")
cat("  DEGs_P4_vs_P16.csv             — gènes P4 vs P16\n")
cat("  cluster_summary.csv            — résumé\n")
cat("Par sous-groupe :\n")
cat("  P4_counts.csv                  — cellules proliférantes\n")
cat("  P16_cluster_X_counts.csv       — par cluster de sénescence\n")
cat("  DEGs_P16_cluster_X.csv         — DEGs intra-P16\n")

# ==============================================================================
# EXPORT POUR pySCENIC (à exécuter après la section 11)
# ==============================================================================

scenic_dir <- file.path(STGP_ROOT, "data/pyscenic")
dir.create(scenic_dir, recursive = TRUE, showWarnings = FALSE)

# Récupérer la count matrix
counts <- GetAssayData(seu, layer = "counts")

# Filtrer pour garder les 5000 gènes variables (comme dans l'article)
var_genes <- VariableFeatures(seu)
counts_filtered <- counts[rownames(counts) %in% var_genes, ]

# IMPORTANT : pySCENIC attend cellules en lignes, gènes en colonnes
# Transposer et convertir en dataframe
counts_df <- as.data.frame(as.matrix(t(counts_filtered)))

# Sauvegarder la matrice d'expression
write.csv(counts_df, file.path(scenic_dir, "expr_matrix_P16.csv"))

# Sauvegarder les métadonnées (clusters, scores, pseudotime...)
write.csv(seu@meta.data, file.path(scenic_dir, "cell_metadata_P16.csv"))

cat("Export terminé :", nrow(counts_df), "cellules ×", ncol(counts_df), "gènes\n")
cat("Fichiers dans :", scenic_dir, "\n")

################################################################################
# NOTES IMPORTANTES
################################################################################
#
# 1. NOMBRE DE CLUSTERS :
#    Si tu n'obtiens pas exactement 4 clusters avec résolution 0.3, ajuste
#    légèrement (0.25 ou 0.35). Le nombre exact dépend de la version de
#    Seurat et des paramètres aléatoires (set.seed).
#
# 2. SCENIC (non inclus ici) :
#    La partie SCENIC se fait en Python avec pySCENIC.
#    Tu auras besoin d'exporter la count matrix :
#
#    # Exporter la matrice pour pySCENIC
#    counts <- GetAssayData(seu, slot = "counts")
#    writeMM(counts, file = file.path(output_dir, "counts_P16.mtx"))
#    write.csv(rownames(counts), file.path(output_dir, "gene_names.csv"))
#    write.csv(colnames(counts), file.path(output_dir, "cell_barcodes.csv"))
#
# 3. REPRODUCTIBILITÉ :
#    Ajoute set.seed(42) avant RunUMAP et FindClusters pour obtenir
#    des résultats identiques à chaque exécution.
#
# 4. POUR LE GNN HÉTÉROGÈNE :
#    Les données extraites ici (clusters, DEGs, GSEA, trajectoire) serviront
#    comme couches dans ton graphe hétérogène :
#      - Noeuds "gène" : les DEGs avec leurs log2FC par cluster
#      - Noeuds "cellule" / "type cellulaire" : les 4 clusters
#      - Arêtes "interaction" : corrélation d'expression, regulons SCENIC
#      - Arêtes "enrichissement" : gène → pathway (GSEA)
#
################################################################################