################################################################################
#
#  EXPORT COMPLET POUR LE GNN — Tous les gènes (pas seulement les HVG)
#  ====================================================================
#
#  Ce script réutilise le clustering P16 déjà calculé dans sene_clusteringR.R
#  (via le .rds sauvegardé) mais exporte TOUS les gènes pour le GNN.
#
#  Différences avec sene_clusteringR.R :
#    - Pas de restriction à 5000 HVG
#    - DEGs calculés avec seuils relâchés (min.pct=0.1, logfc.threshold=0)
#    - Pas de figures (export données brutes uniquement)
#    - Inclut explicitement les gènes non-DE (vrais négatifs pour le GNN)
#
################################################################################

library(Seurat)
library(dplyr)
library(Matrix)

# MAST est nécessaire pour le test DE alternatif (modèle hurdle adapté au
# scRNA-seq qui modélise le dropout et peut inclure le taux de détection
# cellulaire comme covariable — plus conservateur que Wilcoxon).
if (!requireNamespace("MAST", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
  BiocManager::install("MAST")
}
library(MAST)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

base_dir <- "//wsl.localhost/Ubuntu/home/USER/M2/S2/Stage/Projet_Colin/huvec_gnn"
data_dir <- file.path(base_dir, "data")
dropseq_dir <- file.path(data_dir, "DROPseq")
rds_dir <- file.path(base_dir, "output", "r_clustering")
out_dir <- file.path(data_dir, "gnn_data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Seuils relâchés pour les DEGs (capturer plus de signal)
MIN_PCT <- 0.1          # Au moins 10% des cellules expriment le gène (vs 25%)
LOGFC_THRESH <- 0.0     # Pas de seuil sur le log2FC (on garde tout, le GNN décidera)
PADJ_THRESH <- 0.05     # Seuil de significativité

# ── Paramètres de robustesse des DEGs ────────────────────────────────────────
# CONTEXTE : GSE102090 ne fournit qu'un échantillon par condition (n=1 P4, n=1 P16),
# ce qui rend le pseudo-bulk gold standard (agrégation par réplicat + DESeq2/edgeR)
# impossible. Pour compenser, on fiabilise les labels DEG via :
#   1. Consensus multi-méthodes (Wilcoxon + MAST) — réduit les faux positifs
#   2. Bootstrap de stabilité — évalue la robustesse de chaque DEG par sous-échantillonnage
N_BOOTSTRAP <- 100       # Nombre d'itérations de sous-échantillonnage
BOOT_FRAC <- 0.8         # Fraction de cellules tirées à chaque itération
BOOT_STABILITY_THRESH <- 0.7  # Seuil : un DEG est "stable" s'il est DE dans ≥70% des bootstraps

cat("=====================================================================\n")
cat("EXPORT COMPLET POUR LE GNN — Tous les gènes\n")
cat("=====================================================================\n\n")

# ==============================================================================
# 2. CHARGEMENT OU RECONSTRUCTION DE L'OBJET SEURAT P16 (CLUSTERISÉ)
# ==============================================================================
cat("--- Vérification de l'objet Seurat P16 clusterisé ---\n")

rds_p16 <- file.path(rds_dir, "seurat_P16_clustered.rds")

if (file.exists(rds_p16)) {
  cat("--- Chargement de l'objet P16 clusterisé sauvegardé ---\n")
  seu_p16 <- readRDS(rds_p16)
  cat("P16 clusterisé :", ncol(seu_p16), "cellules,", nrow(seu_p16), "gènes\n")
  cat("Clusters :\n")
  print(table(Idents(seu_p16)))
} else {
  cat("\n--- Objet P16 non trouvé → reconstruction depuis les matrices brutes ---\n")
  
  # Chargement des fichiers bruts (mêmes que pour P4, mais pour le GSM P16)
  genes <- read.table(
    file.path(dropseq_dir, "GSE102090_genes.tsv"),
    header = FALSE, stringsAsFactors = FALSE
  )
  gene_names <- make.unique(genes$V2)
  
  barcodes_p16 <- read.table(
    file.path(dropseq_dir, "GSM2723762_D2P16_barcodes.tsv"),
    header = FALSE, stringsAsFactors = FALSE
  )
  matrix_p16 <- readMM(file.path(dropseq_dir, "GSM2723762_D2P16_matrix.mtx"))
  matrix_p16 <- as(matrix_p16, "dgCMatrix")
  
  rownames(matrix_p16) <- gsub("_", "-", gene_names)
  colnames(matrix_p16) <- paste0("P16_", barcodes_p16$V1)
  
  seu_p16 <- CreateSeuratObject(
    counts = matrix_p16, 
    project = "P16_senescent",
    min.cells = 3, 
    min.features = 200
  )
  seu_p16$passage <- "P16"
  seu_p16[["percent.mt"]] <- PercentageFeatureSet(seu_p16, pattern = "^MT-")
  
  # Mêmes filtres QC que dans sene_clusteringR.R (identiques à P4)
  seu_p16 <- subset(seu_p16,
                    subset = nFeature_RNA > 1500 & nFeature_RNA < 5000 &
                      nCount_RNA > 300 & nCount_RNA < 30000 & percent.mt < 6)
  
  cat("P16 après QC :", ncol(seu_p16), "cellules,", nrow(seu_p16), "gènes\n")
  
  # Sauvegarde pour les prochains lancements (le clustering complet devra être 
  # réappliqué si nécessaire après cette reconstruction de base)
  saveRDS(seu_p16, rds_p16)
  cat("→ Objet P16 (post-QC) sauvegardé dans :", rds_p16, "\n")
}

# ==============================================================================
# 3. CHARGEMENT OU RECONSTRUCTION DE L'OBJET SEURAT P4
# ==============================================================================
cat("\n--- Vérification de l'objet Seurat P4 ---\n")

# On définit un nom de fichier RDS cohérent pour P4 (vous pouvez le changer si besoin)
rds_p4 <- file.path(rds_dir, "seurat_P4.rds")

if (file.exists(rds_p4)) {
  cat("--- Chargement de l'objet P4 sauvegardé ---\n")
  seurat_p4 <- readRDS(rds_p4)
  cat("P4 chargé :", ncol(seurat_p4), "cellules,", nrow(seurat_p4), "gènes\n")
} else {
  cat("--- Objet P4 non trouvé → reconstruction depuis les matrices brutes ---\n")
  
  # Chargement des fichiers bruts (code original, légèrement réorganisé)
  genes <- read.table(
    file.path(dropseq_dir, "GSE102090_genes.tsv"),
    header = FALSE, stringsAsFactors = FALSE
  )
  gene_names <- make.unique(genes$V2)
  
  barcodes_p4 <- read.table(
    file.path(dropseq_dir, "GSM2723761_D2P4_barcodes.tsv"),
    header = FALSE, stringsAsFactors = FALSE
  )
  matrix_p4 <- readMM(file.path(dropseq_dir, "GSM2723761_D2P4_matrix.mtx"))
  matrix_p4 <- as(matrix_p4, "dgCMatrix")
  
  rownames(matrix_p4) <- gsub("_", "-", gene_names)
  colnames(matrix_p4) <- paste0("P4_", barcodes_p4$V1)
  
  seurat_p4 <- CreateSeuratObject(
    counts = matrix_p4, 
    project = "P4_young",
    min.cells = 3, 
    min.features = 200
  )
  seurat_p4$passage <- "P4"
  seurat_p4[["percent.mt"]] <- PercentageFeatureSet(seurat_p4, pattern = "^MT-")
  
  # Mêmes filtres QC que sene_clusteringR.R
  seurat_p4 <- subset(seurat_p4,
                      subset = nFeature_RNA > 1500 & nFeature_RNA < 5000 &
                        nCount_RNA > 300 & nCount_RNA < 30000 & percent.mt < 6)
  
  cat("P4 après QC :", ncol(seurat_p4), "cellules,", nrow(seurat_p4), "gènes\n")
  
  # Sauvegarde automatique pour les prochains lancements
  saveRDS(seurat_p4, rds_p4)
  cat("→ Objet P4 sauvegardé dans :", rds_p4, "\n")
}

# ==============================================================================
# 4. FUSION P4 + P16 AVEC TOUS LES GÈNES (ou chargement si déjà existant)
# ==============================================================================
cat("\n--- Fusion P4 + P16 (ou chargement de seurat_full) ---\n")

rds_full <- file.path(rds_dir, "seurat_full.rds")

if (file.exists(rds_full)) {
  cat("--- Objet seurat_full trouvé → chargement direct ---\n")
  seu_full <- readRDS(rds_full)
  
  cat("Dataset fusionné chargé :", ncol(seu_full), "cellules ×", 
      nrow(seu_full), "gènes\n")
  cat(" P4 :", sum(seu_full$passage == "P4"), "cellules\n")
  cat(" P16 :", sum(seu_full$passage == "P16"), "cellules\n")
  cat(" États cellulaires :\n")
  print(table(seu_full$cell_state))
  
} else {
  cat("--- Objet seurat_full non trouvé → création de la fusion P4 + P16 ---\n")
  
  # Ajouter les labels
  seurat_p4$cluster_P16 <- NA
  seurat_p4$cell_state <- "P4_proliferative"
  
  seu_p16$cluster_P16 <- as.character(Idents(seu_p16))
  seu_p16$cell_state <- paste0("P16_cluster_", Idents(seu_p16))
  
  # Merger
  seu_full <- merge(seurat_p4, seu_p16)
  seu_full <- JoinLayers(seu_full)
  
  # Normaliser TOUS les gènes (pas de restriction aux HVG)
  seu_full <- NormalizeData(seu_full, 
                            normalization.method = "LogNormalize",
                            scale.factor = 10000)
  
  cat("Dataset fusionné créé :", ncol(seu_full), "cellules ×",
      nrow(seu_full), "gènes\n")
  cat(" P4 :", sum(seu_full$passage == "P4"), "cellules\n")
  cat(" P16 :", sum(seu_full$passage == "P16"), "cellules\n")
  cat(" États cellulaires :\n")
  print(table(seu_full$cell_state))
  
  # Sauvegarde pour les prochaines exécutions
  saveRDS(seu_full, rds_full)
  cat("→ Objet seurat_full sauvegardé dans :", rds_full, "\n")
}


# Gènes communs
common_genes <- intersect(rownames(seurat_p4), rownames(seu_p16))
cat("Gènes communs P4/P16 :", length(common_genes), "\n")

# ==============================================================================
# 5. EXPORT MATRICE NORMALISÉE (tous les gènes)
# ==============================================================================

cat("\n--- Export de la matrice normalisée ---\n")

norm_data <- GetAssayData(seu_full, layer = "data")[common_genes, ]
norm_df <- as.data.frame(as.matrix(t(norm_data)))

norm_df <- cbind(
  barcode     = rownames(norm_df),
  passage     = seu_full$passage,
  cluster_P16 = seu_full$cluster_P16,
  cell_state  = seu_full$cell_state,
  norm_df
)

write.csv(norm_df, file.path(out_dir, "merged_P4_P16_normalized.csv"),
          row.names = FALSE)
cat("  -> merged_P4_P16_normalized.csv (",
    nrow(norm_df), "cellules x", length(common_genes), "genes)\n")

# ==============================================================================
# 6. EXPORT MÉTADONNÉES
# ==============================================================================

meta_full <- seu_full@meta.data
meta_full$barcode <- rownames(meta_full)

write.csv(meta_full, file.path(out_dir, "merged_P4_P16_metadata.csv"),
          row.names = FALSE)
cat("  -> merged_P4_P16_metadata.csv\n")

# ==============================================================================
# 7. DEGs P4 vs P16 — CONSENSUS MULTI-MÉTHODES (Wilcoxon + MAST)
# ==============================================================================
# JUSTIFICATION : Avec n=1 par condition (GSE102090), le test Wilcoxon cell-level
# traite chaque cellule comme indépendante, gonflant artificiellement la puissance
# statistique et produisant des faux positifs. Pour fiabiliser les labels :
#   - On exécute DEUX tests (Wilcoxon + MAST) et on ne retient que le consensus
#   - MAST (hurdle model) modélise le dropout propre au scRNA-seq et inclut
#     le taux de détection (nFeature_RNA) comme covariable technique
#   - Un gène DE dans les deux tests a une bien meilleure chance d'être un vrai positif
# ==============================================================================

cat("\n--- Calcul des DEGs P4 vs P16 (consensus Wilcoxon + MAST) ---\n")

Idents(seu_full) <- "passage"

# ── 7a. Wilcoxon (test non-paramétrique, rapide) ─────────────────────────────
cat("  [1/2] Test Wilcoxon...\n")
degs_p4_p16 <- FindMarkers(
  seu_full,
  ident.1 = "P16",
  ident.2 = "P4",
  test.use = "wilcox",
  min.pct = MIN_PCT,
  logfc.threshold = LOGFC_THRESH,
  return.thresh = 1.0
)
degs_p4_p16$gene <- rownames(degs_p4_p16)
degs_p4_p16$significant <- degs_p4_p16$p_val_adj < PADJ_THRESH

write.csv(degs_p4_p16, file.path(out_dir, "DEGs_P4_vs_P16.csv"),
          row.names = FALSE)
cat("    Wilcoxon — DEGs P4 vs P16 :", sum(degs_p4_p16$significant),
    "significatifs /", nrow(degs_p4_p16), "testés\n")

# ── 7b. MAST (hurdle model, adapté au scRNA-seq) ─────────────────────────────
# MAST modélise conjointement le taux de dropout (composante discrète) et le
# niveau d'expression (composante continue). L'inclusion de nFeature_RNA comme
# variable latente absorbe une partie de la variabilité technique (profondeur
# de séquençage, qualité cellulaire), ce que Wilcoxon ne fait pas.
cat("  [2/2] Test MAST (avec covariable nFeature_RNA)...\n")
degs_mast <- FindMarkers(
  seu_full,
  ident.1 = "P16",
  ident.2 = "P4",
  test.use = "MAST",
  latent.vars = "nFeature_RNA",
  min.pct = MIN_PCT,
  logfc.threshold = LOGFC_THRESH,
  return.thresh = 1.0
)
degs_mast$gene <- rownames(degs_mast)
degs_mast$significant <- degs_mast$p_val_adj < PADJ_THRESH

write.csv(degs_mast, file.path(out_dir, "DEGs_P4_vs_P16_MAST.csv"),
          row.names = FALSE)
cat("    MAST — DEGs P4 vs P16 :", sum(degs_mast$significant),
    "significatifs /", nrow(degs_mast), "testés\n")

# ── 7c. Consensus : gènes DE dans les deux tests ─────────────────────────────
# Le consensus élimine les gènes que seul l'un des tests considère comme DE,
# réduisant les faux positifs propres à chaque méthode.
wilcox_sig <- degs_p4_p16$gene[degs_p4_p16$significant]
mast_sig <- degs_mast$gene[degs_mast$significant]
consensus_genes <- intersect(wilcox_sig, mast_sig)

# Score de consensus par gène : 0 (ni l'un ni l'autre), 0.5 (un seul), 1 (les deux)
all_tested <- union(degs_p4_p16$gene, degs_mast$gene)
consensus_df <- data.frame(
  gene = all_tested,
  wilcox_sig = all_tested %in% wilcox_sig,
  mast_sig = all_tested %in% mast_sig,
  stringsAsFactors = FALSE
)
consensus_df$consensus_score <- (as.numeric(consensus_df$wilcox_sig) +
                                   as.numeric(consensus_df$mast_sig)) / 2
consensus_df$is_consensus <- consensus_df$consensus_score == 1

write.csv(consensus_df, file.path(out_dir, "consensus_P4_vs_P16.csv"),
          row.names = FALSE)
cat("    Consensus (Wilcox ∩ MAST) :", length(consensus_genes), "gènes\n")
cat("    Wilcox seul :", sum(consensus_df$consensus_score == 0.5 & consensus_df$wilcox_sig), "\n")
cat("    MAST seul   :", sum(consensus_df$consensus_score == 0.5 & consensus_df$mast_sig), "\n")

# ==============================================================================
# 8. BOOTSTRAP DE STABILITÉ DES DEGs P4 vs P16
# ==============================================================================
# JUSTIFICATION : Même avec le consensus multi-méthodes, certains DEGs peuvent
# dépendre d'un petit sous-ensemble de cellules (outliers, doublets non filtrés).
# Le bootstrap de stabilité sous-échantillonne 80% des cellules N fois et mesure
# dans quelle fraction des itérations chaque gène reste significatif.
# Un gène stable (>70% des bootstraps) est robuste à la composition cellulaire.
# Ce score sera utilisé comme pondération de confiance dans la loss du GNN.
# ==============================================================================

cat("\n--- Bootstrap de stabilité (P4 vs P16,", N_BOOTSTRAP, "itérations) ---\n")

cells_p4 <- WhichCells(seu_full, idents = "P4")
cells_p16 <- WhichCells(seu_full, idents = "P16")
n_p4 <- length(cells_p4)
n_p16 <- length(cells_p16)

# Compteur : combien de fois chaque gène est DE significatif
# On utilise les gènes testés par Wilcoxon comme univers
boot_genes <- degs_p4_p16$gene
boot_counts <- setNames(rep(0, length(boot_genes)), boot_genes)

set.seed(42)
for (b in 1:N_BOOTSTRAP) {
  # Sous-échantillonnage stratifié (même fraction dans P4 et P16)
  sub_p4 <- sample(cells_p4, size = round(BOOT_FRAC * n_p4), replace = FALSE)
  sub_p16 <- sample(cells_p16, size = round(BOOT_FRAC * n_p16), replace = FALSE)
  sub_seu <- subset(seu_full, cells = c(sub_p4, sub_p16))
  Idents(sub_seu) <- "passage"

  # Test rapide (Wilcoxon) sur le sous-échantillon
  sub_degs <- tryCatch(
    FindMarkers(sub_seu, ident.1 = "P16", ident.2 = "P4",
                test.use = "wilcox", min.pct = MIN_PCT,
                logfc.threshold = LOGFC_THRESH, return.thresh = PADJ_THRESH),
    error = function(e) data.frame()
  )

  if (nrow(sub_degs) > 0) {
    sig_genes <- rownames(sub_degs)
    sig_in_universe <- sig_genes[sig_genes %in% boot_genes]
    boot_counts[sig_in_universe] <- boot_counts[sig_in_universe] + 1
  }

  if (b %% 20 == 0) cat("    Bootstrap", b, "/", N_BOOTSTRAP, "\n")
}

# Score de stabilité = fraction des bootstraps où le gène est DE
boot_stability <- boot_counts / N_BOOTSTRAP

stability_df <- data.frame(
  gene = names(boot_stability),
  bootstrap_stability = as.numeric(boot_stability),
  is_stable = as.numeric(boot_stability) >= BOOT_STABILITY_THRESH,
  stringsAsFactors = FALSE
)

write.csv(stability_df, file.path(out_dir, "bootstrap_stability_P4_vs_P16.csv"),
          row.names = FALSE)
cat("  Stabilité bootstrap :\n")
cat("    Gènes stables (≥", BOOT_STABILITY_THRESH*100, "%) :",
    sum(stability_df$is_stable), "/", nrow(stability_df), "\n")
cat("    Stabilité moyenne :", round(mean(stability_df$bootstrap_stability), 3), "\n")
cat("    Stabilité médiane :", round(median(stability_df$bootstrap_stability), 3), "\n")

# ==============================================================================
# 9. DEGs PAR CLUSTER P16 — CONSENSUS MULTI-MÉTHODES
# ==============================================================================
# Même logique de consensus Wilcoxon + MAST appliquée aux comparaisons
# intra-P16 (chaque cluster vs le reste).
# ==============================================================================

cat("\n--- Calcul des DEGs par cluster P16 (consensus Wilcoxon + MAST) ---\n")

Idents(seu_p16) <- seu_p16$cluster_P16
seu_p16 <- NormalizeData(seu_p16)

# ── 9a. Wilcoxon par cluster ─────────────────────────────────────────────────
cat("  [1/2] Wilcoxon par cluster...\n")
markers_wilcox <- FindAllMarkers(
  seu_p16,
  only.pos = FALSE,
  min.pct = MIN_PCT,
  logfc.threshold = LOGFC_THRESH,
  test.use = "wilcox",
  return.thresh = 1.0
)
markers_wilcox$significant <- markers_wilcox$p_val_adj < PADJ_THRESH

# ── 9b. MAST par cluster ─────────────────────────────────────────────────────
cat("  [2/2] MAST par cluster...\n")
markers_mast <- FindAllMarkers(
  seu_p16,
  only.pos = FALSE,
  min.pct = MIN_PCT,
  logfc.threshold = LOGFC_THRESH,
  test.use = "MAST",
  latent.vars = "nFeature_RNA",
  return.thresh = 1.0
)
markers_mast$significant <- markers_mast$p_val_adj < PADJ_THRESH

# ── 9c. Consensus et export par cluster ──────────────────────────────────────
cluster_ids <- unique(markers_wilcox$cluster)
all_cluster_consensus <- list()

for (cl in sort(cluster_ids)) {
  # Wilcoxon
  degs_w <- markers_wilcox %>% filter(cluster == cl)
  write.csv(degs_w,
            file.path(out_dir, paste0("DEGs_P16_cluster_", cl, ".csv")),
            row.names = FALSE)
  w_sig <- degs_w$gene[degs_w$significant]

  # MAST
  degs_m <- markers_mast %>% filter(cluster == cl)
  write.csv(degs_m,
            file.path(out_dir, paste0("DEGs_P16_cluster_", cl, "_MAST.csv")),
            row.names = FALSE)
  m_sig <- degs_m$gene[degs_m$significant]

  # Consensus
  consensus_cl <- intersect(w_sig, m_sig)
  all_cl_genes <- union(degs_w$gene, degs_m$gene)

  cl_consensus_df <- data.frame(
    gene = all_cl_genes,
    cluster = cl,
    wilcox_sig = all_cl_genes %in% w_sig,
    mast_sig = all_cl_genes %in% m_sig,
    stringsAsFactors = FALSE
  )
  cl_consensus_df$consensus_score <- (as.numeric(cl_consensus_df$wilcox_sig) +
                                        as.numeric(cl_consensus_df$mast_sig)) / 2
  all_cluster_consensus[[as.character(cl)]] <- cl_consensus_df

  cat("  Cluster", cl, ": Wilcox=", length(w_sig), "MAST=", length(m_sig),
      "Consensus=", length(consensus_cl), "\n")
}

# Exporter le consensus cluster en un seul fichier
consensus_clusters_df <- do.call(rbind, all_cluster_consensus)
write.csv(consensus_clusters_df,
          file.path(out_dir, "consensus_clusters_P16.csv"),
          row.names = FALSE)

markers_all <- markers_wilcox  # pour compatibilité avec la section résumé

# ==============================================================================
# 10. RÉSUMÉ
# ==============================================================================

summary_df <- data.frame(
  group   = c("P4_proliferative", paste0("P16_cluster_", sort(cluster_ids))),
  passage = c("P4", rep("P16", length(cluster_ids))),
  n_cells = c(
    sum(seu_full$passage == "P4"),
    sapply(sort(cluster_ids), function(cl) sum(seu_p16$cluster_P16 == cl))
  ),
  n_DEGs_wilcox = c(
    sum(degs_p4_p16$significant),
    sapply(sort(cluster_ids), function(cl)
      sum(markers_all$cluster == cl & markers_all$significant))
  ),
  n_DEGs_mast = c(
    sum(degs_mast$significant),
    sapply(sort(cluster_ids), function(cl)
      sum(markers_mast$cluster == cl & markers_mast$significant))
  ),
  n_DEGs_consensus = c(
    length(consensus_genes),
    sapply(sort(cluster_ids), function(cl) {
      cl_df <- all_cluster_consensus[[as.character(cl)]]
      sum(cl_df$consensus_score == 1)
    })
  )
)

write.csv(summary_df, file.path(out_dir, "cluster_summary.csv"),
          row.names = FALSE)

cat("\n=====================================================================\n")
cat("RÉSUMÉ\n")
cat("======================================================================\n")
print(summary_df)
cat("\nGènes totaux dans la matrice     :", length(common_genes), "\n")
cat("Gènes testés (DEG P4 vs P16)    :", nrow(degs_p4_p16), "\n")
cat("DEGs Wilcoxon (P4 vs P16)       :", sum(degs_p4_p16$significant), "\n")
cat("DEGs MAST (P4 vs P16)           :", sum(degs_mast$significant), "\n")
cat("DEGs consensus (P4 vs P16)      :", length(consensus_genes), "\n")
cat("DEGs stables (bootstrap ≥", BOOT_STABILITY_THRESH*100, "%) :",
    sum(stability_df$is_stable), "\n")
cat("\nFichiers exportés dans :", out_dir, "\n")
cat("  - DEGs_P4_vs_P16.csv (Wilcoxon)\n")
cat("  - DEGs_P4_vs_P16_MAST.csv (MAST)\n")
cat("  - consensus_P4_vs_P16.csv (scores consensus)\n")
cat("  - bootstrap_stability_P4_vs_P16.csv (stabilité bootstrap)\n")
cat("  - DEGs_P16_cluster_*.csv (Wilcoxon par cluster)\n")
cat("  - DEGs_P16_cluster_*_MAST.csv (MAST par cluster)\n")
cat("  - consensus_clusters_P16.csv (consensus par cluster)\n")
cat("======================================================================\n")
