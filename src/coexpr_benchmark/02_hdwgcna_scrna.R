#!/usr/bin/env Rscript
# =============================================================================
# 02_hdwgcna_scrna.R — hdWGCNA on HUVEC scRNA-seq (Drop-seq P4 vs P16)
# -----------------------------------------------------------------------------
# Single-cell co-expression via hdWGCNA (Morabito 2023, Cell Reports Methods):
# aggregates cells into *metacells* to overcome scRNA sparsity, then runs the
# WGCNA machinery on the metacell expression. Outputs the same gene->module +
# kME schema as 01_wgcna_bulk.R for the cross-method comparison.
#
# Input: project-normalized matrix merged_P4_P16_normalized.csv (cells x genes,
#        Seurat log-normalized) + metadata (cell_state: P4_proliferative / P16_*).
#
# Usage: Rscript 02_hdwgcna_scrna.R
# =============================================================================

suppressPackageStartupMessages({
  library(Seurat); library(hdWGCNA); library(WGCNA)
  library(Matrix); library(data.table); library(dplyr)
})
options(stringsAsFactors = FALSE)
set.seed(42)
disableWGCNAThreads()   # CRITICAL: this host has only 7GB RAM; forking would OOM
N_VARGENES <- 3000      # pre-filter genes before building Seurat (memory)
P4_SUBSAMPLE <- 3000    # cap the dominant P4 pool to balance groups + save RAM

ROOT   <- Sys.getenv("STGP_ROOT", unset = getwd())  # lancer depuis la racine du depot
OUTDIR <- file.path(ROOT, "output/coexpr_benchmark/hdwgcna_scrna")
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# 1. Build Seurat object from the project-normalized matrix
# ---------------------------------------------------------------------------
cat("Loading normalized scRNA matrix...\n")
mat <- fread(file.path(ROOT, "data/gnn_data/merged_P4_P16_normalized.csv"))
meta_cols <- intersect(c("barcode", "passage", "cluster_P16", "cell_state"), colnames(mat))
cells   <- mat$barcode
passage <- mat$passage; cluster_P16 <- mat$cluster_P16; cell_state <- mat$cell_state

# --- memory frugal: subsample dominant P4 pool, keep all P16 -----------------
is_p4 <- passage == "P4"
keep_cells <- rep(TRUE, length(cells))
if (sum(is_p4) > P4_SUBSAMPLE) {
  drop <- sample(which(is_p4), sum(is_p4) - P4_SUBSAMPLE)
  keep_cells[drop] <- FALSE
}
gene_cols <- setdiff(colnames(mat), meta_cols)
expr <- t(as.matrix(mat[keep_cells, ..gene_cols]))   # genes x cells (subsampled)
rownames(expr) <- gene_cols; colnames(expr) <- cells[keep_cells]
rm(mat); gc(verbose = FALSE)

# --- pre-filter to top-variable genes BEFORE Seurat (cuts RAM ~5x) -----------
gv <- apply(expr, 1, var)
topg <- order(gv, decreasing = TRUE)[seq_len(min(N_VARGENES, nrow(expr)))]
expr <- expr[topg, , drop = FALSE]
expr <- as(expr, "CsparseMatrix")
cat(sprintf("  %d genes x %d cells (var-filtered, P4 subsampled)\n", nrow(expr), ncol(expr)))

md <- data.frame(row.names = colnames(expr),
                 passage    = passage[keep_cells],
                 cell_state = cell_state[keep_cells],
                 cluster_P16= cluster_P16[keep_cells])
# coarse senescence group used as the hdWGCNA grouping variable
md$group <- ifelse(md$passage == "P4", "P4_proliferative",
             ifelse(is.na(md$cluster_P16), "P16", paste0("P16_c", md$cluster_P16)))
md$senescent <- ifelse(md$passage == "P4", 0, 1)

seu <- CreateSeuratObject(counts = expr, meta.data = md)
seu <- SetAssayData(seu, slot = "data", new.data = expr)  # already log-normalized
seu <- FindVariableFeatures(seu, nfeatures = 3000)
seu <- ScaleData(seu, features = VariableFeatures(seu))
seu <- RunPCA(seu, features = VariableFeatures(seu), npcs = 30, verbose = FALSE)
seu <- FindNeighbors(seu, dims = 1:30, verbose = FALSE)
cat("  Seurat object ready; groups:\n"); print(table(seu$group))

# ---------------------------------------------------------------------------
# 2. hdWGCNA: setup + metacells
# ---------------------------------------------------------------------------
seu <- SetupForWGCNA(seu, gene_select = "variable", wgcna_name = "huvec",
                     fraction = 0.05)
cat(sprintf("  WGCNA genes selected: %d\n", length(GetWGCNAGenes(seu))))

seu <- MetacellsByGroups(seu, group.by = c("group"), ident.group = "group",
                         k = 25, max_shared = 10, min_cells = 50,
                         reduction = "pca", assay = "RNA", slot = "data")
seu <- NormalizeMetacells(seu)
mc <- GetMetacellObject(seu)
cat(sprintf("  metacells built: %d (across %d groups)\n", ncol(mc), length(unique(mc$group))))

# Use ALL groups together to learn one shared co-expression network
seu <- SetDatExpr(seu, group_name = unique(as.character(seu$group)),
                  group.by = "group", assay = "RNA", slot = "data")

# ---------------------------------------------------------------------------
# 3. Soft power + network construction
# ---------------------------------------------------------------------------
seu <- TestSoftPowers(seu, networkType = "signed")
pt  <- GetPowerTable(seu)
fwrite(pt, file.path(OUTDIR, "softpower_fit.tsv"), sep = "\t")
power <- pt$Power[which(pt$SFT.R.sq > 0.8)[1]]
if (is.na(power)) power <- 9   # hdWGCNA default for signed scRNA metacells
cat(sprintf("  soft power = %d\n", power))

seu <- ConstructNetwork(seu, soft_power = power, networkType = "signed",
                        TOMType = "signed", minModuleSize = 30,
                        mergeCutHeight = 0.25, deepSplit = 4,
                        setDatExpr = FALSE, tom_name = "huvec",
                        overwrite_tom = TRUE)

# ---------------------------------------------------------------------------
# 4. Module eigengenes, connectivity (kME), trait correlation
# ---------------------------------------------------------------------------
seu <- ModuleEigengenes(seu)
seu <- ModuleConnectivity(seu)

modules <- GetModules(seu)        # gene_name, module, color, kME_*
mods_tab <- data.frame(gene = modules$gene_name,
                       module = as.character(modules$module),
                       color  = modules$color)
# own-module kME
kme_cols <- grep("^kME_", colnames(modules), value = TRUE)
mods_tab$kME_own <- mapply(function(g_i, mod) {
  col <- paste0("kME_", mod)
  if (col %in% colnames(modules)) modules[g_i, col] else NA
}, seq_len(nrow(modules)), modules$module)

# harmonized module eigengenes per cell, then average per senescence group
MEs <- GetMEs(seu, harmonized = TRUE)
me_grp <- aggregate(MEs, by = list(senescent = seu$senescent), FUN = mean)
# correlation eigengene ~ senescent (per cell)
sen <- seu$senescent
mt_cor <- cor(MEs, sen, use = "p")
mt_p   <- corPvalueStudent(mt_cor, length(sen))
mt <- data.frame(module = rownames(mt_cor),
                 cor_senescence = as.numeric(mt_cor),
                 p_value = as.numeric(mt_p))
mt <- mt[order(-abs(mt$cor_senescence)), ]

# ---------------------------------------------------------------------------
# 5. Write outputs (schema parallel to 01_wgcna_bulk.R)
# ---------------------------------------------------------------------------
fwrite(mods_tab, file.path(OUTDIR, "gene_module.tsv"), sep = "\t")
fwrite(modules[, c("gene_name", "module", kme_cols)], file.path(OUTDIR, "kME.tsv"), sep = "\t")
fwrite(me_grp, file.path(OUTDIR, "module_eigengenes_by_group.tsv"), sep = "\t")
fwrite(mt, file.path(OUTDIR, "module_trait.tsv"), sep = "\t")
nMod <- length(setdiff(unique(mods_tab$module), c("grey", "0")))
writeLines(c("dataset\tscRNA_HUVEC_dropseq",
             sprintf("n_cells\t%d", ncol(seu)),
             sprintf("n_metacells\t%d", ncol(mc)),
             sprintf("n_genes\t%d", length(GetWGCNAGenes(seu))),
             sprintf("soft_power\t%d", power),
             sprintf("n_modules\t%d", nMod),
             sprintf("top_module_trait\t%s (r=%.3f, p=%.2g)", mt$module[1], mt$cor_senescence[1], mt$p_value[1])),
           file.path(OUTDIR, "summary.txt"))
saveRDS(seu, file.path(OUTDIR, "seurat_hdwgcna.rds"))
cat("hdWGCNA done ->", OUTDIR, "\n"); print(head(mt, 8))
