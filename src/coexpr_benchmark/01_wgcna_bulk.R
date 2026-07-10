#!/usr/bin/env Rscript
# =============================================================================
# 01_wgcna_bulk.R — Classic WGCNA on bulk HUVEC RNA-seq (pro vs senescent)
# -----------------------------------------------------------------------------
# Co-expression benchmark (vs hdWGCNA on scRNA, vs GRNBoost2 GNN channel).
# Builds a signed WGCNA network, detects modules, computes eigengenes and the
# module-trait (senescence) correlation. Outputs gene->module + kME tables for
# the downstream cross-method comparison (03_compare.py).
#
# Usage: Rscript 01_wgcna_bulk.R <GSE98440|GSE163251>
#
# Caveat: WGCNA is designed for >=15 samples; GSE98440 has 6, GSE163251 has 8.
# Soft-power scale-free fit is therefore unreliable -> we fall back to the
# WGCNA-recommended default power for small signed networks when no power
# reaches the R^2 threshold (Langfelder & Horvath 2008, FAQ).
# =============================================================================

suppressPackageStartupMessages({
  library(WGCNA)
  library(data.table)
})
options(stringsAsFactors = FALSE)
disableWGCNAThreads()   # avoid WGCNA-fork x BLAS oversubscription on this host

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript 01_wgcna_bulk.R <GSE98440|GSE163251>")
DATASET <- args[1]

ROOT    <- "/home/USER/M2/S2/Stage/petry_project/gnn_huvec"
OUTDIR  <- file.path(ROOT, "output/coexpr_benchmark", paste0("wgcna_", DATASET))
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)
N_TOP_GENES <- 3500   # most-variable genes kept for WGCNA tractability/power

# ---------------------------------------------------------------------------
# 1. Load expression matrix -> genes x samples (symbol rownames), + trait
# ---------------------------------------------------------------------------
load_dataset <- function(ds) {
  if (ds == "GSE98440") {
    f <- file.path(ROOT, "data/bulkRNAseq/GSE984440_huvec/GSE98440_norm_counts_HUVECpro_sen.csv")
    d <- fread(f, header = TRUE)                           # first col = ENSG (empty header)
    ids <- d[[1]]; m <- as.data.frame(d[, -1]); rownames(m) <- ids   # ENSG x 6 samples
    # map ENSG -> symbol via Drop-seq genes.tsv
    g <- fread(file.path(ROOT, "data/DROPseq/GSE102090_genes.tsv"), header = FALSE)
    map <- setNames(g$V2, g$V1)
    sym <- map[rownames(m)]
    keep <- !is.na(sym) & sym != ""
    m <- m[keep, ]; sym <- sym[keep]
    m <- aggregate(m, by = list(symbol = sym), FUN = sum)   # collapse dup symbols
    rownames(m) <- m$symbol; m$symbol <- NULL
    state <- ifelse(grepl("sen", colnames(m)), "sen", "pro")
    trait <- ifelse(state == "sen", 1, 0)
  } else if (ds == "GSE163251") {
    f <- file.path(ROOT, "data/bulkRNAseq/GSE163251_huvec/GSE163251_fpkm_all.txt")
    d <- fread(f)
    expr_cols <- grep("^Sample", colnames(d), value = TRUE)
    m <- as.data.frame(d[, ..expr_cols])
    sym <- d$Tracking_id
    keep <- !is.na(sym) & sym != ""
    m <- m[keep, ]; sym <- sym[keep]
    m <- aggregate(m, by = list(symbol = sym), FUN = mean)  # collapse dup symbols (FPKM->mean)
    rownames(m) <- m$symbol; m$symbol <- NULL
    meta <- fread(file.path(ROOT, "data/bulkRNAseq/GSE163251_huvec/GSE163251_metadata.tsv"))
    st   <- setNames(meta$state, meta$sample)[colnames(m)]
    trait <- c(pro = 0, mid = 0.5, sen = 1)[st]             # ordinal senescence axis
    state <- st
  } else stop("unknown dataset")
  list(m = as.matrix(m), trait = trait, state = state)
}

dat <- load_dataset(DATASET)
m <- dat$m
cat(sprintf("[%s] raw matrix: %d genes x %d samples\n", DATASET, nrow(m), ncol(m)))
cat("  samples:", paste(colnames(m), collapse=", "), "\n")
cat("  state  :", paste(dat$state,  collapse=", "), "\n")

# log-transform (counts/FPKM -> log2), drop all-zero / undetected genes
m <- log2(m + 1)
m <- m[rowSums(m > 0) >= max(2, ceiling(0.3 * ncol(m))), ]
# keep top-variable genes (MAD)
v <- apply(m, 1, mad)
m <- m[order(v, decreasing = TRUE)[seq_len(min(N_TOP_GENES, nrow(m)))], ]
datExpr <- t(m)   # samples x genes (WGCNA convention)
cat(sprintf("  WGCNA input: %d samples x %d genes\n", nrow(datExpr), ncol(datExpr)))

gsg <- goodSamplesGenes(datExpr, verbose = 0)
if (!gsg$allOK) datExpr <- datExpr[gsg$goodSamples, gsg$goodGenes]

# ---------------------------------------------------------------------------
# 2. Soft-thresholding power (signed) with small-n fallback
# ---------------------------------------------------------------------------
powers <- c(1:10, seq(12, 30, 2))
sft <- pickSoftThreshold(datExpr, powerVector = powers, networkType = "signed",
                         RsquaredCut = 0.8, blockSize = ncol(datExpr), verbose = 0)
power <- sft$powerEstimate
if (is.na(power)) {
  # Langfelder & Horvath FAQ: default for signed nets, n<20 -> 18
  n <- nrow(datExpr)
  power <- if (n < 20) 18 else if (n < 30) 16 else 14
  cat(sprintf("  pickSoftThreshold: no power reached R^2=0.8 -> fallback power=%d (n=%d)\n", power, n))
} else {
  cat(sprintf("  selected soft power = %d (R^2=%.3f)\n", power,
              sft$fitIndices$SFT.R.sq[sft$fitIndices$Power == power]))
}
fwrite(sft$fitIndices, file.path(OUTDIR, "softpower_fit.tsv"), sep = "\t")

# ---------------------------------------------------------------------------
# 3. Blockwise signed network + module detection
# ---------------------------------------------------------------------------
net <- blockwiseModules(datExpr, power = power,
                        networkType = "signed", TOMType = "signed",
                        minModuleSize = 30, mergeCutHeight = 0.25,
                        deepSplit = 2, numericLabels = TRUE,
                        pamRespectsDendro = FALSE, maxBlockSize = 6000,
                        saveTOMs = FALSE, verbose = 0)
moduleColors <- labels2colors(net$colors)
names(moduleColors) <- colnames(datExpr)
nMod <- length(unique(moduleColors[moduleColors != "grey"]))
cat(sprintf("  modules detected: %d (+ grey) ; grey genes: %d\n",
            nMod, sum(moduleColors == "grey")))

# ---------------------------------------------------------------------------
# 4. Module eigengenes, trait correlation, intramodular kME
# ---------------------------------------------------------------------------
MEs <- moduleEigengenes(datExpr, colors = moduleColors)$eigengenes
MEs <- orderMEs(MEs)
trait <- dat$trait
mt_cor <- cor(MEs, trait, use = "p")
mt_p   <- corPvalueStudent(mt_cor, nrow(datExpr))
mt <- data.frame(module = sub("^ME", "", rownames(mt_cor)),
                 cor_senescence = as.numeric(mt_cor),
                 p_value = as.numeric(mt_p))
mt <- mt[order(-abs(mt$cor_senescence)), ]

kME <- signedKME(datExpr, MEs)   # gene x module kME
gene_module <- data.frame(gene = colnames(datExpr), module = moduleColors)
gene_module$kME_own <- mapply(function(g, mod) {
  col <- paste0("kME", mod)
  if (col %in% colnames(kME)) kME[g, col] else NA
}, gene_module$gene, gene_module$module)

# ---------------------------------------------------------------------------
# 5. Write outputs
# ---------------------------------------------------------------------------
fwrite(gene_module, file.path(OUTDIR, "gene_module.tsv"), sep = "\t")
fwrite(data.frame(gene = rownames(kME), kME), file.path(OUTDIR, "kME.tsv"), sep = "\t")
fwrite(data.frame(sample = rownames(MEs), MEs), file.path(OUTDIR, "module_eigengenes.tsv"), sep = "\t")
fwrite(mt, file.path(OUTDIR, "module_trait.tsv"), sep = "\t")
writeLines(c(sprintf("dataset\t%s", DATASET),
             sprintf("n_samples\t%d", nrow(datExpr)),
             sprintf("n_genes\t%d", ncol(datExpr)),
             sprintf("soft_power\t%d", power),
             sprintf("n_modules\t%d", nMod),
             sprintf("n_grey\t%d", sum(moduleColors == "grey")),
             sprintf("top_module_trait\t%s (r=%.3f, p=%.2g)", mt$module[1], mt$cor_senescence[1], mt$p_value[1])),
           file.path(OUTDIR, "summary.txt"))
cat(sprintf("[%s] done -> %s\n", DATASET, OUTDIR))
print(head(mt, 6))
