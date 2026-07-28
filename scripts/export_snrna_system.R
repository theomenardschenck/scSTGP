#!/usr/bin/env Rscript
# =============================================================================
# export_snrna_system.R — V6 : adapte un dataset snRNA (RDS Seurat/dgCMatrix,
# genes×cells + metadata cellule) aux entrees du pipeline V6.
#
# Produit DEUX agregations (consommateurs differents) :
#   (a) CELLULE-NIVEAU sous-echantillonne, restreint aux genes detectes
#       (>= --min-frac des cellules) → coexpr/SCENIC (GRNBoost2).
#       Sortie sparse : <out>/cells/{matrix.mtx.gz,genes.tsv,barcodes.tsv}
#       + <out>/cell_group.tsv (barcode \t groupe).  Converti en expr_<grp>.csv
#       par scripts/mtx_to_expr_csv.py.
#   (b) PSEUDOBULK par donneur (somme des comptes) → HuMess.
#       Sortie : <out>/abundance_table.tsv (genes × <grp>_<donneur>),
#       <out>/samplesheet.tsv (sample \t groupe), <out>/comp_file.tsv.
#
# Meme script pour le jeu EC (counts_endo) et pour un CONTROLE non-EC extrait
# de count_all_celltype via --celltype-col/--celltype.
#
# Usage (EC AD vs CT) :
#   Rscript scripts/export_snrna_system.R \
#     --counts <dir>/GSM8010381_counts_endo.rds \
#     --meta   <dir>/GSM8010381_metadata_endo.rds \
#     --group-col type --donor-col donor \
#     --out data/pyscenic/GSE252921_endo
#
# Usage (controle Oligo) : ajouter
#     --counts <dir>/GSM8010381_count_all_celltype.rds \
#     --meta   <dir>/GSM8010381_metadata_all_celltype.rds \
#     --celltype-col celltype_global --celltype Oligo \
#     --out data/pyscenic/GSE252921_oligo
# =============================================================================
suppressMessages(library(Matrix))

# ---- parse --key value -------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
getarg <- function(k, default = NULL) {
  i <- match(paste0("--", k), args)
  if (is.na(i)) return(default)
  args[i + 1]
}
counts_path  <- getarg("counts")
meta_path    <- getarg("meta")
group_col    <- getarg("group-col", "type")
donor_col    <- getarg("donor-col", "donor")
celltype_col <- getarg("celltype-col", NA)
celltype_val <- getarg("celltype", NA)
out_dir      <- getarg("out")
subsample    <- as.integer(getarg("subsample", "8000"))
min_frac     <- as.numeric(getarg("min-frac", "0.05"))
seed         <- as.integer(getarg("seed", "1"))
stopifnot(!is.null(counts_path), !is.null(meta_path), !is.null(out_dir))
set.seed(seed)
dir.create(file.path(out_dir, "cells"), recursive = TRUE, showWarnings = FALSE)

msg <- function(...) cat(sprintf(...), "\n")

# ---- load --------------------------------------------------------------------
msg("[load] counts : %s", counts_path)
cnt <- readRDS(counts_path)                       # genes x cells (sparse, PAS encore coerce)
if (is.null(colnames(cnt))) stop("counts sans colnames (barcodes)")
msg("[load] meta   : %s", meta_path)
meta <- readRDS(meta_path)
stopifnot(is.data.frame(meta))
if (length(intersect(colnames(cnt), rownames(meta))) < 2 &&
    "barcode" %in% colnames(meta)) {
  rownames(meta) <- meta$barcode                  # fallback colonne barcode
}

# Frugal memoire : selectionner les colonnes cibles (cellules communes ∩ type
# cellulaire) AVANT toute coercition/densification. Sur un objet 153k cellules
# (~6 Go en dgTMatrix) la coercition du plein est le pic memoire → on la reporte
# apres le subset (~43k cellules pour un type donne).
sel <- intersect(colnames(cnt), rownames(meta))
if (!is.na(celltype_col)) {
  stopifnot(celltype_col %in% colnames(meta))
  ok <- rownames(meta)[as.character(meta[[celltype_col]]) == celltype_val]
  msg("[celltype] %s == %s : %d cellules", celltype_col, celltype_val, length(ok))
  sel <- intersect(sel, ok)
}
stopifnot(length(sel) >= 100)
cnt  <- cnt[, sel, drop = FALSE]                  # subset UNIQUE
if (!inherits(cnt, "dgCMatrix")) {                # coercition sur le petit bloc
  cnt <- if (inherits(cnt, "sparseMatrix")) as(cnt, "CsparseMatrix")
         else as(as.matrix(cnt), "dgCMatrix")
}
meta <- meta[sel, , drop = FALSE]
msg("[align] %d cellules retenues, %d genes", ncol(cnt), nrow(cnt))

stopifnot(group_col %in% colnames(meta), donor_col %in% colnames(meta))
grp   <- as.character(meta[[group_col]])
donor <- as.character(meta[[donor_col]])
msg("[group] %s", paste(sprintf("%s=%d", names(table(grp)), table(grp)),
                        collapse = "  "))

# ---- filtre genes detectes (>= min_frac des cellules) ------------------------
det <- Matrix::rowSums(cnt > 0) / ncol(cnt)
gkeep <- det >= min_frac
msg("[genes] detectes >= %.0f%% : %d / %d retenus", 100 * min_frac,
    sum(gkeep), length(gkeep))
cnt <- cnt[gkeep, , drop = FALSE]

# =============================================================================
# (b) PSEUDOBULK par donneur (avant sous-echantillonnage : toutes les cellules)
# =============================================================================
donor_grp <- paste0(grp, "__", donor)            # ex. AD__02
uniq <- sort(unique(donor_grp))
# matrice indicatrice cellules x pseudobulk (somme des comptes)
ind <- sparse.model.matrix(~ 0 + factor(donor_grp, levels = uniq))
colnames(ind) <- uniq
pb <- as.matrix(cnt %*% ind)                     # genes x pseudobulk
# noms de colonne lisibles <groupe>_<donneur>
sample_names <- gsub("__", "_", uniq)
colnames(pb) <- sample_names
sample_grp   <- sub("__.*$", "", uniq)
ab <- data.frame(check.names = FALSE, matrix(nrow = nrow(pb), ncol = 0))
ab <- cbind(data.frame(gene = rownames(cnt), check.names = FALSE),
            as.data.frame(pb, check.names = FALSE))
write.table(ab, file.path(out_dir, "abundance_table.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
# samplesheet sample \t groupe
ss <- data.frame(sample = sample_names, group = sample_grp)
write.table(ss, file.path(out_dir, "samplesheet.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
# comp_file : une comparaison entre les deux groupes majeurs
g2 <- names(sort(table(grp), decreasing = TRUE))[1:2]
writeLines(sprintf("%s\t%s", g2[1], g2[2]), file.path(out_dir, "comp_file.tsv"))
msg("[humess] abundance_table %d genes x %d pseudobulk (%s) → %s",
    nrow(pb), ncol(pb), paste(g2, collapse = " vs "), out_dir)

# =============================================================================
# (a) CELLULE-NIVEAU sous-echantillonne pour coexpr/SCENIC
# =============================================================================
sel <- unlist(lapply(split(seq_len(ncol(cnt)), grp), function(idx) {
  if (length(idx) > subsample) sample(idx, subsample) else idx
}), use.names = FALSE)
sel <- sort(sel)
sub <- cnt[, sel, drop = FALSE]
msg("[cells] sous-echantillon : %d cellules (<= %d/groupe)", length(sel), subsample)
Matrix::writeMM(sub, file.path(out_dir, "cells", "matrix.mtx"))
system2("gzip", c("-f", shQuote(file.path(out_dir, "cells", "matrix.mtx"))))
writeLines(rownames(sub), file.path(out_dir, "cells", "genes.tsv"))
writeLines(colnames(sub), file.path(out_dir, "cells", "barcodes.tsv"))
cg <- data.frame(barcode = colnames(sub), group = grp[sel])
write.table(cg, file.path(out_dir, "cell_group.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
msg("[cells] sparse mtx + genes/barcodes/cell_group → %s/cells", out_dir)
msg("[done] %s", out_dir)
