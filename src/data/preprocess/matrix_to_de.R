#!/usr/bin/env Rscript
# =============================================================================
# matrix_to_de.R — Matrice d'expression (counts|FPKM) → table DE canonique.
# =============================================================================
# Produit un fichier DE au schéma lisible par src/data/loaders/bulk_rna.py
# (colonnes : gene_symbol, log2FoldChange, stat, pvalue, padj), pour servir
# d'AXE V6 DE-ancré (perturb_top_genes --de-axis-file) ou de cible de probe.
#
# Affectation des échantillons à un état (jeune/prolif vs sénescent), par ordre
# de priorité :
#   1. --metadata <fichier>  : 2 colonnes (sample, state) — croisement explicite.
#   2. --state-from-name     : déduit l'état du NOM de l'échantillon (défaut) :
#        token explicite  pro|prolif|young|ctrl|control  → "pro"
#                          sen|senescent|old              → "sen"
#        sinon jour  d<NN> / D<NN> / day<NN>  → "pro" si jour<=--young-max-day,
#                                               "sen" si jour>=--sene-min-day,
#                                               (intermédiaire = exclu).
#   3. interactif            : si rien ne matche, demande à l'utilisateur (stdin).
#
# DE : DESeq2 si counts (et dispo) ; limma-trend si FPKM/continu (et dispo) ;
#      sinon fallback log2-ratio des moyennes + t de Welch + BH (suffisant pour
#      définir une DIRECTION d'axe, cf. design_log §14bis.8.1 : rank=stat,
#      N≥150 ancres → cos~0.9). Convention <A>_vs_<B> : log_fc>0 ⇔ up dans A
#      (A = sénescent par défaut ⇒ sen_vs_pro).
#
# Exemples :
#   # GSE163251 (FPKM, cinétique j5→j74, auto par jour) :
#   Rscript src/data/preprocess/matrix_to_de.R \
#     --matrix data/bulkRNAseq/GSE163251_huvec/GSE163251_fpkm_all.txt \
#     --gene-col Tracking_id --data-type fpkm \
#     --young-max-day 5 --sene-min-day 60 \
#     --out data/bulkRNAseq/GSE163251_huvec/GSE163251_DE_sen_vs_pro.tsv
#
#   # GSE98440 (counts, états dans le nom *_pro / *_sen) :
#   Rscript src/data/preprocess/matrix_to_de.R \
#     --matrix data/bulkRNAseq/GSE984440_huvec/GSE98440_norm_counts_HUVECpro_sen.csv \
#     --data-type counts \
#     --out data/bulkRNAseq/GSE984440_huvec/GSE98440_DE_recomputed.tsv
# =============================================================================

# ------------------------------- arg parsing --------------------------------
args <- commandArgs(trailingOnly = TRUE)
get_opt <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i)) return(default)
  if (i == length(args) || startsWith(args[i + 1], "--")) return(TRUE)  # flag booléen
  args[i + 1]
}
opt <- list(
  matrix        = get_opt("--matrix"),
  out           = get_opt("--out"),
  gene_col      = get_opt("--gene-col", NULL),
  sample_cols   = get_opt("--sample-cols", NULL),   # regex optionnelle
  data_type     = get_opt("--data-type", "auto"),   # auto|counts|fpkm
  metadata      = get_opt("--metadata", NULL),
  young_max_day = as.numeric(get_opt("--young-max-day", 5)),
  sene_min_day  = as.numeric(get_opt("--sene-min-day", 60)),
  pro_label     = get_opt("--young-label", "pro"),
  sen_label     = get_opt("--sene-label", "sen"),
  cond_label    = get_opt("--condition-label", "sen_vs_pro"),
  keep_middle   = isTRUE(get_opt("--keep-middle", FALSE)),
  min_expr      = as.numeric(get_opt("--min-expr", 1))   # filtre bruit bas-exprimé
)
if (is.null(opt$matrix) || is.null(opt$out))
  stop("--matrix et --out sont requis.")
msg <- function(...) cat(sprintf(...), "\n")
edgeR_cpm <- function(m) sweep(m, 2, colSums(m), "/") * 1e6 + 1  # CPM sans edgeR

# ------------------------------- lecture ------------------------------------
# sniff du séparateur (l'extension ment : des .csv sont tab-séparés)
first <- readLines(opt$matrix, n = 1)
sep <- if (nchar(gsub("[^\t]", "", first)) >= nchar(gsub("[^,]", "", first))) "\t" else ","
df <- read.delim(opt$matrix, sep = sep, check.names = FALSE,
                 stringsAsFactors = FALSE)
# en-tête de 1re colonne vide (matrices counts type "\tS1\tS2…") → nomme-la
blank <- which(is.na(colnames(df)) | colnames(df) == "")
if (length(blank)) colnames(df)[blank] <- paste0("gene_id_col", seq_along(blank))
msg("[matrix] %s : %d lignes × %d colonnes (sep='%s')",
    basename(opt$matrix), nrow(df), ncol(df), if (sep == "\t") "TAB" else sep)

# colonne gène : explicite > en-tête connu > rownames (1re colonne sans nom)
gene_candidates <- c("Tracking_id", "hgnc_symbol", "gene_symbol", "gene_name",
                     "symbol", "gene", "GeneName", "Keys")
rn_are_genes <- !identical(rownames(df), as.character(seq_len(nrow(df))))
hit <- intersect(gene_candidates, colnames(df))
if (!is.null(opt$gene_col)) {
  gcol <- opt$gene_col; genes <- as.character(df[[gcol]])
} else if (length(hit)) {
  gcol <- hit[1]; genes <- as.character(df[[gcol]])
} else if (rn_are_genes) {
  gcol <- "(rownames)"; genes <- rownames(df)
} else {
  gcol <- colnames(df)[1]; genes <- as.character(df[[gcol]])
}
msg("[matrix] colonne gène = '%s' (%d gènes, ex. %s)", gcol, length(genes),
    paste(head(genes, 3), collapse = ", "))

# colonnes échantillons : regex fournie, sinon toutes les colonnes numériques
num_cols <- names(df)[vapply(df, is.numeric, logical(1))]
if (!is.null(opt$sample_cols)) {
  scols <- grep(opt$sample_cols, colnames(df), value = TRUE)
} else {
  drop <- c(gcol, "Locus", "Keys", "baseMean", "lfcSE")
  scols <- setdiff(num_cols, drop)
}
if (length(scols) < 2) stop("Moins de 2 colonnes d'échantillons détectées.")
mat <- as.matrix(df[, scols, drop = FALSE])
rownames(mat) <- genes
mode(mat) <- "numeric"
msg("[matrix] %d échantillons : %s", length(scols), paste(scols, collapse = ", "))

# ------------------------- affectation des états ----------------------------
classify_name <- function(nm, young_max, sene_min) {
  s <- tolower(nm)
  if (grepl("(pro|prolif|young|ctrl|control)", s)) return("pro")
  if (grepl("(senescent|sen|old)", s))             return("sen")
  d <- regmatches(s, regexpr("(d|day)_?([0-9]+)", s))
  if (length(d)) {
    day <- as.numeric(regmatches(d, regexpr("[0-9]+", d)))
    if (day <= young_max) return("pro")
    if (day >= sene_min)  return("sen")
    return("middle")
  }
  NA_character_
}

states <- NULL
if (!is.null(opt$metadata)) {                                  # 1. metadata
  md <- read.delim(opt$metadata, sep = if (grepl("\\.csv$", opt$metadata)) "," else "\t",
                   check.names = FALSE, stringsAsFactors = FALSE)
  key <- colnames(md)[1]; val <- colnames(md)[2]
  raw <- md[[val]][match(scols, md[[key]])]
  # la 2e colonne peut être l'état direct (pro/sen) OU un titre GEO (HUVEC-C1_d5)
  # → on la passe par classify_name (token explicite, sinon jour).
  states <- vapply(raw, classify_name, character(1),
                   young_max = opt$young_max_day, sene_min = opt$sene_min_day)
  msg("[state] croisé avec metadata '%s' (%s → %s, via classify_name)",
      basename(opt$metadata), key, val)
} else {                                                        # 2. nom
  states <- vapply(scols, classify_name, character(1),
                   young_max = opt$young_max_day, sene_min = opt$sene_min_day)
}
# 3. interactif pour les échantillons non résolus
unresolved <- scols[is.na(states)]
if (length(unresolved) && interactive()) {
  for (s in unresolved) {
    a <- readline(sprintf("État de '%s' [%s/%s/skip] : ", s, opt$pro_label, opt$sen_label))
    states[scols == s] <- if (a %in% c(opt$pro_label, opt$sen_label)) a else "middle"
  }
} else if (length(unresolved)) {
  stop(sprintf("Échantillons non résolus (%s). Fournir --metadata ou lancer en interactif.",
               paste(unresolved, collapse = ", ")))
}

# normalise les labels et exclut les intermédiaires
states[states == "pro"] <- opt$pro_label
states[states == "sen"] <- opt$sen_label
names(states) <- scols
keep <- states %in% c(opt$pro_label, opt$sen_label)
if (!opt$keep_middle) { mat <- mat[, keep, drop = FALSE]; states <- states[keep] }
tab <- table(states)
msg("[state] %s", paste(sprintf("%s=%d", names(tab), tab), collapse = "  "))
if (length(unique(states)) < 2) stop("Il faut au moins 2 états distincts.")
pro_i <- which(states == opt$pro_label); sen_i <- which(states == opt$sen_label)

# filtre expression : retire les gènes trop bas (sinon variance ~0 → t explose)
gm_pro <- rowMeans(mat[, pro_i, drop = FALSE])
gm_sen <- rowMeans(mat[, sen_i, drop = FALSE])
expressed <- pmax(gm_pro, gm_sen) >= opt$min_expr
msg("[filter] expression ≥ %.2g (max des 2 groupes) : %d/%d gènes gardés",
    opt$min_expr, sum(expressed), length(expressed))
mat <- mat[expressed, , drop = FALSE]
# Symboles dans l'ORDRE des lignes (source de vérité). DESeq2/limma remplacent
# les rownames par des indices entiers quand il y a des doublons (même symbole à
# plusieurs loci) → on récupère TOUJOURS gene_symbol par position via GENE_IDS,
# jamais via rownames(results/topTable). rownames(mat) uniquifiés pour ces libs.
GENE_IDS <- rownames(mat)
rownames(mat) <- make.unique(as.character(GENE_IDS))

# ------------------------------- type de données ----------------------------
dtype <- opt$data_type
if (dtype == "auto") {
  frac_int <- mean(abs(mat - round(mat)) < 1e-8, na.rm = TRUE)
  dtype <- if (frac_int > 0.95) "counts" else "fpkm"
  msg("[type] auto → %s (%.0f%% entiers)", dtype, 100 * frac_int)
}

# ------------------------------- calcul DE ----------------------------------
# Convention sortie : <A>_vs_<B> = sen_vs_pro ⇒ log2(sen/pro).
res <- NULL
try_deseq <- dtype == "counts" && requireNamespace("DESeq2", quietly = TRUE)
try_limma <- is.null(res) && requireNamespace("limma", quietly = TRUE)

if (try_deseq) {
  msg("[DE] DESeq2 (counts)")
  suppressMessages(library(DESeq2))
  cd <- data.frame(state = factor(states, levels = c(opt$pro_label, opt$sen_label)))
  dds <- DESeqDataSetFromMatrix(round(mat), cd, design = ~state)
  dds <- DESeq(dds, quiet = TRUE)
  r <- as.data.frame(results(dds, contrast = c("state", opt$sen_label, opt$pro_label)))
  res <- data.frame(gene_symbol = GENE_IDS, log2FoldChange = r$log2FoldChange,
                    stat = r$stat, pvalue = r$pvalue, padj = r$padj)
} else if (try_limma) {
  msg("[DE] limma-trend (%s)", dtype)
  suppressMessages(library(limma))
  grp <- factor(states, levels = c(opt$pro_label, opt$sen_label))
  design <- model.matrix(~grp)
  v <- if (dtype == "counts") log2(edgeR_cpm(mat)) else log2(mat + 1)
  fit <- eBayes(lmFit(v, design), trend = TRUE)
  tt <- topTable(fit, coef = 2, number = Inf, sort.by = "none")  # ordre préservé
  res <- data.frame(gene_symbol = GENE_IDS, log2FoldChange = tt$logFC,
                    stat = tt$t, pvalue = tt$P.Value, padj = tt$adj.P.Val)
} else {
  msg("[DE] fallback log2-ratio + t de Welch + BH (ni DESeq2 ni limma)")
  X <- log2(mat + 1)
  ms <- rowMeans(X[, sen_i, drop = FALSE]); mp <- rowMeans(X[, pro_i, drop = FALSE])
  vs <- apply(X[, sen_i, drop = FALSE], 1, var); vp <- apply(X[, pro_i, drop = FALSE], 1, var)
  ns <- length(sen_i); np <- length(pro_i)
  se <- sqrt(vs / ns + vp / np)
  s0 <- median(se, na.rm = TRUE)            # modération SAM (Tusher 2001) : évite
  tstat <- (ms - mp) / (se + s0)            # les t explosifs des gènes ~constants
  dfw <- (vs / ns + vp / np)^2 / ((vs / ns)^2 / (ns - 1) + (vp / np)^2 / (np - 1))
  pval <- 2 * pt(-abs(tstat), df = pmax(dfw, 1))
  res <- data.frame(gene_symbol = GENE_IDS, log2FoldChange = ms - mp,
                    stat = tstat, pvalue = pval, padj = p.adjust(pval, "BH"))
}

# nettoyage : drop gènes sans symbole, dédup (garde |stat| max)
res <- res[!is.na(res$gene_symbol) & res$gene_symbol != "", ]
res <- res[order(-abs(res$stat)), ]
res <- res[!duplicated(res$gene_symbol), ]

dir.create(dirname(opt$out), showWarnings = FALSE, recursive = TRUE)
write.table(res, opt$out, sep = "\t", quote = FALSE, row.names = FALSE)
msg("[out] %d gènes → %s", nrow(res), opt$out)
msg("[out] condition_label = %s | top up: %s | top down: %s", opt$cond_label,
    paste(head(res$gene_symbol[res$log2FoldChange > 0], 5), collapse = ","),
    paste(head(res$gene_symbol[res$log2FoldChange < 0], 5), collapse = ","))
