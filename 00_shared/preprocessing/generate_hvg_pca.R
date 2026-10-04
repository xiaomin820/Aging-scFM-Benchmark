#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(data.table)
  library(irlba)
})

parse_cli <- function() {
  tokens <- commandArgs(trailingOnly = TRUE)
  out <- list()
  for (token in tokens) {
    if (!startsWith(token, "--") || !grepl("=", token, fixed = TRUE)) {
      stop("Arguments must use --name=value syntax: ", token)
    }
    pair <- strsplit(sub("^--", "", token), "=", fixed = TRUE)[[1]]
    out[[pair[1]]] <- paste(pair[-1], collapse = "=")
  }
  out
}

required <- function(args, name) {
  value <- args[[name]]
  if (is.null(value) || !nzchar(value)) stop("Missing --", name)
  value
}

split_paths <- function(value) trimws(strsplit(value, ",", fixed = TRUE)[[1]])

filter_and_normalize <- function(object) {
  DefaultAssay(object) <- "RNA"
  object <- subset(object, subset = nFeature_RNA >= 200)
  object <- object[rowSums(GetAssayData(object, assay = "RNA", slot = "counts") > 0) >= 3, ]
  NormalizeData(
    object,
    normalization.method = "LogNormalize",
    scale.factor = 1e4,
    verbose = FALSE
  )
}

aligned_hvg_matrix <- function(object, hvg) {
  object <- filter_and_normalize(object)
  values <- GetAssayData(object, assay = "RNA", slot = "data")
  aligned <- Matrix(
    0,
    nrow = length(hvg),
    ncol = ncol(values),
    sparse = TRUE,
    dimnames = list(hvg, colnames(values))
  )
  present <- intersect(hvg, rownames(values))
  aligned[present, ] <- values[present, , drop = FALSE]
  list(matrix = aligned, metadata = object@meta.data)
}

output_frame <- function(cell_ids, values, metadata, prefix) {
  frame <- data.frame(cell_id = cell_ids, check.names = FALSE)
  value_frame <- as.data.frame(values, check.names = FALSE)
  colnames(value_frame) <- paste0(prefix, colnames(value_frame))
  frame <- cbind(frame, value_frame)
  keep <- intersect(
    c("donorID", "cell_type", "condition", "age", "label", "label_task1", "label_task2"),
    colnames(metadata)
  )
  if (length(keep)) frame <- cbind(frame, metadata[cell_ids, keep, drop = FALSE])
  frame
}

args <- parse_cli()
train_files <- split_paths(required(args, "train"))
evaluation_files <- if (is.null(args$evaluation)) character() else split_paths(args$evaluation)
output_dir <- required(args, "output")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!all(file.exists(c(train_files, evaluation_files)))) stop("One or more RDS files do not exist")
set.seed(as.integer(if (is.null(args$seed)) 42 else args$seed))

message("Selecting 2,000 HVGs from pooled training data")
training_objects <- lapply(train_files, readRDS)
pooled <- Reduce(function(x, y) merge(x, y), training_objects)
pooled <- filter_and_normalize(pooled)
pooled <- FindVariableFeatures(
  pooled,
  selection.method = "vst",
  nfeatures = 2000,
  verbose = FALSE
)
hvg <- VariableFeatures(pooled)
writeLines(hvg, file.path(output_dir, "training_2000_hvg.txt"))
rm(pooled)
gc()

message("Building aligned training matrices")
training_prepared <- lapply(training_objects, aligned_hvg_matrix, hvg = hvg)
training_matrix <- do.call(rbind, lapply(training_prepared, function(x) t(as.matrix(x$matrix))))

message("Fitting the 50-component training-only PCA")
pca_fit <- prcomp_irlba(training_matrix, n = 50, center = TRUE, scale. = TRUE)
colnames(pca_fit$x) <- paste0("PC", seq_len(ncol(pca_fit$x)))
saveRDS(
  list(hvg = hvg, center = pca_fit$center, scale = pca_fit$scale, rotation = pca_fit$rotation),
  file.path(output_dir, "training_pca_transform.rds")
)

write_outputs <- function(path, prepared, pca_values) {
  stem <- tools::file_path_sans_ext(basename(path))
  cell_ids <- colnames(prepared$matrix)
  hvg_values <- t(as.matrix(prepared$matrix))
  colnames(hvg_values) <- hvg
  fwrite(
    output_frame(cell_ids, hvg_values, prepared$metadata, "emb_"),
    file.path(output_dir, paste0(stem, "_2000hvg.csv"))
  )
  fwrite(
    output_frame(cell_ids, pca_values, prepared$metadata, "emb_"),
    file.path(output_dir, paste0(stem, "_50pca.csv"))
  )
}

start <- 1
for (i in seq_along(train_files)) {
  n <- ncol(training_prepared[[i]]$matrix)
  positions <- start:(start + n - 1)
  write_outputs(train_files[i], training_prepared[[i]], pca_fit$x[positions, , drop = FALSE])
  start <- start + n
}

for (path in evaluation_files) {
  prepared <- aligned_hvg_matrix(readRDS(path), hvg)
  values <- t(as.matrix(prepared$matrix))
  scaled <- scale(values, center = pca_fit$center, scale = pca_fit$scale)
  projected <- scaled %*% pca_fit$rotation
  colnames(projected) <- paste0("PC", seq_len(ncol(projected)))
  write_outputs(path, prepared, projected)
}

message("Saved HVG and PCA references to ", normalizePath(output_dir))
