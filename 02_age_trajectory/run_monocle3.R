#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(monocle3)
  library(readr)
})

command <- commandArgs(trailingOnly = FALSE)
this_file <- sub("^--file=", "", command[grepl("^--file=", command)][[1]])
source(file.path(dirname(normalizePath(this_file)), "common.R"))

arguments <- parse_cli()
model <- required_arg(arguments, "model")
cell_type <- required_arg(arguments, "cell_type")
object_path <- required_arg(arguments, "seurat")
embedding_path <- required_arg(arguments, "embedding")
root_path <- required_arg(arguments, "root_cells")
output_dir <- required_arg(arguments, "output")
age_column <- if (is.null(arguments$age_column)) "age" else arguments$age_column
cell_type_column <- if (is.null(arguments$cell_type_column)) "cell_type" else arguments$cell_type_column
donor_column <- if (is.null(arguments$donor_column)) "donorID" else arguments$donor_column
minimum_cells <- integer_arg(arguments, "min_cells", 1000, minimum = 3)
seed <- integer_arg(arguments, "seed", 42)

input_paths <- c(object_path, embedding_path, root_path)
if (!all(file.exists(input_paths))) {
  stop("Input files do not exist: ", paste(input_paths[!file.exists(input_paths)], collapse = ", "))
}
set.seed(seed)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

object <- readRDS(object_path)
if (!inherits(object, "Seurat")) stop("--seurat must contain a Seurat object")
required_metadata <- c(age_column, cell_type_column, donor_column)
missing_metadata <- setdiff(required_metadata, colnames(object@meta.data))
if (length(missing_metadata)) {
  stop("Missing Seurat metadata columns: ", paste(missing_metadata, collapse = ", "))
}

cell_type_values <- as.character(object@meta.data[[cell_type_column]])
cell_type_cells <- rownames(object@meta.data)[!is.na(cell_type_values) & cell_type_values == cell_type]
if (!length(cell_type_cells)) stop("No cells found for cell type: ", cell_type)

embedding_frame <- read_csv(embedding_path, show_col_types = FALSE)
if (!"cell_id" %in% colnames(embedding_frame)) stop("Embedding CSV requires a cell_id column")
embedding_frame$cell_id <- as.character(embedding_frame$cell_id)
if (anyNA(embedding_frame$cell_id) || any(!nzchar(embedding_frame$cell_id))) {
  stop("Embedding cell_id values cannot be missing or empty")
}
if (anyDuplicated(embedding_frame$cell_id)) stop("Embedding cell_id values must be unique")

embedding_columns <- grep("^(emb_|embedding_|dim_)", colnames(embedding_frame), value = TRUE)
if (!length(embedding_columns)) {
  stop("No embedding columns found; expected names beginning with emb_, embedding_, or dim_")
}
non_numeric <- embedding_columns[!vapply(embedding_frame[embedding_columns], is.numeric, logical(1))]
if (length(non_numeric)) stop("Embedding columns must be numeric: ", paste(non_numeric, collapse = ", "))
embedding <- as.matrix(embedding_frame[embedding_columns])
if (any(!is.finite(embedding))) stop("Embedding dimensions contain missing or non-finite values")
rownames(embedding) <- embedding_frame$cell_id

cells <- intersect(cell_type_cells, rownames(embedding))
if (length(cells) < minimum_cells) {
  stop(
    "Only ", length(cells), " matched cells were available for ", cell_type,
    "; --min_cells=", minimum_cells, " is required"
  )
}
object <- subset(object, cells = cells)
embedding <- embedding[colnames(object), , drop = FALSE]
object@meta.data[[age_column]] <- numeric_metadata(object@meta.data[[age_column]], age_column)
if (sum(is.finite(object@meta.data[[age_column]])) < 3) {
  stop("Fewer than three matched cells have finite values in metadata column '", age_column, "'")
}

object[["benchmark"]] <- CreateDimReducObject(
  embeddings = embedding,
  key = "BENCH_",
  assay = DefaultAssay(object)
)
neighbor_dimensions <- seq_len(min(20, ncol(embedding)))
object <- FindNeighbors(object, reduction = "benchmark", dims = neighbor_dimensions, verbose = FALSE)
object <- FindClusters(object, resolution = 0.5, verbose = FALSE)
object <- RunUMAP(
  object,
  reduction = "benchmark",
  dims = neighbor_dimensions,
  min.dist = 0.01,
  seed.use = seed,
  reduction.name = "benchmark.umap",
  reduction.key = "BUMAP_",
  verbose = FALSE
)

counts <- GetAssayData(object, assay = "RNA", slot = "counts")
gene_metadata <- data.frame(gene_short_name = rownames(counts), row.names = rownames(counts))
cds <- new_cell_data_set(
  counts,
  cell_metadata = object@meta.data,
  gene_metadata = gene_metadata
)
pca_dimensions <- min(50L, ncol(cds) - 1L, nrow(cds) - 1L)
if (pca_dimensions < 2) stop("At least three cells and three expressed genes are required")
cds <- preprocess_cds(cds, num_dim = pca_dimensions)
cds <- reduce_dimension(cds, preprocess_method = "PCA")
reducedDims(cds)$UMAP <- Embeddings(object, reduction = "benchmark.umap")[colnames(cds), , drop = FALSE]
cds <- cluster_cells(cds)
cds <- learn_graph(cds)

root_cells <- trimws(readLines(root_path, warn = FALSE))
root_cells <- unique(root_cells[nzchar(root_cells)])
root_cells <- intersect(root_cells, colnames(cds))
if (!length(root_cells)) stop("No recorded root cells matched the analyzed cells")
cds <- order_cells(cds, root_cells = root_cells)

cell_table <- as.data.frame(colData(cds))
cell_table$cell_id <- rownames(cell_table)
cell_table$pseudotime <- monocle3::pseudotime(cds)
umap <- as.data.frame(reducedDims(cds)$UMAP)
umap$cell_id <- rownames(umap)
colnames(umap)[seq_len(2)] <- c("UMAP_1", "UMAP_2")
cell_table <- merge(cell_table, umap, by = "cell_id", all.x = TRUE, sort = FALSE)

metrics <- calculate_task2_metrics(
  age = cell_table[[age_column]],
  pseudotime = cell_table$pseudotime,
  donor = cell_table[[donor_column]]
)
metrics <- cbind(data.frame(model = model, cell_type = cell_type), metrics)

saveRDS(cds, file.path(output_dir, "monocle3_cds.rds"))
write_csv(cell_table, file.path(output_dir, "cell_pseudotime.csv"))
write_csv(metrics, file.path(output_dir, "task2_metrics.csv"))
writeLines(root_cells, file.path(output_dir, "root_cells_used.txt"))
message("Saved Task 2 trajectory outputs to ", normalizePath(output_dir))
