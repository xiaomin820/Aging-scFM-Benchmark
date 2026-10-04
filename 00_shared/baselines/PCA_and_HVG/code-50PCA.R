# ============================================================================
# Memory-efficient implementation: preallocate and fill the matrix to avoid an 84.9 GB allocation
# Fit PCA on training data only, project validation data, and write six CSV files to "50PCA" directory
# ============================================================================

library(irlba)
library(data.table)
library(Seurat)   # Required for Seurat objects

# 1. Create the output directory ---------------------------------------------------------
output_dir <- "50PCA"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
  cat("Created directory:", output_dir, "\n")
}

# 2. Set input paths for the local environment; absolute paths are also accepted----------------------
# Use setwd() if the RDS and HVG files are not in the current directory
# setwd("path/to/input/data")

train_files <- c(
  "Task1_Training_Part1_n50000.rds",
  "Task1_Training_Part2_n50000.rds",
  "Task1_Training_Part3_n50000.rds",
  "Task1_Training_Part4_n50000.rds",
  "Task1_Training_Part5_n40000.rds"
)
test_file <- "Task1_Independent.Test_GSE134355_n32000.rds"
hvg_file <- "Training_2000HVG.txt"

# 3. Read HVGs ---------------------------------------------------------------
hvg_list <- fread(hvg_file, header = FALSE)$V1
cat("Number of HVGs read:", length(hvg_list), "\n")

# ==================== Step 1: determine matrix dimensions ====================
cat("\n===== Step 1: obtain sample counts and common HVGs =====\n")

# Read gene names from the first file and intersect them with the HVGs
temp_obj <- readRDS(train_files[1])
if (inherits(temp_obj, "Seurat")) {
  all_genes <- rownames(GetAssayData(temp_obj, assay = "RNA", slot = "data"))
} else {
  all_genes <- rownames(temp_obj)
}
common_hvg <- intersect(hvg_list, all_genes)
n_hvg <- length(common_hvg)
cat("Number of matched HVGs:", n_hvg, "\n")
if (n_hvg < 2000) warning("Some HVGs were not matched; check gene-name formatting")
rm(temp_obj); gc()

# Read files sequentially to obtain sample counts without loading all data at once
sample_counts <- sapply(train_files, function(f) {
  obj <- readRDS(f)
  if (inherits(obj, "Seurat")) ncol(obj) else ncol(obj)
})
total_n <- sum(sample_counts)
cat("Total training samples:", total_n, "\n")
cat("Samples per training part:", sample_counts, "\n")

# ==================== Step 2: preallocate the combined matrix and fill it sequentially ====================
cat("\n===== Step 2: preallocate the training matrix to avoid copies from rbind =====\n")
# Allocate once: 240k x 2000, approximately 3.8 GB, without temporary copies
train_combined <- matrix(0, nrow = total_n, ncol = n_hvg)
colnames(train_combined) <- common_hvg

start_idx <- 1
for (i in seq_along(train_files)) {
  cat("  Loading and filling:", train_files[i], "...\n")
  obj <- readRDS(train_files[i])
  
  # Extract the expression matrix (genes x cells)
  if (inherits(obj, "Seurat")) {
    mat <- GetAssayData(obj, assay = "RNA", slot = "data")  # or "scale.data"
  } else {
    mat <- as.matrix(obj)
  }
  
  # Subset HVGs and transpose to cells x genes
  mat_sub <- t(as.matrix(mat[common_hvg, , drop = FALSE]))
  
  # Fill the preallocated combined matrix
  end_idx <- start_idx + nrow(mat_sub) - 1
  train_combined[start_idx:end_idx, ] <- mat_sub
  
  start_idx <- end_idx + 1
  rm(obj, mat, mat_sub); gc()  # Release memory promptly
}
cat("Combined training matrix dimensions:", dim(train_combined), "\n")

# ==================== Step 3: fit PCA on the training data ====================
cat("\n===== Step 3: fit PCA (training data only) =====\n")
pca_fit <- prcomp_irlba(
  train_combined,
  n = 50,
  center = TRUE,
  scale. = TRUE
)

train_center <- pca_fit$center
train_scale  <- pca_fit$scale
rotation_mat <- pca_fit$rotation
train_pc_scores <- pca_fit$x
colnames(train_pc_scores) <- paste0("PC", 1:50)

cat("PCA completed; variance explained by the first five PCs:\n",
    round(pca_fit$sdev[1:5]^2 / sum(pca_fit$sdev^2), 4), "\n")

# Release the training matrix to free memory
rm(train_combined); gc()

# ==================== Step 4: split and save the training data with labels ====================
cat("\n===== Step 4: split and save training CSV files to", output_dir, "=====\n")
start_idx <- 1
for (i in seq_along(train_files)) {
  end_idx <- start_idx + sample_counts[i] - 1
  part_pc <- train_pc_scores[start_idx:end_idx, , drop = FALSE]
  
  # ----- Extract ground-truth labels by reloading the original object -----
  orig_obj <- readRDS(train_files[i])

if (!inherits(orig_obj, "Seurat")) {
    stop("The input is not a Seurat object")
}

## Extract cell identifiers
cell_ids <- colnames(orig_obj)

## Extract label metadata
if (!"label" %in% colnames(orig_obj@meta.data)) {
    stop("The Seurat object has no metadata column named label")
}
label_vec <- as.character(orig_obj@meta.data$label)

## Check lengths
stopifnot(length(cell_ids) == nrow(part_pc))
stopifnot(length(label_vec) == nrow(part_pc))

## Construct the output
out_df <- data.frame(
    Cell = cell_ids,
    as.data.frame(part_pc),
    label = label_vec,
    check.names = FALSE
)

out_file <- file.path(
    output_dir,
    paste0("Task1_Training_Part", i,
           "_n", sample_counts[i],
           "_pca.csv")
)

fwrite(out_df, out_file)
  cat("Saved:", out_file, "\n")
  
  start_idx <- end_idx + 1
  rm(orig_obj, part_pc); gc()
}

# ==================== Step 5: project the validation (test) data ====================
cat("\n===== Step 5: load and project the validation data =====\n")
test_obj <- readRDS(test_file)

# Extract the test expression matrix (genes x cells)
if (!inherits(test_obj, "Seurat")) {
    stop("The test input is not a Seurat object")
}
test_mat <- GetAssayData(test_obj, assay = "RNA", slot = "data")  # Use "data" or "scale.data"

# Retain only the HVGs used for training (common_hvg)
test_mat_sub <- test_mat[common_hvg, , drop = FALSE]

# Transpose to cells x genes
test_mat_t <- t(as.matrix(test_mat_sub))

# Standardize with training-derived centering and scaling parameters
test_scaled <- scale(test_mat_t, center = train_center, scale = train_scale)

# Project into PCA space
test_pc_scores <- test_scaled %*% rotation_mat
colnames(test_pc_scores) <- paste0("PC", 1:50)
cat("Test projection completed; dimensions:", dim(test_pc_scores), "\n")

# Extract cell identifiers and labels
test_cell_ids <- colnames(test_obj)
if (!"label" %in% colnames(test_obj@meta.data)) {
    stop("The test metadata has no label column")
}
test_label <- as.character(test_obj@meta.data$label)

# Check lengths
stopifnot(length(test_cell_ids) == nrow(test_pc_scores))
stopifnot(length(test_label) == nrow(test_pc_scores))

# Construct the output data frame
test_out_df <- data.frame(
    Cell = test_cell_ids,
    as.data.frame(test_pc_scores),
    label = test_label,
    check.names = FALSE
)

out_test_file <- file.path(
    output_dir,
    "Independent_Test_GSE134355_pca.csv"
)

fwrite(test_out_df, out_test_file)
cat("Saved:", out_test_file, "\n")

cat("\n====== Completed. All CSV files were saved to '", output_dir, "' directory ======\n")