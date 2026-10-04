# ===========================================
# Process RDS files in batch and export PCA and HVG features
# ===========================================
library(Seurat)
library(Matrix)
library(data.table)

# Set input and output directories
input_dir <- "."   # Set the path to the RDS directory
output_dir <- "noFM"  # Set the output directory for CSV files
dir.create(output_dir, showWarnings = FALSE)

# Find all RDS files
rds_files <- list.files(input_dir, pattern = "\\.rds$", full.names = TRUE)

# Process each file in sequence
for (rds_file in rds_files) {
  
  cat("Processing file:", rds_file, "\n")
  
  # 1. Read the Seurat object
  seu <- readRDS(rds_file)
  
  # 2. QC
  seu <- subset(seu, subset = nFeature_RNA >= 200)
  seu <- seu[rowSums(seu[["RNA"]]@counts > 0) >= 3, ]
  
  # 3. Normalization and HVG selection
  seu <- NormalizeData(seu, normalization.method = "LogNormalize", scale.factor = 1e4)
  seu <- FindVariableFeatures(seu, selection.method = "vst", nfeatures = 2000)
  
  # 4. PCA (50 PCs)
  seu <- ScaleData(seu, features = VariableFeatures(seu))
  seu <- RunPCA(seu, features = VariableFeatures(seu), npcs = 50)
  
  # Base filename
  base_name <- tools::file_path_sans_ext(basename(rds_file))
  
  # Export PCA features
  pca_df <- as.data.frame(Embeddings(seu, reduction = "pca"))
  if ("label" %in% colnames(seu@meta.data)) {
    pca_df$label <- seu@meta.data$label
  }
  fwrite(pca_df, file = file.path(output_dir, paste0(base_name, "_50PCA_with_label.csv")), row.names = TRUE)
  
  # Export HVG features
  hvg_genes <- VariableFeatures(seu)
  hvg_mat <- GetAssayData(seu, assay = "RNA", slot = "data")[hvg_genes, ]
  hvg_df <- as.data.frame(t(as.matrix(hvg_mat)))
  if ("label" %in% colnames(seu@meta.data)) {
    hvg_df$label <- seu@meta.data$label
  }
  fwrite(hvg_df, file = file.path(output_dir, paste0(base_name, "_2000HVG_with_label.csv")), row.names = TRUE)
  
  cat("Completed:", rds_file, "\n\n")
}

cat("All files have been processed.\n")