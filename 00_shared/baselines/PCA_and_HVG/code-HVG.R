##############Stage 1: calculate a common HVG set
library(Seurat)

train_files <- c(
    "Task1_Training_Part1_n50000.rds",
    "Task1_Training_Part2_n50000.rds",
    "Task1_Training_Part3_n50000.rds",
    "Task1_Training_Part4_n50000.rds",
    "Task1_Training_Part5_n40000.rds"
)

objs <- lapply(train_files, readRDS)

## Merge
train <- Reduce(function(x, y) merge(x, y), objs)

## QC
train <- subset(train, subset = nFeature_RNA >= 200)
train <- train[rowSums(train[["RNA"]]@counts > 0) >= 3, ]

## Normalize
train <- NormalizeData(train)

## HVG
train <- FindVariableFeatures(
    train,
    selection.method = "vst",
    nfeatures = 2000
)

hvg <- VariableFeatures(train)

write.table(
    hvg,
    "Training_2000HVG.txt",
    quote = FALSE,
    row.names = FALSE,
    col.names = FALSE
)

##############Stage 2: extract the same genes from all datasets
library(Seurat)
library(data.table)

hvg <- scan("Training_2000HVG.txt",
            what = "",
            quiet = TRUE)

input_dir <- "."
output_dir <- "noFM"

dir.create(output_dir, showWarnings = FALSE)

rds_files <- list.files(
    input_dir,
    pattern="\\.rds$",
    full.names=TRUE
)

for(rds_file in rds_files){

    cat(rds_file,"\n")

    seu <- readRDS(rds_file)

    ## QC
    seu <- subset(seu,
                  subset=nFeature_RNA>=200)

    seu <- seu[rowSums(seu[["RNA"]]@counts>0)>=3,]

    ## Normalize
    seu <- NormalizeData(seu)

    mat <- GetAssayData(
        seu,
        assay="RNA",
        slot="data"
    )

    ## Fill missing genes
    missing_gene <- setdiff(hvg, rownames(mat))

    if(length(missing_gene)>0){

        zero <- Matrix::Matrix(
            0,
            nrow=length(missing_gene),
            ncol=ncol(mat),
            sparse=TRUE
        )

        rownames(zero) <- missing_gene
        colnames(zero) <- colnames(mat)

        mat <- rbind(mat, zero)

    }

    ## Preserve a consistent gene order
    mat <- mat[hvg, ]

    hvg_df <- as.data.frame(t(as.matrix(mat)))

    if("label" %in% colnames(seu@meta.data))
        hvg_df$label <- seu@meta.data$label

    base <- tools::file_path_sans_ext(
        basename(rds_file)
    )

    fwrite(
        hvg_df,
        file=file.path(
            output_dir,
            paste0(base,
            "_2000HVG_with_label.csv")
        ),
        row.names=TRUE
    )

}

