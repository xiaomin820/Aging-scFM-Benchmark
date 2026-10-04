############################################################
## Mouse gene -> Human ortholog
## Support ENSMUSG and MGI symbol
############################################################

library(Seurat)
library(biomaRt)
library(dplyr)
library(Matrix)



############################################################
## Input and output directories
############################################################

input_dir <- "."
output_dir <- "humanized_rds"

dir.create(
    output_dir,
    showWarnings = FALSE
)



############################################################
## Connect Ensembl
############################################################

human <- useMart(
    "ensembl",
    dataset = "hsapiens_gene_ensembl",
    host = "https://dec2021.archive.ensembl.org/"
)


mouse <- useMart(
    "ensembl",
    dataset = "mmusculus_gene_ensembl",
    host = "https://dec2021.archive.ensembl.org/"
)



############################################################
## Mouse -> Human conversion function
############################################################

convert_mouse_to_human <- function(genes){


    # remove Ensembl version
    genes_clean <- sub(
        "\\..*$",
        "",
        genes
    )


    ########################################################
    ## Detect gene type
    ########################################################

    ensembl_ratio <- mean(
        grepl(
            "^ENSMUSG",
            genes_clean
        )
    )


    if(ensembl_ratio > 0.5){


        message(
            "Detected: Mouse Ensembl ID"
        )


        map <- getLDS(

            attributes = "ensembl_gene_id",

            filters = "ensembl_gene_id",

            values = genes_clean,

            mart = mouse,


            attributesL = "hgnc_symbol",

            martL = human,


            uniqueRows = TRUE

        )


    }else{


        message(
            "Detected: Mouse MGI symbol"
        )


        map <- getLDS(

            attributes = "mgi_symbol",

            filters = "mgi_symbol",

            values = genes_clean,

            mart = mouse,


            attributesL = "hgnc_symbol",

            martL = human,


            uniqueRows = TRUE

        )

    }


    colnames(map) <- c(
        "mouse_gene",
        "human_gene"
    )


    map <- map %>%
        filter(
            mouse_gene != "",
            human_gene != "",
            !is.na(mouse_gene),
            !is.na(human_gene)
        )


    return(map)

}




############################################################
## Read RDS files
############################################################

files <- list.files(
    input_dir,
    pattern="\\.rds$",
    full.names = TRUE
)



############################################################
## Loop
############################################################

for(file in files){


    cat("\n=================================\n")

    cat(
        "Processing:",
        basename(file),
        "\n"
    )



    ########################################################
    ## Load Seurat
    ########################################################

    seu <- readRDS(file)



    old_genes <- rownames(seu)


    cat(
        "Original genes:",
        length(old_genes),
        "\n"
    )



    ########################################################
    ## Convert gene
    ########################################################

    gene_map <- convert_mouse_to_human(
        old_genes
    )


    cat(
        "Mapped genes:",
        nrow(gene_map),
        "\n"
    )



    ########################################################
    ## Extract counts (Seurat v4)
    ########################################################

    counts <- GetAssayData(
        seu,
        assay = "RNA",
        slot = "counts"
    )


    counts <- as.matrix(counts)



    ########################################################
    ## Keep mapped genes
    ########################################################

    keep <- intersect(
        old_genes,
        gene_map$mouse_gene
    )


    counts <- counts[
        keep,
        ,
        drop = FALSE
    ]



    ########################################################
    ## Replace mouse gene -> human gene
    ########################################################

    human_names <- gene_map$human_gene[
        match(
            rownames(counts),
            gene_map$mouse_gene
        )
    ]


    rownames(counts) <- human_names



    ########################################################
    ## Merge duplicated human genes
    ########################################################

    if(anyDuplicated(rownames(counts)) > 0){


        cat(
            "Merging duplicated human genes:",
            sum(duplicated(rownames(counts))),
            "\n"
        )


        counts <- rowsum(
            x = counts,
            group = rownames(counts)
        )


    }



    ########################################################
    ## Convert sparse matrix
    ########################################################

    counts <- Matrix::Matrix(
        counts,
        sparse = TRUE
    )



    ########################################################
    ## Keep metadata
    ########################################################

    meta <- seu@meta.data



    ########################################################
    ## Recreate Seurat object
    ########################################################

    seu_human <- CreateSeuratObject(

        counts = counts,

        meta.data = meta,

        project = "mouse_to_human"

    )



    ########################################################
    ## Save
    ########################################################

    outfile <- file.path(

        output_dir,

        paste0(

            tools::file_path_sans_ext(
                basename(file)
            ),

            "_humanGene.rds"

        )

    )


    saveRDS(
        seu_human,
        outfile
    )


    cat(
        "Final human genes:",
        nrow(seu_human),
        "\n"
    )


    cat(
        "Saved:",
        outfile,
        "\n"
    )


}


cat("\n========== Finished ==========\n")