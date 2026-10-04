
############################################################
# Donor-level Age Prediction Evaluation
#
# Statistical unit:
#   DONOR
#
# Workflow:
#   Single-cell predictions
#          (lower)
#   donor x cell_type x condition
#          (lower)
#   mean(pred_age) within donor
#          (lower)
#   donor-level prediction
#          (lower)
#   PCC / MAE / RMSE / R2 ...
#          (lower)
#   donor-level bootstrap 95% CI
############################################################


############################################################
# 1. Libraries
############################################################

library(tidyverse)
library(irr)
library(DescTools)


############################################################
# 2. Parameters
############################################################

input_dir <- "."

metadata_dir <- "metadata"

output_file <- "Age_prediction_metrics_donor_CI.csv"

set.seed(123)

n_boot <- 1000

sample_fraction <- 0.8


############################################################
# 3. Calculate metrics
#
# IMPORTANT:
# Input df must already be DONOR-LEVEL data.
#
# Each row = one donor.
############################################################

calculate_metrics <- function(df) {

    df <- df %>%
        filter(
            !is.na(true_age),
            !is.na(pred_age)
        )

    if (nrow(df) < 3) {

        out <- rep(
            NA_real_,
            13
        )

        names(out) <- c(
            "PCC",
            "Spearman",
            "MAE",
            "RMSE",
            "R2",
            "CCC",
            "ICC",
            "Bias",
            "MedAE",
            "Calibration_slope",
            "Calibration_intercept",
            "Age_difference",
            "Age_difference_SD"
        )

        return(out)
    }


    true <- df$true_age

    pred <- df$pred_age


    ########################################################
    # PCC
    ########################################################

    PCC <- suppressWarnings(
        cor(
            true,
            pred,
            method = "pearson",
            use = "complete.obs"
        )
    )


    ########################################################
    # Spearman
    ########################################################

    Spearman <- suppressWarnings(
        cor(
            true,
            pred,
            method = "spearman",
            use = "complete.obs"
        )
    )


    ########################################################
    # MAE
    ########################################################

    MAE <- mean(
        abs(pred - true)
    )


    ########################################################
    # RMSE
    ########################################################

    RMSE <- sqrt(
        mean(
            (pred - true)^2
        )
    )


    ########################################################
    # R2
    ########################################################

    denominator <- sum(
        (true - mean(true))^2
    )

    if (denominator == 0) {

        R2 <- NA_real_

    } else {

        R2 <- 1 -
            sum(
                (pred - true)^2
            ) /
            denominator
    }


    ########################################################
    # Bias
    ########################################################

    Bias <- mean(
        pred - true
    )


    ########################################################
    # Median absolute error
    ########################################################

    MedAE <- median(
        abs(pred - true)
    )


    ########################################################
    # CCC
    ########################################################

    CCC <- tryCatch(

        DescTools::CCC(
            pred,
            true
        )$rho.c$est,

        error = function(e) {
            NA_real_
        }
    )


    ########################################################
    # ICC
    ########################################################

    ICC <- tryCatch(

        irr::icc(
            data.frame(
                pred = pred,
                true = true
            ),
            model = "twoway",
            type = "agreement",
            unit = "single"
        )$value,

        error = function(e) {
            NA_real_
        }
    )


    ########################################################
    # Calibration
    ########################################################

    cal_model <- tryCatch(

        lm(
            pred ~ true
        ),

        error = function(e) {
            NULL
        }
    )


    if (is.null(cal_model)) {

        Calibration_slope <- NA_real_

        Calibration_intercept <- NA_real_

    } else {

        coefs <- coef(cal_model)

        Calibration_intercept <- unname(
            coefs[1]
        )

        Calibration_slope <- unname(
            coefs[2]
        )
    }


    ########################################################
    # Age difference
    ########################################################

    age_difference <- pred - true

    Age_difference <- mean(
        age_difference
    )

    Age_difference_SD <- sd(
        age_difference
    )


    ########################################################
    # Return
    ########################################################

    c(
        PCC = PCC,
        Spearman = Spearman,
        MAE = MAE,
        RMSE = RMSE,
        R2 = R2,
        CCC = CCC,
        ICC = ICC,
        Bias = Bias,
        MedAE = MedAE,
        Calibration_slope = Calibration_slope,
        Calibration_intercept = Calibration_intercept,
        Age_difference = Age_difference,
        Age_difference_SD = Age_difference_SD
    )
}



############################################################
# 4. Donor-level bootstrap
#
# IMPORTANT:
# Bootstrap sampling is performed on DONORS.
############################################################

bootstrap_metrics <- function(
    donor_df,
    n_boot = 1000,
    fraction = 0.8
) {

    donor_df <- donor_df %>%
        filter(
            !is.na(true_age),
            !is.na(pred_age)
        )


    n_donor <- nrow(
        donor_df
    )


    metric_names <- names(
        calculate_metrics(
            donor_df
        )
    )


    if (n_donor < 3) {

        return(
            tibble(
                Metric = metric_names,
                Mean = NA_real_,
                Lower95 = NA_real_,
                Upper95 = NA_real_
            )
        )
    }


    ########################################################
    # Number of donors sampled in each bootstrap
    ########################################################

    sample_size <- max(
        3,
        floor(
            n_donor * fraction
        )
    )

    sample_size <- min(
        sample_size,
        n_donor
    )


    ########################################################
    # Bootstrap matrix
    ########################################################

    bootstrap_results <- matrix(
        NA_real_,
        nrow = n_boot,
        ncol = length(metric_names)
    )

    colnames(
        bootstrap_results
    ) <- metric_names


    ########################################################
    # Bootstrap
    ########################################################

    for (i in seq_len(n_boot)) {

        idx <- sample(
            seq_len(n_donor),
            size = sample_size,
            replace = TRUE
        )


        bootstrap_df <- donor_df[
            idx,
            ,
            drop = FALSE
        ]


        bootstrap_results[i, ] <-
            calculate_metrics(
                bootstrap_df
            )
    }


    ########################################################
    # Summary
    ########################################################

    result <- tibble(

        Metric = metric_names,

        Mean = apply(
            bootstrap_results,
            2,
            mean,
            na.rm = TRUE
        ),

        Lower95 = apply(
            bootstrap_results,
            2,
            quantile,
            probs = 0.025,
            na.rm = TRUE
        ),

        Upper95 = apply(
            bootstrap_results,
            2,
            quantile,
            probs = 0.975,
            na.rm = TRUE
        )
    )


    return(result)
}



############################################################
# 5. Read metadata
############################################################

metadata_files <- list.files(
    metadata_dir,
    pattern = "\\.csv$",
    full.names = TRUE
)


if (length(metadata_files) == 0) {

    stop(
        "No metadata CSV files found in: ",
        metadata_dir
    )
}


metadata_list <- list()


for (i in seq_along(metadata_files)) {

    cat(
        "Reading metadata:",
        metadata_files[i],
        "\n"
    )


    tmp <- read.csv(
        metadata_files[i],
        check.names = FALSE,
        stringsAsFactors = FALSE
    )


    ########################################################
    # Check required columns
    ########################################################

    required_metadata <- c(
        "cell",
        "cell_type",
        "condition",
        "donorID"
    )


    missing_columns <- setdiff(
        required_metadata,
        colnames(tmp)
    )


    if (length(missing_columns) > 0) {

        warning(
            "Skipping metadata file because columns are missing: ",
            basename(metadata_files[i]),
            "\nMissing: ",
            paste(
                missing_columns,
                collapse = ", "
            )
        )

        next
    }


    tmp <- tmp %>%
        select(
            cell,
            cell_type,
            condition,
            donorID
        ) %>%
        distinct()


    metadata_list <- append(
        metadata_list,
        list(tmp)
    )
}


if (length(metadata_list) == 0) {

    stop(
        "No valid metadata files were found."
    )
}


metadata <- bind_rows(
    metadata_list
) %>%
    distinct(
        cell,
        .keep_all = TRUE
    )


cat(
    "\nTotal metadata cells:",
    nrow(metadata),
    "\n"
)


############################################################
# 6. Retained cell types
############################################################

keep_cell_types <- c(
    "Excitatory neuron",
    "Inhibitory neuron",
    "Astrocyte",
    "OPC",
    "Oligodendrocyte",
    "Microglia"
)


metadata <- metadata %>%
    filter(
        cell_type %in% keep_cell_types
    )


cat(
    "Metadata cells after cell-type filtering:",
    nrow(metadata),
    "\n"
)


############################################################
# 7. Prediction files
############################################################

prediction_files <- list.files(
    input_dir,
    pattern = "\\.csv$",
    full.names = TRUE
)


############################################################
# Remove output file itself
############################################################

prediction_files <- prediction_files[
    basename(prediction_files) !=
        basename(output_file)
]


if (length(prediction_files) == 0) {

    stop(
        "No prediction CSV files found."
    )
}


cat(
    "\nPrediction files:",
    length(prediction_files),
    "\n\n"
)


############################################################
# 8. Remove previous output
############################################################

if (file.exists(output_file)) {

    file.remove(
        output_file
    )
}


############################################################
# 9. Main loop
############################################################

for (prediction_file in prediction_files) {


    cat(
        "\n==============================================\n"
    )

    cat(
        "Processing:",
        basename(prediction_file),
        "\n"
    )

    cat(
        "==============================================\n"
    )


    ########################################################
    # Read prediction
    ########################################################

    prediction_df <- tryCatch(

        read.csv(
            prediction_file,
            check.names = FALSE,
            stringsAsFactors = FALSE
        ),

        error = function(e) {

            warning(
                "Cannot read prediction file: ",
                basename(prediction_file),
                "\n",
                e$message
            )

            return(NULL)
        }
    )


    if (is.null(prediction_df)) {
        next
    }


    ########################################################
    # Check prediction columns
    ########################################################

    required_prediction <- c(
        "cell_id",
        "true_age",
        "pred_age"
    )


    missing_prediction <- setdiff(
        required_prediction,
        colnames(prediction_df)
    )


    if (length(missing_prediction) > 0) {

        warning(
            "Skipping file: ",
            basename(prediction_file),
            "\nMissing columns: ",
            paste(
                missing_prediction,
                collapse = ", "
            )
        )

        next
    }


    ########################################################
    # Keep required columns
    ########################################################

    prediction_df <- prediction_df %>%
        select(
            cell_id,
            true_age,
            pred_age
        )


    ########################################################
    # Ensure numeric
    ########################################################

    prediction_df <- prediction_df %>%
        mutate(
            true_age = suppressWarnings(
                as.numeric(true_age)
            ),
            pred_age = suppressWarnings(
                as.numeric(pred_age)
            )
        )


    ########################################################
    # Merge metadata
    ########################################################

    merged_df <- prediction_df %>%
        inner_join(
            metadata,
            by = c(
                "cell_id" = "cell"
            )
        )


    ########################################################
    # Remove missing values
    ########################################################

    merged_df <- merged_df %>%
        filter(
            !is.na(true_age),
            !is.na(pred_age),
            !is.na(cell_type),
            !is.na(condition),
            !is.na(donorID)
        )


    cat(
        "Matched cells:",
        nrow(merged_df),
        "\n"
    )


    if (nrow(merged_df) == 0) {

        warning(
            "No matched cells for: ",
            basename(prediction_file)
        )

        next
    }


    ########################################################
    # Check true age consistency within donor
    ########################################################

    age_check <- merged_df %>%
        group_by(
            donorID
        ) %>%
        summarise(
            n_true_age = n_distinct(
                true_age
            ),
            .groups = "drop"
        )


    inconsistent_donors <- age_check %>%
        filter(
            n_true_age > 1
        )


    if (nrow(inconsistent_donors) > 0) {

        warning(
            "Some donors have multiple true_age values in: ",
            basename(prediction_file),
            "\nUsing mean true_age within donor."
        )
    }


    ########################################################
    # DONOR-LEVEL AGGREGATION
    #
    # One row = one donor x cell_type x condition
    ########################################################

    donor_df <- merged_df %>%
        group_by(
            cell_type,
            condition,
            donorID
        ) %>%
        summarise(

            true_age = mean(
                true_age,
                na.rm = TRUE
            ),

            pred_age = mean(
                pred_age,
                na.rm = TRUE
            ),

            N_cell = n(),

            .groups = "drop"
        )


    ########################################################
    # Summary
    ########################################################

    cat(
        "Donor x cell type observations:",
        nrow(donor_df),
        "\n"
    )

    cat(
        "Unique donors:",
        n_distinct(
            donor_df$donorID
        ),
        "\n"
    )


    ########################################################
    # Calculate metrics for each
    # cell_type x condition
    ########################################################

    group_list <- donor_df %>%
        group_by(
            cell_type,
            condition
        ) %>%
        group_split()


    result_list <- list()


    ########################################################
    # Loop groups
    ########################################################

    for (g in group_list) {


        if (nrow(g) < 3) {
            next
        }


        ####################################################
        # Group information
        ####################################################

        group_cell_type <- unique(
            g$cell_type
        )[1]


        group_condition <- unique(
            g$condition
        )[1]


        n_donor <- nrow(
            g
        )


        n_cell <- sum(
            g$N_cell
        )


        ####################################################
        # Donor-level metrics
        ####################################################

        metric_result <- bootstrap_metrics(
            donor_df = g %>%
                select(
                    true_age,
                    pred_age
                ),
            n_boot = n_boot,
            fraction = sample_fraction
        )


        ####################################################
        # Add metadata
        ####################################################

        metric_result <- metric_result %>%
            mutate(

                file = basename(
                    prediction_file
                ),

                cell_type = group_cell_type,

                condition = group_condition,

                N_donor = n_donor,

                N_cell = n_cell

            ) %>%
            select(
                file,
                cell_type,
                condition,
                N_donor,
                N_cell,
                Metric,
                Mean,
                Lower95,
                Upper95
            )


        ####################################################
        # Store
        ####################################################

        result_list <- append(
            result_list,
            list(metric_result)
        )
    }


    ########################################################
    # Combine results
    ########################################################

    if (length(result_list) == 0) {

        warning(
            "No valid donor-level groups for: ",
            basename(prediction_file)
        )

        next
    }


    final_result <- bind_rows(
        result_list
    )


    ########################################################
    # Incremental save
    ########################################################

    if (!file.exists(output_file)) {

        write.csv(
            final_result,
            output_file,
            row.names = FALSE
        )

    } else {

        write.table(
            final_result,
            output_file,
            sep = ",",
            row.names = FALSE,
            col.names = FALSE,
            append = TRUE
        )
    }


    cat(
        "Finished:",
        basename(prediction_file),
        "\n"
    )
}


############################################################
# 10. Finished
############################################################

cat(
    "\n====================================================\n"
)

cat(
    "Donor-level age prediction analysis completed!\n"
)

cat(
    "Output:",
    output_file,
    "\n"
)

cat(
    "====================================================\n"
)

############################################################
# Donor-level Age Prediction Evaluation
#
# Statistical unit:
#   DONOR
#
# Workflow:
#   Single-cell predictions
#          (lower)
#   donor x cell_type x condition
#          (lower)
#   mean(pred_age) within donor
#          (lower)
#   donor-level prediction
#          (lower)
#   PCC / MAE / RMSE / R2 ...
#          (lower)
#   donor-level bootstrap 95% CI
############################################################


############################################################
# 1. Libraries
############################################################

library(tidyverse)
library(irr)
library(DescTools)


############################################################
# 2. Parameters
############################################################

input_dir <- "."

metadata_dir <- "metadata"

output_file <- "Age_prediction_metrics_donor_CI.csv"

set.seed(123)

n_boot <- 1000

sample_fraction <- 0.8


############################################################
# 3. Calculate metrics
#
# IMPORTANT:
# Input df must already be DONOR-LEVEL data.
#
# Each row = one donor.
############################################################

calculate_metrics <- function(df) {

    df <- df %>%
        filter(
            !is.na(true_age),
            !is.na(pred_age)
        )

    if (nrow(df) < 3) {

        out <- rep(
            NA_real_,
            13
        )

        names(out) <- c(
            "PCC",
            "Spearman",
            "MAE",
            "RMSE",
            "R2",
            "CCC",
            "ICC",
            "Bias",
            "MedAE",
            "Calibration_slope",
            "Calibration_intercept",
            "Age_difference",
            "Age_difference_SD"
        )

        return(out)
    }


    true <- df$true_age

    pred <- df$pred_age


    ########################################################
    # PCC
    ########################################################

    PCC <- suppressWarnings(
        cor(
            true,
            pred,
            method = "pearson",
            use = "complete.obs"
        )
    )


    ########################################################
    # Spearman
    ########################################################

    Spearman <- suppressWarnings(
        cor(
            true,
            pred,
            method = "spearman",
            use = "complete.obs"
        )
    )


    ########################################################
    # MAE
    ########################################################

    MAE <- mean(
        abs(pred - true)
    )


    ########################################################
    # RMSE
    ########################################################

    RMSE <- sqrt(
        mean(
            (pred - true)^2
        )
    )


    ########################################################
    # R2
    ########################################################

    denominator <- sum(
        (true - mean(true))^2
    )

    if (denominator == 0) {

        R2 <- NA_real_

    } else {

        R2 <- 1 -
            sum(
                (pred - true)^2
            ) /
            denominator
    }


    ########################################################
    # Bias
    ########################################################

    Bias <- mean(
        pred - true
    )


    ########################################################
    # Median absolute error
    ########################################################

    MedAE <- median(
        abs(pred - true)
    )


    ########################################################
    # CCC
    ########################################################

    CCC <- tryCatch(

        DescTools::CCC(
            pred,
            true
        )$rho.c$est,

        error = function(e) {
            NA_real_
        }
    )


    ########################################################
    # ICC
    ########################################################

    ICC <- tryCatch(

        irr::icc(
            data.frame(
                pred = pred,
                true = true
            ),
            model = "twoway",
            type = "agreement",
            unit = "single"
        )$value,

        error = function(e) {
            NA_real_
        }
    )


    ########################################################
    # Calibration
    ########################################################

    cal_model <- tryCatch(

        lm(
            pred ~ true
        ),

        error = function(e) {
            NULL
        }
    )


    if (is.null(cal_model)) {

        Calibration_slope <- NA_real_

        Calibration_intercept <- NA_real_

    } else {

        coefs <- coef(cal_model)

        Calibration_intercept <- unname(
            coefs[1]
        )

        Calibration_slope <- unname(
            coefs[2]
        )
    }


    ########################################################
    # Age difference
    ########################################################

    age_difference <- pred - true

    Age_difference <- mean(
        age_difference
    )

    Age_difference_SD <- sd(
        age_difference
    )


    ########################################################
    # Return
    ########################################################

    c(
        PCC = PCC,
        Spearman = Spearman,
        MAE = MAE,
        RMSE = RMSE,
        R2 = R2,
        CCC = CCC,
        ICC = ICC,
        Bias = Bias,
        MedAE = MedAE,
        Calibration_slope = Calibration_slope,
        Calibration_intercept = Calibration_intercept,
        Age_difference = Age_difference,
        Age_difference_SD = Age_difference_SD
    )
}



############################################################
# 4. Donor-level bootstrap
#
# IMPORTANT:
# Bootstrap sampling is performed on DONORS.
############################################################

bootstrap_metrics <- function(
    donor_df,
    n_boot = 1000,
    fraction = 0.8
) {

    donor_df <- donor_df %>%
        filter(
            !is.na(true_age),
            !is.na(pred_age)
        )


    n_donor <- nrow(
        donor_df
    )


    metric_names <- names(
        calculate_metrics(
            donor_df
        )
    )


    if (n_donor < 3) {

        return(
            tibble(
                Metric = metric_names,
                Mean = NA_real_,
                Lower95 = NA_real_,
                Upper95 = NA_real_
            )
        )
    }


    ########################################################
    # Number of donors sampled in each bootstrap
    ########################################################

    sample_size <- max(
        3,
        floor(
            n_donor * fraction
        )
    )

    sample_size <- min(
        sample_size,
        n_donor
    )


    ########################################################
    # Bootstrap matrix
    ########################################################

    bootstrap_results <- matrix(
        NA_real_,
        nrow = n_boot,
        ncol = length(metric_names)
    )

    colnames(
        bootstrap_results
    ) <- metric_names


    ########################################################
    # Bootstrap
    ########################################################

    for (i in seq_len(n_boot)) {

        idx <- sample(
            seq_len(n_donor),
            size = sample_size,
            replace = TRUE
        )


        bootstrap_df <- donor_df[
            idx,
            ,
            drop = FALSE
        ]


        bootstrap_results[i, ] <-
            calculate_metrics(
                bootstrap_df
            )
    }


    ########################################################
    # Summary
    ########################################################

    result <- tibble(

        Metric = metric_names,

        Mean = apply(
            bootstrap_results,
            2,
            mean,
            na.rm = TRUE
        ),

        Lower95 = apply(
            bootstrap_results,
            2,
            quantile,
            probs = 0.025,
            na.rm = TRUE
        ),

        Upper95 = apply(
            bootstrap_results,
            2,
            quantile,
            probs = 0.975,
            na.rm = TRUE
        )
    )


    return(result)
}



############################################################
# 5. Read metadata
############################################################

metadata_files <- list.files(
    metadata_dir,
    pattern = "\\.csv$",
    full.names = TRUE
)


if (length(metadata_files) == 0) {

    stop(
        "No metadata CSV files found in: ",
        metadata_dir
    )
}


metadata_list <- list()


for (i in seq_along(metadata_files)) {

    cat(
        "Reading metadata:",
        metadata_files[i],
        "\n"
    )


    tmp <- read.csv(
        metadata_files[i],
        check.names = FALSE,
        stringsAsFactors = FALSE
    )


    ########################################################
    # Check required columns
    ########################################################

    required_metadata <- c(
        "cell",
        "cell_type",
        "condition",
        "donorID"
    )


    missing_columns <- setdiff(
        required_metadata,
        colnames(tmp)
    )


    if (length(missing_columns) > 0) {

        warning(
            "Skipping metadata file because columns are missing: ",
            basename(metadata_files[i]),
            "\nMissing: ",
            paste(
                missing_columns,
                collapse = ", "
            )
        )

        next
    }


    tmp <- tmp %>%
        select(
            cell,
            cell_type,
            condition,
            donorID
        ) %>%
        distinct()


    metadata_list <- append(
        metadata_list,
        list(tmp)
    )
}


if (length(metadata_list) == 0) {

    stop(
        "No valid metadata files were found."
    )
}


metadata <- bind_rows(
    metadata_list
) %>%
    distinct(
        cell,
        .keep_all = TRUE
    )


cat(
    "\nTotal metadata cells:",
    nrow(metadata),
    "\n"
)


############################################################
# 6. Retained cell types
############################################################

keep_cell_types <- c(
    "Excitatory neuron",
    "Inhibitory neuron",
    "Astrocyte",
    "OPC",
    "Oligodendrocyte",
    "Microglia"
)


metadata <- metadata %>%
    filter(
        cell_type %in% keep_cell_types
    )


cat(
    "Metadata cells after cell-type filtering:",
    nrow(metadata),
    "\n"
)


############################################################
# 7. Prediction files
############################################################

prediction_files <- list.files(
    input_dir,
    pattern = "\\.csv$",
    full.names = TRUE
)


############################################################
# Remove output file itself
############################################################

prediction_files <- prediction_files[
    basename(prediction_files) !=
        basename(output_file)
]


if (length(prediction_files) == 0) {

    stop(
        "No prediction CSV files found."
    )
}


cat(
    "\nPrediction files:",
    length(prediction_files),
    "\n\n"
)


############################################################
# 8. Remove previous output
############################################################

if (file.exists(output_file)) {

    file.remove(
        output_file
    )
}


############################################################
# 9. Main loop
############################################################

for (prediction_file in prediction_files) {


    cat(
        "\n==============================================\n"
    )

    cat(
        "Processing:",
        basename(prediction_file),
        "\n"
    )

    cat(
        "==============================================\n"
    )


    ########################################################
    # Read prediction
    ########################################################

    prediction_df <- tryCatch(

        read.csv(
            prediction_file,
            check.names = FALSE,
            stringsAsFactors = FALSE
        ),

        error = function(e) {

            warning(
                "Cannot read prediction file: ",
                basename(prediction_file),
                "\n",
                e$message
            )

            return(NULL)
        }
    )


    if (is.null(prediction_df)) {
        next
    }


    ########################################################
    # Check prediction columns
    ########################################################

    required_prediction <- c(
        "cell_id",
        "true_age",
        "pred_age"
    )


    missing_prediction <- setdiff(
        required_prediction,
        colnames(prediction_df)
    )


    if (length(missing_prediction) > 0) {

        warning(
            "Skipping file: ",
            basename(prediction_file),
            "\nMissing columns: ",
            paste(
                missing_prediction,
                collapse = ", "
            )
        )

        next
    }


    ########################################################
    # Keep required columns
    ########################################################

    prediction_df <- prediction_df %>%
        select(
            cell_id,
            true_age,
            pred_age
        )


    ########################################################
    # Ensure numeric
    ########################################################

    prediction_df <- prediction_df %>%
        mutate(
            true_age = suppressWarnings(
                as.numeric(true_age)
            ),
            pred_age = suppressWarnings(
                as.numeric(pred_age)
            )
        )


    ########################################################
    # Merge metadata
    ########################################################

    merged_df <- prediction_df %>%
        inner_join(
            metadata,
            by = c(
                "cell_id" = "cell"
            )
        )


    ########################################################
    # Remove missing values
    ########################################################

    merged_df <- merged_df %>%
        filter(
            !is.na(true_age),
            !is.na(pred_age),
            !is.na(cell_type),
            !is.na(condition),
            !is.na(donorID)
        )


    cat(
        "Matched cells:",
        nrow(merged_df),
        "\n"
    )


    if (nrow(merged_df) == 0) {

        warning(
            "No matched cells for: ",
            basename(prediction_file)
        )

        next
    }


    ########################################################
    # Check true age consistency within donor
    ########################################################

    age_check <- merged_df %>%
        group_by(
            donorID
        ) %>%
        summarise(
            n_true_age = n_distinct(
                true_age
            ),
            .groups = "drop"
        )


    inconsistent_donors <- age_check %>%
        filter(
            n_true_age > 1
        )


    if (nrow(inconsistent_donors) > 0) {

        warning(
            "Some donors have multiple true_age values in: ",
            basename(prediction_file),
            "\nUsing mean true_age within donor."
        )
    }


    ########################################################
    # DONOR-LEVEL AGGREGATION
    #
    # One row = one donor x cell_type x condition
    ########################################################

    donor_df <- merged_df %>%
        group_by(
            cell_type,
            condition,
            donorID
        ) %>%
        summarise(

            true_age = mean(
                true_age,
                na.rm = TRUE
            ),

            pred_age = mean(
                pred_age,
                na.rm = TRUE
            ),

            N_cell = n(),

            .groups = "drop"
        )


    ########################################################
    # Summary
    ########################################################

    cat(
        "Donor x cell type observations:",
        nrow(donor_df),
        "\n"
    )

    cat(
        "Unique donors:",
        n_distinct(
            donor_df$donorID
        ),
        "\n"
    )


    ########################################################
    # Calculate metrics for each
    # cell_type x condition
    ########################################################

    group_list <- donor_df %>%
        group_by(
            cell_type,
            condition
        ) %>%
        group_split()


    result_list <- list()


    ########################################################
    # Loop groups
    ########################################################

    for (g in group_list) {


        if (nrow(g) < 3) {
            next
        }


        ####################################################
        # Group information
        ####################################################

        group_cell_type <- unique(
            g$cell_type
        )[1]


        group_condition <- unique(
            g$condition
        )[1]


        n_donor <- nrow(
            g
        )


        n_cell <- sum(
            g$N_cell
        )


        ####################################################
        # Donor-level metrics
        ####################################################

        metric_result <- bootstrap_metrics(
            donor_df = g %>%
                select(
                    true_age,
                    pred_age
                ),
            n_boot = n_boot,
            fraction = sample_fraction
        )


        ####################################################
        # Add metadata
        ####################################################

        metric_result <- metric_result %>%
            mutate(

                file = basename(
                    prediction_file
                ),

                cell_type = group_cell_type,

                condition = group_condition,

                N_donor = n_donor,

                N_cell = n_cell

            ) %>%
            select(
                file,
                cell_type,
                condition,
                N_donor,
                N_cell,
                Metric,
                Mean,
                Lower95,
                Upper95
            )


        ####################################################
        # Store
        ####################################################

        result_list <- append(
            result_list,
            list(metric_result)
        )
    }


    ########################################################
    # Combine results
    ########################################################

    if (length(result_list) == 0) {

        warning(
            "No valid donor-level groups for: ",
            basename(prediction_file)
        )

        next
    }


    final_result <- bind_rows(
        result_list
    )


    ########################################################
    # Incremental save
    ########################################################

    if (!file.exists(output_file)) {

        write.csv(
            final_result,
            output_file,
            row.names = FALSE
        )

    } else {

        write.table(
            final_result,
            output_file,
            sep = ",",
            row.names = FALSE,
            col.names = FALSE,
            append = TRUE
        )
    }


    cat(
        "Finished:",
        basename(prediction_file),
        "\n"
    )
}


############################################################
# 10. Finished
############################################################

cat(
    "\n====================================================\n"
)

cat(
    "Donor-level age prediction analysis completed!\n"
)

cat(
    "Output:",
    output_file,
    "\n"
)

cat(
    "====================================================\n"
)
