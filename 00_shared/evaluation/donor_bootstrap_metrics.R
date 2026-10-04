#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(irr)
  library(DescTools)
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

calculate_metrics <- function(frame) {
  frame <- frame %>% filter(!is.na(true_age), !is.na(pred_age))
  names_out <- c(
    "PCC", "Spearman", "MAE", "RMSE", "R2", "CCC", "ICC", "Bias", "MedAE",
    "Calibration_slope", "Calibration_intercept", "Age_difference", "Age_difference_SD"
  )
  if (nrow(frame) < 3) return(setNames(rep(NA_real_, length(names_out)), names_out))
  truth <- frame$true_age
  prediction <- frame$pred_age
  denominator <- sum((truth - mean(truth))^2)
  calibration <- tryCatch(lm(prediction ~ truth), error = function(e) NULL)
  c(
    PCC = suppressWarnings(cor(truth, prediction, method = "pearson")),
    Spearman = suppressWarnings(cor(truth, prediction, method = "spearman")),
    MAE = mean(abs(prediction - truth)),
    RMSE = sqrt(mean((prediction - truth)^2)),
    R2 = if (denominator == 0) NA_real_ else 1 - sum((prediction - truth)^2) / denominator,
    CCC = tryCatch(DescTools::CCC(prediction, truth)$rho.c$est, error = function(e) NA_real_),
    ICC = tryCatch(
      irr::icc(data.frame(prediction, truth), model = "twoway", type = "agreement", unit = "single")$value,
      error = function(e) NA_real_
    ),
    Bias = mean(prediction - truth),
    MedAE = median(abs(prediction - truth)),
    Calibration_slope = if (is.null(calibration)) NA_real_ else unname(coef(calibration)[2]),
    Calibration_intercept = if (is.null(calibration)) NA_real_ else unname(coef(calibration)[1]),
    Age_difference = mean(prediction - truth),
    Age_difference_SD = sd(prediction - truth)
  )
}

bootstrap_metrics <- function(frame, n_boot, fraction) {
  n_donor <- nrow(frame)
  if (n_donor < 3) return(tibble())
  sample_size <- min(n_donor, max(3, floor(n_donor * fraction)))
  values <- replicate(
    n_boot,
    calculate_metrics(frame[sample(seq_len(n_donor), sample_size, replace = TRUE), , drop = FALSE])
  )
  tibble(
    Metric = rownames(values),
    Mean = apply(values, 1, mean, na.rm = TRUE),
    Lower95 = apply(values, 1, quantile, probs = 0.025, na.rm = TRUE),
    Upper95 = apply(values, 1, quantile, probs = 0.975, na.rm = TRUE)
  )
}

args <- parse_cli()
prediction_path <- required(args, "predictions")
metadata_path <- required(args, "metadata")
output_path <- required(args, "output")
n_boot <- as.integer(if (is.null(args$n_boot)) 1000 else args$n_boot)
fraction <- as.numeric(if (is.null(args$fraction)) 0.8 else args$fraction)
seed <- as.integer(if (is.null(args$seed)) 123 else args$seed)
metadata_cell_column <- if (is.null(args$metadata_cell_column)) "cell" else args$metadata_cell_column
set.seed(seed)

prediction <- read_csv(prediction_path, show_col_types = FALSE) %>%
  select(cell_id, true_age, pred_age) %>%
  mutate(across(c(true_age, pred_age), as.numeric))
metadata <- read_csv(metadata_path, show_col_types = FALSE)
required_metadata <- c(metadata_cell_column, "donorID", "cell_type", "condition")
if (!all(required_metadata %in% colnames(metadata))) {
  stop("Missing metadata columns: ", paste(setdiff(required_metadata, colnames(metadata)), collapse = ", "))
}
join_key <- setNames(metadata_cell_column, "cell_id")
merged <- prediction %>%
  inner_join(metadata %>% select(all_of(required_metadata)), by = join_key) %>%
  filter(if_all(c(true_age, pred_age, donorID, cell_type, condition), ~ !is.na(.x)))

donor_frame <- merged %>%
  group_by(cell_type, condition, donorID) %>%
  summarise(
    true_age = mean(true_age),
    pred_age = mean(pred_age),
    N_cell = n(),
    .groups = "drop"
  )
write_csv(donor_frame, paste0(tools::file_path_sans_ext(output_path), "_donor_values.csv"))

results <- donor_frame %>%
  group_by(cell_type, condition) %>%
  group_split() %>%
  map_dfr(function(group) {
    if (nrow(group) < 3) return(tibble())
    bootstrap_metrics(group %>% select(true_age, pred_age), n_boot, fraction) %>%
      mutate(
        cell_type = group$cell_type[1],
        condition = group$condition[1],
        N_donor = nrow(group),
        N_cell = sum(group$N_cell),
        .before = 1
      )
  })
write_csv(results, output_path)
message("Saved ", normalizePath(output_path))
