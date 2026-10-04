#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(readr))

command <- commandArgs(trailingOnly = FALSE)
this_file <- sub("^--file=", "", command[grepl("^--file=", command)][[1]])
trajectory_dir <- dirname(normalizePath(this_file))
source(file.path(trajectory_dir, "common.R"))

arguments <- parse_cli()
manifest_path <- required_arg(arguments, "manifest")
output_dir <- required_arg(arguments, "output")
age_column <- if (is.null(arguments$age_column)) "age" else arguments$age_column
cell_type_column <- if (is.null(arguments$cell_type_column)) "cell_type" else arguments$cell_type_column
donor_column <- if (is.null(arguments$donor_column)) "donorID" else arguments$donor_column
minimum_cells <- integer_arg(arguments, "min_cells", 1000, minimum = 3)
seed <- integer_arg(arguments, "seed", 42)

if (!file.exists(manifest_path)) stop("Manifest does not exist: ", manifest_path)
manifest_path <- normalizePath(manifest_path, mustWork = TRUE)
manifest_dir <- dirname(manifest_path)
manifest <- read_csv(manifest_path, show_col_types = FALSE, col_types = cols(.default = col_character()))
required_columns <- c("model", "cell_type", "seurat", "embedding", "root_cells")
missing_columns <- setdiff(required_columns, colnames(manifest))
if (length(missing_columns)) stop("Manifest is missing columns: ", paste(missing_columns, collapse = ", "))
if (!nrow(manifest)) stop("Manifest contains no runs")
if (anyNA(manifest[required_columns]) || any(!nzchar(as.matrix(manifest[required_columns])))) {
  stop("Required manifest fields cannot be missing or empty")
}
run_keys <- paste(manifest$model, manifest$cell_type, sep = "\r")
if (anyDuplicated(run_keys)) stop("Manifest model and cell_type combinations must be unique")

resolve_path <- function(path) {
  if (!grepl("^([A-Za-z]:[/\\\\]|/)", path)) path <- file.path(manifest_dir, path)
  normalizePath(path, mustWork = FALSE)
}
for (column in c("seurat", "embedding", "root_cells")) {
  manifest[[column]] <- vapply(manifest[[column]], resolve_path, character(1))
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
runs_dir <- file.path(output_dir, "runs")
logs_dir <- file.path(output_dir, "logs")
dir.create(runs_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(logs_dir, recursive = TRUE, showWarnings = FALSE)

rscript <- file.path(R.home("bin"), if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
runner <- file.path(trajectory_dir, "run_monocle3.R")
completed <- list()
failures <- list()

for (index in seq_len(nrow(manifest))) {
  run <- manifest[index, , drop = FALSE]
  run_name <- sprintf(
    "%03d_%s__%s",
    index,
    safe_output_name(run$model[[1]]),
    safe_output_name(run$cell_type[[1]])
  )
  run_output <- file.path(runs_dir, run_name)
  log_path <- file.path(logs_dir, paste0(run_name, ".log"))
  message("[", index, "/", nrow(manifest), "] ", run$model[[1]], " / ", run$cell_type[[1]])
  command_arguments <- c(
    runner,
    paste0("--model=", run$model[[1]]),
    paste0("--cell_type=", run$cell_type[[1]]),
    paste0("--seurat=", run$seurat[[1]]),
    paste0("--embedding=", run$embedding[[1]]),
    paste0("--root_cells=", run$root_cells[[1]]),
    paste0("--output=", run_output),
    paste0("--age_column=", age_column),
    paste0("--cell_type_column=", cell_type_column),
    paste0("--donor_column=", donor_column),
    paste0("--min_cells=", minimum_cells),
    paste0("--seed=", seed)
  )
  status <- system2(
    rscript,
    args = vapply(command_arguments, shQuote, character(1)),
    stdout = log_path,
    stderr = log_path
  )
  metrics_path <- file.path(run_output, "task2_metrics.csv")
  if (identical(status, 0L) && file.exists(metrics_path)) {
    completed[[length(completed) + 1L]] <- read_csv(metrics_path, show_col_types = FALSE)
  } else {
    failures[[length(failures) + 1L]] <- data.frame(
      model = run$model[[1]],
      cell_type = run$cell_type[[1]],
      exit_status = status,
      log = normalizePath(log_path, mustWork = FALSE)
    )
  }
}

if (length(completed)) {
  detailed <- sort_task2_results(do.call(rbind, completed))
  publication <- detailed[c("model", "cell_type", "pearson_r", "spearman_rho", "linear_R2")]
  write_csv(publication, file.path(output_dir, "task2_age_pseudotime_metrics.csv"))
  write_csv(detailed, file.path(output_dir, "task2_age_pseudotime_metrics_detailed.csv"))
}
if (length(failures)) {
  failure_table <- do.call(rbind, failures)
  write_csv(failure_table, file.path(output_dir, "task2_failed_runs.csv"))
  stop(length(failures), " of ", nrow(manifest), " Task 2 runs failed; see task2_failed_runs.csv")
}

message("Completed ", nrow(manifest), " Task 2 runs; summary saved to ", normalizePath(output_dir))
