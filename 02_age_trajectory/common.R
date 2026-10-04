task2_model_order <- c(
  "scGPT",
  "Geneformer",
  "scFoundation",
  "UCE",
  "scPRINT",
  "scLong",
  "TranscriptFormer",
  "CellFM",
  "scCello",
  "SCimilarity",
  "scVI",
  "2000HVG",
  "50PCA"
)

parse_cli <- function(tokens = commandArgs(trailingOnly = TRUE)) {
  arguments <- list()
  for (token in tokens) {
    if (!startsWith(token, "--") || !grepl("=", token, fixed = TRUE)) {
      stop("Arguments must use --name=value syntax: ", token)
    }
    pair <- strsplit(sub("^--", "", token), "=", fixed = TRUE)[[1]]
    name <- pair[[1]]
    if (!nzchar(name)) stop("Argument names cannot be empty")
    arguments[[name]] <- paste(pair[-1], collapse = "=")
  }
  arguments
}

required_arg <- function(arguments, name) {
  value <- arguments[[name]]
  if (is.null(value) || !nzchar(value)) stop("Missing --", name)
  value
}

integer_arg <- function(arguments, name, default, minimum = NULL) {
  raw_value <- if (is.null(arguments[[name]])) as.character(default) else arguments[[name]]
  value <- suppressWarnings(as.integer(raw_value))
  if (length(value) != 1 || is.na(value) || (!is.null(minimum) && value < minimum)) {
    stop("--", name, " must be an integer", if (!is.null(minimum)) paste0(" >= ", minimum))
  }
  value
}

numeric_metadata <- function(values, column_name) {
  if (is.factor(values)) values <- as.character(values)
  converted <- suppressWarnings(as.numeric(values))
  invalid <- !is.na(values) & is.na(converted)
  if (any(invalid)) stop("Metadata column '", column_name, "' contains non-numeric values")
  converted
}

calculate_task2_metrics <- function(age, pseudotime, donor = NULL) {
  age <- numeric_metadata(age, "age")
  pseudotime <- numeric_metadata(pseudotime, "pseudotime")
  complete <- is.finite(age) & is.finite(pseudotime)
  age <- age[complete]
  pseudotime <- pseudotime[complete]
  if (!is.null(donor)) donor <- donor[complete]

  if (length(age) < 3) stop("At least three cells with finite age and pseudotime are required")
  if (length(unique(age)) < 2) stop("Age has no variation among cells with finite pseudotime")
  if (length(unique(pseudotime)) < 2) stop("Pseudotime has no variation among cells with finite age")

  linear_fit <- stats::lm(age ~ pseudotime)
  data.frame(
    pearson_r = unname(stats::cor(age, pseudotime, method = "pearson")),
    spearman_rho = unname(stats::cor(age, pseudotime, method = "spearman")),
    linear_R2 = unname(summary(linear_fit)$r.squared),
    n_cells = length(age),
    n_donors = if (is.null(donor)) NA_integer_ else length(unique(donor[!is.na(donor)])),
    check.names = FALSE
  )
}

sort_task2_results <- function(results) {
  model_position <- match(results$model, task2_model_order)
  unknown <- is.na(model_position)
  model_position[unknown] <- length(task2_model_order) + seq_len(sum(unknown))
  results[order(model_position, results$cell_type), , drop = FALSE]
}

safe_output_name <- function(value) {
  cleaned <- gsub("[^A-Za-z0-9._-]+", "_", value)
  gsub("^_+|_+$", "", cleaned)
}
