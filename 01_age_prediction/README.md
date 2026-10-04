# Task 1 - Age prediction

`models/` contains model-specific embedding extraction, training and prediction scripts. Preserve each model's internal directory structure and configure its input, resource and checkpoint paths before execution.

Training data: `../../aging_benchmark_data/01_age_prediction/training/`. Independent validation: `../../aging_benchmark_data/01_age_prediction/independent_validation/`.

`evaluate_condition_celltype_donor.R` calculates age-prediction metrics. Current tables are in `../data/supplementary/` (S3-S6). CellFM training and scAgeClock run code are not included.
