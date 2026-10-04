# Task 2 - Age trajectories

Use the shared healthy expression data from `../../aging_benchmark_data/01_age_prediction/training/`. The parameterized runner requires a compatible Seurat RDS object, aligned cell embeddings and explicit root-cell IDs.

From the code root:

```bash
Rscript 02_age_trajectory/run_task2_batch.R --manifest=configs/task2_manifest.example.csv --output=outputs/task2
```

Replace the example manifest paths with actual inputs. Configure the age field with `--age_column=label` when appropriate. R dependencies include Seurat, monocle3 and readr. This reference workflow requires the original roots/settings to reproduce S7.
