# Task 4 - Senescence identification

`models/` contains embedding extraction, classifier training and prediction scripts. `map_mouse_genes_to_human_symbols.R` provides gene mapping for mouse inputs. Some model filenames retain `Task3`; within this directory they refer to senescence classification.

Data are in `../../aging_benchmark_data/04_senescence_identification/`. The independent validation inputs are GSE94980 (560 cells), bulk RNA-seq (151 samples) and GSE229553 (3,872 cells). The exact 12,505-cell GSE164241 evaluation and 11,280-cell development split are not included. Training-source files require the original subset and label mapping. DeepScence and SenCID run code are not included.
