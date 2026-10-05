# Task 5 - Regulatory networks

`models/` contains ten gene-embedding extraction scripts. Configure pretrained resources and gene lists for each model. Audited input embeddings and regulatory reference data are in `../../aging_benchmark_data/05_regulatory_network/`.

The primary endpoint is pre-motif Union AP over 163,280 directed non-self pairs. Run `python 06_cross_task_summary/reproduce.py` from the code root as documented there to recalculate the supplied edge statistics. Original GRNBoost2 adjacency and the full motif-inference pipeline are not included.
