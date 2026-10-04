# Shared code

`preprocessing/` provides HVG/PCA and scVI extraction. `baselines/` contains model-specific baseline scripts. `evaluation/` provides donor metrics and the cross-task aggregation kernel.

`downstream/` is an optional standardized MLP workflow with donor-disjoint validation. It differs from the manuscript's model-specific settings and cell-level internal validation. Its defaults are in `config/benchmark_mlp.json`; its dependencies are listed in `requirements.txt` and `requirements-r.txt`.
