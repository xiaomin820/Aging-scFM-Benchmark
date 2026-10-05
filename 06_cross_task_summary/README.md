# Cross-task summary

From the code root, run:

```bash
python 06_cross_task_summary/reproduce.py
```

The workflow reads `data/supplementary/`, `data/task5/` and `configs/reproduction.json`. It checks S15/S16, 33 AP estimates, 11 Union AUROC values, reference labels and 66,000 weighted bootstrap AP values. Outputs are written to `outputs/reproduced/`.

`bootstrap_ap.py` implements weighted AP resampling; `../00_shared/evaluation/cross_task_analysis.py` supplies table parsing and ranking. This is the five-task synthesis, not a sixth benchmark task.
