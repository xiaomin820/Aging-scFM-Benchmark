"""Table parsing and ranking kernel; see provenance/source_files.csv."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import re
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


FOUNDATION_MODELS = [
    "scGPT", "Geneformer", "scFoundation", "UCE", "scPRINT", "scLong",
    "TranscriptFormer", "CellFM", "scCello", "SCimilarity",
]
MODEL_ORDER = [
    *FOUNDATION_MODELS, "scAgeClock", "DeepScence", "SenCID", "scVI",
    "2000HVG", "50PCA", "SCENIC",
]
DISPLAY_NAMES = {"2000HVG": "2,000 HVGs", "50PCA": "50 PCs"}
TASK4_MODELS = [
    *FOUNDATION_MODELS, "DeepScence", "SenCID", "scVI", "2000HVG", "50PCA",
]
A4_ONE_THIRD = (8.27, 11.69 / 3)
MIN_FONT_SIZE = 8
BUBBLE_AREA_MIN = 300
BUBBLE_AREA_MAX = 500
TASKS = ["T1", "T2", "T3", "T4", "T5"]
LABELS = {
    "T1": "T1: Age prediction\nDonor PCC (higher)",
    "T2": "T2: Pseudotime\nPearson r (higher)",
    "T3": "T3: Disease-shift direction\nPositive delta AgeGap fraction (higher)",
    "T4": "T4: Senescence\nF1 (higher)",
    "T5": "T5: GRN edge ranking\nAUPRC (higher)",
}
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
ESTIMATE = re.compile(rf"^\s*({NUMBER})(?:\s*\(\s*({NUMBER})\s*-\s*({NUMBER})\s*\))?\s*$")


def parse_estimate(value: object) -> float:
    if pd.isna(value) or str(value).strip().lower() in {"", "na", "nan", "n/a"}:
        return np.nan
    match = ESTIMATE.fullmatch(str(value).replace("\u2212", "-"))
    if match is None:
        raise ValueError(f"Cannot parse estimate: {value!r}")
    estimate = float(match.group(1))
    if not np.isfinite(estimate):
        raise ValueError(f"Non-finite estimate: {value!r}")
    if match.group(2) is not None:
        lower, upper = float(match.group(2)), float(match.group(3))
        if not np.isfinite([lower, upper]).all() or lower > upper:
            raise ValueError(f"Invalid confidence interval: {value!r}")
    return estimate


def recover_task4_model_ids(table: pd.DataFrame, source_name: str, issues: list) -> pd.DataFrame:
    table = table.copy()
    for context, indices in table.groupby(["OOD", "Dataset"], sort=False).groups.items():
        context_rows = table.loc[indices]
        blank_indices = context_rows.index[context_rows.model.isna() | context_rows.model.fillna("").str.strip().eq("")]
        missing_models = set(TASK4_MODELS) - set(context_rows.model.dropna().str.strip())
        if len(blank_indices) == 1 and len(missing_models) == 1:
            recovered_model = missing_models.pop()
            table.loc[blank_indices[0], "model"] = recovered_model
            issues.append({
                "source": source_name, "csv_line": int(blank_indices[0]) + 2,
                "issue": f"Recovered blank model identifier as {recovered_model}",
                "detail": f"Context {context[0]} | {context[1]} had one blank row and exactly one missing expected Task 4 model",
            })
    return table


def load_tables(root: Path) -> tuple[dict, list, list]:
    tables, manifest, issues = {}, [], []
    for number in [3, 4, 7, 8, 10, 11, 17]:
        matches = sorted({
            *root.glob(f"Supplementary Table S{number}. *.csv"),
            *root.glob(f"Supplementary Table {number}. *.csv"),
        })
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one S{number} CSV; found {len(matches)}")
        source = matches[0]
        table = pd.read_csv(source)
        model_column = "Model" if "Model" in table else "model"
        table = table.rename(columns={model_column: "model"})
        if number == 10:
            table = recover_task4_model_ids(table, source.name, issues)
        missing = table.model.isna() | table.model.fillna("").str.strip().eq("")
        for row_index, row in table[missing].iterrows():
            issues.append({
                "source": source.name, "csv_line": int(row_index) + 2,
                "issue": "Missing model identifier; excluded without imputation",
                "detail": json.dumps(row.where(row.notna(), None).to_dict(), ensure_ascii=False),
            })
        table = table.loc[~missing].copy()
        table["model"] = table.model.str.strip()
        tables[number] = table
        manifest.append({"table": f"S{number}", "file": source.name,
                         "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "rows_read": len(table) + int(missing.sum()), "rows_kept": len(table)})
    return tables, manifest, issues


def metric_records(table: pd.DataFrame, task: str, metric: str, contexts: list[str],
                   higher: bool, source: str, absolute: bool = False) -> pd.DataFrame:
    frame = table[["model", *contexts, metric]].copy()
    frame["context"] = frame[contexts].astype(str).agg(" | ".join, axis=1) if contexts else "matched_input"
    if frame.duplicated(["model", "context"]).any():
        raise ValueError(f"Duplicate model/context observations in {source}: {metric}")
    frame["value"] = frame[metric].map(parse_estimate)
    if absolute:
        frame["value"] = frame.value.abs()
    frame["task"], frame["metric"], frame["higher_is_better"], frame["source"] = task, metric, higher, source
    return frame[["task", "model", "context", "metric", "value", "higher_is_better", "source"]]


def build_records(tables: dict, replacements: dict | None = None) -> pd.DataFrame:
    tables = dict(tables)
    disease = tables[8].copy()
    signed_gap = disease["Delta AgeGap vs Control (years; 95% CI)"].map(parse_estimate)
    disease["positive_delta_age_gap"] = (signed_gap > 0).astype(float).where(signed_gap.notna())
    tables[8] = disease
    specifications = {
        "T1": (4, "PCC", ["dataset"], True, False),
        "T2": (7, "pearson_r", ["cell_type"], True, False),
        "T3": (8, "positive_delta_age_gap", ["Dataset", "Condition"], True, False),
        "T4": (10, "F1", ["OOD", "Dataset"], True, False),
        "T5": (17, "AUPRC", ["evaluation_context"], True, False),
    }
    specifications.update(replacements or {})
    return pd.concat([
        metric_records(tables[number], task, metric, contexts, higher, f"S{number}", absolute)
        for task, (number, metric, contexts, higher, absolute) in specifications.items()
    ], ignore_index=True)


def coverage_table(records: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for task, task_frame in records.groupby("task", sort=False):
        expected = task_frame.context.nunique()
        counts = task_frame[np.isfinite(task_frame.value)].groupby("model").context.nunique()
        for model in models:
            observed = int(counts.get(model, 0))
            rows.append({"model": model, "task": task, "observed": observed,
                         "expected": expected, "complete": observed == expected})
    return pd.DataFrame(rows)


def aggregate(records: pd.DataFrame, models: list[str], require_all: bool = True,
              aggregation: str = "metric_mean") -> tuple[pd.DataFrame, pd.DataFrame]:
    if aggregation not in {"metric_mean", "context_rank"}:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    coverage = coverage_table(records, models)
    scores = pd.DataFrame(index=pd.Index(models, name="model"), columns=TASKS, dtype=float)
    elementary = []
    for task in TASKS:
        complete = coverage[(coverage.task == task) & coverage.complete].model.tolist()
        if require_all and set(complete) != set(models):
            raise ValueError(f"Incomplete coverage in {task}: {sorted(set(models) - set(complete))}")
        if len(complete) < 2:
            continue
        frame = records[(records.task == task) & records.model.isin(complete)].copy()
        if frame.higher_is_better.nunique() != 1:
            raise ValueError(f"Inconsistent metric direction in {task}")
        frame["context_rank"] = frame.groupby("context").value.rank(
            ascending=not bool(frame.higher_is_better.iloc[0]), method="average")
        frame["n_competitors"] = len(complete)
        frame["score"] = (len(complete) - frame.context_rank) / (len(complete) - 1)
        means = frame.groupby("model").value.agg(
            lambda values: sum(Fraction(str(value)) for value in values) / len(values))
        mean_metric_ranks = means.rank(ascending=not bool(frame.higher_is_better.iloc[0]), method="average")
        frame["task_metric_mean"] = frame.model.map(means.map(float))
        frame["task_rank_by_metric_mean"] = frame.model.map(mean_metric_ranks)
        if aggregation == "metric_mean":
            task_scores = (len(complete) - mean_metric_ranks) / (len(complete) - 1)
        else:
            mean_ranks = frame.groupby("model").context_rank.mean()
            task_scores = (len(complete) - mean_ranks) / (len(complete) - 1)
        scores.loc[complete, task] = task_scores.reindex(complete)
        frame["task_score"] = frame.model.map(task_scores)
        frame["aggregation"] = aggregation
        elementary.append(frame)
    return scores, pd.concat(elementary, ignore_index=True) if elementary else pd.DataFrame()

