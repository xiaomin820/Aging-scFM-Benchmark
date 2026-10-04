from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


EMBEDDING_PREFIXES = ("emb_", "embedding_", "dim_")


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def parse_hidden_dims(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        dimensions = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        dimensions = [int(item) for item in value]
    if not dimensions or any(item < 1 for item in dimensions):
        raise ValueError("hidden dimensions must be positive integers")
    return dimensions


def _frame_embedding_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        str(column)
        for column in frame.columns
        if str(column).startswith(EMBEDDING_PREFIXES)
    ]
    if not columns:
        raise ValueError(
            "No embedding columns found; expected prefixes emb_, embedding_ or dim_"
        )
    return columns


def load_embedding(path: Path) -> tuple[np.ndarray, pd.DataFrame | None, np.ndarray | None]:
    suffix = path.suffix.lower()
    embedded_metadata: pd.DataFrame | None = None
    cell_ids: np.ndarray | None = None
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "X" not in payload.files:
                raise KeyError(f"{path} does not contain an X array")
            values = np.asarray(payload["X"], dtype=np.float32)
            if "cell_id" in payload.files:
                cell_ids = np.asarray(payload["cell_id"], dtype=str)
    elif suffix == ".npy":
        values = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    elif suffix in {".parquet", ".csv"}:
        frame = pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)
        embedding_columns = _frame_embedding_columns(frame)
        values = frame[embedding_columns].to_numpy(dtype=np.float32)
        embedded_metadata = frame.drop(columns=embedding_columns)
        if "cell_id" in embedded_metadata.columns:
            cell_ids = embedded_metadata["cell_id"].astype(str).to_numpy()
    else:
        raise ValueError(f"Unsupported embedding format: {path}")
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError(f"Invalid embedding shape {values.shape}: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Embedding contains NaN or Inf: {path}")
    if cell_ids is not None and len(cell_ids) != len(values):
        raise ValueError(f"cell_id length does not match embedding rows: {path}")
    return values, embedded_metadata, cell_ids


def load_pair(
    embedding_path: Path, metadata_path: Path | None
) -> tuple[np.ndarray, pd.DataFrame]:
    values, embedded_metadata, embedded_ids = load_embedding(embedding_path)
    if metadata_path is not None:
        metadata = pd.read_csv(metadata_path)
    elif embedded_metadata is not None:
        metadata = embedded_metadata
    else:
        raise ValueError(f"Metadata is required for {embedding_path}")
    if len(metadata) != len(values):
        raise ValueError(
            f"Row mismatch: {embedding_path} has {len(values)} rows, "
            f"but metadata has {len(metadata)}"
        )
    if embedded_ids is not None and "cell_id" in metadata.columns:
        metadata_ids = metadata["cell_id"].astype(str).to_numpy()
        if not np.array_equal(embedded_ids, metadata_ids):
            raise ValueError(f"cell_id order mismatch: {embedding_path}")
    return values, metadata.reset_index(drop=True)


def split_indices(
    labels: np.ndarray, task: str, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie between 0 and 1")
    rng = np.random.default_rng(seed)
    if task == "classification":
        train_parts: list[np.ndarray] = []
        validation_parts: list[np.ndarray] = []
        for label in (0.0, 1.0):
            indices = np.flatnonzero(labels == label)
            if len(indices) < 2:
                raise ValueError(f"Class {int(label)} has fewer than two observations")
            indices = rng.permutation(indices)
            count = max(1, int(round(len(indices) * validation_fraction)))
            count = min(count, len(indices) - 1)
            validation_parts.append(indices[:count])
            train_parts.append(indices[count:])
        train = rng.permutation(np.concatenate(train_parts))
        validation = rng.permutation(np.concatenate(validation_parts))
        return train, validation
    indices = rng.permutation(len(labels))
    count = max(1, int(round(len(indices) * validation_fraction)))
    count = min(count, len(indices) - 1)
    return indices[count:], indices[:count]


def split_grouped_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    strata: np.ndarray,
    task: str,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole donors while approximating cell-type and class composition."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie between 0 and 1")
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    strata = np.asarray(strata)
    if not (len(labels) == len(groups) == len(strata)):
        raise ValueError("labels, groups and strata must have the same length")
    if np.any(pd.isna(groups)) or np.any(groups.astype(str) == ""):
        raise ValueError("group identifiers must be non-missing")
    if np.any(pd.isna(strata)):
        raise ValueError("stratification values must be non-missing")
    groups = groups.astype(str)
    strata = strata.astype(str)

    unique_groups, group_codes = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)
    if n_groups < 2:
        raise ValueError("At least two donor groups are required")
    n_validation = int(round(n_groups * validation_fraction))
    n_validation = min(max(1, n_validation), n_groups - 1)

    category = strata if task == "regression" else np.char.add(
        np.char.add(strata, "\x1f"), labels.astype(str)
    )
    _, category_codes = np.unique(category, return_inverse=True)
    n_categories = int(category_codes.max()) + 1
    counts = np.zeros((n_groups, n_categories), dtype=np.float64)
    np.add.at(counts, (group_codes, category_codes), 1.0)
    target = counts.sum(axis=0) * validation_fraction
    scale = np.maximum(target, 1.0)

    rng = np.random.default_rng(seed)
    best_selection: np.ndarray | None = None
    best_score = np.inf
    attempts = max(512, min(4096, n_groups * 32))
    for _ in range(attempts):
        candidate = rng.choice(n_groups, size=n_validation, replace=False)
        validation_mask = np.isin(group_codes, candidate)
        if task == "classification":
            if len(np.unique(labels[validation_mask])) < 2:
                continue
            if len(np.unique(labels[~validation_mask])) < 2:
                continue
        observed = counts[candidate].sum(axis=0)
        composition_error = np.mean(((observed - target) / scale) ** 2)
        size_error = (
            validation_mask.mean() - validation_fraction
        ) ** 2
        score = float(composition_error + size_error)
        if score < best_score:
            best_score = score
            best_selection = candidate

    if best_selection is None:
        raise ValueError(
            "Could not form donor-disjoint partitions containing both classes; "
            "check donor labels or increase the validation fraction"
        )
    validation_mask = np.isin(group_codes, best_selection)
    train = rng.permutation(np.flatnonzero(~validation_mask))
    validation = rng.permutation(np.flatnonzero(validation_mask))
    return train, validation


def fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - truth
    denominator = np.sum((truth - truth.mean()) ** 2)
    pcc = float(np.corrcoef(truth, prediction)[0, 1])
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "PCC": pcc,
        "R2": float(1 - np.sum(error**2) / denominator) if denominator > 0 else float("nan"),
    }


def classification_metrics(
    truth: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=np.int64)
    predicted = (np.asarray(probability) >= threshold).astype(np.int64)
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "F1": float(f1),
        "PPV": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float((tp + tn) / len(truth)),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "threshold": float(threshold),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(temporary, path)
