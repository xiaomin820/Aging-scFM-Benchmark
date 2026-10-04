#!/usr/bin/env python3
"""Predict Task3 senescence labels for scLONG independent embeddings."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DROPOUT = 0.2
EXPECTED_INPUT_DIM = 200
EXPECTED_TAG = "Task3_scLONG_Global_MLP"
EXPECTED_INDEPENDENT_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--only", help="Exact dataset id, 'independent', or 'all'.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--expected-independent-count",
        type=int,
        default=EXPECTED_INDEPENDENT_COUNT,
        help="Expected number of independent embedding datasets.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.expected_independent_count < 1:
        parser.error("--expected-independent-count must be positive")
    return args


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.9g")
    os.replace(temporary, path)


class SenescenceMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)


def binary_metrics(
    truth: np.ndarray, predicted: np.ndarray, threshold: float
) -> dict[str, float | int | None]:
    y = np.asarray(truth, dtype=np.int64)
    p = np.asarray(predicted, dtype=np.int64)
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tp = int(np.sum((y == 1) & (p == 1)))
    denominator = 2 * tp + fp + fn
    positive_count = tp + fn
    negative_count = tn + fp
    f1 = float(2 * tp / denominator) if denominator > 0 else 0.0
    precision = float(tp / (tp + fp)) if tp + fp > 0 else None
    sensitivity = float(tp / positive_count) if positive_count > 0 else None
    specificity = float(tn / negative_count) if negative_count > 0 else None
    balanced_accuracy = (
        float((sensitivity + specificity) / 2)
        if sensitivity is not None and specificity is not None
        else None
    )
    mcc_denominator = float(
        np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    return {
        "Accuracy": float((tp + tn) / len(y)),
        "F1": f1,
        "F1_publication": f1 if positive_count > 0 else None,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_positive_rate": (
            float(fp / negative_count) if negative_count > 0 else None
        ),
        "balanced_accuracy": balanced_accuracy,
        "MCC": (
            float((tp * tn - fp * fn) / mcc_denominator)
            if mcc_denominator > 0
            else None
        ),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "threshold": threshold,
    }


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    required = {
        "tag",
        "strategy",
        "seed",
        "input_dim",
        "model_state_dict",
        "embedding_mean",
        "embedding_std",
        "best_epoch",
        "decision_threshold",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint missing keys: {sorted(missing)}")
    if checkpoint["tag"] != EXPECTED_TAG or checkpoint["strategy"] != "Global":
        raise ValueError("Unexpected checkpoint tag or strategy.")
    if int(checkpoint["seed"]) != 42:
        raise ValueError(f"Unexpected seed: {checkpoint['seed']}")
    if int(checkpoint["input_dim"]) != EXPECTED_INPUT_DIM:
        raise ValueError(f"Unexpected input_dim: {checkpoint['input_dim']}")
    threshold = float(checkpoint["decision_threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Invalid decision threshold: {threshold}")
    mean = torch.as_tensor(checkpoint["embedding_mean"]).cpu().numpy().reshape(-1)
    std = torch.as_tensor(checkpoint["embedding_std"]).cpu().numpy().reshape(-1)
    if mean.shape != (EXPECTED_INPUT_DIM,) or std.shape != (EXPECTED_INPUT_DIM,):
        raise ValueError(f"Invalid scaler shapes: {mean.shape}, {std.shape}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Checkpoint scaler statistics are invalid.")
    model = SenescenceMLP(EXPECTED_INPUT_DIM)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    checkpoint["_model"] = model
    checkpoint["_mean_numpy"] = mean.astype(np.float32)
    checkpoint["_std_numpy"] = std.astype(np.float32)
    return checkpoint


def dataset_id_from_npz(path: Path) -> str:
    suffix = "_scLONG_cell_embeddings.npz"
    if not path.name.endswith(suffix):
        raise ValueError(path.name)
    return path.name[: -len(suffix)]


def discover_independent(embedding_dir: Path) -> list[Path]:
    paths = sorted(
        path
        for path in embedding_dir.glob(
            "Independent.Test_*_scLONG_cell_embeddings.npz"
        )
        if path.is_file()
    )
    ids = [dataset_id_from_npz(path) for path in paths]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate independent dataset ids.")
    return paths


def select_paths(paths: list[Path], selector: str) -> list[Path]:
    if selector.lower() in {"all", "independent"}:
        return paths
    exact = [path for path in paths if dataset_id_from_npz(path) == selector]
    if exact:
        return exact
    raise ValueError(f"Unknown --only value: {selector!r}. Run with --list.")


def classify_dataset(name: str) -> dict[str, str]:
    return {
        "species": "mouse" if "_mouse_" in name else "human",
        "observation_type": "bulk_sample" if "_bulk_" in name else "single_cell",
    }


def predict_batches(
    model: nn.Module, values: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device))
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def predict_one(
    npz_path: Path,
    embedding_dir: Path,
    prediction_dir: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    device: torch.device,
    batch_size: int,
    overwrite: bool,
) -> None:
    name = dataset_id_from_npz(npz_path)
    metadata_path = (
        embedding_dir / f"{name}_scLONG_cell_embeddings_metadata.csv"
    )
    manifest_path = embedding_dir / f"{name}_scLONG_embedding_manifest.json"
    prediction_path = (
        prediction_dir
        / f"{name}_scLONG_Global_MLP_senescence_predictions.csv"
    )
    metrics_path = prediction_dir / f"{name}_scLONG_Global_MLP_metrics.json"
    if prediction_path.exists() and metrics_path.exists() and not overwrite:
        print(f"SKIP_COMPLETE: {name}")
        return
    if (prediction_path.exists() or metrics_path.exists()) and not overwrite:
        existing = [
            str(path) for path in (prediction_path, metrics_path) if path.exists()
        ]
        raise FileExistsError(
            f"Partial prediction output exists; inspect or use --overwrite: {existing}"
        )
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing metadata or manifest for {name}")

    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != {"X", "cell_id"}:
            raise ValueError(f"Unexpected NPZ keys: {payload.files}")
        embeddings = np.asarray(payload["X"], dtype=np.float32)
        npz_cell_ids = np.asarray(payload["cell_id"], dtype=str)
    metadata = pd.read_csv(metadata_path, dtype={"cell_id": str})
    if embeddings.ndim != 2 or embeddings.shape[1] != EXPECTED_INPUT_DIM:
        raise ValueError(f"Invalid embedding shape for {name}: {embeddings.shape}")
    if len(metadata) != len(embeddings) or not np.isfinite(embeddings).all():
        raise ValueError(f"Invalid metadata or embedding values for {name}")
    cell_ids = metadata["cell_id"].astype(str).to_numpy()
    if not np.array_equal(npz_cell_ids, cell_ids):
        raise ValueError(f"cell_id order mismatch for {name}")
    if pd.Index(cell_ids).has_duplicates:
        raise ValueError(f"Duplicate cell_id values for {name}")
    labels = pd.to_numeric(metadata["label"], errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"Invalid binary labels for {name}")
    truth = labels.to_numpy(dtype=np.int8)

    standardized = (
        (embeddings - checkpoint["_mean_numpy"]) / checkpoint["_std_numpy"]
    ).astype(np.float32)
    if not np.isfinite(standardized).all():
        raise ValueError(f"Non-finite standardized embeddings for {name}")
    probability = predict_batches(
        checkpoint["_model"], standardized, device, batch_size
    )
    threshold = float(checkpoint["decision_threshold"])
    predicted = (probability >= threshold).astype(np.int8)
    metrics = binary_metrics(truth, predicted, threshold)
    domain = classify_dataset(name)

    frame = metadata.copy()
    frame = frame.rename(columns={"label": "true_label"})
    frame["true_label"] = truth
    frame["senescent_score"] = probability
    frame["predicted_label"] = predicted
    frame["predicted_class"] = np.where(predicted == 1, "senescent", "normal")
    frame["correct"] = predicted == truth
    frame["species"] = domain["species"]
    frame["observation_type"] = domain["observation_type"]
    payload = {
        "metrics_version": "1.0",
        "created_at_utc": now_utc(),
        "dataset_id": name,
        **domain,
        "n_observations": int(len(frame)),
        "label_0_count": int((truth == 0).sum()),
        "label_1_count": int((truth == 1).sum()),
        "metrics_status": (
            "two_class_metrics"
            if len(np.unique(truth)) == 2
            else "single_class_descriptive_metrics"
        ),
        "metrics": metrics,
        "model": {
            "checkpoint": str(checkpoint_path.resolve()),
            "tag": checkpoint["tag"],
            "strategy": checkpoint["strategy"],
            "seed": int(checkpoint["seed"]),
            "best_epoch": int(checkpoint["best_epoch"]),
        },
        "label_definition": {"0": "normal", "1": "senescent"},
        "scientific_note": (
            "scLONG is used as a human single-cell representation model. Bulk and mouse "
            "classification results are cross-domain exploratory outputs and are "
            "reported separately from human single-cell results."
        ),
    }
    atomic_csv(prediction_path, frame)
    atomic_json(metrics_path, payload)
    print(f"DATASET: {name}")
    print(f"PREDICTION_ROWS: {len(frame)}")
    print(f"METRICS: {json.dumps(metrics, ensure_ascii=False)}")
    print(f"SAVED_PREDICTIONS: {prediction_path}")
    print(f"SAVED_METRICS: {metrics_path}")
    print(f"DATASET_PREDICTION_FINISHED: {name}", flush=True)


def update_summary(prediction_dir: Path, expected_independent_count: int) -> None:
    records: list[dict[str, Any]] = []
    pooled_truth: list[np.ndarray] = []
    pooled_predicted: list[np.ndarray] = []
    threshold: float | None = None
    for path in sorted(prediction_dir.glob("*_scLONG_Global_MLP_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        name = payload["dataset_id"]
        records.append(
            {
                "dataset_id": name,
                "species": payload["species"],
                "observation_type": payload["observation_type"],
                "n_observations": payload["n_observations"],
                "label_0_count": payload["label_0_count"],
                "label_1_count": payload["label_1_count"],
                "Accuracy": metrics["Accuracy"],
                "F1": metrics["F1"],
                "F1_publication": metrics["F1_publication"],
                "precision": metrics["precision"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "false_positive_rate": metrics["false_positive_rate"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "MCC": metrics["MCC"],
                "TN": metrics["TN"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "TP": metrics["TP"],
                "metrics_status": payload["metrics_status"],
            }
        )
        prediction_path = (
            prediction_dir
            / f"{name}_scLONG_Global_MLP_senescence_predictions.csv"
        )
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        frame = pd.read_csv(prediction_path)
        pooled_truth.append(frame["true_label"].to_numpy(dtype=np.int8))
        pooled_predicted.append(frame["predicted_label"].to_numpy(dtype=np.int8))
        threshold = float(metrics["threshold"])

    if not records:
        return
    if threshold is None:
        raise RuntimeError("No threshold found while building summary.")
    pooled = binary_metrics(
        np.concatenate(pooled_truth), np.concatenate(pooled_predicted), threshold
    )
    all_expected = len(records) == expected_independent_count
    summary_records = records + [
        {
            "dataset_id": "__POOLED_ALL_INDEPENDENT__"
            if all_expected
            else "__POOLED_AVAILABLE_INDEPENDENT__",
            "species": "mixed",
            "observation_type": "mixed",
            "n_observations": int(sum(r["n_observations"] for r in records)),
            "label_0_count": int(sum(r["label_0_count"] for r in records)),
            "label_1_count": int(sum(r["label_1_count"] for r in records)),
            "Accuracy": pooled["Accuracy"],
            "F1": pooled["F1"],
            "F1_publication": pooled["F1_publication"],
            "precision": pooled["precision"],
            "sensitivity": pooled["sensitivity"],
            "specificity": pooled["specificity"],
            "false_positive_rate": pooled["false_positive_rate"],
            "balanced_accuracy": pooled["balanced_accuracy"],
            "MCC": pooled["MCC"],
            "TN": pooled["TN"],
            "FP": pooled["FP"],
            "FN": pooled["FN"],
            "TP": pooled["TP"],
            "metrics_status": (
                "computed_on_all_expected_independent_datasets"
                if all_expected
                else "computed_on_currently_available_independent_datasets"
            ),
        }
    ]
    atomic_csv(
        prediction_dir
        / "Task3_scLONG_Global_MLP_independent_metrics_summary.csv",
        pd.DataFrame(summary_records),
    )
    atomic_json(
        prediction_dir
        / "Task3_scLONG_Global_MLP_independent_metrics_summary.json",
        {
            "created_at_utc": now_utc(),
            "expected_independent_dataset_count": expected_independent_count,
            "available_independent_dataset_count": len(records),
            "all_expected_independent_datasets_present": all_expected,
            "datasets": records,
            "pooled_metrics": pooled,
        },
    )


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    embedding_dir = root / "outputs" / "embeddings"
    prediction_dir = root / "outputs" / "predictions"
    checkpoint_path = (
        root / "models" / "Task3_scLONG_Global_MLP_senescence_classifier.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    paths = discover_independent(embedding_dir)
    if len(paths) != args.expected_independent_count:
        raise RuntimeError(
            f"Expected {args.expected_independent_count} independent embedding "
            f"datasets, found {len(paths)}."
        )
    if args.list:
        for path in paths:
            print(dataset_id_from_npz(path))
        return 0
    if not args.only:
        raise ValueError("--only is required unless --list is used.")
    selected = select_paths(paths, args.only)
    if not selected:
        raise RuntimeError("No independent embedding NPZ files are available.")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    checkpoint = load_checkpoint(checkpoint_path, device)
    print(f"DEVICE: {device}")
    print("CHECKPOINT_VALID: True")
    print(f"AVAILABLE_INDEPENDENT_EMBEDDING_COUNT: {len(paths)}")
    print(f"SELECTED_DATASET_COUNT: {len(selected)}")
    for path in selected:
        predict_one(
            path,
            embedding_dir,
            prediction_dir,
            checkpoint_path,
            checkpoint,
            device,
            args.batch_size,
            args.overwrite,
        )
        update_summary(prediction_dir, args.expected_independent_count)
    print("ALL_SELECTED_TASK3_SCLONG_PREDICTIONS_FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

