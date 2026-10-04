#!/usr/bin/env python3
"""Train the Task3 Global binary senescence MLP from SCimilarity embeddings."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
LR = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 5
VAL_RATIO = 0.1
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
GRAD_CLIP = 5.0
EXPECTED_INPUT_DIM = 128
THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


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
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def binary_metrics(
    y_true: np.ndarray, probability: np.ndarray, threshold: float = THRESHOLD
) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = (np.asarray(probability, dtype=np.float64) >= threshold).astype(np.int64)
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    accuracy = float((tp + tn) / len(truth))
    denominator = 2 * tp + fp + fn
    f1 = float(2 * tp / denominator) if denominator > 0 else 0.0
    return {
        "Accuracy": accuracy,
        "F1": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "threshold": float(threshold),
    }


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


def discover_training_files(embedding_dir: Path) -> tuple[Path, Path, str]:
    paths = sorted(embedding_dir.glob("Training_*_SCimilarity_cell_embeddings.npz"))
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected one training embedding NPZ, found {len(paths)} in {embedding_dir}"
        )
    npz_path = paths[0]
    suffix = "_SCimilarity_cell_embeddings.npz"
    dataset_id = npz_path.name[: -len(suffix)]
    metadata_path = (
        embedding_dir / f"{dataset_id}_SCimilarity_cell_embeddings_metadata.csv"
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    return npz_path, metadata_path, dataset_id


def stratified_split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        if len(indices) < 2:
            raise ValueError(f"Class {label} has fewer than two observations.")
        indices = rng.permutation(indices)
        val_size = int(len(indices) * VAL_RATIO)
        if val_size < 1:
            val_size = 1
        if val_size >= len(indices):
            raise ValueError(f"Invalid validation size for class {label}: {val_size}")
        val_parts.append(indices[:val_size])
        train_parts.append(indices[val_size:])
    train = rng.permutation(np.concatenate(train_parts))
    validation = rng.permutation(np.concatenate(val_parts))
    return train, validation


def batched_probabilities(
    model: nn.Module, values: np.ndarray, device: torch.device, batch_size: int = 4096
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            outputs.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def main() -> int:
    args = parse_args()
    set_reproducibility(SEED)
    root = args.project_root.expanduser().resolve()
    embedding_dir = root / "outputs" / "embeddings"
    model_path = root / "models" / "Task3_SCimilarity_Global_MLP_senescence_classifier.pt"
    result_dir = root / "outputs" / "training"
    summary_path = result_dir / "Task3_SCimilarity_Global_MLP_training_summary.json"
    history_path = result_dir / "Task3_SCimilarity_Global_MLP_training_history.csv"
    split_path = result_dir / "Task3_SCimilarity_Global_MLP_training_split.csv"
    validation_path = (
        result_dir / "Task3_SCimilarity_Global_MLP_validation_predictions.csv"
    )
    outputs = [model_path, summary_path, history_path, split_path, validation_path]
    if any(path.exists() for path in outputs) and not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        raise FileExistsError(
            f"Training outputs exist; inspect or use --overwrite: {existing}"
        )

    npz_path, metadata_path, dataset_id = discover_training_files(embedding_dir)
    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != {"X", "cell_id"}:
            raise ValueError(f"Unexpected NPZ keys: {payload.files}")
        x = np.asarray(payload["X"], dtype=np.float32)
        npz_cell_ids = np.asarray(payload["cell_id"], dtype=str)
    metadata = pd.read_csv(metadata_path, dtype={"cell_id": str})
    if x.ndim != 2 or x.shape[1] != EXPECTED_INPUT_DIM:
        raise ValueError(f"Expected (n,{EXPECTED_INPUT_DIM}), got {x.shape}")
    if len(metadata) != len(x) or not np.isfinite(x).all():
        raise ValueError("Invalid training embedding or metadata rows.")
    cell_ids = metadata["cell_id"].astype(str).to_numpy()
    if not np.array_equal(npz_cell_ids, cell_ids):
        raise ValueError("NPZ and metadata cell_id order differs.")
    if pd.Index(cell_ids).has_duplicates:
        raise ValueError("Training cell_id values are not unique.")
    numeric = pd.to_numeric(metadata["label"], errors="coerce")
    if numeric.isna().any():
        raise ValueError("Training label contains missing/nonnumeric values.")
    y = numeric.to_numpy(dtype=np.float32)
    if set(np.unique(y).tolist()) != {0.0, 1.0}:
        raise ValueError(f"Training labels must contain both 0 and 1: {np.unique(y)}")

    train_indices, val_indices = stratified_split(y)
    x_train_raw = x[train_indices]
    x_val_raw = x[val_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]
    mean = x_train_raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train_raw.std(axis=0, dtype=np.float64).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Embedding standardization statistics are invalid.")
    x_train = ((x_train_raw - mean) / std).astype(np.float32)
    x_val = ((x_val_raw - mean) / std).astype(np.float32)
    if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
        raise ValueError("Standardized embeddings contain NaN or Inf.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_x_tensor = torch.from_numpy(x_val).to(device)
    val_y_tensor = torch.from_numpy(y_val).to(device)
    model = SenescenceMLP(EXPECTED_INPUT_DIM).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.BCEWithLogitsLoss()
    best_val_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    history: list[dict[str, Any]] = []

    print(f"DATASET_ID: {dataset_id}")
    print(f"TOTAL_OBSERVATIONS: {len(x)}")
    print(f"LABEL_0_COUNT: {int((y == 0).sum())}")
    print(f"LABEL_1_COUNT: {int((y == 1).sum())}")
    print(f"TRAIN_SIZE: {len(train_indices)}")
    print(f"VALIDATION_SIZE: {len(val_indices)}")
    print(f"SEED: {SEED}")
    print(f"DEVICE: {device}")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(batch_x)
            train_count += len(batch_x)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x_tensor)
            val_loss = float(loss_fn(val_logits, val_y_tensor).item())
            val_probability = torch.sigmoid(val_logits).cpu().numpy()
        train_loss = train_loss_sum / train_count
        epoch_metrics = binary_metrics(y_val, val_probability)
        improved = val_loss < best_val_loss
        history.append(
            {
                "epoch": epoch,
                "train_bce": train_loss,
                "val_bce": val_loss,
                "val_accuracy": epoch_metrics["Accuracy"],
                "val_f1": epoch_metrics["F1"],
                "improved": improved,
            }
        )
        print(
            f"epoch={epoch:03d} train_bce={train_loss:.8f} "
            f"val_bce={val_loss:.8f} val_accuracy={epoch_metrics['Accuracy']:.6f} "
            f"val_f1={epoch_metrics['F1']:.6f} improved={improved}",
            flush=True,
        )
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"EARLY_STOPPING_AT_EPOCH: {epoch}")
                break

    if best_state is None:
        raise RuntimeError("No best model state was captured.")
    model.load_state_dict(best_state)
    model.to(device)
    probability = batched_probabilities(model, x_val, device)
    metrics = binary_metrics(y_val, probability)
    predicted = (probability >= THRESHOLD).astype(np.int8)

    split_frame = pd.DataFrame(
        {
            "source_row_index": np.arange(len(x)),
            "cell_id": cell_ids,
            "label": y.astype(np.int8),
            "split": "",
        }
    )
    split_frame.loc[train_indices, "split"] = "train"
    split_frame.loc[val_indices, "split"] = "validation"
    if "donorID" in metadata.columns:
        split_frame["donorID"] = metadata["donorID"].to_numpy()
        train_donors = set(
            metadata.iloc[train_indices]["donorID"].dropna().astype(str)
        )
        val_donors = set(metadata.iloc[val_indices]["donorID"].dropna().astype(str))
        donor_overlap = sorted(train_donors & val_donors)
    else:
        donor_overlap = []
    validation_frame = pd.DataFrame(
        {
            "source_row_index": val_indices,
            "cell_id": cell_ids[val_indices],
            "true_label": y_val.astype(np.int8),
            "senescent_probability": probability,
            "predicted_label": predicted,
            "correct": predicted == y_val.astype(np.int8),
        }
    )

    checkpoint = {
        "checkpoint_format_version": "1.0",
        "task": "Task3_senescence_binary_classification",
        "foundation_model": "SCimilarity",
        "strategy": "Global",
        "tag": "Task3_SCimilarity_Global_MLP",
        "seed": SEED,
        "input_dim": EXPECTED_INPUT_DIM,
        "architecture": [512, 128, 1],
        "dropout": DROPOUT,
        "model_state_dict": best_state,
        "embedding_mean": torch.from_numpy(mean.copy()),
        "embedding_std": torch.from_numpy(std.copy()),
        "decision_threshold": THRESHOLD,
        "label_definition": {0: "normal", 1: "senescent"},
        "best_epoch": best_epoch,
        "best_val_bce": best_val_loss,
        "validation_metrics": metrics,
        "training_dataset_id": dataset_id,
        "training_config": {
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "validation_ratio": VAL_RATIO,
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "loss": "BCEWithLogitsLoss",
            "gradient_clip_max_norm": GRAD_CLIP,
            "sampling": "stratified global random observation-level split",
            "standardization": "mean/std fitted on the 90% training split only",
        },
    }
    summary = {
        "summary_version": "1.0",
        "created_at_utc": now_utc(),
        "dataset_id": dataset_id,
        "seed": SEED,
        "strategy": "Global",
        "device": str(device),
        "input_embedding_shape": [int(v) for v in x.shape],
        "label_0_count": int((y == 0).sum()),
        "label_1_count": int((y == 1).sum()),
        "train_observations": int(len(train_indices)),
        "validation_observations": int(len(val_indices)),
        "train_validation_donor_overlap_count": len(donor_overlap),
        "train_validation_donor_overlap": donor_overlap,
        "best_epoch": best_epoch,
        "best_val_bce": best_val_loss,
        "validation_metrics": metrics,
        "checkpoint": str(model_path.resolve()),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
        "methodological_note": (
            "The fixed benchmark split is observation-level and stratified by label. "
            "Cells from the same donor may occur in both splits; validation is not "
            "donor-independent."
        ),
    }
    atomic_torch_save(model_path, checkpoint)
    atomic_csv(history_path, pd.DataFrame(history))
    atomic_csv(split_path, split_frame)
    atomic_csv(validation_path, validation_frame)
    atomic_json(summary_path, summary)
    print(f"BEST_EPOCH: {best_epoch}")
    print(f"BEST_VAL_BCE: {best_val_loss:.8f}")
    print(f"VALIDATION_METRICS: {json.dumps(metrics, ensure_ascii=False)}")
    print(f"SAVED_CHECKPOINT: {model_path}")
    print("TASK3_SCIMILARITY_GLOBAL_MLP_TRAINING_FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
