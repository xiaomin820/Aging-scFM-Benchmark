#!/usr/bin/env python3
"""
Predict binary labels for all Task3 Independent.Test scCello embeddings with trained MLP models.

Example:
  conda activate sccello
  python task3/scripts/3_predict_task3_tests.py --gpu_ids 0,1
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[2]
TASK3_DIR = PROJECT_DIR / "task3"
DEFAULT_EMBED_DIR = TASK3_DIR / "outputs" / "sccello_embeddings"
DEFAULT_MODEL_DIR = TASK3_DIR / "outputs" / "models"
DEFAULT_PRED_DIR = TASK3_DIR / "outputs" / "predictions"


def parse_gpu_ids(gpu_ids: Optional[str]) -> List[int]:
    if gpu_ids is None or str(gpu_ids).strip() == "":
        return []
    return [int(x.strip()) for x in str(gpu_ids).split(",") if x.strip()]


def resolve_device(device_arg: str, gpu_ids_arg: Optional[str]) -> Tuple[torch.device, List[int]]:
    gpu_ids = parse_gpu_ids(gpu_ids_arg)
    if gpu_ids:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu_ids was provided, but CUDA is not available.")
        return torch.device(f"cuda:{gpu_ids[0]}"), gpu_ids
    return torch.device(device_arg), []


class BinaryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PredictionDataset(Dataset):
    def __init__(self, emb_path: Path, mean: np.ndarray, std: np.ndarray) -> None:
        self.embeddings = np.load(emb_path, mmap_mode="r")
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int):
        x = np.asarray(self.embeddings[idx], dtype=np.float32)
        x = (x - self.mean) / self.std
        return torch.tensor(x.tolist(), dtype=torch.float32), idx


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == ""


def parse_binary_labels(values: Sequence[Any], label_col: str, allow_missing: bool = False) -> np.ndarray:
    parsed = []
    failed = []
    for value in values:
        if is_missing(value):
            parsed.append(np.nan)
            failed.append(value)
            continue
        try:
            label = float(value)
            if label not in (0.0, 1.0):
                raise ValueError
            parsed.append(label)
            continue
        except (TypeError, ValueError):
            parsed.append(np.nan)
            failed.append(value)
            continue
    arr = np.asarray(parsed, dtype=np.float32)
    if np.isnan(arr).any() and not allow_missing:
        examples = [str(v) for v in failed[:5]]
        raise ValueError(f"Could not parse {int(np.isnan(arr).sum())} values from '{label_col}' as binary 0/1 labels. Examples: {examples}")
    return arr


def read_csv_records(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return records, fieldnames


def write_csv_records(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    write_csv_records(path, rows, list(rows[0].keys()))


def discover_test_embeddings(embed_dir: Path) -> List[Tuple[Path, Path]]:
    emb_files = sorted(embed_dir.glob("Independent.Test*_embeddings.npy"))
    if not emb_files:
        raise FileNotFoundError(f"No Task3 independent test embeddings were found in {embed_dir}")
    pairs = []
    for emb_path in emb_files:
        obs_path = embed_dir / emb_path.name.replace("_embeddings.npy", "_obs.csv")
        if not obs_path.exists():
            raise FileNotFoundError(f"Missing obs csv for {emb_path}: {obs_path}")
        pairs.append((emb_path, obs_path))
    return pairs


def discover_models(model_dir: Path) -> List[Path]:
    model_paths = sorted(model_dir.glob("binary_mlp_*_best.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No trained MLP checkpoints were found in {model_dir}")
    return model_paths


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = (np.asarray(y_prob, dtype=np.float64) >= threshold).astype(np.int64)
    accuracy = float((y_true == y_pred).mean()) if y_true.size else float("nan")
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = 2 * tp + fp + fn
    f1 = float(2 * tp / denom) if denom > 0 else 0.0
    return {"Accuracy": accuracy, "F1": f1}


def load_model(ckpt_path: Path, device: torch.device, gpu_ids: Sequence[int]) -> Tuple[nn.Module, Dict]:
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    model = BinaryMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=list(checkpoint["hidden_dims"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=list(gpu_ids), output_device=int(gpu_ids[0]))
        model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_one_model(
    ckpt_path: Path,
    emb_path: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    gpu_ids: Sequence[int],
) -> Tuple[str, np.ndarray]:
    model, checkpoint = load_model(ckpt_path, device, gpu_ids)
    strategy = str(checkpoint.get("strategy", ckpt_path.stem.replace("binary_mlp_", "").replace("_best", "")))
    mean = checkpoint["embedding_mean"]
    std = checkpoint["embedding_std"]
    if torch.is_tensor(mean):
        mean = np.asarray(mean.detach().cpu().tolist(), dtype=np.float32)
    if torch.is_tensor(std):
        std = np.asarray(std.detach().cpu().tolist(), dtype=np.float32)
    dataset = PredictionDataset(emb_path, np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    probs = np.empty(len(dataset), dtype=np.float32)
    for x, idx in loader:
        x = x.to(device, non_blocking=True)
        prob = torch.sigmoid(model(x)).detach().cpu().tolist()
        idx_list = idx.detach().cpu().tolist() if torch.is_tensor(idx) else list(idx)
        probs[np.asarray(idx_list, dtype=np.int64)] = np.asarray(prob, dtype=np.float32)
    return strategy, probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict all Task3 independent test binary labels from scCello embeddings.")
    parser.add_argument("--embed_dir", type=Path, default=DEFAULT_EMBED_DIR)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--pred_dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--embedding_path", type=Path, default=None)
    parser.add_argument("--obs_path", type=Path, default=None)
    parser.add_argument("--model_paths", type=Path, nargs="*", default=None)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated GPU ids, e.g. '0,1,2'. Overrides --device.")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def predict_one_test_file(
    emb_path: Path,
    obs_path: Path,
    model_paths: Sequence[Path],
    args: argparse.Namespace,
    device: torch.device,
    gpu_ids: Sequence[int],
) -> List[Dict[str, Any]]:
    output, output_fields = read_csv_records(obs_path)
    metrics_rows = []
    y_true = None
    metric_mask = None
    if args.label_col in output_fields:
        y_true = parse_binary_labels([row[args.label_col] for row in output], args.label_col, allow_missing=True)
        metric_mask = ~np.isnan(y_true)
        numeric_col = f"{args.label_col}_numeric"
        if numeric_col not in output_fields:
            output_fields.append(numeric_col)
        for row, value in zip(output, y_true):
            row[numeric_col] = float(value)

    for ckpt_path in model_paths:
        strategy, probs = predict_one_model(ckpt_path, emb_path, args.batch_size, args.num_workers, device, gpu_ids)
        prob_col = f"prob_label1_{strategy}"
        pred_col = f"pred_label_{strategy}"
        for col in (prob_col, pred_col):
            if col not in output_fields:
                output_fields.append(col)
        pred_labels = (probs >= 0.5).astype(np.int64)
        for row, prob, pred_label in zip(output, probs, pred_labels):
            row[prob_col] = float(prob)
            row[pred_col] = int(pred_label)
        if y_true is not None and metric_mask is not None and metric_mask.any():
            row = {
                "test_file": emb_path.name.replace("_embeddings.npy", ""),
                "model": ckpt_path.name,
                "strategy": strategy,
                "n_metric_cells": int(metric_mask.sum()),
                "n_missing_label_cells": int((~metric_mask).sum()),
                **binary_metrics(y_true[metric_mask], probs[metric_mask]),
            }
            metrics_rows.append(row)
            print(
                f"{emb_path.stem} | {strategy}: Accuracy={row['Accuracy']:.4f} "
                f"F1={row['F1']:.4f}"
            )

    pred_path = args.pred_dir / f"{emb_path.name.replace('_embeddings.npy', '')}_binary_predictions.csv"
    write_csv_records(pred_path, output, output_fields)
    print(f"Saved predictions: {pred_path}")

    if metrics_rows:
        metrics_path = args.pred_dir / f"{emb_path.name.replace('_embeddings.npy', '')}_binary_prediction_metrics.csv"
        write_csv_rows(metrics_path, metrics_rows)
        print(f"Saved metrics: {metrics_path}")
    return metrics_rows


def main() -> None:
    args = parse_args()
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    if args.embedding_path is not None or args.obs_path is not None:
        if args.embedding_path is None or args.obs_path is None:
            raise ValueError("--embedding_path and --obs_path must be provided together.")
        test_pairs = [(args.embedding_path, args.obs_path)]
    else:
        test_pairs = discover_test_embeddings(args.embed_dir)

    model_paths = args.model_paths if args.model_paths else discover_models(args.model_dir)
    device, gpu_ids = resolve_device(args.device, args.gpu_ids)
    if len(gpu_ids) > 1:
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    all_metrics = []
    for emb_path, obs_path in test_pairs:
        all_metrics.extend(predict_one_test_file(emb_path, obs_path, model_paths, args, device, gpu_ids))

    if all_metrics:
        all_metrics_path = args.pred_dir / "all_test_binary_prediction_metrics.csv"
        write_csv_rows(all_metrics_path, all_metrics)
        print(f"Saved combined metrics: {all_metrics_path}")


if __name__ == "__main__":
    main()
