#!/usr/bin/env python3
"""
Predict ages for Task1_Independent.Test scCello embeddings with trained MLP models.

Example:
  conda activate sccello
  python aging_scripts/3_predict_age.py --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EMBED_DIR = PROJECT_DIR / "outputs" / "sccello_age" / "cell_embeddings"
DEFAULT_MODEL_DIR = PROJECT_DIR / "outputs" / "sccello_age" / "models"
DEFAULT_PRED_DIR = PROJECT_DIR / "outputs" / "sccello_age" / "predictions"
AGE_PATTERN = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(?:-|_|\s)?\s*(day|week|month|year)s?(?:-old)?\s*$", re.IGNORECASE)


class AgeMLP(nn.Module):
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


def parse_age_to_years(values: Sequence[Any], label_col: str) -> np.ndarray:
    parsed = []
    failed = []
    for value in values:
        if is_missing(value):
            parsed.append(np.nan)
            failed.append(value)
            continue
        try:
            parsed.append(float(value))
            continue
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        match = AGE_PATTERN.match(text)
        if match is None:
            parsed.append(np.nan)
            failed.append(value)
            continue
        number = float(match.group(1))
        unit = match.group(2).lower()
        if unit == "day":
            parsed.append(number / 365.25)
        elif unit == "week":
            parsed.append(number / 52.1775)
        elif unit == "month":
            parsed.append(number / 12.0)
        else:
            parsed.append(number)
    arr = np.asarray(parsed, dtype=np.float32)
    if np.isnan(arr).any():
        examples = [str(v) for v in failed[:5]]
        raise ValueError(f"Could not parse {int(np.isnan(arr).sum())} values from '{label_col}' as ages. Examples: {examples}")
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


def discover_test_embedding(embed_dir: Path) -> Tuple[Path, Path]:
    emb_files = sorted(embed_dir.glob("Task1_Independent.Test*_embeddings.npy"))
    if not emb_files:
        raise FileNotFoundError(f"No independent test embeddings were found in {embed_dir}")
    if len(emb_files) > 1:
        print(f"Found multiple test embeddings; using {emb_files[0]}")
    emb_path = emb_files[0]
    obs_path = embed_dir / emb_path.name.replace("_embeddings.npy", "_obs.csv")
    if not obs_path.exists():
        raise FileNotFoundError(f"Missing obs csv for {emb_path}: {obs_path}")
    return emb_path, obs_path


def discover_models(model_dir: Path) -> List[Path]:
    model_paths = sorted(model_dir.glob("age_mlp_*_best.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No trained MLP checkpoints were found in {model_dir}")
    return model_paths


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = math.sqrt(np.mean(err**2))
    pcc = float("nan") if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12 else float(np.corrcoef(y_true, y_pred)[0, 1])
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": float(mae), "RMSE": float(rmse), "PCC": pcc, "R2": r2}


def load_model(ckpt_path: Path, device: torch.device) -> Tuple[AgeMLP, Dict]:
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    model = AgeMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=list(checkpoint["hidden_dims"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_one_model(ckpt_path: Path, emb_path: Path, batch_size: int, num_workers: int, device: torch.device) -> Tuple[str, np.ndarray]:
    model, checkpoint = load_model(ckpt_path, device)
    strategy = str(checkpoint.get("strategy", ckpt_path.stem.replace("age_mlp_", "").replace("_best", "")))
    mean = checkpoint["embedding_mean"]
    std = checkpoint["embedding_std"]
    if torch.is_tensor(mean):
        mean = mean.detach().cpu().numpy()
    if torch.is_tensor(std):
        std = std.detach().cpu().numpy()
    dataset = PredictionDataset(emb_path, np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    preds = np.empty(len(dataset), dtype=np.float32)
    for x, idx in loader:
        x = x.to(device, non_blocking=True)
        pred = model(x).detach().cpu().tolist()
        idx_list = idx.detach().cpu().tolist() if torch.is_tensor(idx) else list(idx)
        preds[np.asarray(idx_list, dtype=np.int64)] = np.asarray(pred, dtype=np.float32)
    return strategy, preds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict independent test ages from scCello embeddings.")
    parser.add_argument("--embed_dir", type=Path, default=DEFAULT_EMBED_DIR)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--pred_dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--embedding_path", type=Path, default=None)
    parser.add_argument("--obs_path", type=Path, default=None)
    parser.add_argument("--model_paths", type=Path, nargs="*", default=None)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    if args.embedding_path is None or args.obs_path is None:
        emb_path, obs_path = discover_test_embedding(args.embed_dir)
    else:
        emb_path, obs_path = args.embedding_path, args.obs_path

    model_paths = args.model_paths if args.model_paths else discover_models(args.model_dir)
    output, output_fields = read_csv_records(obs_path)
    device = torch.device(args.device)

    metrics_rows = []
    y_true = None
    if args.label_col in output_fields:
        y_true = parse_age_to_years([row[args.label_col] for row in output], args.label_col)
        years_col = f"{args.label_col}_years"
        if years_col not in output_fields:
            output_fields.append(years_col)
        for row, value in zip(output, y_true):
            row[years_col] = float(value)

    for ckpt_path in model_paths:
        strategy, preds = predict_one_model(ckpt_path, emb_path, args.batch_size, args.num_workers, device)
        pred_col = f"pred_age_{strategy}"
        if pred_col not in output_fields:
            output_fields.append(pred_col)
        for row, value in zip(output, preds):
            row[pred_col] = float(value)
        if y_true is not None:
            row = {"model": ckpt_path.name, "strategy": strategy, **regression_metrics(y_true, preds)}
            metrics_rows.append(row)
            print(f"{strategy}: MAE={row['MAE']:.4f} RMSE={row['RMSE']:.4f} PCC={row['PCC']:.4f} R2={row['R2']:.4f}")

    pred_path = args.pred_dir / "Task1_Independent.Test_age_predictions.csv"
    write_csv_records(pred_path, output, output_fields)
    print(f"Saved predictions: {pred_path}")

    if metrics_rows:
        metrics_path = args.pred_dir / "Task1_Independent.Test_age_prediction_metrics.csv"
        write_csv_rows(metrics_path, metrics_rows)
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
