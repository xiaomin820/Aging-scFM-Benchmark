#!/usr/bin/env python3
"""
Predict senescent-cell binary labels for all Task3 Independent.Test embeddings.

Outputs:
  prob_label1: predicted probability for label 1, senescent cell
  pred_label: thresholded binary prediction

Run after scripts 1 and 2:

  conda activate geneformer
  python task3/scripts/3_predict_binary.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
DEFAULT_EMBED_DIR = TASK3_DIR / "outputs" / "geneformer_cell_embeddings"
DEFAULT_MODEL_DIR = TASK3_DIR / "outputs" / "geneformer_binary_mlp_models"
DEFAULT_PRED_DIR = TASK3_DIR / "outputs" / "geneformer_binary_predictions"


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
        return int(self.embeddings.shape[0])

    def __getitem__(self, idx: int):
        x = np.asarray(self.embeddings[idx], dtype=np.float32)
        x = (x - self.mean) / self.std
        return torch.from_numpy(x), idx


def discover_test_embeddings(embed_dir: Path) -> List[Tuple[Path, Path]]:
    emb_files = sorted(embed_dir.glob("*Independent.Test*_embeddings.npy"))
    if not emb_files:
        raise FileNotFoundError(f"No Task3 independent test embeddings were found in {embed_dir}")
    pairs = []
    for emb_path in emb_files:
        obs_path = embed_dir / emb_path.name.replace("_embeddings.npy", "_obs.csv")
        if not obs_path.exists():
            raise FileNotFoundError(f"Missing obs csv for {emb_path}: {obs_path}")
        pairs.append((emb_path, obs_path))
    return pairs


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = (np.asarray(y_prob) >= threshold).astype(np.int64)
    accuracy = float((y_pred == y_true).mean())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"Accuracy": accuracy, "F1": float(f1)}


def load_model(ckpt_path: Path, device: torch.device) -> Tuple[BinaryMLP, Dict]:
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
    return model, checkpoint


@torch.no_grad()
def predict_one_embedding(
    model: BinaryMLP,
    checkpoint: Dict,
    emb_path: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
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
    probs = np.empty(len(dataset), dtype=np.float32)
    for x, idx in loader:
        x = x.to(device, non_blocking=True)
        prob = torch.sigmoid(model(x)).detach().cpu().numpy().astype(np.float32)
        probs[np.asarray(idx)] = prob
    return probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict all Task3 independent test binary labels from Geneformer embeddings.")
    parser.add_argument("--embed_dir", type=Path, default=DEFAULT_EMBED_DIR)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--pred_dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--embedding_paths", type=Path, nargs="*", default=None)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_path or (args.model_dir / "binary_mlp_global_best.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained binary MLP checkpoint not found: {model_path}")

    if args.embedding_paths:
        pairs = []
        for emb_path in args.embedding_paths:
            obs_path = args.embed_dir / emb_path.name.replace("_embeddings.npy", "_obs.csv")
            if not obs_path.exists():
                obs_path = emb_path.with_name(emb_path.name.replace("_embeddings.npy", "_obs.csv"))
            if not obs_path.exists():
                raise FileNotFoundError(f"Missing obs csv for {emb_path}")
            pairs.append((emb_path, obs_path))
    else:
        pairs = discover_test_embeddings(args.embed_dir)

    device = torch.device(args.device)
    model, checkpoint = load_model(model_path, device)
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5))
    metrics_rows = []

    for emb_path, obs_path in pairs:
        obs = pd.read_csv(obs_path)
        probs = predict_one_embedding(model, checkpoint, emb_path, args.batch_size, args.num_workers, device)
        pred_labels = (probs >= threshold).astype(np.int64)
        output = obs.copy()
        output["prob_label1"] = probs
        output["pred_label"] = pred_labels
        stem = emb_path.name.replace("_embeddings.npy", "")
        pred_path = args.pred_dir / f"{stem}_binary_predictions.csv"
        output.to_csv(pred_path, index=False)
        print(f"Saved predictions: {pred_path}")

        if args.label_col in output.columns:
            y_true = pd.to_numeric(output[args.label_col], errors="coerce").to_numpy(dtype=np.float32)
            valid = ~np.isnan(y_true) & np.isin(y_true, [0.0, 1.0])
            if valid.any():
                row = {
                    "test_set": stem,
                    "model": model_path.name,
                    "threshold": threshold,
                    "n_cells": int(valid.sum()),
                    "label0": int((y_true[valid] == 0).sum()),
                    "label1": int((y_true[valid] == 1).sum()),
                    **classification_metrics(y_true[valid], probs[valid], threshold),
                }
                metrics_rows.append(row)
                print(f"{stem}: Accuracy={row['Accuracy']:.4f} F1={row['F1']:.4f}")

    run_meta = {
        "model_path": str(model_path),
        "embedding_paths": [str(path) for path, _ in pairs],
        "prediction_dir": str(args.pred_dir),
        "threshold": threshold,
        "positive_class": 1,
        "negative_class": 0,
    }
    (args.pred_dir / "prediction_run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    if metrics_rows:
        metrics_path = args.pred_dir / "Task3_Independent.Test_binary_prediction_metrics.csv"
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
