#!/usr/bin/env python3
"""Predict and evaluate Task3 senescent-cell binary labels on independent tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING_DIR = REPO_ROOT / "task3/outputs/uce_embeddings"
DEFAULT_CHECKPOINT = REPO_ROOT / "task3/outputs/binary_mlp_global/best_binary_mlp.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "task3/outputs/binary_mlp_global/predictions"


class UCEBinaryDataset(Dataset):
    def __init__(self, h5ad_path: Path, label_col: str = "label") -> None:
        self.h5ad_path = h5ad_path
        adata = ad.read_h5ad(h5ad_path)
        if "X_uce" not in adata.obsm:
            raise KeyError(f"{h5ad_path} does not contain adata.obsm['X_uce']")
        if label_col not in adata.obs:
            raise KeyError(f"{h5ad_path} does not contain adata.obs['{label_col}']")

        labels = pd.to_numeric(adata.obs[label_col], errors="coerce").to_numpy(np.float32)
        valid = np.isfinite(labels)
        if valid.sum() == 0:
            raise ValueError(f"No numeric labels found in {h5ad_path}:{label_col}")
        labels = labels[valid]
        unique_labels = set(np.unique(labels).tolist())
        if not unique_labels <= {0.0, 1.0}:
            raise ValueError(f"Expected binary labels 0/1 in {h5ad_path}, found {sorted(unique_labels)}")

        embeddings = np.asarray(adata.obsm["X_uce"], dtype=np.float32)
        self.x = torch.from_numpy(embeddings[valid])
        self.y = torch.from_numpy(labels).view(-1, 1)
        self.cell_ids = np.asarray(adata.obs_names)[valid]
        self.label_raw = adata.obs[label_col].astype(str).to_numpy()[valid]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Task3 binary labels on independent test embeddings.")
    parser.add_argument("--embedding_dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def find_test_embedding_paths(embedding_dir: Path) -> list[Path]:
    paths = sorted(embedding_dir.glob("Independent.Test_task3*_UCE_input_uce_adata.h5ad"))
    if not paths:
        raise FileNotFoundError(f"No Task3 independent test embedding files found in {embedding_dir}")
    return paths


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[MLPBinaryClassifier, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MLPBinaryClassifier(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=checkpoint["hidden_dims"],
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict(model: nn.Module, dataset: Dataset, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_out = []
    labels_out = []
    for x, y in loader:
        logits = model(x.to(device)).cpu().numpy().reshape(-1)
        logits_out.append(logits)
        labels_out.append(y.numpy().reshape(-1))
    logits_np = np.concatenate(logits_out)
    probs_np = 1.0 / (1.0 + np.exp(-logits_np))
    return np.concatenate(labels_out), probs_np


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true_i = y_true.astype(int)
    y_pred_i = y_pred.astype(int)
    tp = int(((y_true_i == 1) & (y_pred_i == 1)).sum())
    tn = int(((y_true_i == 0) & (y_pred_i == 0)).sum())
    fp = int(((y_true_i == 0) & (y_pred_i == 1)).sum())
    fn = int(((y_true_i == 1) & (y_pred_i == 0)).sum())
    accuracy = float((tp + tn) / len(y_true_i)) if len(y_true_i) else float("nan")
    denom = (2 * tp + fp + fn)
    f1 = float((2 * tp) / denom) if denom > 0 else 0.0
    return {
        "accuracy": accuracy,
        "f1": f1,
        "n_cells": int(len(y_true_i)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_dataset_dir = args.output_dir / "predictions_by_dataset"
    per_dataset_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)

    all_predictions = []
    metric_rows = []
    for h5ad_path in find_test_embedding_paths(args.embedding_dir):
        dataset_name = h5ad_path.stem.replace("_uce_adata", "")
        print(f"[predict] {dataset_name}")
        dataset = UCEBinaryDataset(h5ad_path, label_col=args.label_col)
        y_true, prob_1 = predict(model, dataset, args.batch_size, device)
        pred_label = (prob_1 >= args.threshold).astype(np.int64)
        metrics = compute_binary_metrics(y_true, pred_label)

        pred_df = pd.DataFrame(
            {
                "dataset": dataset_name,
                "cell_id": dataset.cell_ids,
                "label_raw": dataset.label_raw,
                "label": y_true.astype(np.int64),
                "probability_label1": prob_1,
                "prediction": pred_label,
            }
        )
        pred_df.to_csv(per_dataset_dir / f"{dataset_name}_predictions.csv", index=False)
        all_predictions.append(pred_df)

        metric_rows.append(
            {
                "dataset": dataset_name,
                **metrics,
                "threshold": args.threshold,
                "checkpoint": str(args.checkpoint),
                "best_epoch": checkpoint.get("best_epoch"),
                "best_val_bce": checkpoint.get("best_val_bce"),
            }
        )

    pd.concat(all_predictions, ignore_index=True).to_csv(
        args.output_dir / "task3_independent_test_binary_predictions.csv", index=False
    )
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.output_dir / "task3_independent_test_binary_metrics.csv", index=False)
    with open(args.output_dir / "task3_independent_test_binary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metric_rows, f, indent=2)

    print(metrics_df.to_string(index=False))
    print(f"Wrote prediction outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
