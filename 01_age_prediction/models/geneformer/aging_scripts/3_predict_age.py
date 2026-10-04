#!/usr/bin/env python3
"""
Predict ages for Task1_Independent.Test embeddings with trained Geneformer MLP models.

Run after scripts 1 and 2:
  conda activate geneformer
  python aging_scripts/3_predict_age.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_EMBED_DIR = PROJECT_DIR / "outputs" / "geneformer_cell_embeddings"
DEFAULT_MODEL_DIR = PROJECT_DIR / "outputs" / "geneformer_age_mlp_models"
DEFAULT_PRED_DIR = PROJECT_DIR / "outputs" / "geneformer_age_predictions"


class AgeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
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
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        pcc = float("nan")
    else:
        pcc = float(np.corrcoef(y_true, y_pred)[0, 1])
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
def predict_one_model(
    ckpt_path: Path,
    emb_path: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[str, np.ndarray]:
    model, checkpoint = load_model(ckpt_path, device)
    strategy = str(checkpoint.get("strategy", ckpt_path.stem.replace("age_mlp_", "").replace("_best", "")))
    mean = checkpoint["embedding_mean"]
    std = checkpoint["embedding_std"]
    if torch.is_tensor(mean):
        mean = mean.detach().cpu().numpy()
    if torch.is_tensor(std):
        std = std.detach().cpu().numpy()
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    dataset = PredictionDataset(emb_path, mean, std)
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
        pred = model(x).detach().cpu().numpy().astype(np.float32)
        preds[np.asarray(idx)] = pred
    return strategy, preds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict independent test ages from Geneformer embeddings.")
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
    obs = pd.read_csv(obs_path)
    output = obs.copy()
    device = torch.device(args.device)

    metrics_rows = []
    y_true = None
    if args.label_col in output.columns:
        y_true_candidate = pd.to_numeric(output[args.label_col], errors="coerce").to_numpy(dtype=np.float32)
        if not np.isnan(y_true_candidate).any():
            y_true = y_true_candidate

    for ckpt_path in model_paths:
        strategy, preds = predict_one_model(ckpt_path, emb_path, args.batch_size, args.num_workers, device)
        pred_col = f"pred_age_{strategy}"
        output[pred_col] = preds
        if y_true is not None:
            row = {"model": ckpt_path.name, "strategy": strategy, **regression_metrics(y_true, preds)}
            metrics_rows.append(row)
            print(
                f"{strategy}: MAE={row['MAE']:.4f} RMSE={row['RMSE']:.4f} "
                f"PCC={row['PCC']:.4f} R2={row['R2']:.4f}"
            )

    pred_path = args.pred_dir / "Task1_Independent.Test_age_predictions.csv"
    output.to_csv(pred_path, index=False)
    print(f"Saved predictions: {pred_path}")

    run_meta = {
        "embedding_path": str(emb_path),
        "obs_path": str(obs_path),
        "model_paths": [str(path) for path in model_paths],
        "prediction_file": str(pred_path),
    }
    (args.pred_dir / "prediction_run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    if metrics_rows:
        metrics_path = args.pred_dir / "Task1_Independent.Test_age_prediction_metrics.csv"
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
