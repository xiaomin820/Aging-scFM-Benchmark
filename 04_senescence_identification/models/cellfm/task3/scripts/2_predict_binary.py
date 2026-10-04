#!/usr/bin/env python3
"""
Predict Task3 senescent-cell labels for all Independent.Test embeddings.

Run after 1_generate_embeddings_and_train_mlp.py:
  conda activate CellFM
  python task3/scripts/2_predict_binary.py --gpu_ids 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
DEFAULT_EMBED_DIR = TASK3_DIR / "outputs" / "cell_embeddings"
DEFAULT_MODEL_DIR = TASK3_DIR / "outputs" / "models"
DEFAULT_PRED_DIR = TASK3_DIR / "outputs" / "predictions"


def resolve_device(gpu_ids: Optional[str], device: str) -> Tuple[torch.device, List[int]]:
    if gpu_ids:
        ids = [int(item.strip()) for item in gpu_ids.split(",") if item.strip()]
        if not ids:
            raise ValueError("--gpu_ids was provided but no valid GPU ids were parsed.")
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu_ids requires CUDA, but CUDA is not available.")
        return torch.device(f"cuda:{ids[0]}"), ids
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is not None:
        return resolved, [resolved.index]
    return resolved, []


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
        return torch.from_numpy(x), idx


def discover_test_embeddings(embed_dir: Path) -> List[Tuple[Path, Path]]:
    emb_files = sorted(embed_dir.glob("Independent.Test*_embeddings.npy"))
    if not emb_files:
        raise FileNotFoundError(f"No Independent.Test embeddings found in {embed_dir}")
    pairs = []
    for emb_path in emb_files:
        obs_path = embed_dir / emb_path.name.replace("_embeddings.npy", "_obs.csv")
        if not obs_path.exists():
            raise FileNotFoundError(f"Missing obs csv for {emb_path}: {obs_path}")
        pairs.append((emb_path, obs_path))
    return pairs


def discover_model(model_dir: Path) -> Path:
    preferred = model_dir / "task3_binary_mlp_global_best.pt"
    if preferred.exists():
        return preferred
    matches = sorted(model_dir.glob("*task3_binary_mlp_global_best.pt"))
    if not matches:
        raise FileNotFoundError(f"No task3 binary MLP checkpoint found in {model_dir}")
    return matches[-1]


def binary_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    true = y_true.astype(np.int64)
    pred = pred.astype(np.int64)
    tp = int(((pred == 1) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    acc = float((tp + tn) / max(len(true), 1))
    denom = 2 * tp + fp + fn
    f1 = float(2 * tp / denom) if denom else 0.0
    return {"Accuracy": acc, "F1": f1, "TP": tp, "TN": tn, "FP": fp, "FN": fn}


def load_model(ckpt_path: Path, device: torch.device, device_ids: List[int]) -> Tuple[nn.Module, Dict, np.ndarray, np.ndarray]:
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    model = BinaryMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=list(checkpoint["hidden_dims"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    state_dict = checkpoint["model_state_dict"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        stripped = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
        model.load_state_dict(stripped)
    if len(device_ids) > 1:
        print(f"Using DataParallel for MLP prediction on GPUs: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    model.eval()

    mean = checkpoint["embedding_mean"]
    std = checkpoint["embedding_std"]
    if torch.is_tensor(mean):
        mean = mean.detach().cpu().numpy()
    if torch.is_tensor(std):
        std = std.detach().cpu().numpy()
    return model, checkpoint, np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    emb_path: Path,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    dataset = PredictionDataset(emb_path, mean, std)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    logits = np.empty(len(dataset), dtype=np.float32)
    for x, idx in loader:
        out = model(x.to(device, non_blocking=True)).detach().cpu().numpy().astype(np.float32)
        logits[np.asarray(idx)] = out
    return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Task3 binary senescent-cell labels from CellFM embeddings.")
    parser.add_argument("--embed_dir", type=Path, default=DEFAULT_EMBED_DIR)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--pred_dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--embedding_path", type=Path, default=None)
    parser.add_argument("--obs_path", type=Path, default=None)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated GPU ids, e.g. 0,1,2. Overrides --device.")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    device, device_ids = resolve_device(args.gpu_ids, args.device)
    ckpt_path = args.model_path if args.model_path else discover_model(args.model_dir)
    model, checkpoint, mean, std = load_model(ckpt_path, device, device_ids)
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5))
    print(f"Using model: {ckpt_path}")
    print(f"Using threshold: {threshold}")

    if args.embedding_path is None or args.obs_path is None:
        test_pairs = discover_test_embeddings(args.embed_dir)
    else:
        test_pairs = [(args.embedding_path, args.obs_path)]

    metrics_rows = []
    for emb_path, obs_path in test_pairs:
        test_name = emb_path.name.replace("_embeddings.npy", "")
        obs = pd.read_csv(obs_path)
        logits = predict_logits(model, emb_path, mean, std, args.batch_size, args.num_workers, device)
        probs = 1.0 / (1.0 + np.exp(-logits))
        pred = (probs >= threshold).astype(np.int64)

        output = obs.copy()
        output["pred_logit"] = logits
        output["pred_prob_label1"] = probs
        output["pred_label"] = pred
        pred_path = args.pred_dir / f"{test_name}_binary_predictions.csv"
        output.to_csv(pred_path, index=False)
        print(f"Saved predictions: {pred_path}")

        if args.label_col in output.columns:
            labels = pd.to_numeric(output[args.label_col], errors="coerce").to_numpy(dtype=np.float32)
            valid = np.isfinite(labels) & np.isin(labels, [0.0, 1.0])
            if valid.any():
                row = {
                    "test_set": test_name,
                    "model": ckpt_path.name,
                    "threshold": threshold,
                    "n_eval": int(valid.sum()),
                    "n_total": int(len(output)),
                    **binary_metrics(labels[valid], pred[valid]),
                }
                metrics_rows.append(row)
                print(f"{test_name}: Accuracy={row['Accuracy']:.4f} F1={row['F1']:.4f}")

    if metrics_rows:
        metrics_path = args.pred_dir / "Task3_independent_test_binary_metrics.csv"
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
