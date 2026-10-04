#!/usr/bin/env python3
"""
Train a lightweight global-batching MLP binary classifier for Task3.

Labels:
  0 = normal cell
  1 = senescent cell

Run after generating embeddings:

  conda activate geneformer
  python task3/scripts/2_train_binary_mlp_global.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
DEFAULT_EMBED_DIR = TASK3_DIR / "outputs" / "geneformer_cell_embeddings"
DEFAULT_MODEL_DIR = TASK3_DIR / "outputs" / "geneformer_binary_mlp_models"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_training_embedding(embed_dir: Path) -> Tuple[Path, Path]:
    emb_files = sorted(embed_dir.glob("Training*_embeddings.npy"))
    if not emb_files:
        raise FileNotFoundError(f"No Task3 training embeddings were found in {embed_dir}")
    if len(emb_files) > 1:
        print(f"Found multiple training embeddings; using {emb_files[0]}")
    emb_path = emb_files[0]
    obs_path = embed_dir / emb_path.name.replace("_embeddings.npy", "_obs.csv")
    if not obs_path.exists():
        raise FileNotFoundError(f"Missing obs csv for {emb_path}: {obs_path}")
    return emb_path, obs_path


def split_indices(n: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    val_idx = np.sort(order[:n_val])
    train_idx = np.sort(order[n_val:])
    return train_idx, val_idx


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        emb_path: Path,
        obs_path: Path,
        label_col: str,
        indices: Optional[np.ndarray] = None,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        drop_invalid_labels: bool = True,
    ) -> None:
        self.emb_path = Path(emb_path)
        self.obs_path = Path(obs_path)
        self.label_col = label_col
        self.embeddings = np.load(self.emb_path, mmap_mode="r")
        self.obs = pd.read_csv(self.obs_path)
        if len(self.obs) != self.embeddings.shape[0]:
            raise ValueError(f"Row mismatch: {self.emb_path} and {self.obs_path}")
        if label_col not in self.obs.columns:
            raise KeyError(f"Column '{label_col}' was not found in {self.obs_path}")

        labels = pd.to_numeric(self.obs[label_col], errors="coerce").to_numpy(dtype=np.float32)
        self.labels = labels
        selected_indices = np.asarray(indices, dtype=np.int64) if indices is not None else np.arange(len(labels), dtype=np.int64)
        selected_labels = self.labels[selected_indices]
        invalid_mask = np.isnan(selected_labels) | ~np.isin(selected_labels, [0.0, 1.0])
        if invalid_mask.any():
            bad = int(invalid_mask.sum())
            if not drop_invalid_labels:
                raise ValueError(f"{bad} labels in '{label_col}' are not valid binary labels 0/1.")
            print(f"Skipping {bad} cells with missing/non-binary '{label_col}' in {self.obs_path.name}.")
            selected_indices = selected_indices[~invalid_mask]
        if len(selected_indices) == 0:
            raise ValueError(f"No valid binary labels remain in {self.obs_path}")

        self.indices = selected_indices
        self.mean = mean.astype(np.float32) if mean is not None else None
        self.std = std.astype(np.float32) if std is not None else None

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        x = np.asarray(self.embeddings[real_idx], dtype=np.float32)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        y = np.asarray(self.labels[real_idx], dtype=np.float32)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def clone_dataset_with_scaler(dataset: EmbeddingDataset, mean: np.ndarray, std: np.ndarray) -> EmbeddingDataset:
    return EmbeddingDataset(dataset.emb_path, dataset.obs_path, dataset.label_col, dataset.indices, mean, std)


def compute_scaler(dataset: EmbeddingDataset, chunk_size: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
    dim = dataset.embedding_dim
    total = 0
    sum_x = np.zeros(dim, dtype=np.float64)
    sum_x2 = np.zeros(dim, dtype=np.float64)
    for start in tqdm(range(0, len(dataset.indices), chunk_size), desc="Scaler"):
        idx = dataset.indices[start : start + chunk_size]
        x = np.asarray(dataset.embeddings[idx], dtype=np.float32)
        sum_x += x.sum(axis=0, dtype=np.float64)
        sum_x2 += np.square(x, dtype=np.float64).sum(axis=0, dtype=np.float64)
        total += x.shape[0]
    mean = sum_x / max(total, 1)
    var = np.maximum(sum_x2 / max(total, 1) - mean**2, 1e-8)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


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


def parse_hidden_dims(value: str) -> List[int]:
    dims = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not dims:
        raise ValueError("--hidden_dims must contain at least one integer")
    return dims


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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> Tuple[float, Dict[str, float]]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_n = 0
    probs = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            total_n += y.numel()
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels)
    return total_loss / max(total_n, 1), classification_metrics(y_true, y_prob, threshold)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    running = 0.0
    total = 0
    for x, y in tqdm(loader, desc="Train global", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        running += loss.item() * y.numel()
        total += y.numel()
    return running / max(total, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
    mean: np.ndarray,
    std: np.ndarray,
    epoch: int,
    val_loss: float,
    metrics: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "strategy": "global",
            "task": "binary_classification",
            "input_dim": input_dim,
            "hidden_dims": list(hidden_dims),
            "dropout": dropout,
            "embedding_mean": torch.from_numpy(mean.astype(np.float32)),
            "embedding_std": torch.from_numpy(std.astype(np.float32)),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_metrics": metrics,
            "label_col": args.label_col,
            "threshold": args.threshold,
            "negative_class": 0,
            "positive_class": 1,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a global binary MLP classifier from Task3 Geneformer embeddings.")
    parser.add_argument("--embed_dir", type=Path, default=DEFAULT_EMBED_DIR)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--embedding_path", type=Path, default=None)
    parser.add_argument("--obs_path", type=Path, default=None)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--hidden_dims", type=str, default="512,128")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize_embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop_invalid_labels", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if args.embedding_path is None or args.obs_path is None:
        emb_path, obs_path = discover_training_embedding(args.embed_dir)
    else:
        emb_path, obs_path = args.embedding_path, args.obs_path

    base = EmbeddingDataset(emb_path, obs_path, args.label_col, drop_invalid_labels=args.drop_invalid_labels)
    train_pos, val_pos = split_indices(len(base), args.val_fraction, args.seed)
    train_idx = base.indices[train_pos]
    val_idx = base.indices[val_pos]
    train_raw = EmbeddingDataset(emb_path, obs_path, args.label_col, train_idx, drop_invalid_labels=args.drop_invalid_labels)
    val_raw = EmbeddingDataset(emb_path, obs_path, args.label_col, val_idx, drop_invalid_labels=args.drop_invalid_labels)
    train_labels = train_raw.labels[train_raw.indices].astype(int)
    val_labels = val_raw.labels[val_raw.indices].astype(int)
    print(
        f"{emb_path.name}: train={len(train_raw)} "
        f"(label0={(train_labels == 0).sum()} label1={(train_labels == 1).sum()}) "
        f"val={len(val_raw)} (label0={(val_labels == 0).sum()} label1={(val_labels == 1).sum()})"
    )

    if args.standardize_embeddings:
        mean, std = compute_scaler(train_raw)
    else:
        mean = np.zeros(train_raw.embedding_dim, dtype=np.float32)
        std = np.ones(train_raw.embedding_dim, dtype=np.float32)
    train_ds = clone_dataset_with_scaler(train_raw, mean, std)
    val_ds = clone_dataset_with_scaler(val_raw, mean, std)

    device = torch.device(args.device)
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    model = BinaryMLP(train_ds.embedding_dim, hidden_dims, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    ckpt_path = args.model_dir / "binary_mlp_global_best.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, device, args.threshold)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics}
        history.append(row)
        print(
            f"[global] epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"Accuracy={val_metrics['Accuracy']:.4f} F1={val_metrics['F1']:.4f}"
        )
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(ckpt_path, model, train_ds.embedding_dim, hidden_dims, args.dropout, mean, std, epoch, val_loss, val_metrics, args)
            print(f"  saved best checkpoint: {ckpt_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    pd.DataFrame(history).to_csv(args.model_dir / "binary_mlp_global_history.csv", index=False)
    run_meta = {
        "embedding_path": str(emb_path),
        "obs_path": str(obs_path),
        "model_dir": str(args.model_dir),
        "label_col": args.label_col,
        "strategy": "global",
        "task": "binary_classification",
        "negative_class": 0,
        "positive_class": 1,
        "val_fraction": args.val_fraction,
        "patience": args.patience,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "threshold": args.threshold,
        "standardize_embeddings": args.standardize_embeddings,
        "drop_invalid_labels": args.drop_invalid_labels,
    }
    (args.model_dir / "training_run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
