#!/usr/bin/env python3
"""
Train lightweight MLP regressors on Geneformer cell embeddings for age prediction.

By default this trains two checkpoints:
  1. global: random mini-batches sampled across all training parts
  2. per_part: iterate over training parts and shuffle batches within each part

Run after generating embeddings:
  conda activate geneformer
  python aging_scripts/2_train_age_mlp.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_EMBED_DIR = PROJECT_DIR / "outputs" / "geneformer_cell_embeddings"
DEFAULT_MODEL_DIR = PROJECT_DIR / "outputs" / "geneformer_age_mlp_models"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_train_parts(embed_dir: Path) -> List[Tuple[Path, Path]]:
    emb_files = sorted(embed_dir.glob("Task1_Training_Part*_embeddings.npy"))
    pairs = []
    for emb_path in emb_files:
        obs_path = embed_dir / emb_path.name.replace("_embeddings.npy", "_obs.csv")
        if not obs_path.exists():
            raise FileNotFoundError(f"Missing obs csv for {emb_path}: {obs_path}")
        pairs.append((emb_path, obs_path))
    if not pairs:
        raise FileNotFoundError(f"No Task1 training embeddings were found in {embed_dir}")
    return pairs


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
    ) -> None:
        self.emb_path = Path(emb_path)
        self.obs_path = Path(obs_path)
        self.label_col = label_col
        self.embeddings = np.load(self.emb_path, mmap_mode="r")
        self.obs = pd.read_csv(self.obs_path)
        if len(self.obs) != self.embeddings.shape[0]:
            raise ValueError(
                f"Row mismatch: {self.emb_path} has {self.embeddings.shape[0]} embeddings, "
                f"{self.obs_path} has {len(self.obs)} rows"
            )
        if label_col not in self.obs.columns:
            raise KeyError(f"Column '{label_col}' was not found in {self.obs_path}")
        labels = pd.to_numeric(self.obs[label_col], errors="coerce").to_numpy(dtype=np.float32)
        if np.isnan(labels).any():
            bad = int(np.isnan(labels).sum())
            raise ValueError(f"{bad} labels in '{label_col}' could not be converted to numeric ages.")
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64) if indices is not None else np.arange(len(labels), dtype=np.int64)
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


def make_splits(
    pairs: List[Tuple[Path, Path]],
    label_col: str,
    val_fraction: float,
    seed: int,
) -> Tuple[List[EmbeddingDataset], List[EmbeddingDataset]]:
    train_datasets = []
    val_datasets = []
    for part_id, (emb_path, obs_path) in enumerate(pairs):
        base = EmbeddingDataset(emb_path, obs_path, label_col)
        train_idx, val_idx = split_indices(len(base.labels), val_fraction, seed + part_id)
        train_datasets.append(EmbeddingDataset(emb_path, obs_path, label_col, train_idx))
        val_datasets.append(EmbeddingDataset(emb_path, obs_path, label_col, val_idx))
        print(f"{emb_path.name}: train={len(train_idx)} val={len(val_idx)}")
    return train_datasets, val_datasets


def compute_scaler(datasets: Sequence[EmbeddingDataset], chunk_size: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
    dim = datasets[0].embedding_dim
    total = 0
    sum_x = np.zeros(dim, dtype=np.float64)
    sum_x2 = np.zeros(dim, dtype=np.float64)
    for dataset in datasets:
        for start in tqdm(range(0, len(dataset.indices), chunk_size), desc=f"Scaler {dataset.emb_path.name}"):
            idx = dataset.indices[start : start + chunk_size]
            x = np.asarray(dataset.embeddings[idx], dtype=np.float32)
            sum_x += x.sum(axis=0, dtype=np.float64)
            sum_x2 += np.square(x, dtype=np.float64).sum(axis=0, dtype=np.float64)
            total += x.shape[0]
    mean = sum_x / max(total, 1)
    var = np.maximum(sum_x2 / max(total, 1) - mean**2, 1e-8)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


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


def parse_hidden_dims(value: str) -> List[int]:
    dims = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not dims:
        raise ValueError("--hidden_dims must contain at least one integer")
    return dims


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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_n = 0
    preds = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            total_loss += criterion(pred, y).item()
            total_n += y.numel()
            preds.append(pred.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)
    return total_loss / max(total_n, 1), regression_metrics(y_true, y_pred)


def train_one_epoch_global(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running = 0.0
    total = 0
    for x, y in tqdm(loader, desc="Train global", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        running += loss.item() * y.numel()
        total += y.numel()
    return running / max(total, 1)


def train_one_epoch_per_part(
    model: nn.Module,
    datasets: Sequence[EmbeddingDataset],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    epoch: int,
) -> float:
    model.train()
    running = 0.0
    total = 0
    order = list(range(len(datasets)))
    random.shuffle(order)
    for part_idx in order:
        generator = torch.Generator()
        generator.manual_seed(10_000 + epoch * 100 + part_idx)
        loader = DataLoader(
            datasets[part_idx],
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            generator=generator,
        )
        for x, y in tqdm(loader, desc=f"Train part {part_idx + 1}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += loss.item() * y.numel()
            total += y.numel()
    return running / max(total, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    strategy: str,
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
            "strategy": strategy,
            "input_dim": input_dim,
            "hidden_dims": list(hidden_dims),
            "dropout": dropout,
            "embedding_mean": torch.from_numpy(mean.astype(np.float32)),
            "embedding_std": torch.from_numpy(std.astype(np.float32)),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_metrics": metrics,
            "label_col": args.label_col,
        },
        path,
    )


def train_strategy(
    strategy: str,
    train_datasets_raw: Sequence[EmbeddingDataset],
    val_datasets_raw: Sequence[EmbeddingDataset],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    print(f"\nTraining strategy: {strategy}")
    if args.standardize_embeddings:
        mean, std = compute_scaler(train_datasets_raw)
    else:
        mean = np.zeros(train_datasets_raw[0].embedding_dim, dtype=np.float32)
        std = np.ones(train_datasets_raw[0].embedding_dim, dtype=np.float32)

    train_datasets = [clone_dataset_with_scaler(ds, mean, std) for ds in train_datasets_raw]
    val_datasets = [clone_dataset_with_scaler(ds, mean, std) for ds in val_datasets_raw]

    input_dim = train_datasets[0].embedding_dim
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    model = AgeMLP(input_dim, hidden_dims, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    val_loader = DataLoader(
        ConcatDataset(val_datasets),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if strategy == "global":
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        train_loader = DataLoader(
            ConcatDataset(train_datasets),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            generator=generator,
        )
    else:
        train_loader = None

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    ckpt_path = args.model_dir / f"age_mlp_{strategy}_best.pt"

    for epoch in range(1, args.epochs + 1):
        if strategy == "global":
            train_loss = train_one_epoch_global(model, train_loader, optimizer, criterion, device)
        elif strategy == "per_part":
            train_loss = train_one_epoch_per_part(
                model, train_datasets, optimizer, criterion, device, args.batch_size, args.num_workers, epoch
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        val_loss, val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics}
        history.append(row)
        print(
            f"[{strategy}] epoch {epoch:03d}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} MAE={val_metrics['MAE']:.4f} "
            f"RMSE={val_metrics['RMSE']:.4f} PCC={val_metrics['PCC']:.4f} R2={val_metrics['R2']:.4f}"
        )

        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                ckpt_path,
                model,
                strategy,
                input_dim,
                hidden_dims,
                args.dropout,
                mean,
                std,
                epoch,
                val_loss,
                val_metrics,
                args,
            )
            print(f"  saved best checkpoint: {ckpt_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    history_path = args.model_dir / f"age_mlp_{strategy}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved history: {history_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train age MLP regressors from Geneformer embeddings.")
    parser.add_argument("--embed_dir", type=Path, default=DEFAULT_EMBED_DIR)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--strategies", nargs="+", choices=("global", "per_part"), default=["global", "per_part"])
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
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize_embeddings", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    pairs = discover_train_parts(args.embed_dir)
    train_datasets, val_datasets = make_splits(pairs, args.label_col, args.val_fraction, args.seed)
    device = torch.device(args.device)
    for strategy in args.strategies:
        train_strategy(strategy, train_datasets, val_datasets, args, device)

    run_meta = {
        "embed_dir": str(args.embed_dir),
        "model_dir": str(args.model_dir),
        "label_col": args.label_col,
        "strategies": args.strategies,
        "val_fraction": args.val_fraction,
        "patience": args.patience,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "standardize_embeddings": args.standardize_embeddings,
    }
    (args.model_dir / "training_run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
