#!/usr/bin/env python3
"""
Train a lightweight global-batching MLP binary classifier on Task3 scCello cell embeddings.

By default this trains only one model:
  global: random mini-batches across the Task3 training embedding file

Example:
  conda activate sccello
  python task3/scripts/2_train_task3_global_mlp.py --gpu_ids 0,1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[2]
TASK3_DIR = PROJECT_DIR / "task3"
DEFAULT_EMBED_DIR = TASK3_DIR / "outputs" / "sccello_embeddings"
DEFAULT_MODEL_DIR = TASK3_DIR / "outputs" / "models"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def discover_train_parts(embed_dir: Path) -> List[Tuple[Path, Path]]:
    emb_files = sorted(embed_dir.glob("Training_task3*_embeddings.npy"))
    pairs = []
    for emb_path in emb_files:
        obs_path = embed_dir / emb_path.name.replace("_embeddings.npy", "_obs.csv")
        if not obs_path.exists():
            raise FileNotFoundError(f"Missing obs csv for {emb_path}: {obs_path}")
        pairs.append((emb_path, obs_path))
    if not pairs:
        raise FileNotFoundError(f"No Task3 training embeddings were found in {embed_dir}")
    return pairs


def split_indices(n: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    val_idx = np.sort(order[:n_val])
    train_idx = np.sort(order[n_val:])
    return train_idx, val_idx


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


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        self.obs, self.obs_columns = read_csv_records(self.obs_path)
        if len(self.obs) != self.embeddings.shape[0]:
            raise ValueError(
                f"Row mismatch: {self.emb_path} has {self.embeddings.shape[0]} embeddings, "
                f"{self.obs_path} has {len(self.obs)} rows"
            )
        if label_col not in self.obs_columns:
            raise KeyError(f"Column '{label_col}' was not found in {self.obs_path}")
        self.labels = parse_binary_labels([row[label_col] for row in self.obs], label_col, allow_missing=True)
        valid_indices = np.nonzero(~np.isnan(self.labels))[0].astype(np.int64)
        if indices is None:
            self.indices = valid_indices
        else:
            requested = np.asarray(indices, dtype=np.int64)
            self.indices = requested[~np.isnan(self.labels[requested])]
        if len(self.indices) == 0:
            raise ValueError(f"No valid numeric labels were found in {self.obs_path} column '{label_col}'")
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
        y = float(self.labels[real_idx])
        return torch.tensor(x.tolist(), dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


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
        train_pos, val_pos = split_indices(len(base.indices), val_fraction, seed + part_id)
        train_idx = base.indices[train_pos]
        val_idx = base.indices[val_pos]
        train_datasets.append(EmbeddingDataset(emb_path, obs_path, label_col, train_idx))
        val_datasets.append(EmbeddingDataset(emb_path, obs_path, label_col, val_idx))
        skipped = len(base.labels) - len(base.indices)
        print(f"{emb_path.name}: train={len(train_idx)} val={len(val_idx)} skipped_missing_labels={skipped}")
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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_n = 0
    preds = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            total_n += y.numel()
            preds.extend(torch.sigmoid(logits).detach().cpu().tolist())
            labels.extend(y.detach().cpu().tolist())
    y_prob = np.asarray(preds, dtype=np.float32)
    y_true = np.asarray(labels, dtype=np.float32)
    return total_loss / max(total_n, 1), binary_metrics(y_true, y_prob)


def train_one_epoch_global(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
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
    model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "strategy": strategy,
            "input_dim": input_dim,
            "hidden_dims": list(hidden_dims),
            "dropout": dropout,
            "embedding_mean": torch.tensor(mean.astype(np.float32).tolist(), dtype=torch.float32),
            "embedding_std": torch.tensor(std.astype(np.float32).tolist(), dtype=torch.float32),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_metrics": metrics,
            "label_col": args.label_col,
            "training_args": vars(args),
        },
        path,
    )


def train_strategy(
    strategy: str,
    train_datasets_raw: Sequence[EmbeddingDataset],
    val_datasets_raw: Sequence[EmbeddingDataset],
    args: argparse.Namespace,
    device: torch.device,
    gpu_ids: Sequence[int],
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
    model = BinaryMLP(input_dim, hidden_dims, args.dropout).to(device)
    if len(gpu_ids) > 1:
        print(f"Using DataParallel on GPUs: {list(gpu_ids)}")
        model = nn.DataParallel(model, device_ids=list(gpu_ids), output_device=int(gpu_ids[0]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

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
    ckpt_path = args.model_dir / f"binary_mlp_{strategy}_best.pt"

    for epoch in range(1, args.epochs + 1):
        if strategy == "global":
            train_loss = train_one_epoch_global(model, train_loader, optimizer, criterion, device)
        elif strategy == "per_part":
            train_loss = train_one_epoch_per_part(model, train_datasets, optimizer, criterion, device, args.batch_size, args.num_workers, epoch)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        val_loss, val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics}
        history.append(row)
        print(
            f"[{strategy}] epoch {epoch:03d}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} Accuracy={val_metrics['Accuracy']:.4f} "
            f"F1={val_metrics['F1']:.4f}"
        )

        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(ckpt_path, model, strategy, input_dim, hidden_dims, args.dropout, mean, std, epoch, val_loss, val_metrics, args)
            print(f"  saved best checkpoint: {ckpt_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    history_path = args.model_dir / f"binary_mlp_{strategy}_history.csv"
    write_csv_rows(history_path, history)
    print(f"Saved history: {history_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Task3 global binary MLP classifier from scCello embeddings.")
    parser.add_argument("--embed_dir", type=Path, default=DEFAULT_EMBED_DIR)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--strategies", nargs="+", choices=("global",), default=["global"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated GPU ids, e.g. '0,1,2'. Overrides --device.")
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
    device, gpu_ids = resolve_device(args.device, args.gpu_ids)
    for strategy in args.strategies:
        train_strategy(strategy, train_datasets, val_datasets, args, device, gpu_ids)

    manifest_path = args.model_dir / "training_manifest.json"
    manifest = {
        "embed_dir": str(args.embed_dir),
        "model_dir": str(args.model_dir),
        "strategies": args.strategies,
        "label_col": args.label_col,
        "patience": args.patience,
        "val_fraction": args.val_fraction,
        "task": "binary_classification",
        "positive_label": 1,
        "negative_label": 0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
