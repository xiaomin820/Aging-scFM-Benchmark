#!/usr/bin/env python3
"""Train a global-batching binary MLP classifier for Task3 senescent-cell labels."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING_DIR = REPO_ROOT / "task3/outputs/uce_embeddings"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "task3/outputs/binary_mlp_global"


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
        labels = labels[valid]
        if labels.size == 0:
            raise ValueError(f"No numeric labels found in {h5ad_path}:{label_col}")
        unique_labels = set(np.unique(labels).tolist())
        if not unique_labels <= {0.0, 1.0}:
            raise ValueError(f"Expected binary labels 0/1 in {h5ad_path}, found {sorted(unique_labels)}")

        embeddings = np.asarray(adata.obsm["X_uce"], dtype=np.float32)
        self.x = torch.from_numpy(embeddings[valid])
        self.y = torch.from_numpy(labels).view(-1, 1)

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
    parser = argparse.ArgumentParser(description="Train Task3 binary MLP from UCE cell embeddings.")
    parser.add_argument("--embedding_dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--train_embedding", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--hidden_dims", type=str, default="512,128")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_hidden_dims(hidden_dims: str) -> list[int]:
    return [int(x.strip()) for x in hidden_dims.split(",") if x.strip()]


def find_train_embedding_path(embedding_dir: Path) -> Path:
    paths = sorted(embedding_dir.glob("Training_task3*_UCE_input_uce_adata.h5ad"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected 1 Task3 training embedding file, found {len(paths)} in {embedding_dir}")
    return paths[0]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_val(dataset: Dataset, val_fraction: float, seed: int):
    val_size = max(1, int(round(len(dataset) * val_fraction)))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("Validation split leaves no training samples")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.shape[0]
        total_n += x.shape[0]
    return total_loss / total_n


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.shape[0]
        total_n += x.shape[0]
    return total_loss / total_n


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_embedding = args.train_embedding or find_train_embedding_path(args.embedding_dir)
    dataset = UCEBinaryDataset(train_embedding, label_col=args.label_col)
    train_subset, val_subset = split_train_val(dataset, args.val_fraction, args.seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
    )

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    device = torch.device(args.device)
    model = MLPBinaryClassifier(input_dim=dataset.x.shape[1], hidden_dims=hidden_dims, dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_path = args.output_dir / "best_binary_mlp.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_bce": train_loss, "val_bce": val_loss})
        print(f"epoch={epoch:03d} train_bce={train_loss:.6f} val_bce={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": dataset.x.shape[1],
                    "hidden_dims": hidden_dims,
                    "dropout": args.dropout,
                    "best_epoch": best_epoch,
                    "best_val_bce": best_val_loss,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
                break

    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    with open(args.output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_bce": best_val_loss,
                "n_train": len(train_subset),
                "n_val": len(val_subset),
                "train_embedding": str(train_embedding),
            },
            f,
            indent=2,
        )

    print(f"Wrote training outputs to {args.output_dir}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
