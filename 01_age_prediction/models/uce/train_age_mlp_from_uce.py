#!/usr/bin/env python3
"""Train an MLP age regressor from UCE cell embeddings."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split

from age_mlp_utils import (
    DEFAULT_EMBEDDING_DIR,
    MLPRegressor,
    load_train_datasets,
    parse_hidden_dims,
)


DEFAULT_OUTPUT_DIR = Path("outputs/aging_age_mlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an age MLP on UCE cell embeddings.")
    parser.add_argument("--embedding_dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--batching", choices=["global", "per_part"], default="global")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset: Dataset, val_fraction: float, seed: int) -> tuple[Subset, Subset]:
    val_size = max(1, int(round(len(dataset) * val_fraction)))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("Validation split leaves no training samples")
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def make_loaders(
    train_sets: list[Dataset],
    args: argparse.Namespace,
) -> tuple[list[DataLoader], DataLoader, int]:
    train_subsets = []
    val_subsets = []
    for dataset in train_sets:
        train_subset, val_subset = split_dataset(dataset, args.val_fraction, args.seed)
        train_subsets.append(train_subset)
        val_subsets.append(val_subset)

    if args.batching == "global":
        train_loaders = [
            DataLoader(
                ConcatDataset(train_subsets),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=args.device.startswith("cuda"),
            )
        ]
    else:
        train_loaders = [
            DataLoader(
                subset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=args.device.startswith("cuda"),
            )
            for subset in train_subsets
        ]

    val_loader = DataLoader(
        ConcatDataset(val_subsets),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
    )
    input_dim = train_sets[0].x.shape[1]
    return train_loaders, val_loader, input_dim


def run_epoch(
    model: nn.Module,
    loaders: list[DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for loader in loaders:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
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
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.shape[0]
        total_n += x.shape[0]
    return total_loss / total_n


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    train_sets = load_train_datasets(args.embedding_dir, args.label_col)
    train_loaders, val_loader, input_dim = make_loaders(train_sets, args)

    device = torch.device(args.device)
    model = MLPRegressor(input_dim=input_dim, hidden_dims=hidden_dims, dropout=args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_path = args.output_dir / "best_age_mlp.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loaders, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        print(f"epoch={epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dims": hidden_dims,
                    "dropout": args.dropout,
                    "best_epoch": best_epoch,
                    "best_val_mse": best_val_loss,
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
        json.dump({"best_epoch": best_epoch, "best_val_mse": best_val_loss}, f, indent=2)

    print(f"Wrote training outputs to {args.output_dir}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
