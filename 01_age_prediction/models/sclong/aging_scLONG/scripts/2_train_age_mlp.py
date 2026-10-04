from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TAG = "scLONG"

LR = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 5
VAL_RATIO = 0.1
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
GRAD_CLIP = 5.0
SEED = 2026

TRAIN_PARTS = [f"train_part{i}" for i in range(1, 6)]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AgeMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_training_embedding(emb_dir, name):
    npz_path = emb_dir / f"{name}_{TAG}_cell_embeddings.npz"
    meta_path = emb_dir / f"{name}_{TAG}_cell_embeddings_metadata.csv"

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    with np.load(npz_path, allow_pickle=True) as data:
        key = "X" if "X" in data.files else data.files[0]
        X = np.asarray(data[key], dtype=np.float32)

    meta = pd.read_csv(meta_path)

    if "label" not in meta.columns:
        raise ValueError(f"{meta_path} is missing the label column")

    y = pd.to_numeric(
        meta["label"],
        errors="coerce",
    ).to_numpy(dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(
            f"{name}: embedding must be a two-dimensional array; observed shape: {X.shape}"
        )

    if X.shape[0] != len(meta):
        raise ValueError(
            f"{name}: embedding rows {X.shape[0]} "
            f"do not match metadata rows {len(meta)} (mismatch)"
        )

    valid_mask = np.isfinite(y)
    X = X[valid_mask]
    y = y[valid_mask]

    if len(y) == 0:
        raise ValueError(f"{name}: No valid age labels")

    if not np.isfinite(X).all():
        raise ValueError(f"{name}: embedding contains NaN or Inf")

    print(
        f"loaded {name}: "
        f"X={X.shape}, y={y.shape}, "
        f"age_range=({y.min():.1f}, {y.max():.1f})"
    )

    return X, y


def split_train_val(X, y):
    n = len(y)

    if n < 2:
        raise ValueError("Training data must contain at least two samples")

    indices = np.random.permutation(n)
    n_val = max(1, int(n * VAL_RATIO))

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
    )


def make_loader(X, y, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
    )


def train_loop(
    model,
    train_loaders,
    val_loader,
    device,
    strategy,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []

        if strategy == "global":
            for xb, yb in train_loaders[0]:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad(set_to_none=True)

                pred = model(xb)
                loss = loss_fn(pred, yb)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                optimizer.step()
                train_losses.append(loss.item())

        elif strategy == "per_part":
            iterators = [
                iter(loader)
                for loader in train_loaders
            ]

            steps = max(
                len(loader)
                for loader in train_loaders
            )

            for _ in range(steps):
                for i, loader in enumerate(train_loaders):
                    try:
                        xb, yb = next(iterators[i])
                    except StopIteration:
                        iterators[i] = iter(loader)
                        xb, yb = next(iterators[i])

                    xb = xb.to(device)
                    yb = yb.to(device)

                    optimizer.zero_grad(set_to_none=True)

                    pred = model(xb)
                    loss = loss_fn(pred, yb)

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        GRAD_CLIP,
                    )

                    optimizer.step()
                    train_losses.append(loss.item())

        else:
            raise ValueError(
                f"Unknown training strategy: {strategy}"
            )

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = loss_fn(pred, yb)

                val_losses.append(loss.item())

        train_mse = float(np.mean(train_losses))
        val_mse = float(np.mean(val_losses))

        print(
            f"[{strategy}] epoch {epoch:03d} "
            f"train_mse={train_mse:.4f} "
            f"val_mse={val_mse:.4f}"
        )

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            bad_epochs = 0

        else:
            bad_epochs += 1

            if bad_epochs >= PATIENCE:
                print(
                    f"[{strategy}] early stopping "
                    f"at epoch {epoch}"
                )
                break

    if best_state is None:
        raise RuntimeError(
            f"{strategy}: No valid model was obtained"
        )

    model.load_state_dict(best_state)

    return {
        "best_epoch": int(best_epoch),
        "best_val_mse": float(best_val),
    }


def save_checkpoint(
    path,
    model,
    input_dim,
    mean,
    std,
    strategy_name,
    train_info,
):
    checkpoint = {
        "model_state": model.state_dict(),
        "input_dim": int(input_dim),
        "mean": mean,
        "std": std,
        "tag": TAG,
        "strategy": strategy_name,
        "train_info": train_info,
        "training_config": {
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "validation_ratio": VAL_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "gradient_clip": GRAD_CLIP,
            "seed": SEED,
            "hidden_layers": [512, 128],
            "activation": "ReLU",
            "normalization": "LayerNorm",
            "optimizer": "AdamW",
            "loss": "MSELoss",
        },
    }

    torch.save(checkpoint, path)
    print(f"[{strategy_name}] saved model: {path}")


def main():
    set_seed()

    root = Path(__file__).resolve().parents[1]
    emb_dir = root / "embeddings"
    model_dir = root / "models"

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("loading training embeddings...")

    train_sets = [
        load_training_embedding(emb_dir, name)
        for name in TRAIN_PARTS
    ]

    input_dim = train_sets[0][0].shape[1]

    for name, (X, _) in zip(
        TRAIN_PARTS,
        train_sets,
    ):
        if X.shape[1] != input_dim:
            raise ValueError(
                f"{name}: embedding dimensions "
                f"{X.shape[1]} != {input_dim}"
            )

    X_all = np.vstack([
        X
        for X, _ in train_sets
    ]).astype(np.float32)

    y_all = np.concatenate([
        y
        for _, y in train_sets
    ]).astype(np.float32)

    # Compute standardization statistics from training data only.
    mean = X_all.mean(
        axis=0,
        keepdims=True,
    ).astype(np.float32)

    std = X_all.std(
        axis=0,
        keepdims=True,
    ).astype(np.float32)

    std[std < 1e-6] = 1.0

    X_all_std = (
        (X_all - mean) / std
    ).astype(np.float32)

    part_std = [
        (
            ((X - mean) / std).astype(np.float32),
            y,
        )
        for X, y in train_sets
    ]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)
    print("input_dim:", input_dim)
    print("combined train:", X_all_std.shape)
    print("training labels:", y_all.shape)
    print("independent_test is not loaded by this script")

    # Global training
    X_train, y_train, X_val, y_val = (
        split_train_val(
            X_all_std,
            y_all,
        )
    )

    print(
        "[Global] train:",
        X_train.shape,
        "validation:",
        X_val.shape,
    )

    global_model = AgeMLP(
        input_dim
    ).to(device)

    global_info = train_loop(
        global_model,
        [
            make_loader(
                X_train,
                y_train,
                shuffle=True,
            )
        ],
        make_loader(
            X_val,
            y_val,
            shuffle=False,
        ),
        device,
        "global",
    )

    global_path = (
        model_dir
        / f"{TAG}_Global_MLP_age_predictor.pt"
    )

    save_checkpoint(
        global_path,
        global_model,
        input_dim,
        mean,
        std,
        "Global",
        global_info,
    )

    # Per-part training
    train_loaders = []
    val_xs = []
    val_ys = []

    for part_name, (X, y) in zip(
        TRAIN_PARTS,
        part_std,
    ):
        X_train, y_train, X_val, y_val = (
            split_train_val(X, y)
        )

        print(
            f"[PerPart] {part_name}: "
            f"train={X_train.shape}, "
            f"validation={X_val.shape}"
        )

        train_loaders.append(
            make_loader(
                X_train,
                y_train,
                shuffle=True,
            )
        )

        val_xs.append(X_val)
        val_ys.append(y_val)

    per_part_val_loader = make_loader(
        np.vstack(val_xs).astype(np.float32),
        np.concatenate(val_ys).astype(np.float32),
        shuffle=False,
    )

    per_part_model = AgeMLP(
        input_dim
    ).to(device)

    per_part_info = train_loop(
        per_part_model,
        train_loaders,
        per_part_val_loader,
        device,
        "per_part",
    )

    per_part_path = (
        model_dir
        / f"{TAG}_PerPart_MLP_age_predictor.pt"
    )

    save_checkpoint(
        per_part_path,
        per_part_model,
        input_dim,
        mean,
        std,
        "PerPart",
        per_part_info,
    )

    summary = {
        "tag": TAG,
        "input_dim": int(input_dim),
        "training_parts": TRAIN_PARTS,
        "number_of_training_cells": int(len(y_all)),
        "independent_test_used": False,
        "global": global_info,
        "per_part": per_part_info,
    }

    summary_path = (
        model_dir
        / f"{TAG}_MLP_training_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("saved training summary:", summary_path)
    print("scLONG Global and PerPart MLP training finished.")
    print("No independent-test prediction was performed.")


if __name__ == "__main__":
    main()