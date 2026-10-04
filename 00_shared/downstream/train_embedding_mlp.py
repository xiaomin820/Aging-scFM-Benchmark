from __future__ import annotations

import argparse
import copy
import math
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import (
    EmbeddingMLP,
    atomic_json,
    classification_metrics,
    fit_standardizer,
    load_pair,
    parse_hidden_dims,
    regression_metrics,
    set_reproducibility,
    split_grouped_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the common embedding MLP.")
    parser.add_argument("--task", choices=("regression", "classification"), required=True)
    parser.add_argument("--embedding", type=Path, action="append", required=True)
    parser.add_argument("--metadata", type=Path, action="append")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--group-column", default="donorID")
    parser.add_argument("--stratify-column", default="cell_type")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dims", default="512,128")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame]:
    metadata_paths = args.metadata or [None] * len(args.embedding)
    if len(metadata_paths) != len(args.embedding):
        raise ValueError("Provide one --metadata for each --embedding, or none for embedded metadata")
    matrices: list[np.ndarray] = []
    frames: list[pd.DataFrame] = []
    for embedding_path, metadata_path in zip(args.embedding, metadata_paths):
        values, metadata = load_pair(embedding_path, metadata_path)
        metadata = metadata.copy()
        metadata["source_embedding"] = str(embedding_path)
        metadata["source_row"] = np.arange(len(metadata))
        matrices.append(values)
        frames.append(metadata)
    dimensions = {matrix.shape[1] for matrix in matrices}
    if len(dimensions) != 1:
        raise ValueError(f"Embedding dimensions differ across inputs: {sorted(dimensions)}")
    return np.concatenate(matrices), pd.concat(frames, ignore_index=True)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_reproducibility(args.seed)

    values, metadata = load_inputs(args)
    if args.label_column not in metadata.columns:
        raise KeyError(f"Missing label column: {args.label_column}")
    labels = pd.to_numeric(metadata[args.label_column], errors="coerce").to_numpy(dtype=np.float32)
    valid = np.isfinite(labels)
    if args.task == "classification":
        valid &= np.isin(labels, [0.0, 1.0])
    if not valid.all():
        values = values[valid]
        metadata = metadata.loc[valid].reset_index(drop=True)
        labels = labels[valid]
    if args.task == "classification" and set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError("Classification labels must contain both 0 and 1")

    missing_split_columns = [
        column
        for column in (args.group_column, args.stratify_column)
        if column not in metadata.columns
    ]
    if missing_split_columns:
        raise KeyError(
            "Missing metadata columns required for donor-disjoint splitting: "
            + ", ".join(missing_split_columns)
        )
    train_indices, validation_indices = split_grouped_indices(
        labels,
        metadata[args.group_column].to_numpy(),
        metadata[args.stratify_column].to_numpy(),
        args.task,
        args.validation_fraction,
        args.seed,
    )
    mean, std = fit_standardizer(values[train_indices])
    standardized = ((values - mean) / std).astype(np.float32)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    model = EmbeddingMLP(standardized.shape[1], hidden_dims, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function: nn.Module = (
        nn.MSELoss() if args.task == "regression" else nn.BCEWithLogitsLoss()
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(standardized[train_indices]),
            torch.from_numpy(labels[train_indices]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_x = torch.from_numpy(standardized[validation_indices]).to(device)
    validation_y = torch.from_numpy(labels[validation_indices]).to(device)

    best_loss = math.inf
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x)
            loss = loss_function(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch_x)
            count += len(batch_x)
        model.eval()
        with torch.no_grad():
            validation_output = model(validation_x)
            validation_loss = float(loss_function(validation_output, validation_y).item())
        improved = validation_loss < best_loss
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / count,
                "validation_loss": validation_loss,
                "improved": improved,
            }
        )
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        output = model(validation_x).cpu().numpy()
    if args.task == "classification":
        probability = torch.sigmoid(torch.from_numpy(output)).numpy()
        validation_metrics = classification_metrics(
            labels[validation_indices], probability, args.threshold
        )
    else:
        validation_metrics = regression_metrics(labels[validation_indices], output)

    checkpoint = {
        "format_version": "1.0",
        "task": args.task,
        "input_dim": int(standardized.shape[1]),
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
        "model_state_dict": best_state,
        "embedding_mean": torch.from_numpy(mean),
        "embedding_std": torch.from_numpy(std),
        "label_column": args.label_column,
        "threshold": args.threshold,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "validation_metrics": validation_metrics,
        "training_config": {
            "validation_fraction": args.validation_fraction,
            "split_unit": args.group_column,
            "composition_stratum": args.stratify_column,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "maximum_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "gradient_clip_max_norm": args.gradient_clip,
            "seed": args.seed,
            "loss": type(loss_function).__name__,
        },
    }
    checkpoint_path = args.output_dir / "best_model.pt"
    temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, checkpoint_path)
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    split = metadata.copy()
    split["split"] = ""
    split.loc[train_indices, "split"] = "train"
    split.loc[validation_indices, "split"] = "validation"
    split.to_csv(args.output_dir / "training_split.csv", index=False)
    atomic_json(
        args.output_dir / "run_summary.json",
        {
            "task": args.task,
            "observations": int(len(labels)),
            "input_dim": int(standardized.shape[1]),
            "train_observations": int(len(train_indices)),
            "validation_observations": int(len(validation_indices)),
            "train_groups": int(metadata.iloc[train_indices][args.group_column].nunique()),
            "validation_groups": int(
                metadata.iloc[validation_indices][args.group_column].nunique()
            ),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "validation_metrics": validation_metrics,
            "checkpoint": str(checkpoint_path.resolve()),
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
            },
        },
    )
    print(f"Saved {checkpoint_path}")
    print(validation_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
