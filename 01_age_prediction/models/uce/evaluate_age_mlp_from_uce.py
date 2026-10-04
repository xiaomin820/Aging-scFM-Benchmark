#!/usr/bin/env python3
"""Evaluate a trained MLP age regressor on the independent UCE test embeddings."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from age_mlp_utils import (
    DEFAULT_EMBEDDING_DIR,
    compute_metrics,
    load_model_from_checkpoint,
    load_test_dataset,
    predict,
)


DEFAULT_CHECKPOINT = Path("outputs/aging_age_mlp/best_age_mlp.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/aging_age_mlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an age MLP on Task1 independent UCE embeddings.")
    parser.add_argument("--embedding_dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    test_set = load_test_dataset(args.embedding_dir, args.label_col)
    y_true, y_pred = predict(model, test_set, args.batch_size, device)
    metrics = compute_metrics(y_true, y_pred)

    pd.DataFrame(
        {
            "cell_id": test_set.cell_ids,
            "label": y_true,
            "prediction": y_pred,
        }
    ).to_csv(args.output_dir / "independent_test_predictions.csv", index=False)

    metrics_payload = {
        **asdict(metrics),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_mse": checkpoint.get("best_val_mse"),
        "checkpoint": str(args.checkpoint),
    }
    with open(args.output_dir / "independent_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(json.dumps(metrics_payload, indent=2))
    print(f"Wrote evaluation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
