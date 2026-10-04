"""
Predict Task3 senescent-cell labels on Independent.Test embedding files.

Example:
    python task3/scripts/03_predict_binary_task3.py --gpu 2
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_EMBEDDING_DIR = "./task3/embedding_results"
DEFAULT_MODEL_PATH = "./task3/output_binary_mlp_global/mlp_binary_global.pt"
DEFAULT_OUTPUT_DIR = "./task3/output_binary_mlp_global"
TRAIN_PREFIX = "Training_task3"


class BinaryMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout=0.3):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 128]
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Task3 binary senescent labels.")
    parser.add_argument("--embedding_dir", default=DEFAULT_EMBEDDING_DIR, help="Task3 embedding parquet directory")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="Trained binary MLP checkpoint")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--gpu", default="0", help="GPU ID for prediction")
    parser.add_argument("--label_col", default="label", help="Binary label column")
    parser.add_argument("--batch_size", type=int, default=1024, help="Prediction batch size")
    parser.add_argument("--threshold", type=float, default=None, help="Probability threshold. Defaults to checkpoint value or 0.5")
    parser.add_argument(
        "--files",
        default="test",
        help="Files to predict: test or comma-separated embedding parquet basenames",
    )
    return parser.parse_args()


def embedding_columns(df):
    cols = [c for c in df.columns if c.startswith("emb_")]
    if not cols:
        raise ValueError("No emb_* columns found")
    return cols


def resolve_embedding_files(embedding_dir, files_arg):
    embedding_dir = Path(embedding_dir)
    all_files = sorted(embedding_dir.glob("*_embeddings.parquet"))
    test_files = [p for p in all_files if not p.name.startswith(TRAIN_PREFIX)]

    if files_arg is None or files_arg.strip().lower() == "test":
        files = test_files
    else:
        files = [embedding_dir / item.strip() for item in files_arg.split(",") if item.strip()]

    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing embedding file(s): {missing}")
    if not files:
        raise FileNotFoundError(f"No embedding files selected from {embedding_dir}")
    return files


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = BinaryMLP(checkpoint["input_dim"], checkpoint["hidden_dims"], checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    threshold = checkpoint.get("training_config", {}).get("threshold", 0.5)
    return model, checkpoint, threshold


def load_embedding_table(path, label_col):
    df = pd.read_parquet(path)
    emb_cols = embedding_columns(df)
    X = df[emb_cols].values.astype(np.float32)
    meta_cols = [c for c in df.columns if not c.startswith("emb_")]
    meta_df = df[meta_cols].copy()
    y = None
    if label_col in df.columns:
        y = pd.to_numeric(df[label_col], errors="coerce").values.astype(np.float32)
    return X, y, meta_df


def predict_probabilities(model, X, device, batch_size):
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(device)
            logits = model(batch)
            probs.extend(torch.sigmoid(logits).cpu().numpy())
    return np.array(probs)


def calculate_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(int)
    if len(y_true) == 0:
        return None
    return {
        "n_cells": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate_true": float(np.mean(y_true == 1)),
        "positive_rate_pred": float(np.mean(y_pred == 1)),
    }


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    output_dir = Path(args.output_dir)
    prediction_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model, checkpoint, checkpoint_threshold = load_model(args.model_path, device)
    threshold = args.threshold if args.threshold is not None else checkpoint_threshold
    logger.info(f"Loaded model: {args.model_path}")
    logger.info(f"Prediction threshold: {threshold}")

    embedding_files = resolve_embedding_files(args.embedding_dir, args.files)
    metrics_rows = []
    all_predictions = []

    for path in embedding_files:
        logger.info(f"Predicting {path.name}")
        X, y, meta_df = load_embedding_table(path, args.label_col)
        probs = predict_probabilities(model, X, device, args.batch_size)
        pred_labels = (probs >= threshold).astype(int)

        pred_df = meta_df.copy()
        pred_df["probability_label1_senescent"] = probs
        pred_df["predicted_label"] = pred_labels
        if y is not None:
            pred_df["true_label_numeric"] = y
            metrics = calculate_metrics(y, pred_labels)
            if metrics is not None:
                metrics_rows.append({"test_file": path.name, **metrics})

        output_path = prediction_dir / f"{path.stem}_binary_predictions.csv"
        pred_df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions: {output_path}")
        all_predictions.append(pred_df)

    if all_predictions:
        combined = pd.concat(all_predictions, ignore_index=True)
        combined.to_csv(output_dir / "all_test_binary_predictions.csv", index=False)
        logger.info(f"Saved combined predictions: {output_dir / 'all_test_binary_predictions.csv'}")

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(output_dir / "test_binary_metrics_summary.csv", index=False)
        save_json(metrics_rows, output_dir / "test_binary_metrics_summary.json")
        logger.info("\nTask3 Binary Test Metrics Summary")
        logger.info(metrics_df.to_string(index=False))

    logger.info("Task3 binary prediction complete.")


if __name__ == "__main__":
    main()
