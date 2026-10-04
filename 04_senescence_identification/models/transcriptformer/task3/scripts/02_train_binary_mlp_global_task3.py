"""
Train a global-sampling binary MLP for Task3 senescent-cell classification.

Run after generating embeddings with:
    python task3/scripts/01_generate_task3_embeddings.py --gpu 2 --batch_size 8

Example:
    python task3/scripts/02_train_binary_mlp_global_task3.py --gpu 2
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


RANDOM_SEED = 42
DEFAULT_EMBEDDING_DIR = "./task3/embedding_results"
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


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Task3 global binary MLP.")
    parser.add_argument("--embedding_dir", default=DEFAULT_EMBEDDING_DIR, help="Task3 embedding parquet directory")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for model/history")
    parser.add_argument("--gpu", default="0", help="GPU ID for MLP training")
    parser.add_argument("--label_col", default="label", help="Binary label column")
    parser.add_argument("--batch_size", type=int, default=256, help="MLP training batch size")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument("--dropout", type=float, default=0.3, help="MLP dropout")
    parser.add_argument("--hidden_dims", default="512,256,128", help="Comma-separated hidden layer sizes")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for positive class")
    return parser.parse_args()


def parse_hidden_dims(value):
    dims = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not dims:
        raise ValueError("--hidden_dims must contain at least one dimension")
    return dims


def train_embedding_file(embedding_dir):
    matches = sorted(Path(embedding_dir).glob(f"{TRAIN_PREFIX}*_embeddings.parquet"))
    if not matches:
        raise FileNotFoundError(f"No training embedding parquet found in {embedding_dir}")
    if len(matches) > 1:
        logger.warning(f"Multiple training embedding files found; using {matches[0]}")
    return matches[0]


def embedding_columns(df):
    cols = [c for c in df.columns if c.startswith("emb_")]
    if not cols:
        raise ValueError("No emb_* columns found")
    return cols


def load_training_data(path, label_col):
    df = pd.read_parquet(path)
    emb_cols = embedding_columns(df)
    labels = pd.to_numeric(df[label_col], errors="coerce") if label_col in df.columns else None
    if labels is None:
        raise ValueError(f"Label column not found: {label_col}")

    X = df[emb_cols].values.astype(np.float32)
    y = labels.values.astype(np.float32)
    finite_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    binary_mask = np.isin(y, [0.0, 1.0])
    keep_mask = finite_mask & binary_mask

    n_removed = int((~keep_mask).sum())
    if n_removed:
        logger.warning(
            "Removing %d rows with invalid labels/embeddings "
            "(non-finite: %d, non-binary labels: %d)",
            n_removed,
            int((~finite_mask).sum()),
            int((finite_mask & ~binary_mask).sum()),
        )

    X = X[keep_mask]
    y = y[keep_mask]
    if len(y) == 0:
        raise ValueError("No valid binary training labels remain after filtering")
    return X, y


def calculate_metrics(y_true, logits, threshold):
    probs = torch.sigmoid(torch.as_tensor(logits)).numpy()
    preds = (probs >= threshold).astype(int)
    y_true = y_true.astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "positive_rate_true": float(np.mean(y_true == 1)),
        "positive_rate_pred": float(np.mean(preds == 1)),
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_loader(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item()
            n_batches += 1
            logits_all.extend(logits.cpu().numpy())
            labels_all.extend(batch_y.cpu().numpy())
    return total_loss / max(n_batches, 1), np.array(logits_all), np.array(labels_all)


def train_model(X_train, y_train, X_val, y_val, input_dim, hidden_dims, args, device):
    train_loader = DataLoader(EmbeddingDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

    model = BinaryMLP(input_dim, hidden_dims, args.dropout).to(device)
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    history = []

    for epoch in range(args.max_epochs):
        start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_logits, val_labels = evaluate_loader(model, val_loader, criterion, device)
        metrics = calculate_metrics(val_labels, val_logits, args.threshold)
        elapsed = time.time() - start

        logger.info(
            f"Epoch {epoch + 1}/{args.max_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {metrics['accuracy']:.4f} | "
            f"Val F1: {metrics['f1']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": metrics["accuracy"],
                "val_f1": metrics["f1"],
                "val_positive_rate_true": metrics["positive_rate_true"],
                "val_positive_rate_pred": metrics["positive_rate_pred"],
            }
        )
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            logger.info(f"  -> New best model! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

    model.load_state_dict(best_model_state)
    return model, history


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_path = train_embedding_file(args.embedding_dir)
    logger.info(f"Loading training embeddings: {train_path}")
    X, y = load_training_data(train_path, args.label_col)
    logger.info(f"Training data: {X.shape[0]} cells, {X.shape[1]} embedding dims")
    logger.info(f"Label counts: normal/0={(y == 0).sum()}, senescent/1={(y == 1).sum()}")

    stratify = y if len(np.unique(y)) == 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=stratify,
    )
    logger.info(f"Train split: {len(X_train)}; validation split: {len(X_val)}")

    model, history = train_model(X_train, y_train, X_val, y_val, X.shape[1], hidden_dims, args, device)
    pd.DataFrame(history).to_csv(output_dir / "training_history_binary_global.csv", index=False)

    model_path = output_dir / "mlp_binary_global.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": X.shape[1],
            "hidden_dims": hidden_dims,
            "dropout": args.dropout,
            "training_config": {
                "task": "task3_senescent_binary_classification",
                "strategy": "global_random_sampling",
                "training_file": train_path.name,
                "label_col": args.label_col,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "val_size": args.val_size,
                "threshold": args.threshold,
            },
        },
        model_path,
    )
    save_json(history[-1] if history else {}, output_dir / "last_validation_metrics.json")
    logger.info(f"Saved model: {model_path}")
    logger.info("Task3 binary MLP training complete.")


if __name__ == "__main__":
    main()
