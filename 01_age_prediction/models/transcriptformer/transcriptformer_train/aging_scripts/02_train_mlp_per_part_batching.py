"""
Step 2: Train MLP for Age Prediction - Per-Part Balanced Batching Strategy

Each training batch draws examples from every training part, so batches are
balanced across the five source files instead of sampled globally.
"""

import os
import time
import json
import logging
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
EMBEDDING_DIR = "./embedding_results"
OUTPUT_DIR = "./output_mlp_per_part"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "mlp_model_per_part.pt")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "test_predictions_per_part.csv")

HIDDEN_DIMS = [512, 256, 128]
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 5

TRAIN_FILES = [
    "cell_embedding_Task1_Training_Part1_n50000_TranscriptFormer_input.parquet",
    "cell_embedding_Task1_Training_Part2_n50000_TranscriptFormer_input.parquet",
    "cell_embedding_Task1_Training_Part3_n50000_TranscriptFormer_input.parquet",
    "cell_embedding_Task1_Training_Part4_n50000_TranscriptFormer_input.parquet",
    "cell_embedding_Task1_Training_Part5_n40000_TranscriptFormer_input.parquet",
]
TEST_FILE = "cell_embedding_Task1_Independent.Test_GSE134355_n32000_TranscriptFormer_input.parquet"


class AgeMLP(nn.Module):
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
    def __init__(self, embeddings, ages):
        self.embeddings = torch.FloatTensor(embeddings)
        self.ages = torch.FloatTensor(ages)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.ages[idx]


class PerPartBatchSampler:
    """Yield batches with roughly equal samples from each training part."""

    def __init__(self, part_indices, batch_size, seed=42, drop_last=False):
        self.part_indices = [np.array(indices) for indices in part_indices if len(indices) > 0]
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        if not self.part_indices:
            raise ValueError("No non-empty training parts found")

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        shuffled_parts = [rng.permutation(indices) for indices in self.part_indices]
        positions = np.zeros(len(shuffled_parts), dtype=int)
        base = self.batch_size // len(shuffled_parts)
        remainder = self.batch_size % len(shuffled_parts)
        per_part_sizes = np.array([base + (i < remainder) for i in range(len(shuffled_parts))])

        while True:
            batch = []
            progressed = False
            for part_id, take in enumerate(per_part_sizes):
                if take == 0:
                    continue
                start = positions[part_id]
                end = min(start + take, len(shuffled_parts[part_id]))
                if start < end:
                    batch.extend(shuffled_parts[part_id][start:end].tolist())
                    positions[part_id] = end
                    progressed = True
            if not progressed:
                break
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                rng.shuffle(batch)
                yield batch

    def __len__(self):
        max_part_len = max(len(indices) for indices in self.part_indices)
        per_part = max(1, math.ceil(self.batch_size / len(self.part_indices)))
        return math.ceil(max_part_len / per_part)


def load_embedding_file(path, part_name=None):
    df = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].values
    y = df["age"].values
    if part_name is not None:
        logger.info(f"{part_name}: {len(X)} samples, age range {y.min():.2f}-{y.max():.2f}")
    return X, y, df


def load_train_parts():
    X_parts = []
    y_parts = []
    for i, filename in enumerate(TRAIN_FILES, start=1):
        path = os.path.join(EMBEDDING_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training embedding not found: {path}")
        X, y, _ = load_embedding_file(path, part_name=f"Part {i}")
        X_parts.append(X)
        y_parts.append(y)
    return X_parts, y_parts


def load_test_data():
    path = os.path.join(EMBEDDING_DIR, TEST_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test embeddings not found: {path}")
    return load_embedding_file(path)


def split_parts(X_parts, y_parts):
    train_X_parts = []
    train_y_parts = []
    val_X_parts = []
    val_y_parts = []
    for X, y in zip(X_parts, y_parts, strict=True):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=RANDOM_SEED
        )
        train_X_parts.append(X_train)
        train_y_parts.append(y_train)
        val_X_parts.append(X_val)
        val_y_parts.append(y_val)
    return train_X_parts, train_y_parts, val_X_parts, val_y_parts


def concat_parts(X_parts, y_parts):
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    part_indices = []
    offset = 0
    for X_part in X_parts:
        indices = np.arange(offset, offset + len(X_part))
        part_indices.append(indices)
        offset += len(X_part)
    return X, y, part_indices


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pcc, _ = pearsonr(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return {"MAE": mae, "RMSE": rmse, "PCC": pcc, "R2": r2}


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()
            n_batches += 1
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    return total_loss / n_batches, np.array(all_preds), np.array(all_targets)


def train_model(X_train, y_train, part_indices, X_val, y_val, device, input_dim):
    logger.info("Training MLP with per-part balanced batching strategy")
    logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}")

    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)
    train_sampler = PerPartBatchSampler(part_indices, BATCH_SIZE, seed=RANDOM_SEED)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AgeMLP(input_dim, HIDDEN_DIMS, DROPOUT).to(device)
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    history = []

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device)
        metrics = calculate_metrics(val_targets, val_preds)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch + 1}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val MAE: {metrics['MAE']:.4f} | "
            f"Val PCC: {metrics['PCC']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": metrics["MAE"],
                "val_rmse": metrics["RMSE"],
                "val_pcc": metrics["PCC"],
                "val_r2": metrics["R2"],
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
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

    model.load_state_dict(best_model_state)
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "training_history_per_part.csv"), index=False)
    return model, history


def predict(model, X, device, batch_size=512):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(device)
            predictions.extend(model(batch).cpu().numpy())
    return np.array(predictions)


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    X_parts, y_parts = load_train_parts()
    X_test, y_test, _ = load_test_data()
    input_dim = X_parts[0].shape[1]

    train_X_parts, train_y_parts, val_X_parts, val_y_parts = split_parts(X_parts, y_parts)
    X_train, y_train, train_part_indices = concat_parts(train_X_parts, train_y_parts)
    X_val = np.concatenate(val_X_parts, axis=0)
    y_val = np.concatenate(val_y_parts, axis=0)

    model, history = train_model(X_train, y_train, train_part_indices, X_val, y_val, device, input_dim)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": HIDDEN_DIMS,
            "dropout": DROPOUT,
            "training_config": {
                "strategy": "per_part_balanced_batching",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
            },
        },
        MODEL_SAVE_PATH,
    )
    logger.info(f"Model saved to {MODEL_SAVE_PATH}")

    logger.info("=" * 60)
    logger.info("Evaluating on independent test set...")
    test_predictions = predict(model, X_test, device)
    test_metrics = calculate_metrics(y_test, test_predictions)
    logger.info("Test Set Metrics (Per-Part Batching):")
    logger.info(f"  MAE:  {test_metrics['MAE']:.4f}")
    logger.info(f"  RMSE: {test_metrics['RMSE']:.4f}")
    logger.info(f"  PCC:  {test_metrics['PCC']:.4f}")
    logger.info(f"  R^2:   {test_metrics['R2']:.4f}")

    with open(os.path.join(OUTPUT_DIR, "test_metrics_per_part.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    pd.DataFrame(
        {
            "predicted_age": test_predictions,
            "true_age": y_test,
        }
    ).to_csv(PREDICTIONS_PATH, index=True)
    logger.info(f"Predictions saved to {PREDICTIONS_PATH}")
    logger.info("Training Complete!")


if __name__ == "__main__":
    main()
