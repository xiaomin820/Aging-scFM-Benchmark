"""
Step 2: Train MLP for Age Prediction - Global Batching Strategy
Train an age-prediction MLP with global random sampling
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
RANDOM_SEED = 42
EMBEDDING_DIR = "./embedding_results"
OUTPUT_DIR = "./output_mlp_global"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "mlp_model_global.pt")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "test_predictions_global.csv")

# MLP Architecture
HIDDEN_DIMS = [512, 256, 128]
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 5


class AgeMLP(nn.Module):
    """Lightweight MLP for age prediction"""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super(AgeMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class EmbeddingDataset(Dataset):
    """Dataset for embedding and age pairs"""
    def __init__(self, embeddings, ages):
        self.embeddings = torch.FloatTensor(embeddings)
        self.ages = torch.FloatTensor(ages)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.ages[idx]


def load_train_data():
    """Load all training embeddings"""
    logger.info("Loading training embeddings...")

    train_file = os.path.join(EMBEDDING_DIR, "train_all_embeddings.parquet")
    if not os.path.exists(train_file):
        # Check for individual files
        train_files = [
            os.path.join(EMBEDDING_DIR, f"cell_embedding_Task1_Training_Part{i}_n50000_TranscriptFormer_input.parquet")
            for i in range(1, 5)
        ]
        train_files.append(os.path.join(EMBEDDING_DIR, "cell_embedding_Task1_Training_Part5_n40000_TranscriptFormer_input.parquet"))

        dfs = []
        for f in train_files:
            if os.path.exists(f):
                df = pd.read_parquet(f)
                dfs.append(df)

        if dfs:
            train_df = pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError(f"Training embeddings not found at {train_file}")
    else:
        train_df = pd.read_parquet(train_file)

    # Extract embeddings and labels
    emb_cols = [c for c in train_df.columns if c.startswith("emb_")]
    X = train_df[emb_cols].values
    y = train_df["age"].values

    logger.info(f"Loaded {len(X)} training samples with {X.shape[1]}-dim embeddings")
    logger.info(f"Age range: {y.min():.2f} - {y.max():.2f}")

    return X, y


def load_test_data():
    """Load test embeddings"""
    logger.info("Loading test embeddings...")

    test_file = os.path.join(
        EMBEDDING_DIR,
        "cell_embedding_Task1_Independent.Test_GSE134355_n32000_TranscriptFormer_input.parquet"
    )

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test embeddings not found at {test_file}")

    test_df = pd.read_parquet(test_file)

    emb_cols = [c for c in test_df.columns if c.startswith("emb_")]
    X_test = test_df[emb_cols].values
    y_test = test_df["age"].values if "age" in test_df.columns else None

    logger.info(f"Loaded {len(X_test)} test samples")

    return X_test, y_test, test_df


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pcc, p_value = pearsonr(y_true, y_pred)

    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "PCC": pcc,
        "R2": r2
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
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
    """Validate the model"""
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

    val_loss = total_loss / n_batches
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    return val_loss, all_preds, all_targets


def train_model(X_train, y_train, X_val, y_val, device, input_dim):
    """Train the MLP model with early stopping"""
    logger.info(f"Training MLP with global batching strategy")
    logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}")

    # Create datasets
    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = AgeMLP(input_dim, HIDDEN_DIMS, DROPOUT).to(device)
    logger.info(f"Model architecture:\n{model}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = []

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device)

        # Calculate metrics
        metrics = calculate_metrics(val_targets, val_preds)

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch+1}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val MAE: {metrics['MAE']:.4f} | "
            f"Val PCC: {metrics['PCC']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": metrics["MAE"],
            "val_rmse": metrics["RMSE"],
            "val_pcc": metrics["PCC"],
            "val_r2": metrics["R2"]
        })

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.info(f"  -> New best model! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    model.load_state_dict(best_model_state)

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(OUTPUT_DIR, "training_history_global.csv"), index=False)

    return model, history


def predict(model, X, device, batch_size=512):
    """Generate predictions"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            pred = model(batch).cpu().numpy()
            predictions.extend(pred)

    return np.array(predictions)


def main():
    # Setup
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    X, y = load_train_data()
    X_test, y_test, test_df = load_test_data()

    input_dim = X.shape[1]

    # Split training data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_SEED
    )
    logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")

    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val, device, input_dim)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dims': HIDDEN_DIMS,
        'dropout': DROPOUT,
        'training_config': {
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'batch_size': BATCH_SIZE,
            'max_epochs': MAX_EPOCHS,
            'patience': PATIENCE
        }
    }, MODEL_SAVE_PATH)
    logger.info(f"Model saved to {MODEL_SAVE_PATH}")

    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating on independent test set...")
    test_predictions = predict(model, X_test, device)

    if y_test is not None:
        test_metrics = calculate_metrics(y_test, test_predictions)
        logger.info("\nTest Set Metrics (Global Batching):")
        logger.info(f"  MAE:  {test_metrics['MAE']:.4f}")
        logger.info(f"  RMSE: {test_metrics['RMSE']:.4f}")
        logger.info(f"  PCC:  {test_metrics['PCC']:.4f}")
        logger.info(f"  R^2:   {test_metrics['R2']:.4f}")

        # Save metrics
        metrics_file = os.path.join(OUTPUT_DIR, "test_metrics_global.json")
        with open(metrics_file, 'w') as f:
            json.dump(test_metrics, f, indent=2)

    # Save predictions
    results_df = pd.DataFrame({
        'predicted_age': test_predictions,
        'true_age': y_test if y_test is not None else [None] * len(test_predictions)
    })
    results_df.to_csv(PREDICTIONS_PATH, index=True)
    logger.info(f"Predictions saved to {PREDICTIONS_PATH}")

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()