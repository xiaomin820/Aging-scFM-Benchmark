from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TRAIN_PARTS = ["train_part1", "train_part2", "train_part3", "train_part4", "train_part5"]
TEST_NAME = "independent_test"

LR = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 5
VAL_RATIO = 0.1
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
GRAD_CLIP = 5.0
SEED = 2026


def load_one(emb_dir, name):
    npz_path = emb_dir / f"{name}_scGPT_cell_embeddings.npz"
    meta_path = emb_dir / f"{name}_scGPT_cell_embeddings_metadata.csv"

    print(f"loading: {name}")
    print("  npz:", npz_path)
    print("  meta:", meta_path)

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype("float32")
    meta = pd.read_csv(meta_path)
    y = pd.to_numeric(meta["label"], errors="coerce").values.astype("float32")

    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    meta = meta.loc[mask].reset_index(drop=True)

    print("  X:", X.shape, "y:", y.shape)
    return X, y, meta


def split_train_val(X, y, meta, rng):
    idx = rng.permutation(len(X))
    val_size = int(len(X) * VAL_RATIO)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return (
        X[train_idx], y[train_idx], meta.iloc[train_idx].reset_index(drop=True),
        X[val_idx], y[val_idx], meta.iloc[val_idx].reset_index(drop=True),
    )


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    pcc = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"MAE": float(mae), "RMSE": float(rmse), "PCC": float(pcc), "R2": float(r2)}


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
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


def per_part_batches(parts, batch_size, rng, device):
    n_parts = len(parts)
    base = batch_size // n_parts
    sizes = [base] * n_parts
    for i in range(batch_size - base * n_parts):
        sizes[i] += 1

    total_n = sum(len(p["X"]) for p in parts)
    steps = math.ceil(total_n / batch_size)

    for _ in range(steps):
        bx_list, by_list = [], []
        for p, size in zip(parts, sizes):
            idx = rng.integers(0, len(p["X"]), size=size)
            bx_list.append(p["X"][idx])
            by_list.append(p["y"][idx])

        bx = np.vstack(bx_list)
        by = np.concatenate(by_list)
        order = rng.permutation(len(by))

        yield torch.tensor(bx[order]).to(device), torch.tensor(by[order]).to(device)


def train_one(strategy):
    print(f"\n===== strategy: {strategy} =====")

    base = Path.home() / "shared/zhujialin/task1_age_prediction/scgpt"
    emb_dir = base / "embeddings"
    model_dir = base / "models"
    result_dir = base / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    train_parts = []
    val_X_list, val_y_list = [], []

    for part in TRAIN_PARTS:
        X, y, meta = load_one(emb_dir, part)
        X_tr, y_tr, meta_tr, X_val, y_val, meta_val = split_train_val(X, y, meta, rng)
        train_parts.append({"name": part, "X": X_tr, "y": y_tr, "meta": meta_tr})
        val_X_list.append(X_val)
        val_y_list.append(y_val)

    X_test, y_test, meta_test = load_one(emb_dir, TEST_NAME)

    X_train_all = np.vstack([p["X"] for p in train_parts])
    y_train_all = np.concatenate([p["y"] for p in train_parts])
    X_val = np.vstack(val_X_list)
    y_val = np.concatenate(val_y_list)

    mean = X_train_all.mean(axis=0, keepdims=True)
    std = X_train_all.std(axis=0, keepdims=True) + 1e-6

    X_train_all = (X_train_all - mean) / std
    X_val = (X_val - mean) / std
    X_test_scaled = (X_test - mean) / std

    for p in train_parts:
        p["X"] = (p["X"] - mean) / std

    print("combined train X:", X_train_all.shape)
    print("validation X:", X_val.shape)
    print("test X:", X_test_scaled.shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = MLP(X_train_all.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    val_x = torch.tensor(X_val).to(device)
    val_y = torch.tensor(y_val).to(device)

    if strategy == "global":
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train_all), torch.tensor(y_train_all)),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []

        if strategy == "global":
            batch_iter = train_loader
        elif strategy == "per_part":
            batch_iter = per_part_batches(train_parts, BATCH_SIZE, np.random.default_rng(SEED + epoch), device)
        else:
            raise ValueError(strategy)

        for batch in batch_iter:
            if strategy == "global":
                bx, by = batch
                bx = bx.to(device)
                by = by.to(device)
            else:
                bx, by = batch

            pred = model(bx)
            loss = loss_fn(pred, by)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_mse = loss_fn(val_pred, val_y).item()

        train_mse = float(np.mean(losses))
        print(f"[{strategy}] epoch {epoch:03d} train_mse={train_mse:.4f} val_mse={val_mse:.4f}")

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"[{strategy}] early stopping")
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_pred = model(torch.tensor(X_test_scaled).to(device)).cpu().numpy()

    metrics = regression_metrics(y_test, test_pred)
    print(f"[{strategy}] test metrics:", metrics)

    tag = "Global" if strategy == "global" else "PerPart"
    model_path = model_dir / f"scGPT_{tag}_MLP_age_predictor.pt"
    pred_path = result_dir / f"Task1_Independent_Test_scGPT_{tag}_MLP_age_predictions.csv"
    metrics_path = result_dir / f"Task1_Independent_Test_scGPT_{tag}_MLP_metrics.json"

    pred_df = meta_test.copy()
    pred_df["predicted_age"] = test_pred
    pred_df["true_age"] = y_test
    pred_df["error"] = pred_df["predicted_age"] - pred_df["true_age"]
    pred_df.to_csv(pred_path, index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": X_train_all.shape[1],
            "mean": mean.astype("float32"),
            "std": std.astype("float32"),
            "train_parts": TRAIN_PARTS,
            "strategy": strategy,
            "params": {
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "val_ratio": VAL_RATIO,
                "hidden_dims": [512, 128],
                "dropout": DROPOUT,
                "optimizer": "AdamW",
                "weight_decay": WEIGHT_DECAY,
                "loss": "MSELoss",
                "grad_clip_max_norm": GRAD_CLIP,
            },
            "metrics": metrics,
        },
        model_path,
    )

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{strategy}] saved model:", model_path)
    print(f"[{strategy}] saved predictions:", pred_path)
    print(f"[{strategy}] saved metrics:", metrics_path)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_one("global")
    train_one("per_part")

    print("\nscGPT global and per_part MLP age prediction OK")


if __name__ == "__main__":
    main()
