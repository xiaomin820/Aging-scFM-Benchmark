from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

TAG = "scFoundation"

BASE = Path.home() / "shared/zhujialin/task1_age_prediction/scfoundation"
EMB_DIR = BASE / "embeddings"
MODEL_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"

LR = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 5
VAL_RATIO = 0.1
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
GRAD_CLIP = 5.0
SEED = 42


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


def rmse_score(y, pred):
    return float(np.sqrt(mean_squared_error(y, pred)))


def load_part(name):
    npz_path = EMB_DIR / f"{name}_{TAG}_cell_embeddings.npz"
    meta_path = EMB_DIR / f"{name}_{TAG}_cell_embeddings_metadata.csv"

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    X = np.load(npz_path)["X"].astype(np.float32)
    meta = pd.read_csv(meta_path)
    y = pd.to_numeric(meta["label"], errors="coerce").values.astype(np.float32)

    good = np.isfinite(y)
    X = X[good]
    y = y[good]
    meta = meta.loc[good].reset_index(drop=True)

    print(f"loaded {name}: X={X.shape}, y={y.shape}")
    return X, y, meta


def load_train_parts():
    return [load_part(f"train_part{i}") for i in range(1, 6)]


def split_tensor_dataset(X, y):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    n_val = max(1, int(len(ds) * VAL_RATIO))
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(SEED)
    return random_split(ds, [n_train, n_val], generator=gen)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    ys = []
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_loss += loss.item() * len(xb)
            n += len(xb)
            preds.append(pred.cpu().numpy())
            ys.append(yb.cpu().numpy())

    return total_loss / max(n, 1), np.concatenate(preds), np.concatenate(ys)


def calc_metrics(y, pred):
    if np.std(y) == 0 or np.std(pred) == 0:
        pcc = float("nan")
    else:
        pcc = pearsonr(y, pred)[0]

    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": rmse_score(y, pred),
        "PCC": float(pcc),
        "R2": float(r2_score(y, pred)),
    }


def train_one(strategy, train_loaders, val_loader, input_dim, mean, std, device):
    model = AgeMLP(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = 0
    bad = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n = 0

        for loader in train_loaders:
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                train_loss += loss.item() * len(xb)
                n += len(xb)

        val_mse, val_pred, val_y = evaluate(model, val_loader, device)
        print(f"{strategy} epoch {epoch:03d}: train_mse={train_loss / n:.4f}, val_mse={val_mse:.4f}")

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"{strategy} early stopping.")
                break

    model.load_state_dict(best_state)
    _, val_pred, val_y = evaluate(model, val_loader, device)
    metrics = calc_metrics(val_y, val_pred)
    metrics["best_epoch"] = int(best_epoch)
    metrics["best_val_mse"] = float(best_val)

    ckpt_path = MODEL_DIR / f"{TAG}_{strategy}_MLP_age_predictor.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": int(input_dim),
            "mean": mean,
            "std": std,
            "metrics": metrics,
            "tag": TAG,
            "strategy": strategy,
        },
        ckpt_path,
    )

    print("saved:", ckpt_path)
    print(f"{strategy} metrics:", metrics)


def train_global(parts, device):
    print("\n=== Train Global MLP ===")
    X_all = np.concatenate([p[0] for p in parts], axis=0)
    y_all = np.concatenate([p[1] for p in parts], axis=0)

    mean = X_all.mean(axis=0, keepdims=True).astype(np.float32)
    std = X_all.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0

    X_all = ((X_all - mean) / std).astype(np.float32)

    train_ds, val_ds = split_tensor_dataset(X_all, y_all)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    train_one("Global", [train_loader], val_loader, X_all.shape[1], mean, std, device)


def train_per_part(parts, device):
    print("\n=== Train PerPart MLP ===")
    X_all_raw = np.concatenate([p[0] for p in parts], axis=0)

    mean = X_all_raw.mean(axis=0, keepdims=True).astype(np.float32)
    std = X_all_raw.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0

    train_loaders = []
    val_sets = []

    for i, (X, y, _) in enumerate(parts, start=1):
        X = ((X - mean) / std).astype(np.float32)
        train_ds, val_ds = split_tensor_dataset(X, y)
        train_loaders.append(DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True))
        val_sets.append(val_ds)
        print(f"part{i}: train={len(train_ds)}, val={len(val_ds)}")

    val_ds = torch.utils.data.ConcatDataset(val_sets)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    train_one("PerPart", train_loaders, val_loader, X_all_raw.shape[1], mean, std, device)


def main():
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    parts = load_train_parts()
    train_global(parts, device)
    train_per_part(parts, device)

    print("\nscFoundation MLP training finished.")


if __name__ == "__main__":
    main()