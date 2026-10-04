from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TAG = "SCimilarity"

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
TEST_NAME = "independent_test"


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


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    if len(y_true) > 1:
        pcc = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pcc = float("nan")

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "PCC": pcc,
        "R2": r2,
    }


def load_embedding(emb_dir, name):
    npz_path = emb_dir / f"{name}_{TAG}_cell_embeddings.npz"
    meta_path = emb_dir / f"{name}_{TAG}_cell_embeddings_metadata.csv"

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    data = np.load(npz_path, allow_pickle=True)
    key = "X" if "X" in data.files else data.files[0]
    X = np.asarray(data[key], dtype=np.float32)

    meta = pd.read_csv(meta_path)
    y = pd.to_numeric(meta["label"], errors="coerce").to_numpy(dtype=np.float32)

    mask = np.isfinite(y)
    X = X[mask]
    y = y[mask]
    meta = meta.loc[mask].reset_index(drop=True)

    print(f"loaded {name}: X={X.shape}, y={y.shape}")
    return X, y, meta


def split_train_val(X, y):
    n = len(y)
    idx = np.random.permutation(n)
    n_val = max(1, int(n * VAL_RATIO))

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def make_loader(X, y, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)


def train_loop(model, train_loaders, val_loader, device, strategy):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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

                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                train_losses.append(loss.item())

        elif strategy == "per_part":
            iters = [iter(loader) for loader in train_loaders]
            steps = max(len(loader) for loader in train_loaders)

            for _ in range(steps):
                for i, loader in enumerate(train_loaders):
                    try:
                        xb, yb = next(iters[i])
                    except StopIteration:
                        iters[i] = iter(loader)
                        xb, yb = next(iters[i])

                    xb = xb.to(device)
                    yb = yb.to(device)

                    optimizer.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()

                    train_losses.append(loss.item())
        else:
            raise ValueError(strategy)

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_loss = loss_fn(model(xb), yb)
                val_losses.append(val_loss.item())

        train_mse = float(np.mean(train_losses))
        val_mse = float(np.mean(val_losses))

        print(
            f"[{strategy}] epoch {epoch:03d} "
            f"train_mse={train_mse:.4f} val_mse={val_mse:.4f}"
        )

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"[{strategy}] early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
    }


def predict(model, X, device):
    model.eval()
    preds = []

    loader = DataLoader(
        torch.from_numpy(X).float(),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)

    return np.concatenate(preds)


def save_test_outputs(strategy_name, model, train_info, X_test, y_test, meta_test, device, results_dir):
    pred = predict(model, X_test, device)

    out = meta_test.copy()
    out["predicted_age"] = pred
    out["true_age"] = y_test
    out["error"] = out["predicted_age"] - out["true_age"]

    metrics = calc_metrics(y_test, pred)
    metrics.update(train_info)

    pred_path = results_dir / f"Task1_Independent_Test_{TAG}_{strategy_name}_MLP_age_predictions.csv"
    metrics_path = results_dir / f"Task1_Independent_Test_{TAG}_{strategy_name}_MLP_metrics.json"

    out.to_csv(pred_path, index=False)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{strategy_name}] metrics:", metrics)
    print(f"[{strategy_name}] saved predictions:", pred_path)
    print(f"[{strategy_name}] saved metrics:", metrics_path)


def main():
    set_seed()

    root = Path(__file__).resolve().parents[1]
    emb_dir = root / "embeddings"
    model_dir = root / "models"
    results_dir = root / "results"

    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_sets = [load_embedding(emb_dir, name) for name in TRAIN_PARTS]
    X_test, y_test, meta_test = load_embedding(emb_dir, TEST_NAME)

    input_dim = train_sets[0][0].shape[1]
    assert all(X.shape[1] == input_dim for X, _, _ in train_sets)
    assert X_test.shape[1] == input_dim

    X_all = np.vstack([X for X, _, _ in train_sets])
    y_all = np.concatenate([y for _, y, _ in train_sets])

    mean = X_all.mean(axis=0, keepdims=True).astype(np.float32)
    std = X_all.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0

    X_all_std = (X_all - mean) / std
    X_test_std = (X_test - mean) / std
    part_std = [((X - mean) / std, y, meta) for X, y, meta in train_sets]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("input_dim:", input_dim)
    print("combined train:", X_all_std.shape)
    print("test:", X_test_std.shape)

    Xtr, ytr, Xv, yv = split_train_val(X_all_std, y_all)

    global_model = AgeMLP(input_dim).to(device)
    global_info = train_loop(
        global_model,
        [make_loader(Xtr, ytr, shuffle=True)],
        make_loader(Xv, yv, shuffle=False),
        device,
        "global",
    )

    torch.save(
        {
            "model_state": global_model.state_dict(),
            "input_dim": input_dim,
            "mean": mean,
            "std": std,
            "tag": TAG,
            "strategy": "Global",
            "train_info": global_info,
        },
        model_dir / f"{TAG}_Global_MLP_age_predictor.pt",
    )

    save_test_outputs(
        "Global",
        global_model,
        global_info,
        X_test_std,
        y_test,
        meta_test,
        device,
        results_dir,
    )

    train_loaders = []
    val_xs = []
    val_ys = []

    for X, y, _ in part_std:
        Xtr, ytr, Xv, yv = split_train_val(X, y)
        train_loaders.append(make_loader(Xtr, ytr, shuffle=True))
        val_xs.append(Xv)
        val_ys.append(yv)

    val_loader = make_loader(
        np.vstack(val_xs),
        np.concatenate(val_ys),
        shuffle=False,
    )

    per_part_model = AgeMLP(input_dim).to(device)
    per_part_info = train_loop(
        per_part_model,
        train_loaders,
        val_loader,
        device,
        "per_part",
    )

    torch.save(
        {
            "model_state": per_part_model.state_dict(),
            "input_dim": input_dim,
            "mean": mean,
            "std": std,
            "tag": TAG,
            "strategy": "PerPart",
            "train_info": per_part_info,
        },
        model_dir / f"{TAG}_PerPart_MLP_age_predictor.pt",
    )

    save_test_outputs(
        "PerPart",
        per_part_model,
        per_part_info,
        X_test_std,
        y_test,
        meta_test,
        device,
        results_dir,
    )

    print("SCimilarity global and per-part MLP training finished.")


if __name__ == "__main__":
    main()