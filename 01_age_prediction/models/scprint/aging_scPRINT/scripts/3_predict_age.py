from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


TAG = "scPRINT"
BATCH_SIZE = 256
DROPOUT = 0.2


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
    pcc = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "PCC": pcc, "R2": r2}


def load_test_embedding(emb_dir):
    npz_path = emb_dir / f"independent_test_{TAG}_cell_embeddings.npz"
    meta_path = emb_dir / f"independent_test_{TAG}_cell_embeddings_metadata.csv"

    data = np.load(npz_path, allow_pickle=True)
    key = "X" if "X" in data.files else data.files[0]
    X = np.asarray(data[key], dtype=np.float32)

    meta = pd.read_csv(meta_path)
    y = pd.to_numeric(meta["label"], errors="coerce").to_numpy(dtype=np.float32)

    mask = np.isfinite(y)
    X = X[mask]
    y = y[mask]
    meta = meta.loc[mask].reset_index(drop=True)

    return X, y, meta


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
            pred = model(xb.to(device)).cpu().numpy()
            preds.append(pred)

    return np.concatenate(preds)


def run_one(strategy, X, y, meta, model_dir, results_dir, device):
    ckpt_path = model_dir / f"{TAG}_{strategy}_MLP_age_predictor.pt"
    print("loading:", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    X_std = (X - ckpt["mean"]) / ckpt["std"]

    model = AgeMLP(int(ckpt["input_dim"])).to(device)
    model.load_state_dict(ckpt["model_state"])

    pred = predict(model, X_std, device)

    out = meta.copy()
    out["predicted_age"] = pred
    out["true_age"] = y
    out["error"] = out["predicted_age"] - out["true_age"]

    metrics = calc_metrics(y, pred)

    if "train_info" in ckpt:
        metrics.update(ckpt["train_info"])

    pred_path = results_dir / f"Task1_Independent_Test_{TAG}_{strategy}_MLP_age_predictions.csv"
    metrics_path = results_dir / f"Task1_Independent_Test_{TAG}_{strategy}_MLP_metrics.json"

    out.to_csv(pred_path, index=False)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(strategy, metrics)
    print("saved predictions:", pred_path)
    print("saved metrics:", metrics_path)


def main():
    root = Path(__file__).resolve().parents[1]

    emb_dir = root / "embeddings"
    model_dir = root / "models"
    results_dir = root / "results"

    results_dir.mkdir(parents=True, exist_ok=True)

    X, y, meta = load_test_embedding(emb_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    print("test X:", X.shape)

    run_one("Global", X, y, meta, model_dir, results_dir, device)
    run_one("PerPart", X, y, meta, model_dir, results_dir, device)

    print("scPRINT prediction finished.")


if __name__ == "__main__":
    main()
