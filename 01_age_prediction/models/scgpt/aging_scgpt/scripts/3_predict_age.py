from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn

DROPOUT = 0.2
TEST_NAME = "independent_test"
STRATEGIES = ["Global", "PerPart"]


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


def load_test(emb_dir):
    npz_path = emb_dir / f"{TEST_NAME}_scGPT_cell_embeddings.npz"
    meta_path = emb_dir / f"{TEST_NAME}_scGPT_cell_embeddings_metadata.csv"

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype("float32")
    meta = pd.read_csv(meta_path)
    y = pd.to_numeric(meta["label"], errors="coerce").values.astype("float32")

    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    meta = meta.loc[mask].reset_index(drop=True)

    print("test X:", X.shape)
    print("test y:", y.shape)
    return X, y, meta


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


def predict_one(strategy, X_test, y_test, meta_test, model_dir, result_dir, device):
    model_path = model_dir / f"scGPT_{strategy}_MLP_age_predictor.pt"
    pred_path = result_dir / f"Task1_Independent_Test_scGPT_{strategy}_MLP_age_predictions.csv"
    metrics_path = result_dir / f"Task1_Independent_Test_scGPT_{strategy}_MLP_metrics.json"

    print(f"\n=== predict {strategy} ===")
    print("model:", model_path)

    ckpt = torch.load(model_path, map_location=device)
    mean = ckpt["mean"]
    std = ckpt["std"]

    X_scaled = (X_test - mean) / std

    model = MLP(ckpt["input_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(X_scaled).to(device)).cpu().numpy()

    metrics = regression_metrics(y_test, pred)
    print("metrics:", metrics)

    pred_df = meta_test.copy()
    pred_df["predicted_age"] = pred
    pred_df["true_age"] = y_test
    pred_df["error"] = pred_df["predicted_age"] - pred_df["true_age"]
    pred_df.to_csv(pred_path, index=False)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("saved predictions:", pred_path)
    print("saved metrics:", metrics_path)


def main():
    base = Path.home() / "shared/zhujialin/task1_age_prediction/scgpt"
    emb_dir = base / "embeddings"
    model_dir = base / "models"
    result_dir = base / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    X_test, y_test, meta_test = load_test(emb_dir)

    for strategy in STRATEGIES:
        predict_one(strategy, X_test, y_test, meta_test, model_dir, result_dir, device)

    print("\nscGPT age prediction OK")


if __name__ == "__main__":
    main()
