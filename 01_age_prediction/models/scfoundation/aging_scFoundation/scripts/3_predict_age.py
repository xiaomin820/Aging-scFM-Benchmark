from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TAG = "scFoundation"

BASE = Path.home() / "shared/zhujialin/task1_age_prediction/scfoundation"
EMB_DIR = BASE / "embeddings"
MODEL_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"

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


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def rmse_score(y, pred):
    return float(np.sqrt(mean_squared_error(y, pred)))


def load_test_embedding():
    npz_path = EMB_DIR / f"independent_test_{TAG}_cell_embeddings.npz"
    meta_path = EMB_DIR / f"independent_test_{TAG}_cell_embeddings_metadata.csv"

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    X = np.load(npz_path)["X"].astype(np.float32)
    meta = pd.read_csv(meta_path)

    print("test X:", X.shape)
    print("metadata:", meta.shape)

    return X, meta


def predict(model, X, device):
    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    preds = []
    model.eval()

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)

    return np.concatenate(preds)


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


def run_one(strategy, X, meta, device):
    ckpt_path = MODEL_DIR / f"{TAG}_{strategy}_MLP_age_predictor.pt"
    print("loading:", ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch_load(ckpt_path)

    mean = ckpt["mean"]
    std = ckpt["std"]
    X_std = ((X - mean) / std).astype(np.float32)

    model = AgeMLP(int(ckpt["input_dim"])).to(device)
    model.load_state_dict(ckpt["model_state"])

    pred = predict(model, X_std, device)

    out = meta.copy()
    out["predicted_age"] = pred

    pred_path = RESULTS_DIR / f"Task1_Independent_Test_{TAG}_{strategy}_MLP_age_predictions.csv"
    out.to_csv(pred_path, index=False)

    metrics = {}
    if "label" in out.columns:
        y = pd.to_numeric(out["label"], errors="coerce").values.astype(np.float32)
        good = np.isfinite(y) & np.isfinite(pred)
        if good.sum() > 1:
            metrics = calc_metrics(y[good], pred[good])

    if "metrics" in ckpt:
        metrics["best_epoch"] = ckpt["metrics"].get("best_epoch")
        metrics["best_val_mse"] = ckpt["metrics"].get("best_val_mse")

    metrics_path = RESULTS_DIR / f"Task1_Independent_Test_{TAG}_{strategy}_MLP_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(strategy, metrics)
    print("saved predictions:", pred_path)
    print("saved metrics:", metrics_path)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    X, meta = load_test_embedding()

    run_one("Global", X, meta, device)
    run_one("PerPart", X, meta, device)

    print("\nscFoundation prediction finished.")


if __name__ == "__main__":
    main()