"""Shared utilities for age regression from UCE cell embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_EMBEDDING_DIR = Path("outputs/aging_uce_embeddings")


@dataclass
class Metrics:
    mae: float
    rmse: float
    pcc: float
    r2: float


class EmbeddingDataset(Dataset):
    def __init__(self, h5ad_path: Path, label_col: str = "label") -> None:
        self.h5ad_path = h5ad_path
        adata = ad.read_h5ad(h5ad_path)
        if "X_uce" not in adata.obsm:
            raise KeyError(f"{h5ad_path} does not contain adata.obsm['X_uce']")
        if label_col not in adata.obs:
            raise KeyError(f"{h5ad_path} does not contain adata.obs['{label_col}']")

        labels = pd.to_numeric(adata.obs[label_col], errors="coerce").to_numpy(np.float32)
        valid = np.isfinite(labels)
        if valid.sum() == 0:
            raise ValueError(f"No numeric labels found in {h5ad_path}:{label_col}")

        embeddings = np.asarray(adata.obsm["X_uce"], dtype=np.float32)
        self.x = torch.from_numpy(embeddings[valid])
        self.y = torch.from_numpy(labels[valid]).view(-1, 1)
        self.cell_ids = np.asarray(adata.obs_names)[valid]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_hidden_dims(hidden_dims: str) -> list[int]:
    return [int(x.strip()) for x in hidden_dims.split(",") if x.strip()]


def find_train_embedding_paths(embedding_dir: Path) -> list[Path]:
    train_paths = sorted(embedding_dir.glob("Task1_Training_Part*_UCE_input_uce_adata.h5ad"))
    if len(train_paths) != 5:
        raise FileNotFoundError(f"Expected 5 training embedding files, found {len(train_paths)} in {embedding_dir}")
    return train_paths


def find_test_embedding_path(embedding_dir: Path) -> Path:
    test_paths = sorted(embedding_dir.glob("Task1_Independent.Test*_UCE_input_uce_adata.h5ad"))
    if len(test_paths) != 1:
        raise FileNotFoundError(f"Expected 1 independent test embedding file, found {len(test_paths)} in {embedding_dir}")
    return test_paths[0]


def load_train_datasets(embedding_dir: Path, label_col: str) -> list[EmbeddingDataset]:
    return [EmbeddingDataset(path, label_col=label_col) for path in find_train_embedding_paths(embedding_dir)]


def load_test_dataset(embedding_dir: Path, label_col: str) -> EmbeddingDataset:
    return EmbeddingDataset(find_test_embedding_path(embedding_dir), label_col=label_col)


@torch.no_grad()
def predict(model: nn.Module, dataset: Dataset, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    preds = []
    labels = []
    for x, y in loader:
        pred = model(x.to(device)).cpu().numpy().reshape(-1)
        preds.append(pred)
        labels.append(y.numpy().reshape(-1))
    return np.concatenate(labels), np.concatenate(preds)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    pcc = pearsonr(y_true, y_pred).statistic if len(y_true) > 1 else float("nan")
    abs_err = np.abs(y_true - y_pred)
    sq_err = np.square(y_true - y_pred)
    ss_res = float(np.sum(sq_err))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return Metrics(
        mae=float(np.mean(abs_err)),
        rmse=float(math.sqrt(float(np.mean(sq_err)))),
        pcc=float(pcc),
        r2=float(r2),
    )


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[MLPRegressor, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MLPRegressor(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=checkpoint["hidden_dims"],
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
