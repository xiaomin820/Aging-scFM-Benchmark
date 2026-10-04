#!/usr/bin/env python3
"""
Generate CellFM cell embeddings and train a binary MLP for Task3 senescent-cell classification.

Run in the CellFM conda environment:
  conda activate CellFM
  python task3/scripts/1_generate_embeddings_and_train_mlp.py --gpu_ids 0 --batch_size_embed 16
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
PROJECT_DIR = TASK3_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import csr_matrix, issparse
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from layers.utils import Config_80M
from model import Cell_FM


DEFAULT_DATA_DIR = TASK3_DIR / "data" / "task3_Senescent_CellFM_input"
DEFAULT_EMBED_DIR = TASK3_DIR / "outputs" / "cell_embeddings"
DEFAULT_MODEL_DIR = TASK3_DIR / "outputs" / "models"
DEFAULT_CKPT = PROJECT_DIR / "checkpoint" / "CellFM_80M_weight.ckpt"
GENE_INFO = PROJECT_DIR / "csv" / "expand_gene_info.csv"
HGNC_INFO = PROJECT_DIR / "csv" / "updated_hgcn.tsv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(gpu_ids: Optional[str], device: str) -> Tuple[torch.device, List[int]]:
    if gpu_ids:
        ids = [int(item.strip()) for item in gpu_ids.split(",") if item.strip()]
        if not ids:
            raise ValueError("--gpu_ids was provided but no valid GPU ids were parsed.")
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu_ids requires CUDA, but CUDA is not available.")
        return torch.device(f"cuda:{ids[0]}"), ids
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is not None:
        return resolved, [resolved.index]
    return resolved, []


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def discover_h5ad_files(data_dir: Path) -> List[Path]:
    train_files = sorted(data_dir.glob("Training*_CellFM_input.h5ad"))
    test_files = sorted(data_dir.glob("Independent.Test*_CellFM_input.h5ad"))
    return train_files + test_files


def load_gene_maps() -> Tuple[pd.DataFrame, dict]:
    gene_info = pd.read_csv(GENE_INFO, index_col=0)
    geneset = set(map(str, gene_info.index))
    alias_to_approved = {gene: gene for gene in geneset}

    if HGNC_INFO.exists():
        hgnc = pd.read_csv(HGNC_INFO, sep="\t", index_col=1)
        hgnc = hgnc[hgnc["Status"] == "Approved"]
        for approved, row in hgnc.iterrows():
            approved = str(approved)
            if approved in geneset:
                alias_to_approved[approved] = approved
            for col in ("Alias symbols", "Previous symbols"):
                value = row.get(col)
                if pd.notna(value):
                    for alias in str(value).split(", "):
                        if alias and alias not in geneset:
                            alias_to_approved[alias] = approved
    return gene_info, alias_to_approved


def build_gene_selection(
    var_names: Sequence[str],
    gene_info: pd.DataFrame,
    alias_to_approved: dict,
    max_genes: int,
) -> Tuple[List[str], np.ndarray]:
    gene_to_id = {str(gene): idx + 1 for idx, gene in enumerate(gene_info.index)}
    source_genes: List[str] = []
    gene_ids: List[int] = []
    seen_approved = set()

    for raw_gene in map(str, var_names):
        approved = alias_to_approved.get(raw_gene)
        if approved is None or approved not in gene_to_id or approved in seen_approved:
            continue
        source_genes.append(raw_gene)
        gene_ids.append(gene_to_id[approved])
        seen_approved.add(approved)
        if len(gene_ids) >= max_genes:
            break

    if not gene_ids:
        raise ValueError("No h5ad var_names could be mapped to CellFM genes.")
    padded_gene_ids = np.asarray(gene_ids + [0] * (max_genes - len(gene_ids)), dtype=np.int64)
    return source_genes, padded_gene_ids


def normalize_like_cellfm(adata: sc.AnnData) -> sc.AnnData:
    data = adata.X.astype(np.float32)
    totals = np.asarray(data.sum(1)).reshape(-1)
    denom = np.maximum(1, totals / 1e5).astype(np.float32)
    if issparse(data):
        data = data.multiply(1.0 / denom[:, None]).tocsr()
        data.data = np.round(data.data).astype(np.float32)
    else:
        data = csr_matrix(np.round(data / denom[:, None]).astype(np.float32))
    data.eliminate_zeros()
    adata.X = data
    return adata


class CellFMEmbeddingDataset(Dataset):
    def __init__(self, h5ad_path: Path, label_col: str, max_genes: int, normalize_counts: bool) -> None:
        self.h5ad_path = Path(h5ad_path)
        print(f"Reading {self.h5ad_path}")
        self.adata = sc.read_h5ad(self.h5ad_path)
        print(f"  origin shape: {self.adata.shape}")
        if normalize_counts:
            self.adata = normalize_like_cellfm(self.adata)

        gene_info, alias_to_approved = load_gene_maps()
        source_genes, gene_ids = build_gene_selection(self.adata.var_names.tolist(), gene_info, alias_to_approved, max_genes)
        self.gene_ids = gene_ids
        self.selected_gene_len = len(source_genes)
        self.X = self.adata[:, source_genes].X.tocsr() if issparse(self.adata.X) else csr_matrix(self.adata[:, source_genes].X)
        self.totals = np.asarray(self.X.sum(1)).reshape(-1).astype(np.float32)

        self.obs = self.adata.obs.copy()
        self.obs.insert(0, "cell_id", self.obs.index.astype(str))
        self.obs["source_file"] = self.h5ad_path.name
        self.obs["source_stem"] = self.h5ad_path.stem
        if label_col in self.obs.columns:
            labels = pd.to_numeric(self.obs[label_col], errors="coerce")
            invalid = int(labels.isna().sum())
            non_binary = int((~labels.dropna().isin([0, 1])).sum())
            if invalid or non_binary:
                print(f"  warning: {invalid} NaN/non-numeric and {non_binary} non-binary labels in '{label_col}'.")
        else:
            print(f"  warning: obs column '{label_col}' was not found.")
        print(f"  mapped genes: {self.selected_gene_len}/{self.adata.n_vars}; cells: {len(self)}")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        row = np.asarray(self.X[idx].toarray()).reshape(-1).astype(np.float32)
        return row, self.gene_ids, self.totals[idx], idx


class EmbeddingCollator:
    def __init__(self, pad_len: int) -> None:
        self.pad_len = pad_len

    @staticmethod
    def normalize(data: np.ndarray, read_count: float) -> np.ndarray:
        read_count = max(float(read_count), 1.0)
        return np.log1p(data / read_count * 1e4).astype(np.float32)

    @staticmethod
    def cat_st(sampled_count: float, total_count: float) -> np.ndarray:
        return np.log1p(np.asarray([sampled_count, total_count], dtype=np.float32) / 1000.0)

    def __call__(self, samples):
        dw_nzdata_batch = []
        st_feat_batch = []
        nonz_gene_batch = []
        zero_idx_batch = []
        index_batch = []
        for data, gene_ids, total_count, original_idx in samples:
            nonz = data.nonzero()[0]
            if len(nonz) > self.pad_len:
                weights = np.log1p(data[nonz])
                weights = weights / weights.sum()
                chosen = np.random.choice(np.arange(len(nonz)), self.pad_len, replace=False, p=weights)
                nonz = np.sort(nonz[chosen])
            seq_len = len(nonz)
            values = data[nonz]
            genes = gene_ids[nonz]
            sampled_count = float(values.sum())

            padded_values = np.zeros(self.pad_len, dtype=np.float32)
            padded_genes = np.zeros(self.pad_len, dtype=np.int64)
            zero_idx = np.zeros(self.pad_len, dtype=np.float32)
            if seq_len > 0:
                padded_values[:seq_len] = self.normalize(values, sampled_count)
                padded_genes[:seq_len] = genes
                zero_idx[:seq_len] = 1.0

            dw_nzdata_batch.append(torch.tensor(padded_values, dtype=torch.float32))
            st_feat_batch.append(torch.tensor(self.cat_st(sampled_count, total_count), dtype=torch.float32))
            nonz_gene_batch.append(torch.tensor(padded_genes, dtype=torch.long))
            zero_idx_batch.append(torch.tensor(zero_idx, dtype=torch.float32))
            index_batch.append(int(original_idx))
        return {
            "dw_nzdata": torch.stack(dw_nzdata_batch),
            "ST_feat": torch.stack(st_feat_batch),
            "nonz_gene": torch.stack(nonz_gene_batch),
            "zero_idx": torch.stack(zero_idx_batch),
            "index": np.asarray(index_batch, dtype=np.int64),
        }


def build_cellfm(ckpt_path: Path, device: torch.device) -> Cell_FM:
    cfg = Config_80M()
    cfg.ecs_threshold = 0.8
    cfg.ecs = True
    cfg.add_zero = True
    cfg.pad_zero = True
    cfg.mask_ratio = 0.0
    cfg.ckpt_path = str(ckpt_path)
    cfg.device = str(device)
    model = Cell_FM(27855, cfg, ckpt_path=str(ckpt_path), device=str(device)).to(device)
    model.load_model(weight=True, moment=False)
    model.eval()
    return model


class CellFMEncoder(nn.Module):
    def __init__(self, cellfm: Cell_FM) -> None:
        super().__init__()
        self.net = cellfm.net

    def forward(self, dw_nzdata: torch.Tensor, nonz_gene: torch.Tensor, st_feat: torch.Tensor, zero_idx: torch.Tensor) -> torch.Tensor:
        emb, _ = self.net.encode(dw_nzdata, nonz_gene, st_feat, zero_idx)
        return emb[:, 0]


@torch.no_grad()
def extract_one_file(
    h5ad_path: Path,
    output_dir: Path,
    cellfm: Cell_FM,
    encoder: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = h5ad_path.stem
    emb_path = output_dir / f"{stem}_embeddings.npy"
    obs_path = output_dir / f"{stem}_obs.csv"
    meta_path = output_dir / f"{stem}_embedding_meta.json"
    if emb_path.exists() and obs_path.exists() and not args.overwrite_embeddings:
        print(f"Skipping existing embeddings for {stem}")
        return emb_path, obs_path

    dataset = CellFMEmbeddingDataset(h5ad_path, args.label_col, args.max_genes, not args.no_normalize_counts)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_embed,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=EmbeddingCollator(args.max_genes),
        pin_memory=device.type == "cuda",
    )
    embeddings = np.empty((len(dataset), cellfm.cfg.enc_dims), dtype=np.float32)
    encoder.eval()
    for batch in tqdm(loader, desc=f"Embedding {stem}"):
        cls_token = encoder(
            batch["dw_nzdata"].to(device, non_blocking=True),
            batch["nonz_gene"].to(device, non_blocking=True),
            batch["ST_feat"].to(device, non_blocking=True),
            batch["zero_idx"].to(device, non_blocking=True),
        )
        embeddings[batch["index"]] = cls_token.detach().float().cpu().numpy()

    np.save(emb_path, embeddings.astype(np.float16 if args.embedding_dtype == "float16" else np.float32))
    dataset.obs.to_csv(obs_path, index=False)
    meta = {
        "input_h5ad": str(h5ad_path),
        "embedding_file": str(emb_path),
        "obs_file": str(obs_path),
        "n_cells": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "label_col": args.label_col,
        "selected_gene_count": int(dataset.selected_gene_len),
        "normalize_counts": not args.no_normalize_counts,
        "dtype": args.embedding_dtype,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {emb_path}")
    print(f"Saved {obs_path}")
    return emb_path, obs_path


def generate_embeddings(args: argparse.Namespace, device: torch.device, device_ids: List[int]) -> None:
    files = args.h5ad_files if args.h5ad_files else discover_h5ad_files(args.data_dir)
    if not files:
        raise FileNotFoundError(f"No Task3 h5ad files found in {args.data_dir}")
    missing = [str(path) for path in files if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing h5ad files: {missing}")

    cellfm = build_cellfm(args.ckpt_path, device)
    encoder: nn.Module = CellFMEncoder(cellfm)
    if len(device_ids) > 1:
        print(f"Using DataParallel for CellFM embedding on GPUs: {device_ids}")
        encoder = nn.DataParallel(encoder, device_ids=device_ids, output_device=device_ids[0])
    encoder.to(device)
    for h5ad_path in files:
        extract_one_file(Path(h5ad_path), args.embed_dir, cellfm, encoder, device, args)


class EmbeddingBinaryDataset(Dataset):
    def __init__(
        self,
        emb_path: Path,
        obs_path: Path,
        label_col: str,
        indices: Optional[np.ndarray] = None,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> None:
        self.emb_path = Path(emb_path)
        self.obs_path = Path(obs_path)
        self.embeddings = np.load(self.emb_path, mmap_mode="r")
        self.obs = pd.read_csv(self.obs_path)
        if len(self.obs) != self.embeddings.shape[0]:
            raise ValueError(f"Row mismatch: {self.emb_path} vs {self.obs_path}")
        if label_col not in self.obs.columns:
            raise KeyError(f"Column '{label_col}' was not found in {self.obs_path}")
        labels = pd.to_numeric(self.obs[label_col], errors="coerce").to_numpy(dtype=np.float32)
        base_indices = np.asarray(indices, dtype=np.int64) if indices is not None else np.arange(len(labels), dtype=np.int64)
        valid = np.isfinite(labels[base_indices]) & np.isin(labels[base_indices], [0.0, 1.0])
        invalid_count = int((~valid).sum())
        if invalid_count:
            print(f"Filtering {invalid_count} cells with invalid binary labels from {self.obs_path.name}")
        self.labels = labels
        self.indices = base_indices[valid]
        if len(self.indices) == 0:
            raise ValueError(f"No valid binary labels remain in {self.obs_path}")
        self.mean = mean.astype(np.float32) if mean is not None else None
        self.std = std.astype(np.float32) if std is not None else None

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        x = np.asarray(self.embeddings[real_idx], dtype=np.float32)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        y = float(self.labels[real_idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def split_indices(n: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    return np.sort(order[n_val:]), np.sort(order[:n_val])


def discover_training_embeddings(embed_dir: Path) -> List[Tuple[Path, Path]]:
    emb_files = sorted(embed_dir.glob("Training*_embeddings.npy"))
    pairs = []
    for emb_path in emb_files:
        obs_path = embed_dir / emb_path.name.replace("_embeddings.npy", "_obs.csv")
        if not obs_path.exists():
            raise FileNotFoundError(f"Missing obs csv for {emb_path}: {obs_path}")
        pairs.append((emb_path, obs_path))
    if not pairs:
        raise FileNotFoundError(f"No Training*_embeddings.npy files found in {embed_dir}")
    return pairs


def make_splits(pairs: List[Tuple[Path, Path]], args: argparse.Namespace) -> Tuple[List[EmbeddingBinaryDataset], List[EmbeddingBinaryDataset]]:
    train_datasets = []
    val_datasets = []
    for part_id, (emb_path, obs_path) in enumerate(pairs):
        base = EmbeddingBinaryDataset(emb_path, obs_path, args.label_col)
        train_idx, val_idx = split_indices(len(base.labels), args.val_fraction, args.seed + part_id)
        train_ds = EmbeddingBinaryDataset(emb_path, obs_path, args.label_col, train_idx)
        val_ds = EmbeddingBinaryDataset(emb_path, obs_path, args.label_col, val_idx)
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        print(f"{emb_path.name}: train={len(train_ds)} val={len(val_ds)}")
    return train_datasets, val_datasets


def compute_scaler(datasets: Sequence[EmbeddingBinaryDataset], chunk_size: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
    dim = datasets[0].embedding_dim
    total = 0
    sum_x = np.zeros(dim, dtype=np.float64)
    sum_x2 = np.zeros(dim, dtype=np.float64)
    for dataset in datasets:
        for start in tqdm(range(0, len(dataset.indices), chunk_size), desc=f"Scaler {dataset.emb_path.name}"):
            idx = dataset.indices[start : start + chunk_size]
            x = np.asarray(dataset.embeddings[idx], dtype=np.float32)
            sum_x += x.sum(axis=0, dtype=np.float64)
            sum_x2 += np.square(x, dtype=np.float64).sum(axis=0, dtype=np.float64)
            total += x.shape[0]
    mean = sum_x / max(total, 1)
    var = np.maximum(sum_x2 / max(total, 1) - mean ** 2, 1e-8)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


class BinaryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_hidden_dims(value: str) -> List[int]:
    dims = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not dims:
        raise ValueError("--hidden_dims must contain at least one integer")
    return dims


def binary_metrics(y_true: np.ndarray, logits: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= threshold).astype(np.int64)
    true = y_true.astype(np.int64)
    acc = float((pred == true).mean())
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    denom = 2 * tp + fp + fn
    f1 = float(2 * tp / denom) if denom else 0.0
    return {"Accuracy": acc, "F1": f1}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> Tuple[float, Dict[str, float]]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_n = 0
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            total_n += y.numel()
            logits_all.append(logits.detach().cpu().numpy())
            labels_all.append(y.detach().cpu().numpy())
    logits_np = np.concatenate(logits_all)
    labels_np = np.concatenate(labels_all)
    return total_loss / max(total_n, 1), binary_metrics(labels_np, logits_np, threshold)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for x, y in tqdm(loader, desc="Train global", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * y.numel()
        total_n += y.numel()
    return total_loss / max(total_n, 1)


def clone_with_scaler(dataset: EmbeddingBinaryDataset, args: argparse.Namespace, mean: np.ndarray, std: np.ndarray) -> EmbeddingBinaryDataset:
    return EmbeddingBinaryDataset(dataset.emb_path, dataset.obs_path, args.label_col, dataset.indices, mean, std)


def train_mlp(args: argparse.Namespace, device: torch.device, device_ids: List[int]) -> None:
    args.model_dir.mkdir(parents=True, exist_ok=True)
    train_raw, val_raw = make_splits(discover_training_embeddings(args.embed_dir), args)
    mean, std = compute_scaler(train_raw) if args.standardize_embeddings else (
        np.zeros(train_raw[0].embedding_dim, dtype=np.float32),
        np.ones(train_raw[0].embedding_dim, dtype=np.float32),
    )
    train_datasets = [clone_with_scaler(ds, args, mean, std) for ds in train_raw]
    val_datasets = [clone_with_scaler(ds, args, mean, std) for ds in val_raw]

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=args.batch_size_train,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        ConcatDataset(val_datasets),
        batch_size=args.batch_size_train,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    model = BinaryMLP(train_datasets[0].embedding_dim, hidden_dims, args.dropout).to(device)
    if len(device_ids) > 1:
        print(f"Using DataParallel for MLP training on GPUs: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    prefix = f"{args.run_name}_" if args.run_name else ""
    ckpt_path = args.model_dir / f"{prefix}task3_binary_mlp_global_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, device, args.threshold)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics}
        history.append(row)
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"Accuracy={val_metrics['Accuracy']:.4f} F1={val_metrics['F1']:.4f}"
        )
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "task": "task3_binary_classification",
                    "strategy": "global",
                    "input_dim": train_datasets[0].embedding_dim,
                    "hidden_dims": hidden_dims,
                    "dropout": args.dropout,
                    "embedding_mean": torch.from_numpy(mean.astype(np.float32)),
                    "embedding_std": torch.from_numpy(std.astype(np.float32)),
                    "threshold": args.threshold,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "label_col": args.label_col,
                    "run_name": args.run_name,
                },
                ckpt_path,
            )
            print(f"  saved best checkpoint: {ckpt_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    history_path = args.model_dir / f"{prefix}task3_binary_mlp_global_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved history: {history_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Task3 CellFM embeddings and train a global binary MLP.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--embed_dir", type=Path, default=DEFAULT_EMBED_DIR)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--ckpt_path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--h5ad_files", type=Path, nargs="*", default=None)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated GPU ids, e.g. 0,1,2. Overrides --device.")
    parser.add_argument("--batch_size_embed", type=int, default=16)
    parser.add_argument("--batch_size_train", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--hidden_dims", type=str, default="512,128")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_genes", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding_dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--overwrite_embeddings", action="store_true")
    parser.add_argument("--skip_embedding", action="store_true")
    parser.add_argument("--no_normalize_counts", action="store_true")
    parser.add_argument("--standardize_embeddings", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"CellFM checkpoint not found: {args.ckpt_path}")
    device, device_ids = resolve_device(args.gpu_ids, args.device)
    if not args.skip_embedding:
        generate_embeddings(args, device, device_ids)
    train_mlp(args, device, device_ids)


if __name__ == "__main__":
    main()
