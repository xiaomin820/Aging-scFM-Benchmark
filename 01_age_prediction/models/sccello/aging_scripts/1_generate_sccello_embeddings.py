#!/usr/bin/env python3
"""
Generate scCello cell embeddings for Task1 aging h5ad files.

This script is intentionally implemented inside the scCello project and uses the
scCello tokenizer/model path. The CellFM aging scripts were used only as a
workflow reference.

Default inputs:
  aging_data/Task1_Training_Part*.h5ad
  aging_data/Task1_Independent.Test*.h5ad

Default outputs:
  outputs/sccello_age/cell_embeddings/<stem>_embeddings.npy
  outputs/sccello_age/cell_embeddings/<stem>_obs.csv
  outputs/sccello_age/cell_embeddings/<stem>_embedding_meta.json

Example:
  conda activate sccello
  python aging_scripts/1_generate_sccello_embeddings.py --device cuda:0 --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_sccello")

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


DEFAULT_DATA_DIR = PROJECT_DIR / "aging_data"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "sccello_age" / "cell_embeddings"
DEFAULT_MODEL_DIR = Path("/home/liangyunhao/shared/models/katarinayuan/scCello-zeroshot")
TOKEN_DICT_PATH = PROJECT_DIR / "data" / "token_vocabulary" / "token_dictionary.pkl"
GENE_MEDIAN_PATH = PROJECT_DIR / "data" / "token_vocabulary" / "gene_median_dictionary.pkl"
AGE_PATTERN = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(?:-|_|\s)?\s*(day|week|month|year)s?(?:-old)?\s*$", re.IGNORECASE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_h5ad_files(data_dir: Path) -> List[Path]:
    train_files = sorted(data_dir.glob("Task1_Training_Part*.h5ad"))
    test_files = sorted(data_dir.glob("Task1_Independent.Test*.h5ad"))
    return train_files + test_files


def parse_age_to_years(values: Sequence[Any], label_col: str) -> np.ndarray:
    parsed = []
    failed = []
    for value in values:
        if pd.isna(value):
            parsed.append(np.nan)
            failed.append(value)
            continue
        try:
            parsed.append(float(value))
            continue
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        match = AGE_PATTERN.match(text)
        if match is None:
            parsed.append(np.nan)
            failed.append(value)
            continue
        number = float(match.group(1))
        unit = match.group(2).lower()
        if unit == "day":
            parsed.append(number / 365.25)
        elif unit == "week":
            parsed.append(number / 52.1775)
        elif unit == "month":
            parsed.append(number / 12.0)
        else:
            parsed.append(number)

    arr = np.asarray(parsed, dtype=np.float32)
    if np.isnan(arr).any():
        examples = [str(v) for v in failed[:5]]
        raise ValueError(f"Could not parse {int(np.isnan(arr).sum())} values from '{label_col}' as ages. Examples: {examples}")
    return arr


def load_pickle(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_vocab() -> Tuple[Dict[str, int], Dict[str, float], int, int]:
    token_dict = load_pickle(TOKEN_DICT_PATH)
    gene_median = load_pickle(GENE_MEDIAN_PATH)
    if "<cls>" not in token_dict:
        token_dict = dict(token_dict)
        token_dict["<cls>"] = len(token_dict)
    pad_id = int(token_dict.get("<pad>", 0))
    cls_id = int(token_dict["<cls>"])
    return token_dict, gene_median, pad_id, cls_id


def get_gene_ids(adata: ad.AnnData, gene_col: Optional[str]) -> np.ndarray:
    if gene_col is not None:
        if gene_col not in adata.var.columns:
            raise KeyError(f"Column '{gene_col}' was not found in adata.var")
        return adata.var[gene_col].astype(str).to_numpy()
    if "ensembl_id" in adata.var.columns:
        return adata.var["ensembl_id"].astype(str).to_numpy()
    return adata.var_names.astype(str).to_numpy()


class H5adScCelloEmbeddingDataset(Dataset):
    def __init__(
        self,
        h5ad_path: Path,
        label_col: str,
        gene_col: Optional[str],
        max_length: int,
        token_dict: Dict[str, int],
        gene_median: Dict[str, float],
        cls_id: int,
    ) -> None:
        self.h5ad_path = Path(h5ad_path)
        self.max_length = max_length
        self.cls_id = cls_id
        print(f"Reading {self.h5ad_path}")
        self.adata = ad.read_h5ad(self.h5ad_path)
        print(f"  original shape: {self.adata.shape}")
        if label_col not in self.adata.obs.columns:
            raise KeyError(f"Column '{label_col}' was not found in obs for {self.h5ad_path}")

        gene_ids = get_gene_ids(self.adata, gene_col)
        keep_cols = []
        keep_gene_ids = []
        keep_token_ids = []
        keep_medians = []
        for col_idx, gene_id in enumerate(gene_ids):
            gene_id = str(gene_id).split(".")[0]
            if gene_id not in token_dict or gene_id not in gene_median:
                continue
            keep_cols.append(col_idx)
            keep_gene_ids.append(gene_id)
            keep_token_ids.append(int(token_dict[gene_id]))
            keep_medians.append(float(gene_median[gene_id]))
        if not keep_cols:
            raise ValueError("No h5ad genes could be mapped to scCello token vocabulary.")

        self.gene_ids = np.asarray(keep_gene_ids)
        self.token_ids = np.asarray(keep_token_ids, dtype=np.int64)
        self.gene_medians = np.asarray(keep_medians, dtype=np.float32)
        self.X = self.adata.X[:, np.asarray(keep_cols, dtype=np.int64)]
        if issparse(self.X):
            self.X = self.X.tocsr()
        else:
            self.X = np.asarray(self.X, dtype=np.float32)

        self.obs = self.adata.obs.copy()
        self.obs.insert(0, "cell_id", self.obs.index.astype(str))
        self.obs["source_file"] = self.h5ad_path.name
        self.obs["source_stem"] = self.h5ad_path.stem
        self.labels = parse_age_to_years(self.obs[label_col].tolist(), label_col)
        self.obs[f"{label_col}_years"] = self.labels
        print(f"  mapped genes: {len(self.gene_ids)}/{self.adata.n_vars}; cells: {len(self)}")

    def __len__(self) -> int:
        return self.adata.n_obs

    def _row_arrays(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if issparse(self.X):
            row = self.X.getrow(idx)
            values = row.data.astype(np.float32)
            cols = row.indices.astype(np.int64)
        else:
            row = np.asarray(self.X[idx], dtype=np.float32)
            cols = np.nonzero(row)[0].astype(np.int64)
            values = row[cols]
        return values, cols

    def __getitem__(self, idx: int) -> List[int]:
        values, cols = self._row_arrays(idx)
        if len(values) == 0:
            return [self.cls_id]

        total = float(values.sum())
        if total <= 0:
            return [self.cls_id]
        expr_norm = values * (10000.0 / total)
        rank_values = expr_norm / np.maximum(self.gene_medians[cols], 1e-12)
        order = np.argsort(-rank_values)
        token_ids = self.token_ids[cols][order].tolist()
        return [self.cls_id] + token_ids[: self.max_length - 1]


class PadCollator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, examples: Sequence[List[int]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x) for x in examples)
        input_ids = torch.full((len(examples), max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
        for i, ids in enumerate(examples):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_model(model_dir: Path, device: torch.device):
    try:
        from sccello.src.model_prototype_contrastive import PrototypeContrastiveForMaskedLM
    except ImportError as exc:
        raise ImportError(
            "Failed to import the scCello model code. This usually means the current "
            "conda environment has incompatible scCello dependencies, especially "
            "transformers/accelerate. Please fix the sccello environment before "
            "running embedding generation."
        ) from exc

    model = PrototypeContrastiveForMaskedLM.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def encode_batch(model, batch: Dict[str, torch.Tensor], device: torch.device, embedding_mode: str) -> np.ndarray:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    if embedding_mode == "pooler":
        emb = outputs.pooler_output
    elif embedding_mode == "cls":
        emb = outputs.last_hidden_state[:, 0, :]
    elif embedding_mode == "cell_cls":
        emb = model.cell_cls(outputs.last_hidden_state)[:, 0, :]
    else:
        raise ValueError(f"Unknown embedding_mode: {embedding_mode}")
    return emb.detach().float().cpu().numpy()


@torch.no_grad()
def extract_one_file(
    h5ad_path: Path,
    output_dir: Path,
    model,
    device: torch.device,
    label_col: str,
    gene_col: Optional[str],
    batch_size: int,
    num_workers: int,
    max_length: int,
    dtype: str,
    embedding_mode: str,
    overwrite: bool,
    write_h5ad: bool,
    token_dict: Dict[str, int],
    gene_median: Dict[str, float],
    pad_id: int,
    cls_id: int,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = h5ad_path.stem
    emb_path = output_dir / f"{stem}_embeddings.npy"
    obs_path = output_dir / f"{stem}_obs.csv"
    meta_path = output_dir / f"{stem}_embedding_meta.json"
    h5ad_out = output_dir / f"{stem}_embeddings.h5ad"

    if emb_path.exists() and obs_path.exists() and not overwrite:
        print(f"Skipping existing embeddings for {stem}")
        return emb_path, obs_path

    dataset = H5adScCelloEmbeddingDataset(
        h5ad_path=h5ad_path,
        label_col=label_col,
        gene_col=gene_col,
        max_length=max_length,
        token_dict=token_dict,
        gene_median=gene_median,
        cls_id=cls_id,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=PadCollator(pad_id),
        pin_memory=device.type == "cuda",
    )

    embeddings = np.empty((len(dataset), model.config.hidden_size), dtype=np.float32)
    cursor = 0
    for batch in tqdm(loader, desc=f"Embedding {stem}"):
        batch_emb = encode_batch(model, batch, device, embedding_mode)
        embeddings[cursor : cursor + batch_emb.shape[0]] = batch_emb
        cursor += batch_emb.shape[0]

    np.save(emb_path, embeddings.astype(np.float16 if dtype == "float16" else np.float32))
    dataset.obs.to_csv(obs_path, index=False)

    meta = {
        "input_h5ad": str(h5ad_path),
        "embedding_file": str(emb_path),
        "obs_file": str(obs_path),
        "n_cells": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "model_dir": str(DEFAULT_MODEL_DIR),
        "label_col": label_col,
        "gene_col": gene_col,
        "mapped_gene_count": int(len(dataset.gene_ids)),
        "max_length": int(max_length),
        "embedding_mode": embedding_mode,
        "dtype": dtype,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if write_h5ad:
        adata_emb = ad.AnnData(X=embeddings.astype(np.float32), obs=dataset.obs.copy())
        adata_emb.write_h5ad(h5ad_out, compression="gzip")
        print(f"Saved {h5ad_out}")

    print(f"Saved {emb_path}")
    print(f"Saved {obs_path}")
    return emb_path, obs_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scCello cell embeddings for Task1 aging h5ad files.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--h5ad_files", type=Path, nargs="*", default=None)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--gene_col", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--embedding_mode", choices=("pooler", "cls", "cell_cls"), default="cell_cls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write_h5ad", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    h5ad_files = args.h5ad_files if args.h5ad_files else default_h5ad_files(args.data_dir)
    if not h5ad_files:
        raise FileNotFoundError(f"No Task1 h5ad files found in {args.data_dir}")
    missing = [str(path) for path in h5ad_files if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing h5ad files: {missing}")

    token_dict, gene_median, pad_id, cls_id = load_vocab()
    device = torch.device(args.device)
    model = load_model(args.model_dir, device)

    for h5ad_path in h5ad_files:
        extract_one_file(
            h5ad_path=Path(h5ad_path),
            output_dir=args.output_dir,
            model=model,
            device=device,
            label_col=args.label_col,
            gene_col=args.gene_col,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_length=args.max_length,
            dtype=args.dtype,
            embedding_mode=args.embedding_mode,
            overwrite=args.overwrite,
            write_h5ad=args.write_h5ad,
            token_dict=token_dict,
            gene_median=gene_median,
            pad_id=pad_id,
            cls_id=cls_id,
        )


if __name__ == "__main__":
    main()
