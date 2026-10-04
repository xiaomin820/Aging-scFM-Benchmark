#!/usr/bin/env python3
"""
Generate Geneformer cell embeddings for the Task1 aging h5ad files.

Default inputs:
  aging_data/geneformer_input/geneformer_input/Task1_Training_Part1-5_*.h5ad
  aging_data/geneformer_input/geneformer_input/Task1_Independent.Test_*.h5ad

Outputs, per input h5ad:
  outputs/geneformer_cell_embeddings/<stem>_embeddings.npy
  outputs/geneformer_cell_embeddings/<stem>_obs.csv
  outputs/geneformer_cell_embeddings/<stem>_embedding_meta.json

Run:
  conda activate geneformer
  python aging_scripts/1_generate_cell_embeddings.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from datasets import load_from_disk
from scipy.sparse import issparse
from tqdm import tqdm
from transformers import BertForMaskedLM

from geneformer import (
    ENSEMBL_DICTIONARY_FILE,
    ENSEMBL_DICTIONARY_FILE_30M,
    TOKEN_DICTIONARY_FILE,
    TOKEN_DICTIONARY_FILE_30M,
    TranscriptomeTokenizer,
)


DEFAULT_DATA_DIR = PROJECT_DIR / "aging_data" / "geneformer_input" / "geneformer_input"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "geneformer_cell_embeddings"
DEFAULT_MODEL_DIR = PROJECT_DIR / "Geneformer-V2-104M"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_h5ad_files(data_dir: Path) -> List[Path]:
    train_files = sorted(data_dir.glob("Task1_Training_Part*.h5ad"))
    test_files = sorted(data_dir.glob("Task1_Independent.Test*.h5ad"))
    return train_files + test_files


def read_pickle(path: Path) -> Dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def gene_name_dictionary_file(model_version: str) -> Path:
    return ENSEMBL_DICTIONARY_FILE_30M if model_version == "V1" else ENSEMBL_DICTIONARY_FILE


def token_dictionary_file(model_version: str) -> Path:
    return TOKEN_DICTIONARY_FILE_30M if model_version == "V1" else TOKEN_DICTIONARY_FILE


def normalize_ensembl_id(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value:
        return None
    return value.split(".")[0].upper()


def looks_like_ensembl(values: Sequence[object], min_fraction: float = 0.5) -> bool:
    sample = [normalize_ensembl_id(v) for v in values[: min(len(values), 1000)]]
    sample = [v for v in sample if v]
    if not sample:
        return False
    return sum(v.startswith("ENSG") for v in sample) / len(sample) >= min_fraction


def infer_ensembl_ids(
    adata: ad.AnnData,
    gene_id_source: str,
    ensembl_col: str,
    gene_name_id: Dict[str, str],
) -> Tuple[List[Optional[str]], str]:
    def map_symbol(value: object) -> Optional[str]:
        symbol = str(value).strip()
        return gene_name_id.get(symbol) or gene_name_id.get(symbol.upper())

    if gene_id_source == "var_column":
        if ensembl_col not in adata.var.columns:
            raise KeyError(f"Requested var column '{ensembl_col}' was not found.")
        return [normalize_ensembl_id(v) for v in adata.var[ensembl_col]], f"var.{ensembl_col}"

    if gene_id_source == "var_index":
        return [normalize_ensembl_id(v) for v in adata.var_names], "var_names"

    if gene_id_source == "symbol_to_ensembl":
        return [normalize_ensembl_id(map_symbol(v)) for v in adata.var_names], "var_names_symbol_map"

    if gene_id_source != "auto":
        raise ValueError(f"Unknown gene_id_source: {gene_id_source}")

    if ensembl_col in adata.var.columns and looks_like_ensembl(list(adata.var[ensembl_col])):
        return [normalize_ensembl_id(v) for v in adata.var[ensembl_col]], f"var.{ensembl_col}"

    for col in ("ensembl_id", "ensembl", "gene_id", "gene_ids", "gene_ids-0", "feature_id"):
        if col in adata.var.columns and looks_like_ensembl(list(adata.var[col])):
            return [normalize_ensembl_id(v) for v in adata.var[col]], f"var.{col}"

    if looks_like_ensembl(list(adata.var_names)):
        return [normalize_ensembl_id(v) for v in adata.var_names], "var_names"

    mapped = [normalize_ensembl_id(map_symbol(v)) for v in adata.var_names]
    if sum(v is not None for v in mapped) > 0:
        return mapped, "var_names_symbol_map"

    raise ValueError(
        "Could not infer Ensembl IDs. Provide --ensembl_col or use "
        "--gene_id_source var_index/symbol_to_ensembl if appropriate."
    )


def add_required_geneformer_fields(
    h5ad_path: Path,
    prepared_dir: Path,
    label_col: str,
    metadata_cols: Sequence[str],
    gene_id_source: str,
    ensembl_col: str,
    model_version: str,
    overwrite: bool,
) -> Tuple[Path, List[str], Dict[str, object]]:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = prepared_dir / f"{h5ad_path.stem}__geneformer_prepared.h5ad"
    meta_path = prepared_dir / f"{h5ad_path.stem}__geneformer_prepared_meta.json"
    if prepared_path.exists() and meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return prepared_path, list(meta["custom_attr_cols"]), meta

    print(f"Preparing {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    if label_col not in adata.obs.columns:
        raise KeyError(f"Column '{label_col}' was not found in obs for {h5ad_path}")

    gene_name_id = read_pickle(gene_name_dictionary_file(model_version))
    ensembl_ids, source = infer_ensembl_ids(adata, gene_id_source, ensembl_col, gene_name_id)
    adata.var["ensembl_id"] = ensembl_ids

    mapped = pd.Series(adata.var["ensembl_id"]).notna().sum()
    if mapped == 0:
        raise ValueError(f"No genes in {h5ad_path} could be mapped to Geneformer Ensembl IDs.")

    if "n_counts" not in adata.obs.columns:
        totals = np.asarray(adata.X.sum(axis=1)).reshape(-1) if issparse(adata.X) else adata.X.sum(axis=1)
        adata.obs["n_counts"] = np.asarray(totals, dtype=np.float32)
    adata.obs["n_counts"] = pd.to_numeric(adata.obs["n_counts"], errors="coerce").fillna(0).astype(np.float32)
    adata.obs.loc[adata.obs["n_counts"] <= 0, "n_counts"] = 1.0

    adata.obs["cell_id"] = adata.obs_names.astype(str)
    adata.obs["source_file"] = h5ad_path.name
    adata.obs["source_stem"] = h5ad_path.stem

    custom_attr_cols = ["cell_id", label_col, "source_file", "source_stem"]
    for col in metadata_cols:
        if col in adata.obs.columns and col not in custom_attr_cols:
            custom_attr_cols.append(col)

    for col in custom_attr_cols:
        if col == label_col:
            continue
        adata.obs[col] = adata.obs[col].astype(str)

    adata.write_h5ad(prepared_path, compression="gzip")
    meta = {
        "input_h5ad": str(h5ad_path),
        "prepared_h5ad": str(prepared_path),
        "gene_id_source": source,
        "n_input_genes": int(adata.n_vars),
        "n_mapped_genes": int(mapped),
        "n_cells": int(adata.n_obs),
        "custom_attr_cols": custom_attr_cols,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  mapped genes: {mapped}/{adata.n_vars} via {source}")
    return prepared_path, custom_attr_cols, meta


def tokenize_prepared_h5ad(
    prepared_h5ad: Path,
    dataset_dir: Path,
    custom_attr_cols: Sequence[str],
    model_version: str,
    nproc: int,
    chunk_size: int,
    overwrite: bool,
) -> Path:
    # Geneformer's tokenizer saves with Path.with_suffix(".dataset").
    # Dots inside the prefix, e.g. "Task1_Independent.Test...", would otherwise
    # be treated as a suffix and truncate the saved directory name.
    dataset_prefix = prepared_h5ad.stem.replace(".", "_")
    dataset_path = dataset_dir / f"{dataset_prefix}.dataset"
    if dataset_path.exists() and not overwrite:
        return dataset_path

    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    attr_dict = {col: col for col in custom_attr_cols}
    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict=attr_dict,
        nproc=nproc,
        chunk_size=chunk_size,
        model_version=model_version,
    )
    tokenizer.tokenize_data(
        data_directory=prepared_h5ad.parent,
        output_directory=dataset_dir,
        output_prefix=dataset_prefix,
        file_format="h5ad",
        input_identifier=prepared_h5ad.stem,
    )
    return dataset_path


def load_geneformer_model(model_dir: Path, device: torch.device) -> BertForMaskedLM:
    model = BertForMaskedLM.from_pretrained(
        str(model_dir),
        output_hidden_states=True,
        output_attentions=False,
    )
    model.to(device)
    model.eval()
    return model


def quant_layer_index(model: BertForMaskedLM, emb_layer: int) -> int:
    layer_nums = []
    for name, _ in model.named_parameters():
        if "layer." in name:
            layer_nums.append(int(name.split("layer.")[1].split(".")[0]))
    if not layer_nums:
        raise ValueError("Could not infer transformer layer count from model parameters.")
    return max(layer_nums) + 1 + emb_layer


def pad_batch(
    input_ids: Sequence[Sequence[int]],
    lengths: Sequence[int],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = int(max(lengths))
    batch = torch.full((len(input_ids), max_len), int(pad_token_id), dtype=torch.long)
    for i, ids in enumerate(input_ids):
        ids = list(ids)[:max_len]
        batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    length_tensor = torch.tensor(lengths, dtype=torch.long)
    attention_mask = torch.arange(max_len).unsqueeze(0) < length_tensor.unsqueeze(1)
    return batch.to(device), attention_mask.to(device), length_tensor.to(device)


@torch.no_grad()
def extract_embeddings_from_dataset(
    dataset_path: Path,
    output_dir: Path,
    output_prefix: str,
    model: BertForMaskedLM,
    device: torch.device,
    custom_attr_cols: Sequence[str],
    model_version: str,
    emb_mode: str,
    emb_layer: int,
    forward_batch_size: int,
    dtype: str,
    overwrite: bool,
    write_h5ad: bool,
) -> Tuple[Path, Path]:
    emb_path = output_dir / f"{output_prefix}_embeddings.npy"
    obs_path = output_dir / f"{output_prefix}_obs.csv"
    if emb_path.exists() and obs_path.exists() and not overwrite:
        print(f"Skipping existing embeddings for {output_prefix}")
        return emb_path, obs_path

    dataset = load_from_disk(str(dataset_path))
    with open(token_dictionary_file(model_version), "rb") as handle:
        gene_token_dict = pickle.load(handle)
    pad_token_id = gene_token_dict.get("<pad>", 0)
    cls_present = "<cls>" in gene_token_dict
    eos_present = "<eos>" in gene_token_dict
    layer_idx = quant_layer_index(model, emb_layer)

    if emb_mode == "cls" and not cls_present:
        print("CLS token is not present for this model; switching emb_mode to cell mean pooling.")
        emb_mode = "cell"

    hidden_size = int(model.config.hidden_size)
    embeddings = np.empty((len(dataset), hidden_size), dtype=np.float32)

    obs_parts = []
    write_pos = 0
    for start in tqdm(range(0, len(dataset), forward_batch_size), desc=f"Embedding {output_prefix}"):
        end = min(start + forward_batch_size, len(dataset))
        batch = dataset.select(range(start, end))
        input_ids = batch["input_ids"]
        lengths = [int(v) for v in batch["length"]]
        input_tensor, attention_mask, length_tensor = pad_batch(input_ids, lengths, pad_token_id, device)
        outputs = model(input_ids=input_tensor, attention_mask=attention_mask)
        hidden = outputs.hidden_states[layer_idx]

        if emb_mode == "cls":
            emb = hidden[:, 0, :]
        elif emb_mode == "cell":
            if cls_present:
                token_hidden = hidden[:, 1:, :]
                mean_lengths = length_tensor - (2 if eos_present else 1)
            else:
                token_hidden = hidden
                mean_lengths = length_tensor
            mean_lengths = torch.clamp(mean_lengths, min=1)
            max_token_len = token_hidden.shape[1]
            mask = torch.arange(max_token_len, device=device).unsqueeze(0) < mean_lengths.unsqueeze(1)
            emb = token_hidden.masked_fill(~mask.unsqueeze(2), 0.0).sum(dim=1) / mean_lengths.unsqueeze(1).float()
        else:
            raise ValueError("--emb_mode must be 'cls' or 'cell'")

        embeddings[write_pos:end] = emb.detach().float().cpu().numpy()
        obs_parts.append(pd.DataFrame({col: batch[col] for col in custom_attr_cols if col in batch.column_names}))
        write_pos = end

    obs = pd.concat(obs_parts, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings.astype(np.float16 if dtype == "float16" else np.float32))
    obs.to_csv(obs_path, index=False)

    meta = {
        "dataset_path": str(dataset_path),
        "embedding_file": str(emb_path),
        "obs_file": str(obs_path),
        "n_cells": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "emb_mode": emb_mode,
        "emb_layer": int(emb_layer),
        "dtype": dtype,
    }
    (output_dir / f"{output_prefix}_embedding_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if write_h5ad:
        adata_emb = ad.AnnData(X=embeddings.astype(np.float32), obs=obs.copy())
        adata_emb.write_h5ad(output_dir / f"{output_prefix}_embeddings.h5ad", compression="gzip")

    print(f"Saved {emb_path}")
    print(f"Saved {obs_path}")
    return emb_path, obs_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Geneformer cell embeddings for Task1 aging h5ad files.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--h5ad_files", type=Path, nargs="*", default=None)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--metadata_cols", nargs="*", default=["sex", "donorID"])
    parser.add_argument("--gene_id_source", choices=("auto", "var_column", "var_index", "symbol_to_ensembl"), default="auto")
    parser.add_argument("--ensembl_col", type=str, default="ensembl_id")
    parser.add_argument("--model_version", choices=("V1", "V2"), default="V2")
    parser.add_argument("--emb_mode", choices=("cls", "cell"), default="cls")
    parser.add_argument("--emb_layer", type=int, default=-1, help="-1 is the second-to-last hidden layer; 0 is the last hidden layer.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--forward_batch_size", type=int, default=16)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write_h5ad", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Geneformer model directory not found: {args.model_dir}")

    h5ad_files = args.h5ad_files if args.h5ad_files else default_h5ad_files(args.data_dir)
    if not h5ad_files:
        raise FileNotFoundError(f"No Task1 h5ad files found in {args.data_dir}")
    missing = [str(path) for path in h5ad_files if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing h5ad files: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = args.output_dir / "_prepared_h5ad"
    dataset_dir = args.output_dir / "_tokenized_dataset"

    device = torch.device(args.device)
    model = load_geneformer_model(args.model_dir, device)

    for h5ad_path in h5ad_files:
        h5ad_path = Path(h5ad_path)
        prepared_h5ad, custom_attr_cols, _ = add_required_geneformer_fields(
            h5ad_path=h5ad_path,
            prepared_dir=prepared_dir,
            label_col=args.label_col,
            metadata_cols=args.metadata_cols,
            gene_id_source=args.gene_id_source,
            ensembl_col=args.ensembl_col,
            model_version=args.model_version,
            overwrite=args.overwrite,
        )
        tokenized_path = tokenize_prepared_h5ad(
            prepared_h5ad=prepared_h5ad,
            dataset_dir=dataset_dir,
            custom_attr_cols=custom_attr_cols,
            model_version=args.model_version,
            nproc=args.nproc,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
        )
        extract_embeddings_from_dataset(
            dataset_path=tokenized_path,
            output_dir=args.output_dir,
            output_prefix=h5ad_path.stem,
            model=model,
            device=device,
            custom_attr_cols=custom_attr_cols,
            model_version=args.model_version,
            emb_mode=args.emb_mode,
            emb_layer=args.emb_layer,
            forward_batch_size=args.forward_batch_size,
            dtype=args.dtype,
            overwrite=args.overwrite,
            write_h5ad=args.write_h5ad,
        )


if __name__ == "__main__":
    main()
