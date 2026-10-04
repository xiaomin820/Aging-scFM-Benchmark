#!/usr/bin/env python3
"""
Generate Geneformer cell embeddings for Task3 senescent-cell classification h5ad files.

The embedding forward pass supports multiple GPUs by splitting each tokenized
dataset by cell rows. Pass GPU ids only, for example:

  conda activate geneformer
  python task3/scripts/1_generate_cell_embeddings_multigpu.py --gpu_ids 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import random
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
PROJECT_DIR = TASK3_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from datasets import Dataset, load_from_disk
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


DEFAULT_DATA_DIR = TASK3_DIR / "data" / "task3_Senescent_geneformer_input"
DEFAULT_OUTPUT_DIR = TASK3_DIR / "outputs" / "geneformer_cell_embeddings"
DEFAULT_MODEL_DIR = PROJECT_DIR / "Geneformer-V2-104M"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_h5ad_files(data_dir: Path) -> List[Path]:
    train_files = sorted(data_dir.glob("Training*.h5ad"))
    test_files = sorted(data_dir.glob("*Independent.Test*.h5ad"))
    return train_files + test_files


def parse_gpu_ids(value: str) -> List[int]:
    value = value.strip()
    if value.lower() in {"cpu", "none"}:
        return [-1]
    ids: List[int] = []
    for item in value.replace(",", " ").split():
        if item:
            ids.append(int(item))
    if not ids:
        raise ValueError("--gpu_ids must contain at least one GPU id, e.g. 0 or 0,1,2")
    return ids


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
    raise ValueError("Could not infer Ensembl IDs. Provide --ensembl_col or --gene_id_source explicitly.")


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
        if col != label_col:
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
    dataset_prefix = prepared_h5ad.stem.replace(".", "_")
    dataset_path = dataset_dir / f"{dataset_prefix}.dataset"
    if dataset_path.exists() and not overwrite:
        return dataset_path
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict={col: col for col in custom_attr_cols},
        nproc=nproc,
        chunk_size=chunk_size,
        model_version=model_version,
    )
    tokenized_cells, cell_metadata, tokenized_counts = tokenizer.tokenize_files(
        data_directory=prepared_h5ad.parent,
        file_format="h5ad",
        input_identifier=prepared_h5ad.stem,
    )
    tokenized_dataset = create_dataset_without_multiprocess_map(tokenizer, tokenized_cells, cell_metadata, tokenized_counts)
    tokenized_dataset.save_to_disk(str(dataset_path))
    return dataset_path


def create_dataset_without_multiprocess_map(
    tokenizer: TranscriptomeTokenizer,
    tokenized_cells: Sequence[Sequence[int]],
    cell_metadata: Optional[Dict[str, Sequence[object]]],
    tokenized_counts: Sequence[Sequence[float]],
) -> Dataset:
    dataset_dict: Dict[str, List] = {"input_ids": []}
    if tokenizer.keep_counts:
        dataset_dict["counts"] = []
    if cell_metadata is not None:
        dataset_dict.update({key: list(value) for key, value in cell_metadata.items()})

    cls_token = tokenizer.gene_token_dict.get("<cls>")
    eos_token = tokenizer.gene_token_dict.get("<eos>")
    lengths: List[int] = []
    for i, cell in enumerate(tokenized_cells):
        input_ids = list(cell)
        if tokenizer.special_token:
            input_ids = input_ids[0 : tokenizer.model_input_size - 2]
            input_ids = [cls_token] + input_ids + [eos_token]
        else:
            input_ids = input_ids[0 : tokenizer.model_input_size]
        dataset_dict["input_ids"].append(input_ids)
        lengths.append(len(input_ids))

        if tokenizer.keep_counts:
            counts = list(tokenized_counts[i])
            if tokenizer.special_token:
                counts = counts[0 : tokenizer.model_input_size - 2]
                counts = [0.0] + counts + [0.0]
            else:
                counts = counts[0 : tokenizer.model_input_size]
            dataset_dict["counts"].append(counts)

    dataset_dict["length"] = lengths
    return Dataset.from_dict(dataset_dict)


def load_geneformer_model(model_dir: Path, device: torch.device) -> BertForMaskedLM:
    model = BertForMaskedLM.from_pretrained(str(model_dir), output_hidden_states=True, output_attentions=False)
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


def pad_batch(input_ids: Sequence[Sequence[int]], lengths: Sequence[int], pad_token_id: int, device: torch.device):
    max_len = int(max(lengths))
    batch = torch.full((len(input_ids), max_len), int(pad_token_id), dtype=torch.long)
    for i, ids in enumerate(input_ids):
        ids = list(ids)[:max_len]
        batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    length_tensor = torch.tensor(lengths, dtype=torch.long)
    attention_mask = torch.arange(max_len).unsqueeze(0) < length_tensor.unsqueeze(1)
    return batch.to(device), attention_mask.to(device), length_tensor.to(device)


@torch.no_grad()
def extract_part_worker(config: Dict[str, object]) -> None:
    try:
        gpu_id = int(config["gpu_id"])
        device = torch.device("cpu" if gpu_id < 0 else f"cuda:{gpu_id}")
        dataset_path = Path(str(config["dataset_path"]))
        model_dir = Path(str(config["model_dir"]))
        part_path = Path(str(config["part_path"]))
        start = int(config["start"])
        end = int(config["end"])
        model_version = str(config["model_version"])
        emb_mode = str(config["emb_mode"])
        emb_layer = int(config["emb_layer"])
        batch_size = int(config["forward_batch_size"])
        dtype = str(config["dtype"])

        dataset = load_from_disk(str(dataset_path))
        with open(token_dictionary_file(model_version), "rb") as handle:
            gene_token_dict = pickle.load(handle)
        pad_token_id = gene_token_dict.get("<pad>", 0)
        cls_present = "<cls>" in gene_token_dict
        eos_present = "<eos>" in gene_token_dict
        if emb_mode == "cls" and not cls_present:
            emb_mode = "cell"

        model = load_geneformer_model(model_dir, device)
        layer_idx = quant_layer_index(model, emb_layer)
        embeddings = np.empty((end - start, int(model.config.hidden_size)), dtype=np.float32)
        out_pos = 0
        desc = f"GPU {gpu_id} rows {start}:{end}" if gpu_id >= 0 else f"CPU rows {start}:{end}"
        for batch_start in tqdm(range(start, end, batch_size), desc=desc, position=int(config["rank"]), leave=False):
            batch_end = min(batch_start + batch_size, end)
            batch = dataset.select(range(batch_start, batch_end))
            lengths = [int(v) for v in batch["length"]]
            input_tensor, attention_mask, length_tensor = pad_batch(batch["input_ids"], lengths, pad_token_id, device)
            outputs = model(input_ids=input_tensor, attention_mask=attention_mask)
            hidden = outputs.hidden_states[layer_idx]
            if emb_mode == "cls":
                emb = hidden[:, 0, :]
            else:
                if cls_present:
                    token_hidden = hidden[:, 1:, :]
                    mean_lengths = length_tensor - (2 if eos_present else 1)
                else:
                    token_hidden = hidden
                    mean_lengths = length_tensor
                mean_lengths = torch.clamp(mean_lengths, min=1)
                mask = torch.arange(token_hidden.shape[1], device=device).unsqueeze(0) < mean_lengths.unsqueeze(1)
                emb = token_hidden.masked_fill(~mask.unsqueeze(2), 0.0).sum(dim=1) / mean_lengths.unsqueeze(1).float()
            rows = batch_end - batch_start
            embeddings[out_pos : out_pos + rows] = emb.detach().float().cpu().numpy()
            out_pos += rows
            del outputs, hidden, emb
            if device.type == "cuda":
                torch.cuda.empty_cache()

        part_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(part_path, embeddings.astype(np.float16 if dtype == "float16" else np.float32))
    except Exception:
        error_path = Path(str(config["part_path"]) + ".error.txt")
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def write_obs_from_dataset(dataset_path: Path, obs_path: Path, custom_attr_cols: Sequence[str]) -> int:
    dataset = load_from_disk(str(dataset_path))
    obs = pd.DataFrame({col: dataset[col] for col in custom_attr_cols if col in dataset.column_names})
    obs.to_csv(obs_path, index=False)
    return len(dataset)


def extract_embeddings_multigpu(
    dataset_path: Path,
    output_dir: Path,
    output_prefix: str,
    model_dir: Path,
    gpu_ids: Sequence[int],
    custom_attr_cols: Sequence[str],
    model_version: str,
    emb_mode: str,
    emb_layer: int,
    forward_batch_size: int,
    dtype: str,
    overwrite: bool,
) -> Tuple[Path, Path]:
    emb_path = output_dir / f"{output_prefix}_embeddings.npy"
    obs_path = output_dir / f"{output_prefix}_obs.csv"
    if emb_path.exists() and obs_path.exists() and not overwrite:
        print(f"Skipping existing embeddings for {output_prefix}")
        return emb_path, obs_path

    dataset = load_from_disk(str(dataset_path))
    n_cells = len(dataset)
    active_gpu_ids = list(gpu_ids)[: max(1, min(len(gpu_ids), n_cells))]
    splits = np.array_split(np.arange(n_cells), len(active_gpu_ids))
    part_dir = output_dir / "_embedding_parts" / output_prefix.replace(".", "_")
    if part_dir.exists() and overwrite:
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    for rank, (gpu_id, split) in enumerate(zip(active_gpu_ids, splits)):
        if len(split) == 0:
            continue
        configs.append(
            {
                "rank": rank,
                "gpu_id": int(gpu_id),
                "dataset_path": str(dataset_path),
                "model_dir": str(model_dir),
                "part_path": str(part_dir / f"part_{rank:03d}.npy"),
                "start": int(split[0]),
                "end": int(split[-1]) + 1,
                "model_version": model_version,
                "emb_mode": emb_mode,
                "emb_layer": emb_layer,
                "forward_batch_size": forward_batch_size,
                "dtype": dtype,
            }
        )

    print(f"Embedding {output_prefix}: {n_cells} cells across {len(configs)} worker(s): {active_gpu_ids}")
    if len(configs) == 1:
        extract_part_worker(configs[0])
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(configs)) as pool:
            pool.map(extract_part_worker, configs)

    parts = [np.load(config["part_path"], mmap_mode="r") for config in configs]
    embeddings = np.concatenate([np.asarray(part) for part in parts], axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings.astype(np.float16 if dtype == "float16" else np.float32))
    n_obs = write_obs_from_dataset(dataset_path, obs_path, custom_attr_cols)
    if n_obs != embeddings.shape[0]:
        raise ValueError(f"Obs rows ({n_obs}) do not match embeddings ({embeddings.shape[0]})")

    meta = {
        "dataset_path": str(dataset_path),
        "embedding_file": str(emb_path),
        "obs_file": str(obs_path),
        "n_cells": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "gpu_ids": list(active_gpu_ids),
        "emb_mode": emb_mode,
        "emb_layer": int(emb_layer),
        "dtype": dtype,
    }
    (output_dir / f"{output_prefix}_embedding_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {emb_path}")
    print(f"Saved {obs_path}")
    return emb_path, obs_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Task3 Geneformer cell embeddings with optional multi-GPU forward pass.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--h5ad_files", type=Path, nargs="*", default=None)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--metadata_cols", nargs="*", default=["sex", "donorID", "cell_type", "tissue", "species", "dataset", "sample", "batch"])
    parser.add_argument("--gene_id_source", choices=("auto", "var_column", "var_index", "symbol_to_ensembl"), default="auto")
    parser.add_argument("--ensembl_col", type=str, default="ensembl_id")
    parser.add_argument("--model_version", choices=("V1", "V2"), default="V2")
    parser.add_argument("--emb_mode", choices=("cls", "cell"), default="cls")
    parser.add_argument("--emb_layer", type=int, default=-1)
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma/space separated GPU ids, e.g. 0,1,2. Use 'cpu' for CPU.")
    parser.add_argument("--forward_batch_size", type=int, default=16)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Geneformer model directory not found: {args.model_dir}")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if gpu_ids != [-1] and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --gpu_ids cpu to run on CPU.")
    if gpu_ids != [-1]:
        device_count = torch.cuda.device_count()
        bad_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= device_count]
        if bad_gpu_ids:
            raise ValueError(f"Invalid GPU id(s) {bad_gpu_ids}; this machine reports {device_count} CUDA device(s).")

    h5ad_files = args.h5ad_files if args.h5ad_files else default_h5ad_files(args.data_dir)
    if not h5ad_files:
        raise FileNotFoundError(f"No Task3 h5ad files found in {args.data_dir}")
    missing = [str(path) for path in h5ad_files if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing h5ad files: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = args.output_dir / "_prepared_h5ad"
    dataset_dir = args.output_dir / "_tokenized_dataset"

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
        extract_embeddings_multigpu(
            dataset_path=tokenized_path,
            output_dir=args.output_dir,
            output_prefix=h5ad_path.stem,
            model_dir=args.model_dir,
            gpu_ids=gpu_ids,
            custom_attr_cols=custom_attr_cols,
            model_version=args.model_version,
            emb_mode=args.emb_mode,
            emb_layer=args.emb_layer,
            forward_batch_size=args.forward_batch_size,
            dtype=args.dtype,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
