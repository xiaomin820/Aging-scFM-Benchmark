"""
Step 1: Generate Cell Embeddings using TranscriptFormer
Generate cell embeddings for downstream MLP age-prediction training

Usage examples:
    # Use one GPU (GPU 0)
    python aging_scripts/01_generate_embeddings.py --gpu 0

    # Use multiple GPUs (GPU 0,1)
    python aging_scripts/01_generate_embeddings.py --gpu 0,1

    # Specify the batch size
    python aging_scripts/01_generate_embeddings.py --gpu 0,1 --batch_size 128
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path


def _get_arg_value(argv, name, default=None):
    """Read a simple CLI option before argparse is initialized."""
    prefix = f"{name}="
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return default


# CUDA device visibility must be configured before torch/lightning imports.
_early_gpu_arg = _get_arg_value(sys.argv, "--gpu")
if _early_gpu_arg:
    os.environ["CUDA_VISIBLE_DEVICES"] = _early_gpu_arg

# Disable PyTorch JIT optimizations before importing torch
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
torch._dynamo.config.disable = True
torch._dynamo.reset()

# Mock torch.compile
_original_compile = torch.compile
def _disabled_compile(*args, **kwargs):
    return args[0] if args else None
torch.compile = _disabled_compile

import numpy as np
import pandas as pd
import anndata as ad
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir

from transcriptformer.model.inference import run_inference

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Configuration
CHECKPOINT_PATH = "./checkpoints/tf_sapiens"
DATA_DIR = "./aging_data/TranscriptFormer_train_data"
OUTPUT_DIR = "./embedding_results"
EMBEDDING_PREFIX = "cell_embedding"


# File mapping
TRAIN_FILES = [
    "Task1_Training_Part1_n50000_TranscriptFormer_input.h5ad",
    "Task1_Training_Part2_n50000_TranscriptFormer_input.h5ad",
    "Task1_Training_Part3_n50000_TranscriptFormer_input.h5ad",
    "Task1_Training_Part4_n50000_TranscriptFormer_input.h5ad",
    "Task1_Training_Part5_n40000_TranscriptFormer_input.h5ad",
]
TEST_FILE = "Task1_Independent.Test_GSE134355_n32000_TranscriptFormer_input.h5ad"
TRAIN_FILE_ALIASES = {f"train{i + 1}": filename for i, filename in enumerate(TRAIN_FILES)}
FILE_ALIASES = {
    **TRAIN_FILE_ALIASES,
    "test": TEST_FILE,
}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate cell embeddings using TranscriptFormer"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device IDs to use, comma-separated (e.g., '0' or '0,1,2,3')"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Per-GPU batch size for inference (default: 64)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate embeddings even if they exist"
    )
    parser.add_argument(
        "--files",
        type=str,
        default="all",
        help=(
            "Comma-separated file aliases or h5ad basenames to process. "
            "Use all, train, test, train1..train5 (default: all)"
        )
    )
    parser.add_argument(
        "--no_combine",
        action="store_true",
        help="Do not write train_all_embeddings.parquet after processing"
    )
    parser.add_argument(
        "--combine_only",
        action="store_true",
        help="Only combine existing training parquet files and exit"
    )
    return parser.parse_args()


def resolve_requested_files(files_arg):
    """Resolve --files aliases into train files and whether to run the test file."""
    if files_arg is None or files_arg.strip().lower() == "all":
        return list(TRAIN_FILES), True

    selected_train_files = []
    include_test = False
    unknown_items = []

    for raw_item in files_arg.split(","):
        item = raw_item.strip()
        item_lower = item.lower()
        if not item:
            continue
        if item_lower == "all":
            selected_train_files.extend(TRAIN_FILES)
            include_test = True
        elif item_lower == "train":
            selected_train_files.extend(TRAIN_FILES)
        elif item_lower == "test":
            include_test = True
        elif item_lower in FILE_ALIASES:
            resolved = FILE_ALIASES[item_lower]
            if resolved == TEST_FILE:
                include_test = True
            else:
                selected_train_files.append(resolved)
        elif item in TRAIN_FILES:
            selected_train_files.append(item)
        elif item == TEST_FILE:
            include_test = True
        else:
            unknown_items.append(item)

    if unknown_items:
        valid_aliases = ", ".join(["all", "train", "test", *TRAIN_FILE_ALIASES.keys()])
        raise ValueError(f"Unknown --files item(s): {unknown_items}. Valid aliases: {valid_aliases}")

    selected_train_files = list(dict.fromkeys(selected_train_files))
    return selected_train_files, include_test


def output_path_for(data_file):
    return os.path.join(
        OUTPUT_DIR,
        EMBEDDING_PREFIX + "_" + data_file.replace(".h5ad", ".parquet")
    )


def combine_training_embeddings(force=False):
    train_dfs = []
    missing_files = []

    for train_file in TRAIN_FILES:
        parquet_file = output_path_for(train_file)
        if os.path.exists(parquet_file):
            train_dfs.append(pd.read_parquet(parquet_file))
        else:
            missing_files.append(parquet_file)

    if missing_files:
        logger.warning("Missing training embedding files, skipping them:")
        for missing_file in missing_files:
            logger.warning(f"  {missing_file}")

    if not train_dfs:
        logger.warning("No training embedding parquet files found to combine")
        return None

    train_combined = pd.concat(train_dfs, ignore_index=True)
    train_combined_file = os.path.join(OUTPUT_DIR, "train_all_embeddings.parquet")
    if os.path.exists(train_combined_file) and not force:
        logger.info(f"Combined file already exists: {train_combined_file} (use --force to overwrite)")
        return train_combined

    train_combined.to_parquet(train_combined_file, index=True)
    logger.info(f"Combined training embeddings: {train_combined.shape}")
    return train_combined


def setup_hydra_config(args):
    """Setup Hydra configuration for inference"""
    # Step 1: Load base inference config from project's hydra yaml (has all fields)
    config_dir = os.path.abspath("src/transcriptformer/cli/conf")
    with initialize_config_dir(config_dir=config_dir):
        cfg = compose(config_name="inference_config.yaml")

    # Disable struct mode so we can freely modify
    OmegaConf.set_struct(cfg, False)

    # Step 2: Load model-specific config from checkpoint
    checkpoint_config_path = os.path.join(CHECKPOINT_PATH, "config.json")
    with open(checkpoint_config_path) as f:
        model_config_dict = json.load(f)

    # Step 3: Override model config values from checkpoint (skip nulls so defaults stay)
    for section in ["data_config", "model_config", "loss_config"]:
        if section in model_config_dict.get("model", {}):
            for key, val in model_config_dict["model"][section].items():
                if val is not None:
                    # OmegaConf merge syntax
                    OmegaConf.update(cfg, f"model.{section}.{key}", val, merge=False)

    # Step 4: Set runtime inference parameters
    num_gpus = len(args.gpu.split(','))
    cfg.model.checkpoint_path = CHECKPOINT_PATH
    cfg.model.inference_config.load_checkpoint = os.path.join(CHECKPOINT_PATH, "model_weights.pt")
    cfg.model.inference_config.batch_size = args.batch_size
    cfg.model.inference_config.emb_type = "cell"
    cfg.model.inference_config.output_keys = ["embeddings"]
    cfg.model.inference_config.obs_keys = ["all"]
    cfg.model.inference_config.pretrained_embedding = None
    cfg.model.inference_config.device = "cuda"
    cfg.model.inference_config.num_gpus = num_gpus
    cfg.model.data_config.n_data_workers = args.num_workers

    # Also set paths from checkpoint
    cfg.model.data_config.aux_vocab_path = os.path.join(CHECKPOINT_PATH, "vocabs")
    cfg.model.data_config.esm2_mappings_path = os.path.join(CHECKPOINT_PATH, "vocabs")

    # Disable compile_block_mask to avoid issues
    cfg.model.model_config.compile_block_mask = False

    logger.info("Configuration loaded successfully")
    return cfg


def generate_embeddings(cfg, data_file, output_file, force=False):
    """Generate cell embeddings for a given data file"""
    full_path = os.path.join(DATA_DIR, data_file)
    logger.info(f"Processing: {data_file}")

    # Run inference
    adata_output = run_inference(cfg, data_files=[full_path])

    # Extract embeddings
    if "embeddings" in adata_output.obsm:
        embeddings = adata_output.obsm["embeddings"]
    else:
        raise ValueError("Embeddings not found in output")

    # Extract labels (age)
    if "label" in adata_output.obs.columns:
        labels = adata_output.obs["label"].values
    else:
        labels = None

    # Extract other features if available
    sex = adata_output.obs.get("sex", None)

    # Create output dataframe
    n_cells, emb_dim = embeddings.shape
    embedding_cols = [f"emb_{i}" for i in range(emb_dim)]

    df = pd.DataFrame(embeddings, columns=embedding_cols)
    if labels is not None:
        df["age"] = labels
    if sex is not None:
        df["sex"] = sex.values
    df["source_file"] = data_file

    # Save embeddings
    df.to_parquet(output_file, index=True)
    logger.info(f"Saved {n_cells} cells with {emb_dim}-dim embeddings to {output_file}")

    return df


def main():
    args = parse_args()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    num_gpus = len(args.gpu.split(','))
    logger.info(f"Using GPU(s): {args.gpu} ({num_gpus} GPU(s))")
    logger.info(f"Batch size per GPU: {args.batch_size}")
    logger.info(f"Effective total batch size: {args.batch_size * num_gpus}")

    # Setup output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.combine_only:
        combine_training_embeddings(force=args.force)
        return

    selected_train_files, include_test = resolve_requested_files(args.files)
    logger.info(f"Selected training files: {len(selected_train_files)}")
    for selected_file in selected_train_files:
        logger.info(f"  {selected_file}")
    logger.info(f"Selected test file: {'yes' if include_test else 'no'}")

    # Setup config
    cfg = setup_hydra_config(args)

    logger.info("=" * 60)
    logger.info("TranscriptFormer Cell Embedding Generation")
    logger.info("=" * 60)

    # Generate training embeddings
    logger.info("\n--- Processing Training Files ---")
    all_train_dfs = []
    for train_file in selected_train_files:
        if not os.path.exists(os.path.join(DATA_DIR, train_file)):
            logger.warning(f"File not found, skipping: {train_file}")
            continue

        output_file = output_path_for(train_file)

        # Check if already exists (skip unless --force is set)
        if os.path.exists(output_file) and not args.force:
            logger.info(f"Skipping {train_file} - already exists (use --force to regenerate)")
            df = pd.read_parquet(output_file)
        else:
            df = generate_embeddings(cfg, train_file, output_file, args.force)

        all_train_dfs.append(df)

    # Combined training embeddings
    if selected_train_files and not args.no_combine:
        combine_training_embeddings(force=args.force)

    # Generate test embeddings
    if include_test:
        logger.info("\n--- Processing Test File ---")
        test_output_file = output_path_for(TEST_FILE)

        if os.path.exists(test_output_file) and not args.force:
            logger.info(f"Skipping {TEST_FILE} - already exists (use --force to regenerate)")
        else:
            generate_embeddings(cfg, TEST_FILE, test_output_file, args.force)

    logger.info("\n" + "=" * 60)
    logger.info("Embedding Generation Complete!")
    logger.info("=" * 60)

    # Print summary
    logger.info("\nOutput files:")
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith(EMBEDDING_PREFIX) or f == "train_all_embeddings.parquet":
            fpath = os.path.join(OUTPUT_DIR, f)
            size_mb = os.path.getsize(fpath) / 9542 / 548
            logger.info(f"  {f} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
