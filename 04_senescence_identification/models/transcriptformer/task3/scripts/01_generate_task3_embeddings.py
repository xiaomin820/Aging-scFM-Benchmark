"""
Generate TranscriptFormer cell embeddings for Task3 senescent-cell classification.

Example:
    python task3/scripts/01_generate_task3_embeddings.py --gpu 2 --batch_size 8
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def _get_arg_value(argv, name, default=None):
    prefix = f"{name}="
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return default


_early_gpu_arg = _get_arg_value(sys.argv, "--gpu")
if _early_gpu_arg:
    os.environ["CUDA_VISIBLE_DEVICES"] = _early_gpu_arg

os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from transcriptformer.model.inference import run_inference


torch._dynamo.config.disable = True
torch._dynamo.reset()
torch.compile = lambda *args, **kwargs: args[0] if args else None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_INPUT_DIR = "./task3/data/task3_Senescent_TranscriptFormer_input"
DEFAULT_OUTPUT_DIR = "./task3/embedding_results"
DEFAULT_CHECKPOINT_PATH = "./checkpoints/tf_sapiens"
TRAIN_PREFIX = "Training_task3"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Task3 TranscriptFormer cell embeddings.")
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Directory containing Task3 h5ad files")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output parquet embeddings")
    parser.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT_PATH, help="TranscriptFormer checkpoint directory")
    parser.add_argument("--gpu", default="0", help="GPU IDs, for example: 2 or 2,3")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--files",
        default="all",
        help="Files to process: all, train, test, or comma-separated h5ad basenames",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate embeddings if output exists")
    return parser.parse_args()


def resolve_input_files(input_dir, files_arg):
    input_dir = Path(input_dir)
    all_files = sorted(input_dir.glob("*.h5ad"))
    train_files = [p for p in all_files if p.name.startswith(TRAIN_PREFIX)]
    test_files = [p for p in all_files if not p.name.startswith(TRAIN_PREFIX)]

    if files_arg is None or files_arg.strip().lower() == "all":
        files = all_files
    elif files_arg.strip().lower() == "train":
        files = train_files
    elif files_arg.strip().lower() == "test":
        files = test_files
    else:
        files = [input_dir / item.strip() for item in files_arg.split(",") if item.strip()]

    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")
    if not files:
        raise FileNotFoundError(f"No h5ad files selected from {input_dir}")
    return files


def setup_hydra_config(args):
    config_dir = os.path.abspath("src/transcriptformer/cli/conf")
    with initialize_config_dir(config_dir=config_dir):
        cfg = compose(config_name="inference_config.yaml")

    OmegaConf.set_struct(cfg, False)

    checkpoint_config_path = os.path.join(args.checkpoint_path, "config.json")
    with open(checkpoint_config_path) as f:
        model_config_dict = json.load(f)

    for section in ["data_config", "model_config", "loss_config"]:
        if section in model_config_dict.get("model", {}):
            for key, val in model_config_dict["model"][section].items():
                if val is not None:
                    OmegaConf.update(cfg, f"model.{section}.{key}", val, merge=False)

    num_gpus = len([gpu for gpu in args.gpu.split(",") if gpu.strip()])
    cfg.model.checkpoint_path = args.checkpoint_path
    cfg.model.inference_config.load_checkpoint = os.path.join(args.checkpoint_path, "model_weights.pt")
    cfg.model.inference_config.batch_size = args.batch_size
    cfg.model.inference_config.emb_type = "cell"
    cfg.model.inference_config.output_keys = ["embeddings"]
    cfg.model.inference_config.obs_keys = ["all"]
    cfg.model.inference_config.pretrained_embedding = None
    cfg.model.inference_config.device = "cuda"
    cfg.model.inference_config.num_gpus = num_gpus
    cfg.model.data_config.n_data_workers = args.num_workers
    cfg.model.data_config.aux_vocab_path = os.path.join(args.checkpoint_path, "vocabs")
    cfg.model.data_config.esm2_mappings_path = os.path.join(args.checkpoint_path, "vocabs")
    cfg.model.model_config.compile_block_mask = False
    return cfg


def output_path_for(output_dir, input_path):
    return Path(output_dir) / f"{input_path.stem}_embeddings.parquet"


def generate_embeddings_for_file(cfg, input_path, output_path):
    logger.info(f"Processing {input_path.name}")
    adata_output = run_inference(cfg, data_files=[str(input_path)])
    if "embeddings" not in adata_output.obsm:
        raise ValueError(f"Embeddings not found for {input_path}")

    embeddings = adata_output.obsm["embeddings"]
    obs_df = adata_output.obs.copy()
    obs_df.insert(0, "cell_id", obs_df.index.astype(str))
    obs_df.insert(1, "source_file", input_path.name)
    obs_df = obs_df.reset_index(drop=True)

    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    output_df = pd.concat([obs_df, emb_df], axis=1)
    output_df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(output_df)} cells to {output_path}")


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = resolve_input_files(args.input_dir, args.files)
    logger.info(f"Selected {len(input_files)} file(s)")
    for path in input_files:
        logger.info(f"  {path.name}")
    logger.info(f"GPU(s): {args.gpu}; batch size per GPU: {args.batch_size}")

    cfg = setup_hydra_config(args)
    for input_path in input_files:
        output_path = output_path_for(output_dir, input_path)
        if output_path.exists() and not args.force:
            logger.info(f"Skipping existing output: {output_path}")
            continue
        generate_embeddings_for_file(cfg, input_path, output_path)

    logger.info("Task3 embedding generation complete.")


if __name__ == "__main__":
    main()
