#!/usr/bin/env python3
"""Batch-generate UCE cell embeddings for the Task1 aging h5ad files."""
"""
Generate foundation-model cell embeddings for Task1_Training_Part1-5 and Task1_Independent.Test
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_INPUT_DIR = Path("aging_data/UCE_h5ad")
DEFAULT_OUTPUT_DIR = Path("outputs/aging_uce_embeddings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run eval_single_anndata.py on all Task1 aging h5ad files."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--species", type=str, default="human")
    parser.add_argument("--model_loc", type=Path, default=Path("model_files/33l_8ep_1024t_1280.torch"))
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--nlayers", type=int, default=33)
    parser.add_argument("--filter", type=str, default="False", choices=["True", "False"])
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of Accelerate processes/GPUs to use per h5ad file.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU ids passed to Accelerate, e.g. 0,1,2,3.",
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to invoke eval_single_anndata.py.",
    )
    return parser.parse_args()


def expected_output(output_dir: Path, h5ad_path: Path) -> Path:
    return output_dir / f"{h5ad_path.stem}_uce_adata.h5ad"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    h5ad_paths = sorted(args.input_dir.glob("Task1_Training_Part*_UCE_input.h5ad"))
    h5ad_paths.extend(sorted(args.input_dir.glob("Task1_Independent.Test*_UCE_input.h5ad")))
    if not h5ad_paths:
        raise FileNotFoundError(f"No Task1 h5ad files found in {args.input_dir}")

    if not args.model_loc.exists() or args.model_loc.stat().st_size == 0:
        raise FileNotFoundError(f"Model file is missing or empty: {args.model_loc}")

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/uce_matplotlib")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/uce_numba_cache")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    for h5ad_path in h5ad_paths:
        out_path = expected_output(output_dir, h5ad_path)
        if args.skip_existing and out_path.exists():
            print(f"[skip] {out_path}")
            continue

        print(f"[embed] {h5ad_path} -> {out_path}")
        script_args = [
            "eval_single_anndata.py",
            "--adata_path",
            str(h5ad_path),
            "--dir",
            str(output_dir) + "/",
            "--species",
            args.species,
            "--model_loc",
            str(args.model_loc),
            "--batch_size",
            str(args.batch_size),
            "--nlayers",
            str(args.nlayers),
            "--filter",
            args.filter,
        ]
        if args.num_processes > 1:
            cmd = [
                args.python,
                "-m",
                "accelerate.commands.launch",
                "--multi_gpu",
                "--num_processes",
                str(args.num_processes),
            ]
            if args.gpu_ids:
                cmd.extend(["--gpu_ids", args.gpu_ids])
            cmd.extend([*script_args, "--multi_gpu", "True"])
        else:
            cmd = [args.python, *script_args]
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
