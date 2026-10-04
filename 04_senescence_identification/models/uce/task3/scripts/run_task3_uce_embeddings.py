#!/usr/bin/env python3
"""Generate UCE cell embeddings for Task3 senescent-cell classification data."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "task3/data/task3_Senescent_UCE_input"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "task3/outputs/uce_embeddings"
DEFAULT_MODEL_LOC = REPO_ROOT / "model_files/33l_8ep_1024t_1280.torch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UCE embedding generation for Task3 h5ad files.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_loc", type=Path, default=DEFAULT_MODEL_LOC)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--nlayers", type=int, default=33)
    parser.add_argument("--filter", type=str, default="False", choices=["True", "False"])
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    return parser.parse_args()


def infer_species(h5ad_path: Path) -> str:
    name = h5ad_path.name.lower()
    if "_mouse_" in name:
        return "mouse"
    return "human"


def expected_output(output_dir: Path, h5ad_path: Path) -> Path:
    return output_dir / f"{h5ad_path.stem}_uce_adata.h5ad"


def task3_h5ad_paths(input_dir: Path) -> list[Path]:
    train_paths = sorted(input_dir.glob("Training_task3*_UCE_input.h5ad"))
    test_paths = sorted(input_dir.glob("Independent.Test_task3*_UCE_input.h5ad"))
    if len(train_paths) != 1:
        raise FileNotFoundError(f"Expected 1 Task3 training h5ad file, found {len(train_paths)} in {input_dir}")
    if not test_paths:
        raise FileNotFoundError(f"No Task3 independent test h5ad files found in {input_dir}")
    return [train_paths[0], *test_paths]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.model_loc.exists() or args.model_loc.stat().st_size == 0:
        raise FileNotFoundError(f"Model file is missing or empty: {args.model_loc}")

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/uce_matplotlib")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/uce_numba_cache")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    for h5ad_path in task3_h5ad_paths(args.input_dir):
        out_path = expected_output(args.output_dir, h5ad_path)
        if args.skip_existing and out_path.exists():
            print(f"[skip] {out_path}")
            continue

        species = infer_species(h5ad_path)
        print(f"[embed] species={species} {h5ad_path} -> {out_path}")
        script_args = [
            "eval_single_anndata.py",
            "--adata_path",
            str(h5ad_path),
            "--dir",
            str(args.output_dir) + "/",
            "--species",
            species,
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
        subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    main()
