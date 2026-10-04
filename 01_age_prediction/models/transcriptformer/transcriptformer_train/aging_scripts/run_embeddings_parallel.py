"""
Launch multiple single-GPU embedding workers from one terminal.

Example:
    python aging_scripts/run_embeddings_parallel.py --gpus 2,3,4 --batch_size 8
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


OUTPUT_DIR = Path("./embedding_results")
LOG_DIR = OUTPUT_DIR / "logs"

TRAIN_ALIASES = ["train1", "train2", "train3", "train4", "train5"]
ALL_ALIASES = [*TRAIN_ALIASES, "test"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run TranscriptFormer embedding generation in parallel by file.")
    parser.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated physical GPU IDs, for example: 2,3,4",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for each single-GPU worker (default: 8)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers per GPU process (default: 4)",
    )
    parser.add_argument(
        "--files",
        default="all",
        help="Files to process: all, train, test, train1..train5, comma-separated (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate embeddings even if output files already exist",
    )
    parser.add_argument(
        "--skip_combine",
        action="store_true",
        help="Do not combine training parquet files after all workers finish",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print worker commands without running them",
    )
    return parser.parse_args()


def resolve_file_aliases(files_arg):
    if files_arg is None or files_arg.strip().lower() == "all":
        return list(ALL_ALIASES)

    resolved = []
    unknown = []
    for raw_item in files_arg.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item == "all":
            resolved.extend(ALL_ALIASES)
        elif item == "train":
            resolved.extend(TRAIN_ALIASES)
        elif item == "test" or item in TRAIN_ALIASES:
            resolved.append(item)
        else:
            unknown.append(raw_item.strip())

    if unknown:
        valid = ", ".join(["all", "train", "test", *TRAIN_ALIASES])
        raise ValueError(f"Unknown --files item(s): {unknown}. Valid aliases: {valid}")

    return list(dict.fromkeys(resolved))


def split_round_robin(items, n_groups):
    groups = [[] for _ in range(n_groups)]
    for i, item in enumerate(items):
        groups[i % n_groups].append(item)
    return groups


def build_worker_command(gpu, file_aliases, args):
    cmd = [
        sys.executable,
        "aging_scripts/01_generate_embeddings.py",
        "--gpu",
        gpu,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--files",
        ",".join(file_aliases),
        "--no_combine",
    ]
    if args.force:
        cmd.append("--force")
    return cmd


def print_log_tail(log_path, max_lines=30):
    if not log_path.exists():
        return
    lines = log_path.read_text(errors="replace").splitlines()
    print(f"\n--- Last {min(max_lines, len(lines))} lines from {log_path} ---")
    for line in lines[-max_lines:]:
        print(line)


def main():
    args = parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("No GPUs specified")

    file_aliases = resolve_file_aliases(args.files)
    assignments = split_round_robin(file_aliases, len(gpus))

    OUTPUT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    processes = []
    print("Launching embedding workers:")
    for gpu, worker_files in zip(gpus, assignments, strict=False):
        if not worker_files:
            continue
        cmd = build_worker_command(gpu, worker_files, args)
        log_path = LOG_DIR / f"embed_gpu{gpu}_{stamp}.log"
        print(f"  GPU {gpu}: {','.join(worker_files)}")
        print(f"    log: {log_path}")
        print(f"    cmd: {' '.join(cmd)}")

        if args.dry_run:
            continue

        log_file = log_path.open("w")
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((gpu, worker_files, process, log_file, log_path))

    if args.dry_run:
        return

    failed = []
    for gpu, worker_files, process, log_file, log_path in processes:
        return_code = process.wait()
        log_file.close()
        if return_code == 0:
            print(f"GPU {gpu} finished: {','.join(worker_files)}")
        else:
            print(f"GPU {gpu} failed with exit code {return_code}: {','.join(worker_files)}")
            print_log_tail(log_path)
            failed.append((gpu, return_code))

    if failed:
        failed_text = ", ".join([f"GPU {gpu} exit {code}" for gpu, code in failed])
        raise SystemExit(f"One or more workers failed: {failed_text}")

    has_training_files = any(alias in TRAIN_ALIASES for alias in file_aliases)
    if has_training_files and not args.skip_combine:
        combine_cmd = [sys.executable, "aging_scripts/01_generate_embeddings.py", "--combine_only"]
        if args.force:
            combine_cmd.append("--force")
        print("\nCombining training embeddings:")
        print(f"  cmd: {' '.join(combine_cmd)}")
        subprocess.run(combine_cmd, check=True)

    print("\nParallel embedding generation complete.")


if __name__ == "__main__":
    main()
