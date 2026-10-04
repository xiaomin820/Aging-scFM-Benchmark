#!/usr/bin/env python3
"""Generate audited scFoundation cell embeddings for Task3 senescence datasets."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
import torch


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = Path("/home/zhujialin/zjl/aging_benchmark/repos/scFoundation")
TAG = "scFoundation"
INPUT_SUFFIX = "_subset_1"
EXPECTED_DATASETS = 5
EXPECTED_GENES = 19264
METADATA_COLUMNS = ["label", "sex", "donorID"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--only", help="Dataset id, training, independent, or all")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--read-chunk-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.read_chunk_size < 1:
        parser.error("--read-chunk-size must be positive")
    return args


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.9g")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def dataset_id(path: Path) -> str:
    if not path.stem.endswith(INPUT_SUFFIX):
        raise ValueError("Unexpected input filename: {}".format(path.name))
    return path.stem[: -len(INPUT_SUFFIX)]


def is_training(path: Path) -> bool:
    return dataset_id(path).startswith("Training_")


def discover_inputs(input_root: Path) -> list[Path]:
    paths = sorted(input_root.rglob("*{}.csv".format(INPUT_SUFFIX)))
    ids = [dataset_id(path) for path in paths]
    if len(paths) != EXPECTED_DATASETS:
        raise RuntimeError("Expected 5 Task3 scFoundation CSV files, found {}".format(len(paths)))
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate Task3 scFoundation dataset ids")
    if sum(is_training(path) for path in paths) != 1:
        raise RuntimeError("Expected exactly one Task3 scFoundation training dataset")
    return paths


def select_inputs(paths: list[Path], selector: str) -> list[Path]:
    key = selector.strip()
    lowered = key.lower()
    if lowered == "all":
        return paths
    if lowered == "training":
        return [path for path in paths if is_training(path)]
    if lowered == "independent":
        return [path for path in paths if not is_training(path)]
    exact = [path for path in paths if dataset_id(path) == key]
    if exact:
        return exact
    raise ValueError("Unknown --only value {!r}; run with --list".format(selector))


def load_official_genes(gene_index_path: Path) -> list[str]:
    if not gene_index_path.is_file():
        raise FileNotFoundError(gene_index_path)
    frame = pd.read_csv(gene_index_path, sep="\t")
    if list(frame.columns) != ["gene_name", "index"]:
        raise ValueError("Unexpected official gene-index columns: {}".format(list(frame.columns)))
    genes = frame["gene_name"].astype(str).tolist()
    indices = pd.to_numeric(frame["index"], errors="raise").to_numpy(dtype=np.int64)
    if len(genes) != EXPECTED_GENES or len(set(genes)) != EXPECTED_GENES:
        raise ValueError("Invalid official scFoundation gene list")
    if not np.array_equal(indices, np.arange(EXPECTED_GENES)):
        raise ValueError("Official scFoundation gene indices are not 0..19263")
    return genes


def inspect_header(path: Path, official_genes: list[str]) -> None:
    columns = list(pd.read_csv(path, nrows=0).columns)
    if len(columns) != EXPECTED_GENES + 4:
        raise ValueError("{} has {} columns".format(path.name, len(columns)))
    if columns[0] != "barcode" or columns[-3:] != METADATA_COLUMNS:
        raise ValueError("Unexpected identifier/metadata columns in {}".format(path.name))
    if columns[1:-3] != official_genes:
        mismatch = next(
            (i for i, pair in enumerate(zip(columns[1:-3], official_genes)) if pair[0] != pair[1]),
            None,
        )
        raise ValueError("Gene order mismatch in {} at index {}".format(path.name, mismatch))


def validate_prepared(expression_path: Path, metadata_path: Path) -> tuple[int, int]:
    matrix = sparse.load_npz(expression_path)
    metadata = pd.read_csv(metadata_path, dtype={"cell_id": str})
    if matrix.shape != (len(metadata), EXPECTED_GENES):
        raise ValueError("Prepared matrix/metadata mismatch: {} vs {}".format(matrix.shape, len(metadata)))
    if list(metadata.columns) != ["cell_id"] + METADATA_COLUMNS:
        raise ValueError("Invalid prepared metadata columns")
    ids = metadata["cell_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError("Prepared metadata contains duplicate cell_id values")
    if matrix.data.size and (
        not np.isfinite(matrix.data).all()
        or np.any(matrix.data < 0)
        or not np.isclose(matrix.data, np.round(matrix.data), atol=1e-6).all()
    ):
        raise ValueError("Prepared matrix is not finite nonnegative integer-like raw counts")
    return int(matrix.shape[0]), int(matrix.nnz)


def prepare_input(
    path: Path,
    name: str,
    official_genes: list[str],
    prepared_dir: Path,
    chunk_size: int,
    overwrite: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    inspect_header(path, official_genes)
    expression_path = prepared_dir / "{}_expression_{}.npz".format(name, EXPECTED_GENES)
    metadata_path = prepared_dir / "{}_metadata.csv".format(name)
    if expression_path.is_file() and metadata_path.is_file() and not overwrite:
        rows, nnz = validate_prepared(expression_path, metadata_path)
        print("SKIP_VALIDATED_PREPARED: {}".format(name))
        return expression_path, metadata_path, {"rows": rows, "nnz": nnz, "reused": True}
    if (expression_path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError("Partial prepared input exists for {}".format(name))

    prepared_dir.mkdir(parents=True, exist_ok=True)
    expression_tmp = expression_path.with_name(expression_path.stem + ".tmp.npz")
    metadata_tmp = metadata_path.with_name(metadata_path.name + ".tmp")
    for temporary in (expression_tmp, metadata_tmp):
        if temporary.exists():
            temporary.unlink()

    matrices = []
    seen = set()
    metadata_written = False
    rows = 0
    nonzero_min = np.inf
    nonzero_max = -np.inf
    reader = pd.read_csv(
        path,
        chunksize=chunk_size,
        index_col=0,
        dtype={"barcode": str},
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        if list(chunk.columns[-3:]) != METADATA_COLUMNS:
            raise ValueError("Metadata columns changed in chunk {}".format(chunk_number))
        if list(chunk.columns[:-3]) != official_genes:
            raise ValueError("Gene order changed in chunk {}".format(chunk_number))
        ids = np.asarray(chunk.index.astype(str).tolist(), dtype=str)
        if pd.Index(ids).has_duplicates or any(value in seen for value in ids):
            raise ValueError("Duplicate cell_id in {}".format(name))
        seen.update(ids.tolist())
        values = chunk.iloc[:, :-3].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Non-finite or negative expression in {}".format(name))
        nonzero = values[values != 0]
        if nonzero.size:
            if not np.isclose(nonzero, np.round(nonzero), atol=1e-6).all():
                raise ValueError("Input is not raw integer-like counts: {}".format(name))
            nonzero_min = min(nonzero_min, float(nonzero.min()))
            nonzero_max = max(nonzero_max, float(nonzero.max()))
        matrices.append(sparse.csr_matrix(values))
        metadata = chunk.iloc[:, -3:].copy()
        metadata.insert(0, "cell_id", ids)
        metadata.to_csv(
            metadata_tmp,
            mode="w" if not metadata_written else "a",
            header=not metadata_written,
            index=False,
        )
        metadata_written = True
        rows += len(chunk)
        if chunk_number % 20 == 0:
            print("PREPARE_PROGRESS: {} rows={}".format(name, rows), flush=True)
        del values, chunk

    if not matrices:
        raise ValueError("No rows in {}".format(path))
    matrix = sparse.vstack(matrices, format="csr", dtype=np.float32)
    if matrix.shape != (rows, EXPECTED_GENES):
        raise ValueError("Unexpected prepared shape: {}".format(matrix.shape))
    sparse.save_npz(expression_tmp, matrix, compressed=True)
    os.replace(expression_tmp, expression_path)
    os.replace(metadata_tmp, metadata_path)
    audit = {
        "rows": int(rows),
        "columns": EXPECTED_GENES,
        "nnz": int(matrix.nnz),
        "nonzero_min": None if not np.isfinite(nonzero_min) else nonzero_min,
        "nonzero_max": None if not np.isfinite(nonzero_max) else nonzero_max,
        "finite_nonnegative_integer_like": True,
        "reused": False,
    }
    print("PREPARED_SHAPE: {}".format(matrix.shape))
    print("PREPARED_NNZ: {}".format(matrix.nnz))
    del matrices, matrix
    gc.collect()
    return expression_path, metadata_path, audit


def load_raw_embedding(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=True)
    values = np.asarray(values)
    if values.dtype == object:
        values = np.asarray(values.tolist())
    if values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    if values.ndim != 2:
        raise ValueError("Unexpected raw embedding shape {}".format(values.shape))
    return values.astype(np.float32, copy=False)


def validate_complete(npz_path: Path, metadata_path: Path, manifest_path: Path) -> tuple[int, int]:
    metadata = pd.read_csv(metadata_path, dtype={"cell_id": str})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != {"X", "cell_id"}:
            raise ValueError("Unexpected NPZ keys in {}".format(npz_path))
        values = np.asarray(payload["X"], dtype=np.float32)
        ids = np.asarray(payload["cell_id"], dtype=str)
    if values.ndim != 2 or values.shape[0] != len(metadata) or values.shape[1] < 1:
        raise ValueError("Invalid existing embedding shape: {}".format(values.shape))
    if not np.isfinite(values).all():
        raise ValueError("Non-finite existing embeddings")
    if not np.array_equal(ids, metadata["cell_id"].astype(str).to_numpy()):
        raise ValueError("Existing NPZ/metadata order mismatch")
    if manifest.get("embedding", {}).get("shape") != list(values.shape):
        raise ValueError("Existing manifest shape mismatch")
    return int(values.shape[0]), int(values.shape[1])


def process_one(
    path: Path,
    model_dir: Path,
    get_embedding: Path,
    inference_runner: Path,
    checkpoint: Path,
    official_genes: list[str],
    prepared_dir: Path,
    work_dir: Path,
    output_dir: Path,
    chunk_size: int,
    device: str,
    resource_hashes: dict[str, str],
    overwrite: bool,
) -> None:
    name = dataset_id(path)
    npz_path = output_dir / "{}_{}_cell_embeddings.npz".format(name, TAG)
    metadata_path = output_dir / "{}_{}_cell_embeddings_metadata.csv".format(name, TAG)
    manifest_path = output_dir / "{}_{}_embedding_manifest.json".format(name, TAG)
    outputs = [npz_path, metadata_path, manifest_path]
    if all(item.is_file() for item in outputs) and not overwrite:
        rows, dimensions = validate_complete(npz_path, metadata_path, manifest_path)
        print("SKIP_VALIDATED_COMPLETE: {} shape=({}, {})".format(name, rows, dimensions))
        return
    if any(item.exists() for item in outputs) and not overwrite:
        raise FileExistsError("Partial embedding outputs exist for {}".format(name))

    expression_path, prepared_metadata_path, preparation = prepare_input(
        path, name, official_genes, prepared_dir, chunk_size, overwrite
    )
    metadata = pd.read_csv(prepared_metadata_path, dtype={"cell_id": str})
    source_ids = np.asarray(metadata["cell_id"].astype(str).tolist(), dtype=str)
    raw_dir = work_dir / name / "raw_embedding"
    raw_dir.mkdir(parents=True, exist_ok=True)
    task_name = "task2_{}".format(name)
    before = set(raw_dir.glob("*.npy"))
    command = [
        sys.executable,
        str(inference_runner),
        str(get_embedding),
        "--task_name", task_name,
        "--input_type", "singlecell",
        "--output_type", "cell",
        "--pool_type", "all",
        "--tgthighres", "t4",
        "--data_path", str(expression_path),
        "--save_path", str(raw_dir) + os.sep,
        "--pre_normalized", "F",
        "--version", "ce",
    ]
    print("DATASET: {}".format(name), flush=True)
    print("INPUT_ROWS: {}".format(len(metadata)), flush=True)
    print("SCFOUNDATION_COMMAND: {}".format(" ".join(command)), flush=True)
    subprocess.run(command, check=True, cwd=str(model_dir))
    after = set(raw_dir.glob("*.npy"))
    candidates = sorted(after - before)
    if not candidates:
        candidates = sorted(raw_dir.glob("{}*.npy".format(task_name)))
    if not candidates:
        candidates = sorted(raw_dir.glob("*.npy"))
    if not candidates:
        raise FileNotFoundError("scFoundation produced no .npy file in {}".format(raw_dir))
    raw_path = max(candidates, key=lambda item: item.stat().st_mtime)
    values = load_raw_embedding(raw_path)
    if values.shape[0] != len(metadata) or values.shape[1] < 1:
        raise ValueError("Embedding/metadata shape mismatch: {} vs {}".format(values.shape, len(metadata)))
    if not np.isfinite(values).all():
        raise ValueError("Non-finite scFoundation embeddings for {}".format(name))
    atomic_npz(npz_path, X=values, cell_id=source_ids)
    atomic_csv(metadata_path, metadata)
    norms = np.linalg.norm(values, axis=1)
    manifest = {
        "manifest_version": "1.0",
        "created_at_utc": now_utc(),
        "dataset_id": name,
        "role": "training" if is_training(path) else "independent_test",
        "input": {
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
            "gene_count": EXPECTED_GENES,
            "gene_order_matches_official_index": True,
            **preparation,
        },
        "model": {
            "repository": str(model_dir.parent.resolve()),
            "get_embedding": str(get_embedding.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "sha256": resource_hashes,
        },
        "embedding_call": {
            "input_type": "singlecell",
            "output_type": "cell",
            "pool_type": "all",
            "tgthighres": "t4",
            "pre_normalized": "F",
            "version": "ce",
            "device": device,
            "pytorch_mha_fastpath": False,
            "pytorch_sdpa_backends": ["flash", "memory_efficient", "math_fallback"],
        },
        "embedding": {
            "shape": [int(values.shape[0]), int(values.shape[1])],
            "dtype": str(values.dtype),
            "finite": True,
            "l2_norm_min": float(norms.min()),
            "l2_norm_median": float(np.median(norms)),
            "l2_norm_max": float(norms.max()),
            "raw_output": str(raw_path.resolve()),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "torch": torch.__version__,
        },
    }
    atomic_json(manifest_path, manifest)
    print("EMBEDDING_SHAPE: {}".format(values.shape))
    print("EMBEDDING_DTYPE: {}".format(values.dtype))
    print("ALL_FINITE: True")
    print("SAVED_NPZ: {}".format(npz_path))
    print("SAVED_METADATA: {}".format(metadata_path))
    print("DATASET_FINISHED: {}".format(name), flush=True)
    del values, metadata
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    repo = args.repo.expanduser().resolve()
    model_dir = repo / "model"
    get_embedding = model_dir / "get_embedding.py"
    inference_runner = Path(__file__).resolve().with_name(
        "run_scfoundation_memory_efficient.py"
    )
    checkpoint = model_dir / "models" / "models.ckpt"
    gene_index = model_dir / "OS_scRNA_gene_index.19264.tsv"
    for required in (get_embedding, checkpoint, gene_index, inference_runner):
        if not required.is_file():
            raise FileNotFoundError(required)
    input_root = project_root / "scfoundation" / "data"
    method_root = project_root / "scfoundation"
    output_dir = method_root / "outputs" / "embeddings"
    prepared_dir = method_root / "tmp" / "prepared"
    work_dir = method_root / "tmp" / "embedding_work"
    paths = discover_inputs(input_root)
    if args.list:
        for path in paths:
            print("{}\t{}".format(dataset_id(path), path))
        return 0
    if not args.only:
        print("ERROR: --only is required unless --list is used", file=sys.stderr)
        return 2
    official_genes = load_official_genes(gene_index)
    selected = select_inputs(paths, args.only)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resource_hashes = {
        "models.ckpt": sha256(checkpoint),
        "get_embedding.py": sha256(get_embedding),
        "OS_scRNA_gene_index.19264.tsv": sha256(gene_index),
    }
    print("DEVICE: {}".format(device))
    print("SELECTED_DATASET_COUNT: {}".format(len(selected)))
    print("MODEL_RESOURCES_VALID: True")
    print("OFFICIAL_GENE_ORDER_VALID: True")
    for path in selected:
        process_one(
            path,
            model_dir,
            get_embedding,
            inference_runner,
            checkpoint,
            official_genes,
            prepared_dir,
            work_dir,
            output_dir,
            args.read_chunk_size,
            device,
            resource_hashes,
            args.overwrite,
        )
    print("ALL_SELECTED_TASK3_SCFOUNDATION_EMBEDDINGS_FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
