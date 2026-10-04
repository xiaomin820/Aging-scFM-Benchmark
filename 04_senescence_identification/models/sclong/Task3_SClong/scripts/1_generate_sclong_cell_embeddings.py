#!/usr/bin/env python3
"""Generate audited 200-dimensional mean-pooled scLONG embeddings for Task3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = Path("/home/zhujialin/zjl/aging_benchmark/repos/scLong")
DEFAULT_RESOURCE_ROOT = Path(
    "/home/zhujialin/shared/zhujialin/sclong/checkpoints"
)
INPUT_SUFFIX = "_ensembl_aligned"
TAG = "scLONG"
EXPECTED_GENE_COUNT = 27874
EXPECTED_EMBED_DIM = 200
EXPECTED_EMBED_PY_SHA256 = (
    "1f294832bacdaa96862940f5e8fa8aa4bf9095a0fd4ca0f9e285baa7adb333b9"
)
AUDIT_CHUNK_SIZE = 256
CSV_CHUNK_SIZE = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--resource-root", type=Path, default=DEFAULT_RESOURCE_ROOT)
    parser.add_argument(
        "--only",
        help="Exact dataset id, 'training', 'independent', or 'all'. Use --list for ids.",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--export-training-csv", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be positive")
    return args


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def dataset_id(path: Path) -> str:
    stem = path.stem
    if not stem.endswith(INPUT_SUFFIX):
        raise ValueError(f"Unexpected scLONG input filename: {path.name}")
    return stem[: -len(INPUT_SUFFIX)]


def is_training(path: Path) -> bool:
    return dataset_id(path).startswith("Training_")


def classify_dataset(name: str) -> Dict[str, str]:
    species = "mouse" if "_mouse_" in name else "human"
    if "_bulk_" in name:
        observation_type = "bulk_sample"
    elif "_excise_" in name:
        observation_type = "excised_sample"
    else:
        observation_type = "single_cell"
    return {"species": species, "observation_type": observation_type}


def discover_inputs(input_root: Path) -> List[Path]:
    paths = sorted(input_root.rglob("*.h5ad"))
    ids = [dataset_id(path) for path in paths]
    if len(paths) != 5:
        raise RuntimeError(f"Expected 5 H5AD files, found {len(paths)} under {input_root}")
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate Task3 scLONG dataset ids were discovered")
    if sum(is_training(path) for path in paths) != 1:
        raise RuntimeError("Expected exactly one Task3 scLONG training H5AD")
    return paths


def select_inputs(paths: List[Path], selector: str) -> List[Path]:
    key = selector.strip()
    lowered = key.lower()
    if lowered == "all":
        return paths
    if lowered == "training":
        return [path for path in paths if is_training(path)]
    if lowered == "independent":
        return [path for path in paths if not is_training(path)]
    selected = [path for path in paths if dataset_id(path) == key]
    if selected:
        return selected
    raise ValueError(f"Unknown --only value {selector!r}; run with --list")


def resolve_resources(repo: Path, resource_root: Path) -> Dict[str, Path]:
    resources = {
        "embed_py": repo / "embed.py",
        "gene2vec": repo / "selected_gene2vec_27k.npy",
        "selected_genes": repo / "selected_genes_27k.txt",
        "hyper_parameters": resource_root / "gocont_4096_48m_pretrain_1b_mix.pkl",
        "foundation_checkpoint": resource_root
        / "gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth",
    }
    missing = [str(path) for path in resources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing scLONG resources:\n" + "\n".join(missing))
    embed_hash = sha256_file(resources["embed_py"])
    if embed_hash != EXPECTED_EMBED_PY_SHA256:
        raise ValueError(
            "Unexpected embed.py SHA256; refusing to run an unaudited implementation: "
            f"{embed_hash}"
        )
    return resources


def load_selected_genes(path: Path) -> np.ndarray:
    genes = np.asarray(
        [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()],
        dtype=str,
    )
    if len(genes) != EXPECTED_GENE_COUNT:
        raise ValueError(f"Selected gene count {len(genes)} != {EXPECTED_GENE_COUNT}")
    if len(set(genes.tolist())) != len(genes):
        raise ValueError("selected_genes_27k.txt contains duplicate genes")
    return genes


def matrix_nonzero_values(block: Any) -> np.ndarray:
    if sparse.issparse(block):
        return np.asarray(block.data)
    values = np.asarray(block)
    return values[values != 0]


def audit_and_metadata(
    path: Path, selected_genes: np.ndarray
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    backed = ad.read_h5ad(path, backed="r")
    try:
        if not backed.obs_names.is_unique:
            raise ValueError(f"obs_names are not unique: {path}")
        if not backed.var_names.is_unique:
            raise ValueError(f"var_names are not unique: {path}")
        if "label" not in backed.obs.columns:
            raise ValueError(f"Missing obs['label']: {path}")
        genes = np.asarray(backed.var_names, dtype=str)
        if not np.array_equal(genes, selected_genes):
            raise ValueError(f"Gene order does not exactly match selected_genes_27k.txt: {path}")

        metadata = backed.obs.copy().reset_index(drop=True)
        metadata.insert(0, "cell_id", np.asarray(backed.obs_names, dtype=str))
        numeric_labels = pd.to_numeric(metadata["label"], errors="coerce")
        if numeric_labels.isna().any():
            raise ValueError(f"Missing or nonnumeric binary labels: {path}")
        unique_labels = set(numeric_labels.astype(float).unique().tolist())
        if not unique_labels.issubset({0.0, 1.0}):
            raise ValueError(f"Labels outside {{0,1}} in {path}: {sorted(unique_labels)}")
        if is_training(path) and unique_labels != {0.0, 1.0}:
            raise ValueError(f"Training data must contain both binary classes: {path}")

        finite = True
        negative_count = 0
        nonzero_count = 0
        integer_count = 0
        nonzero_min: Optional[float] = None
        nonzero_max: Optional[float] = None
        for start in range(0, backed.n_obs, AUDIT_CHUNK_SIZE):
            stop = min(start + AUDIT_CHUNK_SIZE, backed.n_obs)
            values = matrix_nonzero_values(backed.X[start:stop, :])
            if values.size:
                finite = finite and bool(np.isfinite(values).all())
                negative_count += int(np.count_nonzero(values < 0))
                nonzero_count += int(values.size)
                integer_count += int(np.count_nonzero(values == np.rint(values)))
                current_min = float(values.min())
                current_max = float(values.max())
                nonzero_min = current_min if nonzero_min is None else min(nonzero_min, current_min)
                nonzero_max = current_max if nonzero_max is None else max(nonzero_max, current_max)
        if not finite:
            raise ValueError(f"Expression contains NaN or Inf: {path}")
        if negative_count:
            raise ValueError(f"Expression contains {negative_count} negative values: {path}")
        if nonzero_count == 0:
            raise ValueError(f"Expression matrix is entirely zero: {path}")
        integer_fraction = integer_count / nonzero_count
        if integer_fraction > 0.99:
            raise ValueError(
                f"Input appears to contain raw counts, not log1p values: {path}; "
                f"integer_fraction={integer_fraction}"
            )

        audit = {
            "path": str(path.resolve()),
            "file_size_bytes": int(path.stat().st_size),
            "shape": [int(backed.n_obs), int(backed.n_vars)],
            "unique_cell_ids": True,
            "unique_gene_ids": True,
            "gene_count": int(len(genes)),
            "gene_order_sha256": sha256_strings(genes.tolist()),
            "selected_gene_order_exact_match": True,
            "numeric_label_count": int(numeric_labels.notna().sum()),
            "missing_or_nonnumeric_label_count": int(numeric_labels.isna().sum()),
            "label_0_count": int((numeric_labels == 0).sum()),
            "label_1_count": int((numeric_labels == 1).sum()),
            "expression": {
                "interpretation": "nonnegative continuous log1p expression",
                "finite": finite,
                "negative_count": negative_count,
                "nonzero_count": nonzero_count,
                "nonzero_integer_fraction": integer_fraction,
                "nonzero_min": nonzero_min,
                "nonzero_max": nonzero_max,
            },
        }
        return metadata, audit
    finally:
        backed.file.close()


def validate_embedding(
    values: np.ndarray, expected_rows: int, source: Path
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (expected_rows, EXPECTED_EMBED_DIM):
        raise ValueError(
            f"{source}: embedding shape {array.shape} != ({expected_rows}, {EXPECTED_EMBED_DIM})"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{source}: embedding contains NaN or Inf")
    return array


def write_embedding_csv(path: Path, values: np.ndarray, cell_ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary CSV exists: {temporary}")
    columns = [f"embedding_{index:03d}" for index in range(values.shape[1])]
    wrote_header = False
    for start in range(0, len(values), CSV_CHUNK_SIZE):
        stop = min(start + CSV_CHUNK_SIZE, len(values))
        frame = pd.DataFrame(values[start:stop], columns=columns)
        frame.insert(0, "cell_id", cell_ids[start:stop])
        frame.to_csv(
            temporary,
            mode="a",
            header=not wrote_header,
            index=False,
            float_format="%.9g",
        )
        wrote_header = True
    os.replace(temporary, path)


def validate_existing(
    npz_path: Path, metadata_path: Path, expected_rows: int, cell_ids: np.ndarray
) -> None:
    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != {"X", "cell_id"}:
            raise ValueError(f"Unexpected keys in {npz_path}: {payload.files}")
        values = validate_embedding(payload["X"], expected_rows, npz_path)
        del values
        stored_ids = np.asarray(payload["cell_id"], dtype=str)
    if not np.array_equal(stored_ids, cell_ids):
        raise ValueError(f"Stored cell_id order differs: {npz_path}")
    stored_metadata = pd.read_csv(metadata_path, dtype={"cell_id": str})
    if not np.array_equal(stored_metadata["cell_id"].astype(str).to_numpy(), cell_ids):
        raise ValueError(f"Stored metadata cell_id order differs: {metadata_path}")


def process_one(
    path: Path,
    project_root: Path,
    repo: Path,
    resources: Dict[str, Path],
    selected_genes: np.ndarray,
    batch_size: int,
    export_training_csv: bool,
    overwrite: bool,
) -> None:
    name = dataset_id(path)
    training = is_training(path)
    method_root = project_root
    embedding_dir = method_root / "outputs" / "embeddings"
    temp_dir = method_root / "tmp" / name
    npz_path = embedding_dir / f"{name}_{TAG}_cell_embeddings.npz"
    metadata_path = embedding_dir / f"{name}_{TAG}_cell_embeddings_metadata.csv"
    csv_path = embedding_dir / f"{name}_{TAG}_cell_embeddings.csv"
    manifest_path = embedding_dir / f"{name}_{TAG}_embedding_manifest.json"
    csv_required = (not training) or export_training_csv
    required = [npz_path, metadata_path, manifest_path]
    if csv_required:
        required.append(csv_path)

    print(f"\n===== DATASET: {name} =====", flush=True)
    metadata, input_audit = audit_and_metadata(path, selected_genes)
    cell_ids = metadata["cell_id"].astype(str).to_numpy(dtype=str)
    if all(output.is_file() for output in required) and not overwrite:
        validate_existing(npz_path, metadata_path, len(metadata), cell_ids)
        print(f"SKIP_VALIDATED_COMPLETE: {name}")
        return
    if any(output.exists() for output in required) and not overwrite:
        existing = [str(output) for output in required if output.exists()]
        raise FileExistsError(
            f"Partial outputs already exist for {name}; inspect or use --overwrite: {existing}"
        )

    temp_dir.mkdir(parents=True, exist_ok=True)
    raw_path = temp_dir / f"{name}_{TAG}_raw_200d.npy"
    if raw_path.is_file() and not overwrite:
        values = validate_embedding(np.load(raw_path, mmap_mode="r"), len(metadata), raw_path)
        print(f"REUSING_VALIDATED_RAW_200D: {raw_path}")
    else:
        work_path = temp_dir / f".{name}_{TAG}_raw_200d.{os.getpid()}.partial.npy"
        if work_path.exists():
            raise FileExistsError(f"Stale temporary embedding exists: {work_path}")
        command = [
            sys.executable,
            str(resources["embed_py"]),
            "--target_data_path",
            str(path),
            "--target_embed_path",
            str(work_path),
            "--gene2vec_path",
            str(resources["gene2vec"]),
            "--scfm_hyper_params_path",
            str(resources["hyper_parameters"]),
            "--scfm_ckpt_path",
            str(resources["foundation_checkpoint"]),
            "--scfm_genes_list_path",
            str(resources["selected_genes"]),
            "--batch_size",
            str(batch_size),
        ]
        print("COMMAND:", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=str(repo))
        if not work_path.is_file():
            raise FileNotFoundError(f"embed.py did not create {work_path}")
        values = validate_embedding(np.load(work_path, mmap_mode="r"), len(metadata), work_path)
        os.replace(work_path, raw_path)
        print(f"SAVED_VALIDATED_RAW_200D: {raw_path}")

    atomic_npz(npz_path, X=np.asarray(values, dtype=np.float32), cell_id=cell_ids)
    atomic_csv(metadata_path, metadata)
    if csv_required:
        write_embedding_csv(csv_path, values, cell_ids)

    norms = np.linalg.norm(np.asarray(values, dtype=np.float32), axis=1)
    domain = classify_dataset(name)
    manifest = {
        "manifest_version": "1.0",
        "created_at_utc": now_utc(),
        "dataset_id": name,
        "role": "training" if training else "independent_test",
        **domain,
        "input": input_audit,
        "model": {
            "name": TAG,
            "embed_py": str(resources["embed_py"]),
            "embed_py_sha256": EXPECTED_EMBED_PY_SHA256,
            "foundation_checkpoint": str(resources["foundation_checkpoint"]),
            "mean_pooling": "mean across 27,874 aligned gene embeddings",
        },
        "embedding": {
            "shape": [int(values.shape[0]), int(values.shape[1])],
            "dtype": "float32",
            "finite": True,
            "batch_size": batch_size,
            "l2_norm_min": float(norms.min()),
            "l2_norm_median": float(np.median(norms)),
            "l2_norm_max": float(norms.max()),
        },
        "outputs": {
            "npz": str(npz_path.resolve()),
            "metadata_csv": str(metadata_path.resolve()),
            "embedding_csv": str(csv_path.resolve()) if csv_required else None,
        },
        "scientific_note": (
            "Bulk and mouse inputs are cross-domain for a human single-cell foundation model; "
            "their embeddings and predictions must be interpreted as exploratory."
        ),
    }
    atomic_json(manifest_path, manifest)
    print(f"EMBEDDING_SHAPE: {values.shape}")
    print(f"SAVED_NPZ: {npz_path}")
    print(f"SAVED_METADATA: {metadata_path}")
    if csv_required:
        print(f"SAVED_EMBEDDING_CSV: {csv_path}")
    print(f"DATASET_FINISHED: {name}", flush=True)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    repo = args.repo.expanduser().resolve()
    resource_root = args.resource_root.expanduser().resolve()
    input_root = project_root / "data"
    paths = discover_inputs(input_root)
    if args.list:
        for path in paths:
            print(f"{dataset_id(path)}\t{path}")
        return 0
    if not args.only:
        print("ERROR: --only is required unless --list is used", file=sys.stderr)
        return 2

    resources = resolve_resources(repo, resource_root)
    selected_genes = load_selected_genes(resources["selected_genes"])
    selected = select_inputs(paths, args.only)
    print(f"PROJECT_ROOT: {project_root}")
    print(f"REPO: {repo}")
    print(f"EMBED_PY_SHA256: {EXPECTED_EMBED_PY_SHA256}")
    print(f"SELECTED_DATASET_COUNT: {len(selected)}")
    print(f"EMBEDDING_BATCH_SIZE: {args.embedding_batch_size}")
    print("EMBEDDING_SEMANTICS: mean pooling over 27,874 genes -> 200d")
    for path in selected:
        process_one(
            path,
            project_root,
            repo,
            resources,
            selected_genes,
            args.embedding_batch_size,
            args.export_training_csv,
            args.overwrite,
        )
    print("ALL_SELECTED_TASK3_SCLONG_EMBEDDINGS_FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

