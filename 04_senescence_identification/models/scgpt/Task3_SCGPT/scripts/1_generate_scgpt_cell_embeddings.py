#!/usr/bin/env python3
"""Generate audited 512-dimensional scGPT embeddings for Task3 datasets."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from scgpt.tasks import embed_data


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = Path(
    "/home/zhujialin/shared/zhujialin/scgpt/checkpoints/scGPT_human"
)
INPUT_SUFFIX = "_scGPT_input"
TAG = "scGPT"
EXPECTED_DIM = 512
EXPECTED_MAX_LENGTH = 1200
EXPECTED_N_BINS = 51


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--only", help="Dataset id, training, independent, or all")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=EXPECTED_MAX_LENGTH)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be positive")
    if args.max_length != EXPECTED_MAX_LENGTH:
        parser.error(f"--max-length must remain {EXPECTED_MAX_LENGTH}")
    return args


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


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
        raise ValueError(f"Unexpected input filename: {path.name}")
    return path.stem[: -len(INPUT_SUFFIX)]


def is_training(path: Path) -> bool:
    return dataset_id(path).startswith("Training_")


def discover_inputs(input_root: Path) -> list[Path]:
    paths = sorted(input_root.rglob(f"*{INPUT_SUFFIX}.h5ad"))
    ids = [dataset_id(path) for path in paths]
    if len(paths) != 5:
        raise RuntimeError(f"Expected 5 Task3 scGPT H5AD files, found {len(paths)}")
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate Task3 scGPT dataset ids")
    if sum(is_training(path) for path in paths) != 1:
        raise RuntimeError("Expected exactly one Task3 scGPT training dataset")
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
    raise ValueError(f"Unknown --only value {selector!r}; run with --list")


def load_model_config(model_dir: Path) -> dict[str, Any]:
    required = [model_dir / "args.json", model_dir / "vocab.json", model_dir / "best_model.pt"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing scGPT model files: {missing}")
    config = json.loads((model_dir / "args.json").read_text(encoding="utf-8"))
    if int(config.get("embsize", -1)) != EXPECTED_DIM:
        raise ValueError(f"Unexpected scGPT embsize: {config.get('embsize')}")
    if int(config.get("max_seq_len", -1)) != EXPECTED_MAX_LENGTH:
        raise ValueError(f"Unexpected max_seq_len: {config.get('max_seq_len')}")
    if int(config.get("n_bins", -1)) != EXPECTED_N_BINS:
        raise ValueError(f"Unexpected n_bins: {config.get('n_bins')}")
    return config


def input_metadata(path: Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    backed = ad.read_h5ad(path, backed="r")
    try:
        if not backed.obs_names.is_unique:
            raise ValueError(f"Non-unique obs_names: {path.name}")
        if "label" not in backed.obs.columns:
            raise ValueError(f"Missing obs['label']: {path.name}")
        if "features" not in backed.var.columns:
            raise ValueError(f"Missing var['features']: {path.name}")
        genes = backed.var["features"].astype(str).to_numpy()
        if pd.Index(genes).has_duplicates:
            raise ValueError(f"Duplicate var['features']: {path.name}")
        cell_ids = np.asarray(backed.obs_names, dtype=str)
        metadata = backed.obs.copy().reset_index(drop=True)
        metadata.insert(0, "cell_id", cell_ids)
        numeric = pd.to_numeric(metadata["label"], errors="coerce")
        invalid_label = numeric.isna() | ~numeric.isin([0, 1])
        if invalid_label.any():
            raise ValueError(
                f"Task3 labels must be binary 0/1; invalid rows="
                f"{int(invalid_label.sum())}: {path.name}"
            )
        audit = {
            "shape": [int(backed.n_obs), int(backed.n_vars)],
            "features_unique": True,
            "numeric_label_count": int(numeric.notna().sum()),
            "missing_or_nonnumeric_label_count": int(numeric.isna().sum()),
            "label_0_count": int((numeric == 0).sum()),
            "label_1_count": int((numeric == 1).sum()),
        }
        return metadata, genes, audit
    finally:
        backed.file.close()


def validate_complete(npz_path: Path, metadata_path: Path, manifest_path: Path) -> None:
    metadata = pd.read_csv(metadata_path, dtype={"cell_id": str})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != {"X", "cell_id"}:
            raise ValueError(f"Unexpected NPZ keys in {npz_path}")
        values = payload["X"]
        cell_ids = np.asarray(payload["cell_id"], dtype=str)
    if values.shape != (len(metadata), EXPECTED_DIM):
        raise ValueError(f"Invalid existing embedding shape: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite existing embeddings: {npz_path}")
    if not np.array_equal(cell_ids, metadata["cell_id"].astype(str).to_numpy()):
        raise ValueError(f"Existing NPZ/metadata order mismatch: {npz_path}")
    if manifest.get("embedding", {}).get("shape") != [len(metadata), EXPECTED_DIM]:
        raise ValueError(f"Existing manifest shape mismatch: {manifest_path}")


def process_one(
    path: Path,
    model_dir: Path,
    output_dir: Path,
    model_config: dict[str, Any],
    model_hashes: dict[str, str],
    batch_size: int,
    max_length: int,
    device: str,
    overwrite: bool,
) -> None:
    name = dataset_id(path)
    npz_path = output_dir / f"{name}_{TAG}_cell_embeddings.npz"
    metadata_path = output_dir / f"{name}_{TAG}_cell_embeddings_metadata.csv"
    manifest_path = output_dir / f"{name}_{TAG}_embedding_manifest.json"
    outputs = [npz_path, metadata_path, manifest_path]
    if all(item.is_file() for item in outputs) and not overwrite:
        validate_complete(npz_path, metadata_path, manifest_path)
        print(f"SKIP_VALIDATED_COMPLETE: {name}")
        return
    if any(item.exists() for item in outputs) and not overwrite:
        raise FileExistsError(f"Partial outputs exist for {name}; inspect or use --overwrite")

    metadata, genes, audit = input_metadata(path)
    print(f"\nDATASET: {name}", flush=True)
    print(f"INPUT_SHAPE: {tuple(audit['shape'])}", flush=True)
    print(f"EMBEDDING_BATCH_SIZE: {batch_size}", flush=True)
    embedded = embed_data(
        path,
        model_dir,
        gene_col="features",
        max_length=max_length,
        batch_size=batch_size,
        obs_to_save=["label"],
        device=device,
        use_fast_transformer=True,
        return_new_adata=True,
    )
    values = embedded.X
    if sparse.issparse(values):
        values = values.toarray()
    values = np.asarray(values, dtype=np.float32)
    embedded_ids = np.asarray(embedded.obs_names, dtype=str)
    # Force a fixed-width Unicode array.  Pandas otherwise commonly returns an
    # object-dtype array, which requires pickle and is deliberately rejected by
    # downstream loaders using allow_pickle=False.
    source_ids = np.asarray(metadata["cell_id"].astype(str).tolist(), dtype=str)
    if values.shape != (len(metadata), EXPECTED_DIM):
        raise RuntimeError(f"Unexpected embedding shape {values.shape}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"Non-finite embeddings for {name}")
    if not np.array_equal(embedded_ids, source_ids):
        raise RuntimeError(f"scGPT changed observation order for {name}")

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
            **audit,
            "gene_count": int(len(genes)),
        },
        "model": {
            "directory": str(model_dir.resolve()),
            "args": model_config,
            "sha256": model_hashes,
        },
        "embedding_call": {
            "gene_col": "features",
            "max_length": max_length,
            "batch_size": batch_size,
            "device": device,
            "use_fast_transformer": True,
        },
        "embedding": {
            "shape": [int(values.shape[0]), int(values.shape[1])],
            "dtype": str(values.dtype),
            "finite": True,
            "l2_norm_min": float(norms.min()),
            "l2_norm_median": float(np.median(norms)),
            "l2_norm_max": float(norms.max()),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "anndata": package_version("anndata"),
            "scgpt": package_version("scgpt"),
        },
    }
    atomic_json(manifest_path, manifest)
    print(f"EMBEDDING_SHAPE: {values.shape}")
    print("ALL_FINITE: True")
    print(f"SAVED_NPZ: {npz_path}")
    print(f"SAVED_METADATA: {metadata_path}")
    print(f"DATASET_FINISHED: {name}", flush=True)
    del embedded, values
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    input_root = project_root / "data" / "task3_Senescent_scGPT_input"
    output_dir = project_root / "outputs" / "embeddings"
    paths = discover_inputs(input_root)
    if args.list:
        for path in paths:
            print(f"{dataset_id(path)}\t{path}")
        return 0
    if not args.only:
        print("ERROR: --only is required unless --list is used", file=os.sys.stderr)
        return 2
    model_config = load_model_config(model_dir)
    selected = select_inputs(paths, args.only)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    model_hashes = {
        filename: sha256(model_dir / filename)
        for filename in ("args.json", "vocab.json", "best_model.pt")
    }
    print(f"DEVICE: {device}")
    print(f"SELECTED_DATASET_COUNT: {len(selected)}")
    print(f"MODEL_CONFIG_VALID: True")
    for path in selected:
        process_one(
            path,
            model_dir,
            output_dir,
            model_config,
            model_hashes,
            args.embedding_batch_size,
            args.max_length,
            device,
            args.overwrite,
        )
    print("ALL_SELECTED_TASK3_SCGPT_EMBEDDINGS_FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
