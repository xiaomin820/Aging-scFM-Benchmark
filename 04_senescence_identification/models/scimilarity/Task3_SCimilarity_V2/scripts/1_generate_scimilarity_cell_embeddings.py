#!/usr/bin/env python3
"""Generate Task3 SCimilarity embeddings from audited raw-count H5AD files.

The script validates every input, applies the official SCimilarity gene
alignment and log-normalization utilities, writes outputs atomically, and
never modifies the source H5AD files.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import torch

from scimilarity import CellEmbedding
from scimilarity.utils import align_dataset, lognorm_counts


SCIMILARITY_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = SCIMILARITY_ROOT.parent
DEFAULT_MODEL_PATH = Path(
    "/home/zhujialin/shared/zhujialin/scimilarity/checkpoints/model_v1.1"
)
INPUT_SUFFIX = "_SCimilarity_input"
EMBEDDING_DIM_EXPECTED = 128
EXPECTED_DATASET_COUNT = 6
EXPECTED_INDEPENDENT_COUNT = 5
INTEGER_ATOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=SCIMILARITY_ROOT,
        help="Task3 SCimilarity working directory (contains outputs/, models/, logs/).",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=TASK_ROOT / "data",
        help="Directory containing the six Task3 SCimilarity H5AD files.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--expected-dataset-count",
        type=int,
        default=EXPECTED_DATASET_COUNT,
        help="Expected total H5AD count, including exactly one training dataset.",
    )
    parser.add_argument(
        "--expected-independent-count",
        type=int,
        default=EXPECTED_INDEPENDENT_COUNT,
        help="Expected independent-test H5AD count.",
    )
    parser.add_argument(
        "--only",
        help="Exact dataset id, 'training', 'independent', or 'all'.",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=1024,
        help="Passed to get_embeddings when supported by the installed API.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--export-training-csv",
        action="store_true",
        help="Also export the large training embedding CSV; NPZ is always written.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be positive")
    return args


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(
    path: Path, frame: pd.DataFrame, *, float_format: str | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, float_format=float_format)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def dataset_id(path: Path) -> str:
    stem = path.stem
    if stem.endswith(INPUT_SUFFIX):
        return stem[: -len(INPUT_SUFFIX)]
    return stem


def is_training(path: Path) -> bool:
    return dataset_id(path).startswith("Training_")


def classify_dataset(name: str) -> dict[str, str]:
    species = "mouse" if "_mouse_" in name else "human"
    observation_type = "bulk_sample" if "_bulk_" in name else "single_cell"
    return {"species": species, "observation_type": observation_type}


def discover_inputs(
    input_root: Path,
    expected_dataset_count: int,
    expected_independent_count: int,
) -> list[Path]:
    paths = sorted(input_root.rglob("*.h5ad"))
    ids = [dataset_id(path) for path in paths]
    if len(paths) != expected_dataset_count:
        raise RuntimeError(
            f"Expected {expected_dataset_count} Task3 H5AD files, found "
            f"{len(paths)} under {input_root}"
        )
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate dataset ids were discovered.")
    training_count = sum(is_training(path) for path in paths)
    if training_count != 1:
        raise RuntimeError(f"Expected exactly one training H5AD, found {training_count}")
    if len(paths) - training_count != expected_independent_count:
        raise RuntimeError(
            f"Expected {expected_independent_count} independent H5AD files."
        )
    return paths


def validate_binary_label(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError(
            f"{name}: label contains {int(numeric.isna().sum())} missing/nonnumeric values"
        )
    unique = set(numeric.astype(float).unique().tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError(f"{name}: label values are not binary 0/1: {sorted(unique)}")
    return numeric.astype(np.int8)


def write_input_catalog(paths: list[Path], output_root: Path) -> None:
    records: list[dict[str, Any]] = []
    training_donors: set[str] = set()
    reference_genes: np.ndarray | None = None
    for path in paths:
        backed = ad.read_h5ad(path, backed="r")
        try:
            if "label" not in backed.obs.columns:
                raise ValueError(f"Missing obs['label']: {path.name}")
            genes = np.asarray(backed.var_names, dtype=str)
            if reference_genes is None:
                reference_genes = genes
            elif not np.array_equal(reference_genes, genes):
                raise ValueError(f"Input gene order differs: {path.name}")
            labels = validate_binary_label(backed.obs["label"], dataset_id(path))
            donors = (
                {
                    str(value)
                    for value in backed.obs["donorID"].dropna().astype(str)
                    if str(value).strip() not in {"", "NA", "nan", "None"}
                }
                if "donorID" in backed.obs.columns
                else set()
            )
            if is_training(path):
                training_donors = donors
            records.append(
                {
                    "dataset_id": dataset_id(path),
                    "filename": path.name,
                    "role": "training" if is_training(path) else "independent_test",
                    **classify_dataset(dataset_id(path)),
                    "shape": [int(backed.n_obs), int(backed.n_vars)],
                    "label_0_count": int((labels == 0).sum()),
                    "label_1_count": int((labels == 1).sum()),
                    "donor_count_excluding_NA": len(donors),
                    "filename_row_count_note": (
                        "Filename says n5213 but H5AD contains 4943 observations; "
                        "all processing uses the actual H5AD row count."
                        if "mouse_n5213" in path.name and int(backed.n_obs) == 4943
                        else None
                    ),
                    "_donors": donors,
                }
            )
        finally:
            backed.file.close()

    for record in records:
        donors = record.pop("_donors")
        overlap = sorted(donors & training_donors) if record["role"] != "training" else []
        record["training_donor_overlap_count"] = len(overlap)
        record["training_donor_overlap"] = overlap
    atomic_json(
        output_root / "manifests" / "Task3_SCimilarity_input_catalog.json",
        {
            "catalog_version": "1.0",
            "created_at_utc": now_utc(),
            "input_dataset_count": len(records),
            "all_input_gene_orders_identical": True,
            "input_gene_count": int(len(reference_genes))
            if reference_genes is not None
            else 0,
            "label_definition": {"0": "normal", "1": "senescent"},
            "datasets": records,
        },
    )


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
    raise ValueError(f"Unknown --only value: {selector!r}. Run with --list.")


def full_raw_count_audit(matrix: Any, n_obs: int) -> dict[str, Any]:
    if sparse.issparse(matrix):
        values = np.asarray(matrix.data)
        finite = bool(np.isfinite(values).all())
        negative_count = int(np.count_nonzero(values < 0))
        integer_count = int(
            np.count_nonzero(
                np.isclose(values, np.rint(values), atol=INTEGER_ATOL, rtol=0.0)
            )
        )
        row_sums = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
        storage = type(matrix).__name__
    else:
        values_list: list[np.ndarray] = []
        row_sum_parts: list[np.ndarray] = []
        finite = True
        negative_count = 0
        integer_count = 0
        for start in range(0, n_obs, 256):
            block = np.asarray(matrix[start : start + 256, :])
            nonzero = block[block != 0]
            values_list.append(nonzero)
            row_sum_parts.append(np.sum(block, axis=1, dtype=np.float64))
            finite = finite and bool(np.isfinite(block).all())
            negative_count += int(np.count_nonzero(block < 0))
            integer_count += int(
                np.count_nonzero(
                    np.isclose(nonzero, np.rint(nonzero), atol=INTEGER_ATOL, rtol=0.0)
                )
            )
        values = np.concatenate(values_list) if values_list else np.array([], dtype=float)
        row_sums = (
            np.concatenate(row_sum_parts) if row_sum_parts else np.array([], dtype=float)
        )
        storage = type(matrix).__name__

    nonzero_count = int(values.size)
    integer_fraction = float(integer_count / nonzero_count) if nonzero_count else None
    zero_sum_rows = int(np.count_nonzero(row_sums <= 0))
    result = {
        "storage_type": storage,
        "finite": finite,
        "negative_count": negative_count,
        "nonzero_count": nonzero_count,
        "nonzero_integer_fraction": integer_fraction,
        "zero_sum_rows": zero_sum_rows,
        "row_sum_min": float(row_sums.min()) if row_sums.size else None,
        "row_sum_median": float(np.median(row_sums)) if row_sums.size else None,
        "row_sum_max": float(row_sums.max()) if row_sums.size else None,
    }
    if not finite:
        raise ValueError("Expression matrix contains NaN or Inf.")
    if negative_count:
        raise ValueError(f"Expression matrix contains {negative_count} negative values.")
    if integer_fraction is None or integer_fraction < 1.0:
        raise ValueError(
            f"Expression matrix is not raw integer counts: {integer_fraction}"
        )
    if zero_sum_rows:
        raise ValueError(f"Expression matrix contains {zero_sum_rows} zero-count rows.")
    return result


def make_encoder(model_path: Path, use_gpu: bool) -> CellEmbedding:
    parameters = inspect.signature(CellEmbedding).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if "model_path" in parameters or accepts_kwargs:
        kwargs["model_path"] = str(model_path)
    if "use_gpu" in parameters or accepts_kwargs:
        kwargs["use_gpu"] = use_gpu
    if "model_path" in kwargs:
        return CellEmbedding(**kwargs)
    optional = {"use_gpu": use_gpu} if "use_gpu" in parameters else {}
    return CellEmbedding(str(model_path), **optional)


def get_embeddings(
    encoder: CellEmbedding, matrix: Any, batch_size: int
) -> np.ndarray:
    parameters = inspect.signature(encoder.get_embeddings).parameters
    kwargs: dict[str, Any] = {}
    if "batch_size" in parameters:
        kwargs["batch_size"] = batch_size
    result = encoder.get_embeddings(matrix, **kwargs)
    if torch.is_tensor(result):
        result = result.detach().cpu().numpy()
    return np.asarray(result, dtype=np.float32)


def process_one(
    path: Path,
    encoder: CellEmbedding,
    model_path: Path,
    output_root: Path,
    batch_size: int,
    export_training_csv: bool,
    overwrite: bool,
    use_gpu: bool,
) -> None:
    name = dataset_id(path)
    training = is_training(path)
    emb_dir = output_root / "embeddings"
    npz_path = emb_dir / f"{name}_SCimilarity_cell_embeddings.npz"
    metadata_path = emb_dir / f"{name}_SCimilarity_cell_embeddings_metadata.csv"
    csv_path = emb_dir / f"{name}_SCimilarity_cell_embeddings.csv"
    manifest_path = emb_dir / f"{name}_SCimilarity_embedding_manifest.json"
    csv_required = (not training) or export_training_csv
    required = [npz_path, metadata_path, manifest_path]
    if csv_required:
        required.append(csv_path)

    if all(item.exists() for item in required) and not overwrite:
        print(f"SKIP_COMPLETE: {name}")
        return
    if any(item.exists() for item in required) and not overwrite:
        existing = [str(item) for item in required if item.exists()]
        raise FileExistsError(
            f"Partial outputs exist for {name}; inspect or use --overwrite: {existing}"
        )

    print(f"\n===== DATASET: {name} =====", flush=True)
    print(f"INPUT: {path}", flush=True)
    adata = ad.read_h5ad(path)
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError(f"Non-unique observation or gene ids: {name}")
    if "label" not in adata.obs.columns:
        raise ValueError(f"Missing obs['label']: {name}")
    labels = validate_binary_label(adata.obs["label"], name)
    original_shape = [int(adata.n_obs), int(adata.n_vars)]
    original_genes = np.asarray(adata.var_names, dtype=str)
    metadata = adata.obs.copy().reset_index(drop=True)
    metadata.insert(0, "cell_id", np.asarray(adata.obs_names, dtype=str))
    metadata["label"] = labels.to_numpy(dtype=np.int8)
    raw_audit = full_raw_count_audit(adata.X, int(adata.n_obs))
    print(f"RAW_COUNTS_VALID: True; shape={tuple(original_shape)}", flush=True)
    print(
        f"LABEL_COUNTS: label0={int((labels == 0).sum())}, "
        f"label1={int((labels == 1).sum())}",
        flush=True,
    )

    adata = align_dataset(adata, encoder.gene_order)
    expected_gene_order = np.asarray(encoder.gene_order, dtype=str)
    if not np.array_equal(np.asarray(adata.var_names, dtype=str), expected_gene_order):
        raise RuntimeError("Aligned genes do not exactly match model gene order.")
    adata.layers["counts"] = adata.X.copy()
    adata = lognorm_counts(adata)
    if "counts" in adata.layers:
        del adata.layers["counts"]
    embeddings = get_embeddings(encoder, adata.X, batch_size)
    if embeddings.shape != (original_shape[0], EMBEDDING_DIM_EXPECTED):
        raise RuntimeError(
            f"Unexpected embedding shape {embeddings.shape}; expected "
            f"({original_shape[0]}, {EMBEDDING_DIM_EXPECTED})"
        )
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Embeddings contain NaN or Inf.")

    cell_ids = metadata["cell_id"].astype(str).to_numpy(dtype=str)
    atomic_npz(npz_path, X=embeddings, cell_id=cell_ids)
    atomic_csv(metadata_path, metadata)
    if csv_required:
        columns = [f"embedding_{index:03d}" for index in range(embeddings.shape[1])]
        frame = pd.DataFrame(embeddings, columns=columns)
        frame.insert(0, "cell_id", cell_ids)
        atomic_csv(csv_path, frame, float_format="%.9g")

    norms = np.linalg.norm(embeddings, axis=1)
    atomic_json(
        manifest_path,
        {
            "manifest_version": "1.0",
            "created_at_utc": now_utc(),
            "dataset_id": name,
            "role": "training" if training else "independent_test",
            **classify_dataset(name),
            "label_definition": {"0": "normal", "1": "senescent"},
            "input": {
                "path": str(path.resolve()),
                "file_size_bytes": int(path.stat().st_size),
                "shape": original_shape,
                "original_gene_count": int(len(original_genes)),
                "model_gene_count": int(len(expected_gene_order)),
                "raw_count_full_audit": raw_audit,
                "label_0_count": int((labels == 0).sum()),
                "label_1_count": int((labels == 1).sum()),
            },
            "preprocessing": [
                "align_dataset(input, encoder.gene_order)",
                "copy aligned raw counts to layers['counts']",
                "lognorm_counts (SCimilarity-compatible normalization and log1p)",
            ],
            "model": {
                "path": str(model_path.resolve()),
                "class": type(encoder).__name__,
                "use_gpu_requested": use_gpu,
            },
            "embedding": {
                "shape": [int(v) for v in embeddings.shape],
                "dtype": str(embeddings.dtype),
                "finite": True,
                "l2_norm_min": float(norms.min()),
                "l2_norm_median": float(np.median(norms)),
                "l2_norm_max": float(norms.max()),
            },
            "outputs": {
                "npz": str(npz_path.resolve()),
                "metadata_csv": str(metadata_path.resolve()),
                "embedding_csv": str(csv_path.resolve()) if csv_required else None,
            },
            "software": {
                "python": platform.python_version(),
                "anndata": package_version("anndata"),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "torch": torch.__version__,
                "scimilarity": package_version("scimilarity"),
            },
            "scientific_note": (
                "SCimilarity model_v1.1 is a human single-cell representation model. "
                "Bulk and mouse embeddings are cross-domain exploratory outputs and "
                "must be reported separately."
            ),
        },
    )
    print(f"EMBEDDING_SHAPE: {embeddings.shape}")
    print(f"SAVED_NPZ: {npz_path}")
    print(f"SAVED_METADATA: {metadata_path}")
    if csv_required:
        print(f"SAVED_EMBEDDING_CSV: {csv_path}")
    print(f"DATASET_FINISHED: {name}", flush=True)


def main() -> int:
    args = parse_args()
    if args.expected_dataset_count < 2 or args.expected_independent_count < 1:
        raise ValueError("Expected dataset counts must be positive.")
    if args.expected_dataset_count != args.expected_independent_count + 1:
        raise ValueError(
            "--expected-dataset-count must equal "
            "--expected-independent-count + 1."
        )
    project_root = args.project_root.expanduser().resolve()
    input_root = args.input_root.expanduser().resolve()
    output_root = project_root / "outputs"
    model_path = args.model_path.expanduser().resolve()
    paths = discover_inputs(
        input_root,
        args.expected_dataset_count,
        args.expected_independent_count,
    )
    if args.list:
        for path in paths:
            print(f"{dataset_id(path)}\t{path}")
        return 0
    if not args.only:
        print("ERROR: --only is required unless --list is used.", file=sys.stderr)
        return 2
    if not model_path.is_dir():
        print(f"ERROR: model directory does not exist: {model_path}", file=sys.stderr)
        return 2

    selected = select_inputs(paths, args.only)
    write_input_catalog(paths, output_root)
    use_gpu = (not args.cpu) and torch.cuda.is_available()
    print(f"PROJECT_ROOT: {project_root}")
    print(f"INPUT_ROOT: {input_root}")
    print(f"MODEL_PATH: {model_path}")
    print(f"USE_GPU: {use_gpu}")
    print(f"SELECTED_DATASET_COUNT: {len(selected)}")
    encoder = make_encoder(model_path, use_gpu)
    for path in selected:
        process_one(
            path,
            encoder,
            model_path,
            output_root,
            args.embedding_batch_size,
            args.export_training_csv,
            args.overwrite,
            use_gpu,
        )
    print("ALL_SELECTED_TASK3_SCIMILARITY_EMBEDDINGS_FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
