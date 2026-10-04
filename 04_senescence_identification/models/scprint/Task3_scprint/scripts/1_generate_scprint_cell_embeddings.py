#!/usr/bin/env python3
"""Generate audited 256-dimensional scPRINT embeddings for Task3 datasets."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import DataLoader

from scdataloader import Collator
from scdataloader.data import SimpleAnnDataset
from scdataloader.utils import load_genes
from scprint import scPrint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path(
    "/home/zhujialin/shared/zhujialin/scprint/checkpoints/medium-v1.5.ckpt"
)
TAG = "scPRINT"
EXPECTED_DIM = 256
EXPECTED_INPUT_GENE_COUNT = 25424
MODEL_INPUT_ORGANISM = "NCBITaxon:9606"
DEFAULT_MAX_LEN = 2000
SEED = 42
PRED_EMBEDDING = [
    "cell_type_ontology_term_id",
    "disease_ontology_term_id",
    "self_reported_ethnicity_ontology_term_id",
    "sex_ontology_term_id",
]
EXPECTED_DATASETS = {
    "Independent.Test_task3_Senescent_bulk_n151label072_label1.79": {
        "rows": 151,
        "label_counts": {0: 72, 1: 79},
    },
    "Independent.Test_task3_Senescent_human_n6959_label0.2455_label1.4504": {
        "rows": 6959,
        "label_counts": {0: 2455, 1: 4504},
    },
    "Independent.Test_task3_Senescent_mouse_n4693_label1": {
        "rows": 4693,
        "label_counts": {0: 0, 1: 4693},
    },
    "Independent.Test_task3_Senescent_mouse_n5213_label0": {
        "rows": 4932,
        "label_counts": {0: 4932, 1: 0},
    },
    "Training_task3_Senescent_n11280_label0.5607_label1.5673": {
        "rows": 11280,
        "label_counts": {0: 5607, 1: 5673},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--only", help="Exact dataset id, training, independent, or all")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--transformer",
        choices=["checkpoint", "normal"],
        default="checkpoint",
        help="Keep the checkpoint backend or explicitly use the official normal backend.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be positive")
    if args.max_len != DEFAULT_MAX_LEN:
        parser.error(f"--max-len must remain {DEFAULT_MAX_LEN}")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
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


def sequence_sha256(values: np.ndarray) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
    return path.stem


def is_training(path: Path) -> bool:
    return dataset_id(path).startswith("Training_")


def classify_dataset(name: str) -> dict[str, str]:
    return {
        "source_species": "mouse" if "_mouse_" in name else "human",
        "observation_type": "bulk_sample" if "_bulk_" in name else "single_cell",
    }


def discover_inputs(input_root: Path) -> list[Path]:
    paths = sorted(input_root.rglob("*.h5ad"))
    ids = [dataset_id(path) for path in paths]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate Task3 scPRINT dataset ids")
    expected = set(EXPECTED_DATASETS)
    found = set(ids)
    if found != expected:
        raise RuntimeError(
            "Task3 scPRINT input set mismatch. "
            f"Missing={sorted(expected - found)}; unexpected={sorted(found - expected)}"
        )
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
    selected = [path for path in paths if dataset_id(path) == key]
    if selected:
        return selected
    raise ValueError(f"Unknown --only value {selector!r}; run with --list")


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_gene_ids(values: pd.Series | pd.Index) -> np.ndarray:
    genes = np.asarray(values.astype(str), dtype=str)
    return np.asarray([gene.split(".", 1)[0] for gene in genes], dtype=str)


def inspect_input(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    name = dataset_id(path)
    expected = EXPECTED_DATASETS[name]
    backed = ad.read_h5ad(path, backed="r")
    try:
        if backed.shape != (expected["rows"], EXPECTED_INPUT_GENE_COUNT):
            raise ValueError(
                f"Unexpected input shape for {name}: {backed.shape}; "
                f"expected {(expected['rows'], EXPECTED_INPUT_GENE_COUNT)}"
            )
        if not backed.obs_names.is_unique:
            raise ValueError(f"Non-unique obs_names: {path.name}")
        if not backed.var_names.is_unique:
            raise ValueError(f"Non-unique var_names: {path.name}")
        if "label" not in backed.obs.columns:
            raise ValueError(f"Missing obs['label']: {path.name}")
        if "ensembl_id" not in backed.var.columns:
            raise ValueError(f"Missing var['ensembl_id']: {path.name}")

        genes = normalize_gene_ids(backed.var["ensembl_id"])
        if len(set(genes.tolist())) != len(genes):
            raise ValueError(f"Duplicate normalized Ensembl ids: {path.name}")
        is_ensg = np.char.startswith(genes, "ENSG")
        if not np.all(is_ensg):
            raise ValueError(
                "Input is not the audited human-Ensembl ortholog space: "
                f"{genes[~is_ensg][:10].tolist()}"
            )

        metadata = backed.obs.copy().reset_index(drop=True)
        metadata.insert(
            0, "cell_id", np.asarray(backed.obs_names.astype(str), dtype=str)
        )
        numeric = pd.to_numeric(metadata["label"], errors="coerce")
        if numeric.isna().any() or not set(numeric.unique()).issubset({0, 1}):
            raise ValueError(f"Invalid binary labels: {path.name}")
        counts = {
            0: int((numeric == 0).sum()),
            1: int((numeric == 1).sum()),
        }
        if counts != expected["label_counts"]:
            raise ValueError(
                f"Unexpected label counts for {name}: {counts}; "
                f"expected {expected['label_counts']}"
            )
        metadata["label"] = numeric.astype(np.int8)

        if "donorID" in metadata.columns:
            donor_text = metadata["donorID"].astype(str).str.strip()
            missing = metadata["donorID"].isna() | donor_text.str.lower().isin(
                ["", "nan", "none", "na", "n/a", "unknown"]
            )
            known_donor_count = int(donor_text.loc[~missing].nunique())
            missing_donor_rows = int(missing.sum())
        else:
            known_donor_count = 0
            missing_donor_rows = int(len(metadata))

        audit = {
            "shape": [int(backed.n_obs), int(backed.n_vars)],
            "x_storage": type(backed.X).__name__,
            "obs_names_unique": True,
            "var_names_unique": True,
            "gene_id_source": "var['ensembl_id']",
            "normalized_gene_order_sha256": sequence_sha256(genes),
            "label_0_count": counts[0],
            "label_1_count": counts[1],
            "known_donor_count": known_donor_count,
            "missing_donor_rows": missing_donor_rows,
        }
        return metadata, audit
    finally:
        backed.file.close()


def prepare_adata(
    adata: ad.AnnData,
    model: scPrint,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Align audited human-ortholog counts to scDataLoader's human gene order."""
    adata.obs["organism_ontology_term_id"] = MODEL_INPUT_ORGANISM
    gene_frame = load_genes(organisms=MODEL_INPUT_ORGANISM)
    target_genes = gene_frame.index.astype(str).to_numpy()
    target_position = {gene: index for index, gene in enumerate(target_genes)}
    input_genes = normalize_gene_ids(adata.var["ensembl_id"])

    keep_columns: list[int] = []
    kept_genes: list[str] = []
    seen: set[str] = set()
    for column, gene in enumerate(input_genes):
        if gene in target_position and gene not in seen:
            keep_columns.append(column)
            kept_genes.append(gene)
            seen.add(gene)
    if not keep_columns:
        raise ValueError("No genes match scDataLoader's human gene list")

    matrix = adata.X
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    else:
        matrix = matrix.tocsr()
    raw_values = matrix.data
    if not np.isfinite(raw_values).all():
        raise ValueError("Input count matrix contains NaN or Inf")
    if np.any(raw_values < 0):
        raise ValueError("Input count matrix contains negative values")
    if not np.allclose(raw_values, np.rint(raw_values), atol=1e-6, rtol=0):
        raise ValueError("Input matrix is not integer-like raw counts")

    kept_matrix = matrix[:, keep_columns].tocoo()
    new_columns = np.asarray(
        [target_position[gene] for gene in kept_genes], dtype=np.int64
    )
    aligned_matrix = sparse.csr_matrix(
        (kept_matrix.data, (kept_matrix.row, new_columns[kept_matrix.col])),
        shape=(adata.n_obs, len(target_genes)),
    )
    aligned = ad.AnnData(
        X=aligned_matrix,
        obs=adata.obs.copy(),
        var=gene_frame.copy(),
    )
    aligned.obs_names = adata.obs_names.astype(str)
    aligned.var_names = target_genes
    library_sizes = np.asarray(aligned_matrix.sum(axis=1)).reshape(-1)
    aligned.obs["n_counts"] = library_sizes

    model_genes = set(map(str, model.genes))
    overlap = sum(gene in model_genes for gene in target_genes)
    if overlap < 1:
        raise ValueError("No aligned genes overlap the checkpoint vocabulary")
    audit = {
        "model_input_organism": MODEL_INPUT_ORGANISM,
        "original_gene_count": int(len(input_genes)),
        "scdataloader_human_gene_count": int(len(target_genes)),
        "matched_unique_gene_count": int(len(keep_columns)),
        "dropped_or_duplicate_gene_count": int(len(input_genes) - len(keep_columns)),
        "checkpoint_overlap_count": int(overlap),
        "checkpoint_overlap_fraction": float(overlap / len(target_genes)),
        "input_already_exact_scdataloader_human_gene_order": bool(
            len(input_genes) == len(target_genes)
            and np.array_equal(input_genes, target_genes)
        ),
        "aligned_shape": [int(aligned.n_obs), int(aligned.n_vars)],
        "input_nnz": int(matrix.nnz),
        "input_nonzero_min": float(raw_values.min()) if raw_values.size else 0.0,
        "input_nonzero_max": float(raw_values.max()) if raw_values.size else 0.0,
        "input_all_finite_nonnegative_integer_like": True,
        "aligned_library_size_min": float(library_sizes.min()),
        "aligned_library_size_max": float(library_sizes.max()),
        "aligned_library_size_mean": float(library_sizes.mean()),
    }
    return aligned, audit


def validate_complete(npz_path: Path, metadata_path: Path, manifest_path: Path) -> None:
    metadata = pd.read_csv(metadata_path, dtype={"cell_id": str})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != {"X", "cell_id"}:
            raise ValueError(f"Unexpected NPZ keys in {npz_path}")
        values = np.asarray(payload["X"])
        cell_ids = np.asarray(payload["cell_id"], dtype=str)
    if values.shape != (len(metadata), EXPECTED_DIM):
        raise ValueError(f"Invalid existing embedding shape: {values.shape}")
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError(f"Invalid existing embeddings: {npz_path}")
    if not np.array_equal(cell_ids, metadata["cell_id"].astype(str).to_numpy()):
        raise ValueError(f"Existing NPZ/metadata order mismatch: {npz_path}")
    if manifest.get("embedding", {}).get("shape") != [len(metadata), EXPECTED_DIM]:
        raise ValueError(f"Existing manifest shape mismatch: {manifest_path}")


def load_model(
    checkpoint: Path,
    device: torch.device,
    transformer: str,
) -> scPrint:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    kwargs: dict[str, Any] = {"precpt_gene_emb": None, "map_location": "cpu"}
    if transformer == "normal":
        kwargs["transformer"] = "normal"
    model = scPrint.load_from_checkpoint(str(checkpoint), **kwargs)
    if int(model.d_model) != EXPECTED_DIM:
        raise ValueError(f"Unexpected checkpoint d_model: {model.d_model}")
    if set(model.organisms) != {"NCBITaxon:9606", "NCBITaxon:10090"}:
        raise ValueError(f"Unexpected checkpoint organisms: {model.organisms}")
    missing = sorted(set(PRED_EMBEDDING) - set(model.classes))
    if missing:
        raise ValueError(f"Checkpoint lacks embedding heads: {missing}")
    model.to(device)
    model.eval()
    return model


def embed_adata(
    adata: ad.AnnData,
    model: scPrint,
    device: torch.device,
    batch_size: int,
    max_len: int,
    num_workers: int,
) -> np.ndarray:
    adata.obs["organism_ontology_term_id"] = MODEL_INPUT_ORGANISM
    dataset = SimpleAnnDataset(
        adata, obs_to_output=["organism_ontology_term_id"]
    )
    collator = Collator(
        organisms=model.organisms,
        valid_genes=model.genes,
        how="random expr",
        max_len=max_len,
        add_zero_genes=0,
    )
    loader = DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    model.on_predict_epoch_start()
    chunks: list[np.ndarray] = []
    context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), context:
        for batch_index, batch in enumerate(loader, start=1):
            gene_pos = batch["genes"].to(device, non_blocking=True)
            expression = batch["x"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            output = model._predict(
                gene_pos,
                expression,
                depth,
                predict_mode="none",
                pred_embedding=PRED_EMBEDDING,
                keep_output=False,
            )
            values = output["embs"].detach().float().cpu().numpy()
            chunks.append(values)
            if batch_index == 1 or batch_index % 100 == 0:
                completed = sum(len(chunk) for chunk in chunks)
                print(
                    f"EMBEDDING_PROGRESS: {completed}/{adata.n_obs}", flush=True
                )
    result = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if result.shape != (adata.n_obs, EXPECTED_DIM):
        raise RuntimeError(f"Unexpected embedding shape: {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError("scPRINT embeddings contain NaN or Inf")
    return result


def process_one(
    path: Path,
    model: scPrint,
    checkpoint: Path,
    checkpoint_hash: str,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    max_len: int,
    num_workers: int,
    transformer_mode: str,
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
        raise FileExistsError(
            f"Partial outputs exist for {name}; inspect before any overwrite"
        )

    metadata, audit = inspect_input(path)
    domain = classify_dataset(name)
    print(f"\nDATASET: {name}", flush=True)
    print(f"SOURCE_SPECIES: {domain['source_species']}", flush=True)
    print(f"OBSERVATION_TYPE: {domain['observation_type']}", flush=True)
    print(f"MODEL_INPUT_ORGANISM: {MODEL_INPUT_ORGANISM}", flush=True)
    print(f"INPUT_SHAPE: {tuple(audit['shape'])}", flush=True)
    print(f"EMBEDDING_BATCH_SIZE: {batch_size}", flush=True)
    print(f"NUM_WORKERS: {num_workers}", flush=True)
    set_reproducibility(SEED)
    adata = ad.read_h5ad(path)
    prepared: ad.AnnData | None = None
    try:
        prepared, gene_audit = prepare_adata(adata, model)
        print(
            f"GENE_ALIGNMENT: {gene_audit['matched_unique_gene_count']}/"
            f"{gene_audit['scdataloader_human_gene_count']}",
            flush=True,
        )
        print(
            f"CHECKPOINT_GENE_OVERLAP: {gene_audit['checkpoint_overlap_count']}",
            flush=True,
        )
        print("INPUT_RAW_COUNTS_VALID: True", flush=True)
        values = embed_adata(
            prepared, model, device, batch_size, max_len, num_workers
        )
    finally:
        if prepared is not None:
            del prepared
        del adata
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cell_ids = np.asarray(metadata["cell_id"].astype(str).tolist(), dtype=str)
    atomic_npz(npz_path, X=values, cell_id=cell_ids)
    atomic_csv(metadata_path, metadata)
    norms = np.linalg.norm(values, axis=1)
    manifest = {
        "manifest_version": "1.0",
        "created_at_utc": now_utc(),
        "dataset_id": name,
        "role": "training" if is_training(path) else "independent_test",
        **domain,
        "input": {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            **audit,
            **gene_audit,
            "scientific_note": (
                "The supplied matrix uses audited human ENSG ortholog identifiers. "
                "Mouse-derived data are embedded in that supplied human-ortholog "
                "gene space; source species is recorded separately. Bulk is a "
                "cross-modality exploratory input with substantially higher depth."
            ),
        },
        "model": {
            "foundation_model": TAG,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_gene_embedding_override": None,
            "checkpoint_loader_note": (
                "precpt_gene_emb=None avoids the stale absolute parquet path stored "
                "in the checkpoint; checkpoint state-dict weights are still loaded."
            ),
            "checkpoint_transformer": str(model.transformer),
            "requested_transformer_mode": transformer_mode,
            "d_model": int(model.d_model),
            "organisms": list(model.organisms),
            "pred_embedding_heads": PRED_EMBEDDING,
            "selection": "random expr",
            "max_len": max_len,
            "seed_reset_per_dataset": SEED,
        },
        "embedding": {
            "shape": [int(values.shape[0]), int(values.shape[1])],
            "dtype": str(values.dtype),
            "all_finite": True,
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "norm_mean": float(norms.mean()),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "anndata": package_version("anndata"),
            "scprint": package_version("scprint"),
            "scdataloader": package_version("scdataloader"),
        },
    }
    atomic_json(manifest_path, manifest)
    print(f"EMBEDDING_SHAPE: {values.shape}")
    print(f"EMBEDDING_DTYPE: {values.dtype}")
    print("ALL_FINITE: True")
    print(f"SAVED_NPZ: {npz_path}")
    print(f"SAVED_METADATA: {metadata_path}")
    print(f"SAVED_MANIFEST: {manifest_path}")
    print(f"DATASET_FINISHED: {name}")


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    input_root = root / "data"
    output_dir = root / "outputs" / "embeddings"
    paths = discover_inputs(input_root)
    if args.list:
        for path in paths:
            expected = EXPECTED_DATASETS[dataset_id(path)]
            print(
                f"{dataset_id(path)}\trows={expected['rows']}\t{path.resolve()}"
            )
        print(f"TASK3_SCPRINT_DATASET_COUNT: {len(paths)}")
        return 0
    if not args.only:
        print("ERROR: --only is required unless --list is used", file=os.sys.stderr)
        return 2

    selected = select_inputs(paths, args.only)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    set_reproducibility(SEED)
    model = load_model(checkpoint, device, args.transformer)
    checkpoint_hash = sha256(checkpoint)
    print(f"DEVICE: {device}")
    print(f"SELECTED_DATASET_COUNT: {len(selected)}")
    print("MODEL_CONFIG_VALID: True")
    print(f"MODEL_D_MODEL: {model.d_model}")
    print(f"MODEL_ORGANISMS: {model.organisms}")
    print(f"MODEL_PRED_EMBEDDING_HEADS: {PRED_EMBEDDING}")
    for path in selected:
        process_one(
            path=path,
            model=model,
            checkpoint=checkpoint,
            checkpoint_hash=checkpoint_hash,
            output_dir=output_dir,
            device=device,
            batch_size=args.embedding_batch_size,
            max_len=args.max_len,
            num_workers=args.num_workers,
            transformer_mode=args.transformer,
            overwrite=args.overwrite,
        )
    print("ALL_SELECTED_TASK3_SCPRINT_EMBEDDINGS_FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
