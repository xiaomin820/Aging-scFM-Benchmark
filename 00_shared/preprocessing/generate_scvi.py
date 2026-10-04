from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one scVI reference per input cohort.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-column", action="append", default=[])
    parser.add_argument("--latent-dim", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def cells_with_fractional_values(values) -> np.ndarray:
    if sp.issparse(values):
        flags = values.tocsr(copy=True)
        flags.data = (np.abs(flags.data - np.rint(flags.data)) > 1e-8).astype(np.int8)
        flags.eliminate_zeros()
        return np.asarray(flags.getnnz(axis=1) > 0).ravel()
    array = np.asarray(values)
    return np.any(np.abs(array - np.rint(array)) > 1e-8, axis=1)


def main() -> int:
    args = parse_args()
    try:
        import scanpy as sc
        import scvi
    except ImportError as error:
        raise SystemExit(
            "scVI extraction requires scanpy and scvi-tools; install organized_code/requirements.txt"
        ) from error
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scvi.settings.seed = args.seed
    summaries: list[dict] = []

    for input_path in args.input:
        adata = sc.read_h5ad(input_path)
        original_cells = adata.n_obs
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        fractional = cells_with_fractional_values(adata.X)
        fractional_cells = int(fractional.sum())
        adata = adata[~fractional].copy()
        if adata.n_obs == 0:
            raise ValueError(f"No integer-count cells remain in {input_path}")

        scvi.model.SCVI.setup_anndata(adata)
        model = scvi.model.SCVI(adata, n_latent=args.latent_dim)
        model.train(max_epochs=args.epochs, early_stopping=True)
        latent = model.get_latent_representation()

        frame = pd.DataFrame(
            latent,
            index=adata.obs_names,
            columns=[f"emb_{i + 1}" for i in range(latent.shape[1])],
        )
        frame.insert(0, "cell_id", frame.index.astype(str))
        for column in args.metadata_column:
            if column not in adata.obs:
                raise KeyError(f"Missing metadata column {column!r} in {input_path}")
            frame[column] = adata.obs.loc[frame.index, column].to_numpy()

        stem = input_path.stem
        frame.to_csv(args.output_dir / f"{stem}_scvi.csv", index=False)
        model.save(args.output_dir / f"{stem}_model", overwrite=True)
        summaries.append(
            {
                "input": str(input_path.resolve()),
                "original_cells": original_cells,
                "retained_cells": int(adata.n_obs),
                "retained_genes": int(adata.n_vars),
                "fractional_cells_removed": fractional_cells,
                "latent_dim": args.latent_dim,
                "maximum_epochs": args.epochs,
            }
        )

    payload = {
        "seed": args.seed,
        "fit_scope": "one independent scVI model per input cohort",
        "batch_covariate": None,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scanpy": sc.__version__,
            "scvi_tools": scvi.__version__,
        },
        "cohorts": summaries,
    }
    (args.output_dir / "scvi_run_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
