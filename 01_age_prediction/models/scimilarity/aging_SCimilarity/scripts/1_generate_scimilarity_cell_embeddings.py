from pathlib import Path
import re

import numpy as np
import pandas as pd
import scanpy as sc
from scimilarity import CellAnnotation
from scimilarity.utils import align_dataset, lognorm_counts


TAG = "SCimilarity"

BASE = Path.home() / "shared/zhujialin/task1_age_prediction"
MODEL_PATH = Path.home() / "shared/zhujialin/scimilarity/checkpoints/model_v1.1"

DATA_DIR = BASE / "scimilarity/data"
OUT_DIR = BASE / "scimilarity/embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def output_name(path):
    name = path.name
    if "Independent" in name:
        return "independent_test"

    m = re.search(r"Training_Part(\d+)", name)
    if m:
        return f"train_part{m.group(1)}"

    raise ValueError(f"Cannot parse dataset name from: {path}")


def build_metadata(adata):
    obs = adata.obs.copy()

    meta = pd.DataFrame({
        "cell_id": adata.obs_names.astype(str),
        "label": obs["label"].values,
        "sex": obs["sex"].values if "sex" in obs.columns else "",
        "donorID": obs["donorID"].values if "donorID" in obs.columns else "",
    })

    return meta


def prepare_for_scimilarity(adata, gene_order):
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    adata = align_dataset(adata, gene_order)

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    adata = lognorm_counts(adata)
    return adata


def main():
    print("data dir:", DATA_DIR)
    print("model path:", MODEL_PATH)
    print("model exists:", MODEL_PATH.exists())

    files = sorted(DATA_DIR.rglob("*.h5ad"))
    if not files:
        raise FileNotFoundError(f"No h5ad files found under {DATA_DIR}")

    print("found h5ad files:")
    for f in files:
        print(" ", f)

    ca = CellAnnotation(model_path=str(MODEL_PATH))
    print("gene_order length:", len(ca.gene_order))

    for data_path in files:
        name = output_name(data_path)
        print(f"\n=== Processing {name} ===")
        print("input:", data_path)

        adata = sc.read_h5ad(data_path)
        print("original shape:", adata.shape)
        print("obs columns:", list(adata.obs.columns))
        print("var columns:", list(adata.var.columns))

        adata = prepare_for_scimilarity(adata, ca.gene_order)
        print("aligned/lognorm shape:", adata.shape)

        emb = ca.get_embeddings(adata.X)
        emb = np.asarray(emb, dtype=np.float32)
        print("embedding shape:", emb.shape)

        emb_path = OUT_DIR / f"{name}_{TAG}_cell_embeddings.npz"
        meta_path = OUT_DIR / f"{name}_{TAG}_cell_embeddings_metadata.csv"

        np.savez_compressed(
            emb_path,
            X=emb,
            cell_id=adata.obs_names.astype(str).to_numpy(),
        )

        meta = build_metadata(adata)
        meta.to_csv(meta_path, index=False)

        print("saved embedding:", emb_path)
        print("saved metadata:", meta_path)

    print("\nSCimilarity cell embeddings finished.")


if __name__ == "__main__":
    main()