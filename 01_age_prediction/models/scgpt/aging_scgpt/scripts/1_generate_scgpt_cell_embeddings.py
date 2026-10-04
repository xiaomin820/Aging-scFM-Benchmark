from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from scgpt.tasks import embed_data

FILES = {
    "train_part1": "Task1_Training_Part1_n37366_scGPT_input.h5ad",
    "train_part2": "Task1_Training_Part2_n37324_scGPT_input.h5ad",
    "train_part3": "Task1_Training_Part3_n37494_scGPT_input.h5ad",
    "train_part4": "Task1_Training_Part4_n37430_scGPT_input.h5ad",
    "train_part5": "Task1_Training_Part5_n29974_scGPT_input.h5ad",
    "independent_test": "Task1_Independent.Test_GSE134355_n22095_scGPT_input.h5ad",
}

def run_one(name, input_path, model_dir, out_dir, batch_size, max_length, device, overwrite):
    output_npz = out_dir / f"{name}_scGPT_cell_embeddings.npz"
    output_meta = out_dir / f"{name}_scGPT_cell_embeddings_metadata.csv"

    if output_npz.exists() and output_meta.exists() and not overwrite:
        print(f"[SKIP] {name}: already exists")
        return

    print("\n===", name, "===")
    print("input:", input_path)

    adata_emb = embed_data(
        input_path,
        model_dir,
        gene_col="features",
        max_length=max_length,
        batch_size=batch_size,
        obs_to_save=["label", "sex", "donorID"],
        device=device,
        use_fast_transformer=True,
        return_new_adata=True,
    )

    emb = adata_emb.X
    if sparse.issparse(emb):
        emb = emb.toarray()

    meta = adata_emb.obs[["label", "sex", "donorID"]].copy()
    meta.insert(0, "cell_id", adata_emb.obs_names.astype(str))

    np.savez_compressed(output_npz, X=emb.astype("float32"), cell_id=meta["cell_id"].astype(str).values)
    meta.to_csv(output_meta, index=False)

    print("embedding shape:", emb.shape)
    print("metadata shape:", meta.shape)
    print("saved:", output_npz)
    print("saved:", output_meta)
    print("[OK]", name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=list(FILES.keys()), choices=list(FILES.keys()))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base = Path.home() / "shared/zhujialin/task1_age_prediction"
    data_dir = base / "data"
    model_dir = Path.home() / "shared/zhujialin/scgpt/checkpoints/scGPT_human"
    out_dir = base / "scgpt/embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.only:
        run_one(name, data_dir / FILES[name], model_dir, out_dir, args.batch_size, args.max_length, args.device, args.overwrite)

    print("All requested scGPT embeddings finished.")

if __name__ == "__main__":
    main()
