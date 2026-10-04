from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import scanpy as sc


TAG = "scLONG"

BASE = Path.home() / "shared/zhujialin/task1_age_prediction"
DATA_DIR = BASE / "sclong/data"
OUT_DIR = BASE / "sclong/embeddings"
TMP_DIR = BASE / "sclong/tmp"

REPO = Path.home() / "zjl/aging_benchmark/repos/scLong"
EMBED_PY = REPO / "embed.py"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)


def output_name(path):
    name = path.name

    if "Independent" in name:
        return "independent_test"

    m = re.search(r"Training_Part(\d+)", name)
    if m:
        return f"train_part{m.group(1)}"

    raise ValueError(f"Cannot parse dataset name from: {path}")


def build_metadata(h5ad_path):
    adata = sc.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()

    return pd.DataFrame({
        "cell_id": adata.obs_names.astype(str),
        "label": obs["label"].values,
        "sex": obs["sex"].values if "sex" in obs.columns else "",
        "donorID": obs["donorID"].values if "donorID" in obs.columns else "",
    })


def load_embedding(path):
    if path.exists():
        if path.suffix == ".npy":
            return np.asarray(np.load(path), dtype=np.float32)

        if path.suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            key = "X" if "X" in data.files else data.files[0]
            return np.asarray(data[key], dtype=np.float32)

        if path.suffix == ".csv":
            return pd.read_csv(path, index_col=0).to_numpy(dtype=np.float32)

    candidates = []
    for suffix in [".npy", ".npz", ".csv"]:
        candidates.extend(sorted(path.parent.glob(path.stem + "*" + suffix)))

    if not candidates:
        raise FileNotFoundError(f"No scLONG embedding output found near {path}")

    print("found candidate output:", candidates[0])
    return load_embedding(candidates[0])


def run_sclong(input_h5ad, output_path):
    if not EMBED_PY.exists():
        raise FileNotFoundError(
            f"Cannot find scLONG embed.py: {EMBED_PY}\n"
            "If your scLONG repo path is different, edit REPO in this script."
        )

    cmd = [
        "python",
        str(EMBED_PY),
        "--target_data_path",
        str(input_h5ad),
        "--target_embed_path",
        str(output_path),
    ]

    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO))


def main():
    print("data dir:", DATA_DIR)
    print("scLONG repo:", REPO)
    print("embed script:", EMBED_PY)
    print("embed script exists:", EMBED_PY.exists())

    files = sorted(DATA_DIR.rglob("*.h5ad"))
    if not files:
        raise FileNotFoundError(f"No h5ad files found under {DATA_DIR}")

    print("found h5ad files:")
    for f in files:
        print(" ", f)

    for data_path in files:
        name = output_name(data_path)

        print(f"\n=== Processing {name} ===")
        print("input:", data_path)

        emb_path = OUT_DIR / f"{name}_{TAG}_cell_embeddings.npz"
        meta_path = OUT_DIR / f"{name}_{TAG}_cell_embeddings_metadata.csv"
        raw_path = TMP_DIR / f"{name}_{TAG}_raw.npy"

        if emb_path.exists() and meta_path.exists():
            print("embedding exists, skip:", emb_path)
            continue

        if not raw_path.exists():
            run_sclong(data_path, raw_path)
        else:
            print("raw embedding exists, skip:", raw_path)

        emb = load_embedding(raw_path)
        meta = build_metadata(data_path)

        if emb.shape[0] != len(meta):
            raise ValueError(f"embedding rows {emb.shape[0]} != metadata rows {len(meta)}")

        np.savez_compressed(
            emb_path,
            X=emb.astype(np.float32),
            cell_id=meta["cell_id"].to_numpy(),
        )
        meta.to_csv(meta_path, index=False)

        print("saved embedding:", emb_path)
        print("saved metadata:", meta_path)
        print("embedding shape:", emb.shape)

    print("\nscLONG cell embeddings finished.")


if __name__ == "__main__":
    main()