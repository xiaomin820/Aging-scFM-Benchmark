from pathlib import Path
import subprocess
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse

TAG = "scFoundation"

BASE = Path.home() / "shared/zhujialin/task1_age_prediction"
WORK_DIR = BASE / "scfoundation"
RAW_ZIP = BASE / "raw_zip/scFoundation_input.zip"
DATA_DIR = WORK_DIR / "data"
TMP_DIR = WORK_DIR / "tmp"
EMB_DIR = WORK_DIR / "embeddings"

REPO = Path.home() / "zjl/aging_benchmark/repos/scFoundation"
MODEL_DIR = REPO / "model"
GET_EMBEDDING = MODEL_DIR / "get_embedding.py"

CHUNK_SIZE = 128


def output_name(name):
    low = name.lower()
    if "independent" in low:
        return "independent_test"
    for i in range(1, 6):
        if f"part{i}" in low:
            return f"train_part{i}"
    raise ValueError(f"Cannot infer output name from {name}")


def list_csv_sources():
    extracted = sorted((DATA_DIR / "scFoundation_input").glob("*.csv"))
    if extracted:
        return [("file", p) for p in extracted]

    if not RAW_ZIP.exists():
        raise FileNotFoundError(f"missing zip: {RAW_ZIP}")

    sources = []
    with zipfile.ZipFile(RAW_ZIP) as zf:
        for name in zf.namelist():
            if name.endswith(".csv"):
                sources.append(("zip", name))
    return sorted(sources, key=lambda x: str(x[1]))


def prepare_npz(source, name):
    npz_path = TMP_DIR / f"{name}_expression_19264.npz"
    meta_path = EMB_DIR / f"{name}_{TAG}_cell_embeddings_metadata.csv"

    if npz_path.exists() and meta_path.exists():
        print("prepared exists:", npz_path)
        return npz_path

    print(f"\n=== Preparing {name} ===")
    print("source:", source[1])

    kind, item = source
    if kind == "file":
        reader = pd.read_csv(item, chunksize=CHUNK_SIZE, index_col=0)
    else:
        zf = zipfile.ZipFile(RAW_ZIP)
        reader = pd.read_csv(zf.open(item), chunksize=CHUNK_SIZE, index_col=0)

    matrices = []
    meta_written = False
    n_rows = 0

    for chunk_id, chunk in enumerate(reader, start=1):
        if chunk.shape[1] < 4:
            raise ValueError(f"{item} has too few columns: {chunk.shape}")

        expr = chunk.iloc[:, :-3]
        meta = chunk.iloc[:, -3:].copy()
        meta.insert(0, "cell_id", chunk.index.astype(str))
        meta.columns = ["cell_id", "label", "sex", "donorID"]

        meta.to_csv(
            meta_path,
            mode="w" if not meta_written else "a",
            header=not meta_written,
            index=False,
        )
        meta_written = True

        expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
        matrices.append(sparse.csr_matrix(expr.values))

        n_rows += len(chunk)
        if chunk_id % 20 == 0:
            print(f"  chunks={chunk_id}, rows={n_rows}")

    if not matrices:
        raise ValueError(f"no rows read from {item}")

    X = sparse.vstack(matrices, format="csr")
    print("expression shape:", X.shape)

    sparse.save_npz(npz_path, X)
    print("saved expression:", npz_path)
    print("saved metadata:", meta_path)

    return npz_path


def find_raw_embedding(name):
    candidates = sorted(TMP_DIR.glob(f"{name}*embedding*.npy"))
    if not candidates:
        candidates = sorted(TMP_DIR.glob(f"*{name}*.npy"))
    if not candidates:
        raise FileNotFoundError(f"cannot find scFoundation embedding output for {name} in {TMP_DIR}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_scfoundation(npz_path, name):
    final_out = EMB_DIR / f"{name}_{TAG}_cell_embeddings.npz"

    if final_out.exists():
        print("embedding exists:", final_out)
        return

    before = set(TMP_DIR.glob("*.npy"))

    cmd = [
        "python",
        str(GET_EMBEDDING),
        "--task_name",
        name,
        "--input_type",
        "singlecell",
        "--output_type",
        "cell",
        "--pool_type",
        "all",
        "--tgthighres",
        "t4",
        "--data_path",
        str(npz_path),
        "--save_path",
        str(TMP_DIR) + "/",
        "--pre_normalized",
        "F",
        "--version",
        "ce",
    ]

    print("\n=== Running scFoundation ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(MODEL_DIR))

    after = set(TMP_DIR.glob("*.npy"))
    new_files = sorted(after - before)

    if new_files:
        raw_out = max(new_files, key=lambda p: p.stat().st_mtime)
    else:
        raw_out = find_raw_embedding(name)

    print("raw embedding:", raw_out)

    X = np.load(raw_out, allow_pickle=True)
    X = np.asarray(X)

    if X.dtype == object:
        X = np.asarray(X.tolist())

    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    X = X.astype(np.float32)
    print("embedding shape:", X.shape)

    np.savez_compressed(final_out, X=X)
    print("saved:", final_out)


def main():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    print("repo:", REPO)
    print("get_embedding:", GET_EMBEDDING, GET_EMBEDDING.exists())
    print("raw zip:", RAW_ZIP, RAW_ZIP.exists())

    sources = list_csv_sources()
    print("found csv files:")
    for source in sources:
        print(" ", source[1])

    for source in sources:
        name = output_name(str(source[1]))
        npz_path = prepare_npz(source, name)
        run_scfoundation(npz_path, name)

    print("\nscFoundation cell embeddings finished.")


if __name__ == "__main__":
    main()