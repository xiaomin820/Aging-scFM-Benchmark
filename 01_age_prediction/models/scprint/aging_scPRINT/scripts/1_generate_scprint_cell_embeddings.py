from pathlib import Path
import os
import random
import re
from anndata import AnnData
from scipy import sparse
from scdataloader.utils import load_genes
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from scprint.model.model import scPrint
from scprint.tasks.cell_emb import Embedder


TAG = "scPRINT"
SEED = 2026

BASE = Path.home() / "shared/zhujialin/task1_age_prediction"
DATA_DIR = BASE / "scprint/data"
OUT_DIR = BASE / "scprint/embeddings"
TMP_DIR = BASE / "scprint/tmp"
MODEL_DIR = BASE / "scprint/models"

SOURCE_CKPT = Path.home() / "shared/zhujialin/scprint/checkpoints/medium-v1.5.ckpt"
AUG_CKPT = MODEL_DIR / "medium-v1.5.augmented_gene_emb.ckpt"

SOURCE_GENE_EMB = Path("/home/zhujialin/shared/zhujialin/gene_embedding/outputs/scPRINT_gene_embeddings.parquet")
AUG_GENE_EMB = Path("/home/zhujialin/shared/zhujialin/gene_embedding/outputs/scPRINT_gene_embeddings_augmented.parquet")

OLD_GENE_EMB = "/lustre/fswork/projects/rech/xeg/uat95fg/gene_embeddings.parquet"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    return pd.DataFrame({
        "cell_id": adata.obs_names.astype(str),
        "label": obs["label"].values,
        "sex": obs["sex"].values if "sex" in obs.columns else "",
        "donorID": obs["donorID"].values if "donorID" in obs.columns else "",
    })


def prepare_adata(adata):
    organism = "NCBITaxon:9606"

    if "organism_ontology_term_id" not in adata.obs.columns:
        adata.obs["organism_ontology_term_id"] = organism
    adata.obs["organism_ontology_term_id"] = adata.obs["organism_ontology_term_id"].astype(str)

    genedf = load_genes(organisms=organism)
    target_genes = genedf.index.astype(str).tolist()
    target_pos = {g: i for i, g in enumerate(target_genes)}

    if "ensembl_id" in adata.var.columns:
        gene_ids = adata.var["ensembl_id"].astype(str).str.split(".").str[0].tolist()
    else:
        gene_ids = adata.var_names.astype(str).str.split(".").str[0].tolist()

    keep = []
    seen = set()
    kept_gene_ids = []

    for i, g in enumerate(gene_ids):
        if g in target_pos and g not in seen:
            keep.append(i)
            kept_gene_ids.append(g)
            seen.add(g)

    print("adata genes:", adata.n_vars)
    print("scPRINT target genes:", len(target_genes))
    print("matched genes:", len(keep))

    if not keep:
        raise ValueError("No genes matched between h5ad and scPRINT human gene list.")

    X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()

    X_keep = X[:, keep].tocoo()
    new_cols = np.asarray([target_pos[g] for g in kept_gene_ids], dtype=np.int64)

    X_aligned = sparse.csr_matrix(
        (X_keep.data, (X_keep.row, new_cols[X_keep.col])),
        shape=(adata.n_obs, len(target_genes)),
    )

    obs = adata.obs.copy()
    var = genedf.copy()
    var.index = var.index.astype(str)

    aligned = AnnData(X=X_aligned, obs=obs, var=var)
    aligned.obs_names = adata.obs_names.astype(str)

    if "n_counts" not in aligned.obs.columns:
        aligned.obs["n_counts"] = np.asarray(X_aligned.sum(axis=1)).reshape(-1)

    print("aligned shape:", aligned.shape)
    return aligned


def replace_string_in_obj(obj, old_value, new_value, counter):
    if isinstance(obj, str):
        if old_value in obj:
            counter["n"] += 1
            return obj.replace(old_value, new_value)
        return obj

    if isinstance(obj, dict):
        return {
            k: replace_string_in_obj(v, old_value, new_value, counter)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            replace_string_in_obj(v, old_value, new_value, counter)
            for v in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            replace_string_in_obj(v, old_value, new_value, counter)
            for v in obj
        )

    return obj


def replace_any_gene_emb_path(obj, new_value, counter):
    if isinstance(obj, str):
        if obj.endswith("gene_embeddings.parquet") or "scPRINT_gene_embeddings" in obj:
            counter["n"] += 1
            return new_value
        return obj

    if isinstance(obj, dict):
        return {
            k: replace_any_gene_emb_path(v, new_value, counter)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            replace_any_gene_emb_path(v, new_value, counter)
            for v in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            replace_any_gene_emb_path(v, new_value, counter)
            for v in obj
        )

    return obj


def get_checkpoint_genes(ckpt):
    if "hyper_parameters" not in ckpt:
        raise KeyError("checkpoint has no hyper_parameters")

    hp = ckpt["hyper_parameters"]

    if "genes" not in hp:
        raise KeyError("checkpoint hyper_parameters has no genes")

    genes = list(hp["genes"])
    if not genes:
        raise ValueError("checkpoint genes list is empty")

    return [str(g) for g in genes]


def ensure_augmented_gene_embedding_and_checkpoint():
    print("source checkpoint:", SOURCE_CKPT)
    print("source gene embedding:", SOURCE_GENE_EMB)
    print("augmented gene embedding:", AUG_GENE_EMB)
    print("augmented checkpoint:", AUG_CKPT)

    if not SOURCE_CKPT.exists():
        raise FileNotFoundError(SOURCE_CKPT)

    if not SOURCE_GENE_EMB.exists():
        raise FileNotFoundError(SOURCE_GENE_EMB)

    if AUG_CKPT.exists() and AUG_GENE_EMB.exists():
        print("augmented checkpoint and gene embedding exist, use them")
        return AUG_CKPT

    print("loading source checkpoint...")
    ckpt = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)

    genes = get_checkpoint_genes(ckpt)

    print("checkpoint genes:", len(genes))
    print("checkpoint genes head:", genes[:10])

    print("loading source gene embedding parquet...")
    df = pd.read_parquet(SOURCE_GENE_EMB)
    df.index = df.index.astype(str)

    print("source gene embedding shape:", df.shape)
    print("source gene embedding index head:", df.index[:10].tolist())

    missing = [g for g in genes if g not in df.index]

    print("missing genes:", len(missing))
    print("missing genes head:", missing[:10])

    if missing:
        zeros = pd.DataFrame(
            np.zeros((len(missing), df.shape[1]), dtype=np.float32),
            index=missing,
            columns=df.columns,
        )
        df = pd.concat([df, zeros], axis=0)

    df = df[~df.index.duplicated(keep="first")]
    df = df.loc[genes]

    AUG_GENE_EMB.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(AUG_GENE_EMB)

    print("saved augmented gene embedding:", AUG_GENE_EMB)
    print("augmented gene embedding shape:", df.shape)

    counter_old = {"n": 0}
    ckpt = replace_string_in_obj(
        ckpt,
        OLD_GENE_EMB,
        str(AUG_GENE_EMB),
        counter_old,
    )

    counter_any = {"n": 0}
    ckpt = replace_any_gene_emb_path(
        ckpt,
        str(AUG_GENE_EMB),
        counter_any,
    )

    AUG_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, AUG_CKPT)

    print("saved augmented checkpoint:", AUG_CKPT)
    print("replaced old exact gene embedding paths:", counter_old["n"])
    print("replaced any gene embedding paths:", counter_any["n"])

    return AUG_CKPT


def load_model(device):
    ckpt_path = ensure_augmented_gene_embedding_and_checkpoint()

    print("loading checkpoint:", ckpt_path)

    model = scPrint.load_from_checkpoint(
        str(ckpt_path),
        map_location=device,
        weights_only=False,
    )

    model.eval()
    model.to(device)

    print("model loaded")
    print("model device:", model.device)

    return model


def extract_one(model, data_path, name, device):
    print(f"\n=== Processing {name} ===")
    print("input:", data_path)

    emb_path = OUT_DIR / f"{name}_{TAG}_cell_embeddings.npz"
    meta_path = OUT_DIR / f"{name}_{TAG}_cell_embeddings_metadata.csv"

    if emb_path.exists() and meta_path.exists():
        print("embedding exists, skip:", emb_path)
        return

    adata = sc.read_h5ad(data_path)

    print("original shape:", adata.shape)
    print("obs columns:", list(adata.obs.columns))
    print("var columns:", list(adata.var.columns))

    adata = prepare_adata(adata)

    run_dir = TMP_DIR / f"{name}_scprint_api_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    os.chdir(run_dir)

    try:
        embedder = Embedder(
            batch_size=64,
            num_workers=8,
            how="random expr",
            max_len=2000,
            doclass=False,
            doplot=False,
            keep_all_cls_pred=False,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            output_expression="none",
            get_gene_emb=False,
            save_every=40000,
        )

        adata_emb, metrics = embedder(model, adata, cache=False)

    finally:
        os.chdir(old_cwd)

    if "scprint_emb" not in adata_emb.obsm:
        raise KeyError(
            "scprint_emb not found in adata.obsm. "
            f"Available obsm keys: {list(adata_emb.obsm.keys())}"
        )

    emb = np.asarray(adata_emb.obsm["scprint_emb"], dtype=np.float32)
    meta = build_metadata(adata_emb)

    if emb.shape[0] != len(meta):
        raise ValueError(f"embedding rows {emb.shape[0]} != metadata rows {len(meta)}")

    np.savez_compressed(
        emb_path,
        X=emb,
        cell_id=meta["cell_id"].to_numpy(),
    )

    meta.to_csv(meta_path, index=False)

    print("metrics:", metrics)
    print("saved embedding:", emb_path)
    print("saved metadata:", meta_path)
    print("embedding shape:", emb.shape)


def main():
    set_seed()

    print("data dir:", DATA_DIR)
    print("source checkpoint:", SOURCE_CKPT)
    print("source gene embedding:", SOURCE_GENE_EMB)
    print("source checkpoint exists:", SOURCE_CKPT.exists())
    print("source gene embedding exists:", SOURCE_GENE_EMB.exists())

    files = sorted(DATA_DIR.rglob("*.h5ad"))
    if not files:
        raise FileNotFoundError(f"No h5ad files found under {DATA_DIR}")

    print("found h5ad files:")
    for f in files:
        print(" ", f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = load_model(device)

    for data_path in files:
        name = output_name(data_path)
        extract_one(model, data_path, name, device)

    print("\nscPRINT cell embeddings finished.")


if __name__ == "__main__":
    main()