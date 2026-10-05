import os
import sys
from pathlib import Path

import pandas as pd
import torch

project_root = Path.home() / "zjl/aging_benchmark"
repo_model_dir = project_root / "repos/scFoundation/model"
ckpt_path = repo_model_dir / "models/models.ckpt"

genelist_path = Path.home() / "shared/zhujialin/gene_embedding/genelists/genelist_scFoundation_19264_symbol.csv"
output_path = Path.home() / "shared/zhujialin/gene_embedding/outputs/scFoundation_gene_embeddings.csv"

sys.path.insert(0, str(repo_model_dir))
os.chdir(repo_model_dir)

from load import load_model_frommmf

print("=== scFoundation gene embedding extraction ===")
print("repo_model_dir:", repo_model_dir)
print("ckpt_path:", ckpt_path)
print("genelist_path:", genelist_path)
print("output_path:", output_path)

if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

if not genelist_path.exists():
    raise FileNotFoundError(f"Gene list not found: {genelist_path}")

genes = pd.read_csv(genelist_path, header=None)[0].astype(str).tolist()
print("gene list length:", len(genes))

key = "cell"
model, config = load_model_frommmf(str(ckpt_path), key)

if not hasattr(model, "pos_emb"):
    raise AttributeError("Loaded scFoundation model has no attribute 'pos_emb'.")

emb = model.pos_emb.weight.detach().cpu()
print("raw embedding shape:", tuple(emb.shape))

if emb.shape[0] < len(genes):
    raise ValueError(f"Embedding rows {emb.shape[0]} < gene count {len(genes)}")

emb = emb[: len(genes), :]
print("used embedding shape:", tuple(emb.shape))

df = pd.DataFrame(emb.numpy(), columns=[f"dim_{i}" for i in range(emb.shape[1])])
df.insert(0, "gene_symbol", genes)

output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("saved:", output_path)
print("output shape:", df.shape)
print("scFoundation gene embedding extraction OK")
