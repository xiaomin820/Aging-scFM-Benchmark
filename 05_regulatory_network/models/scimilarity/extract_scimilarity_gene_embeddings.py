from pathlib import Path

import pandas as pd
import torch

model_dir = Path.home() / "shared/zhujialin/scimilarity/checkpoints/model_v1.1"
gene_order_path = model_dir / "gene_order.tsv"
decoder_ckpt_path = model_dir / "decoder.ckpt"

output_path = Path.home() / "shared/zhujialin/gene_embedding/outputs/SCimilarity_gene_embeddings.csv"

print("=== SCimilarity gene embedding extraction ===")
print("model_dir:", model_dir)
print("gene_order_path:", gene_order_path)
print("decoder_ckpt_path:", decoder_ckpt_path)
print("output_path:", output_path)

if not gene_order_path.exists():
    raise FileNotFoundError(gene_order_path)

if not decoder_ckpt_path.exists():
    raise FileNotFoundError(decoder_ckpt_path)

genes = pd.read_csv(gene_order_path, sep="\t", header=None)[0].astype(str).tolist()
print("gene count:", len(genes))

ckpt = torch.load(decoder_ckpt_path, map_location="cpu")
state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

key = "network.3.weight"
if key not in state:
    raise KeyError(f"{key} not found in decoder checkpoint")

emb = state[key].detach().cpu()
print("embedding tensor:", key)
print("raw embedding shape:", tuple(emb.shape))

if emb.shape[0] != len(genes):
    raise ValueError(f"Embedding rows {emb.shape[0]} != gene count {len(genes)}")

df = pd.DataFrame(emb.numpy(), columns=[f"dim_{i}" for i in range(emb.shape[1])])
df.insert(0, "gene_symbol", genes)

output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("saved:", output_path)
print("output shape:", df.shape)
print("SCimilarity gene embedding extraction OK")
