from pathlib import Path

import numpy as np
import pandas as pd

base = Path.home() / "shared/zhujialin/sclong/checkpoints"
vec_path = base / "selected_gene2vec_27k.npy"
genes_path = base / "selected_genes_27k.txt"

target_genelist_path = Path.home() / "shared/zhujialin/gene_embedding/genelists/genelist_scLONG_27874_ensembl.csv"
output_path = Path.home() / "shared/zhujialin/gene_embedding/outputs/scLong_gene_embeddings.csv"

print("=== scLong gene embedding extraction ===")
print("vec_path:", vec_path)
print("genes_path:", genes_path)
print("target_genelist_path:", target_genelist_path)
print("output_path:", output_path)

if not vec_path.exists():
    raise FileNotFoundError(vec_path)

if not genes_path.exists():
    raise FileNotFoundError(genes_path)

vec = np.load(vec_path)
genes = [line.strip() for line in open(genes_path) if line.strip()]

print("vec shape:", vec.shape)
print("gene count:", len(genes))

if vec.shape[0] != len(genes):
    if vec.shape[1] == len(genes):
        print("Transposing vector matrix to genes x dims")
        vec = vec.T
    else:
        raise ValueError(f"Vector shape {vec.shape} does not match gene count {len(genes)}")

target_genes = pd.read_csv(target_genelist_path, header=None)[0].astype(str).str.strip().tolist()
target_set = set(target_genes)

print("target genelist length:", len(target_genes))

records = []
gene_to_idx = {gene: i for i, gene in enumerate(genes)}
for gene in target_genes:
    if gene in gene_to_idx:
        records.append((gene, gene_to_idx[gene]))

print("matched genes:", len(records))

if not records:
    raise ValueError("No genes matched between selected_genes_27k.txt and target genelist.")

out = pd.DataFrame(
    vec[[idx for _, idx in records]],
    columns=[f"dim_{i}" for i in range(vec.shape[1])],
)
out.insert(0, "gene_ensembl_id", [gene for gene, _ in records])

output_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(output_path, index=False)

print("saved:", output_path)
print("output shape:", out.shape)
print("scLong gene embedding extraction OK")
