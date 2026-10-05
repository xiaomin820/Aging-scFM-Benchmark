from pathlib import Path

import pandas as pd
from scdataloader.utils import load_genes
from scprint.tokenizers import protein_embeddings_generator

target_genelist_path = Path.home() / "shared/zhujialin/gene_embedding/genelists/genelist_scPRINT_25424__ensembl.csv"
output_path = Path.home() / "shared/zhujialin/gene_embedding/outputs/scPRINT_gene_embeddings.csv"
fasta_path = str(Path.home() / "shared/zhujialin/scprint/fasta") + "/"

print("=== scPRINT gene embedding extraction ===")
print("target_genelist_path:", target_genelist_path)
print("output_path:", output_path)
print("fasta_path:", fasta_path)

target_genes = pd.read_csv(target_genelist_path, header=None)[0].astype(str).str.strip().tolist()
target_set = set(target_genes)

genedf = load_genes(organisms="NCBITaxon:9606")
sub = genedf.loc[genedf.index.astype(str).isin(target_set)].copy()

print("target genelist length:", len(target_genes))
print("matched genes in genedf:", sub.shape)

result = protein_embeddings_generator(
    sub,
    organism="homo_sapiens",
    cache=True,
    embedding_size=1024,
    fasta_path=fasta_path,
)

emb = result[0]
emb.index.name = "gene_ensembl_id"
emb.columns = [f"dim_{i}" for i in range(emb.shape[1])]

out = emb.reset_index()
out.to_csv(output_path, index=False)

print("saved:", output_path)
print("output shape:", out.shape)
print("scPRINT gene embedding extraction OK")
