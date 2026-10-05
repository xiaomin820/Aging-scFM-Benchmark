
# The original scGPT gene list contains 60,697 entries.
# Four entries are not gene names: <cls>, <eoc>, <pad>, and a separator containing repeated "#" characters.
# Exclude these four entries to export 60,693 gene embeddings.
from pathlib import Path
import json
import pandas as pd
import torch

model_dir = Path.home() / "shared/zhujialin/scgpt/checkpoints/scGPT_human"
model_path = model_dir / "best_model.pt"
vocab_path = model_dir / "vocab.json"
target_genelist_path = Path.home() / "shared/zhujialin/gene_embedding/genelists/genelist_scGPT_60697_symbol.csv"
output_path = Path.home() / "shared/zhujialin/gene_embedding/outputs/scGPT_gene_embeddings.csv"

print("=== scGPT gene embedding extraction ===")
print("model_dir:", model_dir)
print("model_path:", model_path)
print("vocab_path:", vocab_path)
print("target_genelist_path:", target_genelist_path)
print("output_path:", output_path)

for path in [model_path, vocab_path, target_genelist_path]:
    if not path.exists():
        raise FileNotFoundError(path)

with open(vocab_path, "r") as f:
    vocab = json.load(f)

print("vocab type:", type(vocab))
print("vocab length:", len(vocab))

if isinstance(vocab, dict):
    gene_to_id = vocab
else:
    raise TypeError("Expected vocab.json to be a dict mapping token/gene to id.")

id_to_gene = {}
for gene, idx in gene_to_id.items():
    try:
        id_to_gene[int(idx)] = str(gene)
    except Exception:
        pass

print("id_to_gene length:", len(id_to_gene))

ckpt = torch.load(model_path, map_location="cpu")
state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

print("checkpoint keys:", len(state))

candidate_keys = [
    "encoder.embedding.weight",
    "module.encoder.embedding.weight",
]

emb = None
used_key = None
for key in candidate_keys:
    if key in state:
        emb = state[key].detach().cpu()
        used_key = key
        break

if emb is None:
    print("Available embedding-like keys:")
    for key in state.keys():
        if "embedding" in key.lower() or "encoder" in key.lower():
            value = state[key]
            if torch.is_tensor(value):
                print(key, tuple(value.shape))
    raise KeyError("Could not find scGPT gene embedding weight.")

print("used embedding key:", used_key)
print("embedding shape:", tuple(emb.shape))

rows = []
missing_vocab = 0
for idx in range(emb.shape[0]):
    gene = id_to_gene.get(idx)
    if gene is None:
        missing_vocab += 1
        continue

    if gene.startswith("<") and gene.endswith(">"):
        continue

    rows.append((gene, idx))

print("non-special gene/token rows:", len(rows))
print("missing vocab rows:", missing_vocab)

target_genes = pd.read_csv(target_genelist_path, header=None)[0].astype(str).tolist()
target_set = set(target_genes)
print("target genelist length:", len(target_genes))

records = []
for gene, idx in rows:
    if gene in target_set:
        records.append((gene, idx))

print("matched genes:", len(records))

if not records:
    raise ValueError("No genes matched between vocab and target genelist.")

out = pd.DataFrame(
    emb[[idx for _, idx in records]].numpy(),
    columns=[f"dim_{i}" for i in range(emb.shape[1])],
)
out.insert(0, "gene_symbol", [gene for gene, _ in records])

output_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(output_path, index=False)

print("saved:", output_path)
print("output shape:", out.shape)
print("scGPT gene embedding extraction OK")
