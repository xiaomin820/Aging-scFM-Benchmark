"""
Standalone script to extract gene embeddings from scCello model.

This script uses minimal dependencies (numpy, torch, csv) to avoid version conflicts.

Usage:
    source /home/liangyunhao/miniforge3/etc/profile.d/conda.sh
    conda activate sccello
    python extract_gene_embeddings_simple.py
"""

import os
import sys
import pickle
import csv
from pathlib import Path

import numpy as np
import torch

# Paths
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = Path("/home/liangyunhao/shared/models/katarinayuan/scCello-zeroshot")
GENE_LIST_PATH = PROJECT_ROOT / "gen_lists/genelist_scCello_25424_ensembl.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/example_data_saved/scCello_gene_embeddings.csv"
DATA_DIR = PROJECT_ROOT / "data"


def load_token_vocabulary():
    """Load token vocabulary mapping (Ensembl ID -> token ID)."""
    token_dict_path = DATA_DIR / "token_vocabulary" / "token_dictionary.pkl"
    with open(token_dict_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab


def load_vocab_id2name():
    """Load vocab ID to gene name mapping using csv module."""
    vocab_path = DATA_DIR / "token_vocabulary" / "vocab_id2name.csv"
    id_to_name = {}
    with open(vocab_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_to_name[row['id']] = row['name']
    return id_to_name


def load_gene_list():
    """Load gene list from CSV file."""
    genes = []
    with open(GENE_LIST_PATH, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    # Remove BOM and CR
    lines = content.replace('\r', '').strip().split('\n')
    for line in lines:
        gene = line.strip()
        if gene:
            genes.append(gene)
    return genes


def main():
    print("=" * 60)
    print("scCello Gene Embedding Extraction")
    print("=" * 60)

    # Step 1: Load gene list
    print(f"\n[1/6] Loading gene list from {GENE_LIST_PATH}...")
    gene_list = load_gene_list()
    print(f"      Loaded {len(gene_list)} genes")

    # Step 2: Load token vocabulary
    print(f"\n[2/6] Loading token vocabulary...")
    token_vocab = load_token_vocabulary()
    print(f"      Vocabulary size (including <cls> token): {len(token_vocab)}")

    # Step 3: Load vocab ID to name mapping
    print(f"\n[3/6] Loading vocab ID to name mapping...")
    ensembl_to_symbol = load_vocab_id2name()
    print(f"      Loaded {len(ensembl_to_symbol)} gene symbol mappings")

    # Step 4: Load model state dict
    print(f"\n[4/6] Loading model from {MODEL_PATH}...")
    model_bin_path = MODEL_PATH / "pytorch_model.bin"
    state_dict = torch.load(model_bin_path, map_location="cpu", weights_only=False)
    print(f"      Loaded state dict with {len(state_dict)} keys")

    # Step 5: Extract gene embeddings
    print(f"\n[5/6] Extracting gene embeddings...")

    # Find embedding weights key
    embedding_key = None
    for key in state_dict.keys():
        if 'word_embeddings.weight' in key:
            embedding_key = key
            break

    if embedding_key is None:
        available_keys = list(state_dict.keys())[:20]
        raise ValueError(f"Could not find word_embeddings in state dict. Available keys: {available_keys}")

    token_emb = state_dict[embedding_key]
    print(f"      Full embedding matrix shape: {token_emb.shape}")
    print(f"      Embedding key: {embedding_key}")

    # Remove <cls> token (last row) to get gene embeddings only
    gene_emb = token_emb[:-1, :].numpy()
    print(f"      Gene embedding matrix shape: {gene_emb.shape}")

    # Step 6: Map embeddings to genes and save
    print(f"\n[6/6] Mapping embeddings and saving to {OUTPUT_PATH}...")

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for saving
    num_dims = gene_emb.shape[1]
    vocab_keys = list(token_vocab.keys())[:-1]  # Exclude <cls>

    # Create mapping from Ensembl ID to row index
    ensembl_to_row_idx = {}
    for idx, ensembl_id in enumerate(vocab_keys):
        ensembl_to_row_idx[ensembl_id] = idx

    # Write CSV with header
    with open(OUTPUT_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        header = ['Ensembl_ID', 'Gene_Symbol'] + [f'dim_{i}' for i in range(num_dims)]
        writer.writerow(header)

        # Write data for genes in the gene list
        count = 0
        for gene in gene_list:
            if gene in ensembl_to_row_idx:
                row_idx = ensembl_to_row_idx[gene]
                gene_symbol = ensembl_to_symbol.get(gene, '')
                embedding = gene_emb[row_idx]
                row = [gene, gene_symbol] + embedding.tolist()
                writer.writerow(row)
                count += 1

        print(f"      Wrote embeddings for {count} genes")

    print(f"\n{'=' * 60}")
    print(f"Done! Gene embeddings saved to: {OUTPUT_PATH}")
    print(f"Embedding matrix shape: {count} genes x {num_dims} dimensions")
    print("=" * 60)

    # Print sample
    print("\nSample output (first 5 rows):")
    with open(OUTPUT_PATH, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            if i >= 5:
                break
            print(f"  {row[0]} ({row[1]}): [{', '.join(row[2:5])}...]")

    # Summary statistics
    print("\nEmbedding statistics:")
    # Convert to numpy if needed, and compute stats using torch to avoid numpy issues
    emb_tensor = gene_emb if isinstance(gene_emb, torch.Tensor) else torch.tensor(gene_emb)
    print(f"  Min value: {float(emb_tensor.min().item()):.4f}")
    print(f"  Max value: {float(emb_tensor.max().item()):.4f}")
    print(f"  Mean value: {float(emb_tensor.mean().item()):.4f}")
    print(f"  Std value: {float(emb_tensor.std().item()):.4f}")


if __name__ == "__main__":
    main()