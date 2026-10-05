#!/usr/bin/env python
"""
Extract gene embeddings from CellFM model.
This script extracts the gene embedding weights from the pretrained CellFM model
and saves them as a CSV file indexed by gene symbol.

Usage:
    python extract_gene_embeddings.py --ckpt_path checkpoint/CellFM_80M_weight.ckpt
    python extract_gene_embeddings.py --ckpt_path checkpoint/base_weight.ckpt --output gene_embeddings_base.csv
"""

import argparse
import os
import sys

import pandas as pd
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layers.utils import Config_80M
from layers.torch_finetune import FinetuneModel


def map_ms_to_pt(ms_key):
    """Map MindSpore checkpoint keys to PyTorch model keys."""
    name = ms_key
    name = name.replace("layer_norm.gamma", "weight")
    name = name.replace("layer_norm.beta", "bias")
    name = name.replace("post_norm1.gamma", "post_norm1.weight")
    name = name.replace("post_norm1.beta", "post_norm1.bias")
    name = name.replace("post_norm2.gamma", "post_norm2.weight")
    name = name.replace("post_norm2.beta", "post_norm2.bias")
    return name


def load_cellfm_model(ckpt_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load CellFM model from MindSpore checkpoint.

    Args:
        ckpt_path: Path to the MindSpore checkpoint file
        device: Device to load the model on

    Returns:
        model: Loaded PyTorch model
    """
    from mindspore.train.serialization import load_checkpoint

    # Create model instance
    cfg = Config_80M()
    cfg.ecs_threshold = 0.8
    cfg.ecs = True
    cfg.add_zero = True
    cfg.pad_zero = True
    cfg.enc_nlayers = 2

    # n_genes = 27855 for the original model
    n_genes = 27855

    model = FinetuneModel(n_genes, cfg)

    # Load MindSpore checkpoint
    ms_ckpt = load_checkpoint(ckpt_path)

    # Convert to PyTorch state dict
    torch_state_dict = {}
    for ms_key, ms_param in ms_ckpt.items():
        pt_key = map_ms_to_pt(ms_key)
        pt_tensor = torch.tensor(ms_param.asnumpy())

        # Skip optimizer states and meta info
        if pt_key.startswith("moment1.") or pt_key.startswith("moment2."):
            continue
        if pt_key in ['global_step', 'learning_rate', 'beta1_power', 'beta2_power',
                      'current_iterator_step', 'last_overflow_iterator_step']:
            continue
        torch_state_dict[pt_key] = pt_tensor

    # Load state dict into model
    missing_keys, unexpected_keys = model.load_state_dict(torch_state_dict, strict=False)
    print(f"[Load Report]")
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    model = model.to(device)
    model.eval()

    return model


def load_gene_info(gene_info_path):
    """
    Load gene information from CSV file.

    Args:
        gene_info_path: Path to gene info CSV file

    Returns:
        gene_info_df: DataFrame with gene information
        geneset: Dictionary mapping gene symbol to index (1-based)
    """
    gene_info_df = pd.read_csv(gene_info_path, index_col=0, header=0)
    # geneset is 1-based indexed (0 is padding token)
    geneset = {j: i + 1 for i, j in enumerate(gene_info_df.index)}
    return gene_info_df, geneset


def extract_gene_embeddings(model, gene_info_df, geneset, exclude_padding=True):
    """
    Extract gene embeddings from the model.

    Args:
        model: CellFM PyTorch model
        gene_info_df: DataFrame with gene information
        geneset: Dictionary mapping gene symbol to index
        exclude_padding: Whether to exclude padding token (index 0)

    Returns:
        gene_emb_df: DataFrame with gene embeddings indexed by GeneID
    """
    # Get gene embedding weights
    # gene_emb shape: [n_genes + 1 + padding_align, enc_dims]
    gene_emb = model.state_dict()['gene_emb'].cpu().numpy()

    print(f"Gene embedding shape: {gene_emb.shape}")
    print(f"Number of genes in model embedding: {gene_emb.shape[0]}")

    # Create embedding DataFrame
    # Note: index 0 is padding token, actual genes start from index 1
    if exclude_padding:
        # Remove padding token (index 0)
        gene_emb = gene_emb[1:, :]

    n_genes = gene_emb.shape[0]

    # Create a mapping from model index to gene symbol
    # geneset is 1-based, so we need to adjust
    idx_to_gene = {v: k for k, v in geneset.items() if v <= n_genes}

    # Build embedding DataFrame with gene symbols
    gene_emb_df = pd.DataFrame(gene_emb)
    gene_emb_df['Symbol'] = [idx_to_gene.get(i + 1, f'Unknown_{i}') for i in range(n_genes)]

    # Merge with gene info to get GeneID
    # Using inner join to only keep genes that exist in both
    gene_emb_df = pd.merge(
        left=gene_info_df,
        right=gene_emb_df,
        on='Symbol',
        how='inner'
    )

    print(f"Matched genes after merge: {gene_emb_df.shape[0]}")

    # Set index to GeneID and drop Symbol column
    gene_emb_df.set_index('Symbol', inplace=True)

    return gene_emb_df


def extract_embeddings_for_genelist(model, gene_info_df, geneset, gene_list_path, output_path):
    """
    Extract embeddings specifically for genes in the provided genelist file.

    Args:
        model: CellFM PyTorch model
        gene_info_df: DataFrame with gene information
        geneset: Dictionary mapping gene symbol to index
        gene_list_path: Path to CSV file with gene symbols
        output_path: Path to save the output CSV

    Returns:
        gene_emb_df: DataFrame with filtered gene embeddings
    """
    # Get gene embedding weights
    gene_emb = model.state_dict()['gene_emb'].cpu().numpy()

    print(f"Gene embedding shape: {gene_emb.shape}")

    # Remove padding token (index 0)
    gene_emb = gene_emb[1:, :]
    n_genes = gene_emb.shape[0]

    # Load gene list from CSV
    genelist_df = pd.read_csv(gene_list_path, header=None)
    gene_list = genelist_df[0].tolist()
    print(f"Number of genes in genelist: {len(gene_list)}")

    # Create mapping from gene symbol to index
    idx_to_gene = {v: k for k, v in geneset.items() if v <= n_genes}

    # Filter genelist to only include genes that exist in the model
    matched_genes = []
    missing_genes = []

    for gene in gene_list:
        if gene in geneset and geneset[gene] <= n_genes:
            matched_genes.append(gene)
        else:
            missing_genes.append(gene)

    print(f"Matched genes: {len(matched_genes)}")
    print(f"Missing genes: {len(missing_genes)}")

    if missing_genes:
        print(f"First 10 missing genes: {missing_genes[:10]}")

    # Extract embeddings for matched genes
    gene_emb_list = []
    gene_ids = []

    for gene in matched_genes:
        idx = geneset[gene] - 1  # Convert to 0-based index
        gene_emb_list.append(gene_emb[idx])

        # Get GeneID from gene_info
        if gene in gene_info_df.index:
            gene_ids.append(gene)
        else:
            gene_ids.append(gene)

    # Create DataFrame
    embedding_dim = gene_emb.shape[1]
    gene_emb_filtered = pd.DataFrame(
        gene_emb_list,
        index=gene_ids,
        columns=[f'dim_{i}' for i in range(embedding_dim)]
    )

    # Save to CSV without header (matching other models' format)
    gene_emb_filtered.to_csv(output_path, header=False)
    print(f"Gene embeddings saved to {output_path}")
    print(f"Output shape: {gene_emb_filtered.shape}")

    return gene_emb_filtered


def main():
    parser = argparse.ArgumentParser(
        description='Extract gene embeddings from CellFM model'
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='checkpoint/CellFM_80M_weight.ckpt',
        help='Path to the MindSpore checkpoint file'
    )
    parser.add_argument(
        '--gene_list',
        type=str,
        default='gen_lists/genelist_CellFM_24078_symbol.csv',
        help='Path to the gene list CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='gene_embeddings_CellFM_24078.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--gene_info',
        type=str,
        default='csv/expand_gene_info.csv',
        help='Path to gene info CSV file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['finetune', 'base'],
        default='finetune',
        help='Model type: finetune or base'
    )

    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.ckpt_path):
        print(f"Error: Checkpoint file not found: {args.ckpt_path}")
        sys.exit(1)

    print(f"Loading model from {args.ckpt_path}...")
    print(f"Device: {args.device}")

    # Load model
    model = load_cellfm_model(args.ckpt_path, args.device)

    # Load gene info
    print(f"Loading gene info from {args.gene_info}...")
    gene_info_df, geneset = load_gene_info(args.gene_info)
    print(f"Number of genes in gene info: {len(geneset)}")

    # Check if genelist exists
    if os.path.exists(args.gene_list):
        print(f"Extracting embeddings for genes in {args.gene_list}...")
        gene_emb_df = extract_embeddings_for_genelist(
            model, gene_info_df, geneset, args.gene_list, args.output
        )
    else:
        print(f"Genelist file not found: {args.gene_list}")
        print("Extracting all gene embeddings...")

        # Extract all embeddings
        gene_emb_df = extract_gene_embeddings(model, gene_info_df, geneset)

        # Save to CSV without header
        gene_emb_df.to_csv(args.output, header=False)
        print(f"Gene embeddings saved to {args.output}")

    print("Done!")


if __name__ == '__main__':
    main()