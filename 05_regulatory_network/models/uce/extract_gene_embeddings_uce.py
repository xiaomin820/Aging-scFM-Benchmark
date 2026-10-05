"""
Extract Gene Embeddings for UCE Model

Extract ESM2 embeddings from UCE protein_embeddings for genes in genelist_UCE_19790_symbol.csv.

Usage:
    python extract_gene_embeddings_uce.py

Output:
    - outputs/gene_embeddings_uce_19790.csv: 5,120-dimensional embeddings for all 19,790 genes
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


# Map species to protein embedding files
SPECIES_TO_PE_FILE = {
    'human': 'Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt',
    'mouse': 'Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt',
    'frog': 'Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt',
    'zebrafish': 'Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt',
    'mouse_lemur': 'Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt',
    'pig': 'Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt',
    'macaca_fascicularis': 'Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt',
    'macaca_mulatta': 'Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt',
}


def extract_gene_embeddings(
    genelist_path: str,
    output_path: str,
    species: str = "human",
    pe_dir: str = "./model_files/protein_embeddings"
):
    """
    Extract gene ESM2 embeddings from UCE protein_embeddings

    Parameters:
    ----------
    genelist_path : str
        Path to a headerless CSV gene list with gene symbols in the first column
    output_path : str
        Output file path
    species : str
        Species name (default: human)
    pe_dir : str
        Path to the protein embeddings directory
    """
    print("=" * 60)
    print("UCE Gene Embedding Extraction")
    print("=" * 60)

    # 1. Load the gene list
    print(f"\n[1/4] Loading gene list from: {genelist_path}")
    gene_list_df = pd.read_csv(genelist_path, header=None)
    gene_list = gene_list_df.iloc[:, 0].tolist()
    print(f"      Loaded {len(gene_list)} genes")

    # 2. Load species-specific protein embeddings
    print(f"\n[2/4] Loading protein embeddings for {species}...")
    pe_file = os.path.join(pe_dir, SPECIES_TO_PE_FILE[species])
    if not os.path.exists(pe_file):
        raise FileNotFoundError(f"Protein embedding file not found: {pe_file}")

    protein_embeddings = torch.load(pe_file)
    print(f"      Loaded {len(protein_embeddings)} genes with embeddings")
    print(f"      Embedding dimension: {list(protein_embeddings.values())[0].shape[0]}")

    # 3. Extract embeddings for the requested genes
    print(f"\n[3/4] Extracting embeddings for target genes...")

    # Create the output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    results = []
    matched_count = 0
    unmatched_genes = []

    for gene in tqdm(gene_list, desc="Processing genes"):
        # protein_embeddings keys are uppercase gene symbols
        gene_upper = gene.upper()

        if gene_upper in protein_embeddings:
            embedding = protein_embeddings[gene_upper].numpy()
            results.append({
                'GeneSymbol': gene,
            })
            # Add embedding columns
            for i, val in enumerate(embedding):
                results[-1][f'emb_{i}'] = val
            matched_count += 1
        else:
            unmatched_genes.append(gene)

    print(f"\n      Matched: {matched_count} genes")
    print(f"      Unmatched: {len(unmatched_genes)} genes")

    if unmatched_genes:
        print(f"      Sample unmatched genes: {unmatched_genes[:5]}")

    # 4. Save results
    print(f"\n[4/4] Saving embeddings to: {output_path}")
    result_df = pd.DataFrame(results)

    if len(result_df) > 0:
        # Sort by GeneSymbol
        result_df = result_df.sort_values('GeneSymbol')

        # Save as CSV
        result_df.to_csv(output_path, index=False)
        print(f"      Total genes saved: {len(result_df)}")
        print(f"      Embedding dimension: {len(embedding)}")
        print(f"\n[SUCCESS] Embeddings saved successfully!")
    else:
        print("\n[ERROR] No genes matched! Please check your gene list format.")

    return result_df


def check_gene_list_format(genelist_path: str):
    """Check the gene-list file format"""
    print(f"\nGene list format check:")
    print(f"  Path: {genelist_path}")
    with open(genelist_path, 'r') as f:
        first_lines = [f.readline().strip() for _ in range(10)]
    print(f"  First 10 lines:")
    for i, line in enumerate(first_lines):
        print(f"    {i+1}. {line}")


def main():
    # Configure paths
    genelist_path = "./gen_lists/genelist_UCE_19790_symbol.csv"
    output_path = "./outputs/gene_embeddings_uce_19790.csv"
    pe_dir = "./model_files/protein_embeddings"
    species = "human"

    # Check input files
    if not os.path.exists(genelist_path):
        print(f"Error: Gene list file not found: {genelist_path}")
        sys.exit(1)

    # Check the protein embeddings directory
    if not os.path.exists(pe_dir):
        print(f"Error: Protein embeddings directory not found: {pe_dir}")
        print("Please extract model_files/protein_embeddings.tar.gz first.")
        sys.exit(1)

    # Display the gene-list format
    check_gene_list_format(genelist_path)

    # Run extraction
    result_df = extract_gene_embeddings(
        genelist_path=genelist_path,
        output_path=output_path,
        species=species,
        pe_dir=pe_dir
    )

    if len(result_df) > 0:
        print("\n" + "=" * 60)
        print("Sample output (first 3 genes, first 5 embedding dims):")
        print("=" * 60)
        cols = ['GeneSymbol'] + [c for c in result_df.columns if c.startswith('emb_')][:5]
        # Identify embedding columns (emb_0, emb_1, and so on)
        emb_cols = [c for c in result_df.columns if c.startswith('emb_')]
        sample_cols = ['GeneSymbol'] + emb_cols[:5]
        print(result_df[sample_cols].head(3).to_string())


if __name__ == "__main__":
    main()