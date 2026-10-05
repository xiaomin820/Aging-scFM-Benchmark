"""
Extract gene embeddings from Geneformer model for given gene lists.

Read gene lists from gen_lists and extract the corresponding gene embeddings.

Usage:
    python extract_gene_embeddings.py [--model_dir .] [--input_dir ./gen_lists] [--output_dir ./gene_embeddings]
"""

import os
import argparse
import pickle
import pandas as pd
import numpy as np
from transformers import BertForMaskedLM
from pathlib import Path


def load_geneformer_model(model_dir):
    """Load the pretrained Geneformer model"""
    print(f"Loading Geneformer model from {model_dir}...")
    model = BertForMaskedLM.from_pretrained(
        model_dir,
        output_attentions=False,
        output_hidden_states=True
    )
    model.eval()
    return model


def load_token_dictionary(dict_dir):
    """Load the token dictionary and gene-ID mappings"""
    # The default dictionaries are stored in the geneformer package directory
    import geneformer

    # Locate the geneformer package
    geneformer_path = os.path.dirname(geneformer.__file__)
    print(f"Geneformer package path: {geneformer_path}")

    # Load token_dictionary (Ensembl ID -> token)
    token_dict_path = os.path.join(geneformer_path, "token_dictionary_gc104M.pkl")

    with open(token_dict_path, "rb") as f:
        gene_token_dict = pickle.load(f)  # {Ensembl_ID: token_id}

    print(f"Token dictionary path: {token_dict_path}")

    # Create the reverse mapping from token to Ensembl_ID
    token_gene_dict = {v: k for k, v in gene_token_dict.items()}  # {token: ENSG}

    # Load gene_name_id_dict, whose observed mapping is {symbol: ENSG}
    gene_name_id_path = os.path.join(geneformer_path, "gene_name_id_dict_gc104M.pkl")

    with open(gene_name_id_path, "rb") as f:
        gene_name_id = pickle.load(f)  # {symbol: ENSG}

    print(f"Loaded token dictionary with {len(gene_token_dict)} entries")
    print(f"Loaded gene name mapping with {len(gene_name_id)} entries")

    return gene_token_dict, token_gene_dict, gene_name_id


def get_gene_embeddings_from_model(model):
    """Obtain embeddings for all genes in the model"""
    # Geneformer token embedding matrix
    token_emb = model.state_dict()['bert.embeddings.word_embeddings.weight']
    return token_emb.cpu().numpy()


def process_ensembl_gene_list(gene_list_df, gene_token_dict, gene_emb_matrix):
    """
    Process a gene list containing Ensembl IDs
    gene_list_df: DataFrame containing Ensembl IDs
    gene_token_dict: mapping from ENSG to token_id
    """
    results = []
    not_found = []

    for idx, row in gene_list_df.iterrows():
        ensembl_id = str(row.iloc[0]).strip()

        try:
            # Look up ENSG -> token directly in gene_token_dict
            token_id = gene_token_dict.get(ensembl_id)

            if token_id is not None and token_id < len(gene_emb_matrix):
                emb = gene_emb_matrix[token_id]
                results.append({
                    'ENSG_ID': ensembl_id,
                    'token_id': token_id,
                    **{f'emb_{i}': emb[i] for i in range(len(emb))}
                })
            else:
                not_found.append(ensembl_id)
        except Exception as e:
            not_found.append(ensembl_id)

    return pd.DataFrame(results), not_found


def process_symbol_gene_list(gene_list_df, gene_name_id, gene_token_dict, gene_emb_matrix):
    """
    Process a gene list containing gene symbols
    gene_list_df: DataFrame containing gene symbols
    gene_name_id: dictionary mapping symbol to ENSG, loaded directly from pickle
    gene_token_dict: mapping from ENSG to token_id
    """
    results = []
    not_found = []

    for idx, row in gene_list_df.iterrows():
        gene_symbol = str(row.iloc[0]).strip()

        try:
            # Map each symbol to Ensembl ID using gene_name_id ({symbol: ENSG})
            ensembl_id = gene_name_id.get(gene_symbol)

            # Map Ensembl ID to token ID
            token_id = gene_token_dict.get(ensembl_id) if ensembl_id else None

            if token_id is not None and token_id < len(gene_emb_matrix):
                emb = gene_emb_matrix[token_id]
                results.append({
                    'Symbol': gene_symbol,
                    'ENSG_ID': ensembl_id,
                    'token_id': token_id,
                    **{f'emb_{i}': emb[i] for i in range(len(emb))}
                })
            else:
                not_found.append(gene_symbol)
        except Exception as e:
            not_found.append(gene_symbol)

    return pd.DataFrame(results), not_found


def determine_file_format(file_path):
    """
    Determine the gene-list format
    Return 'ensembl' or 'symbol'
    """
    # Inspect the first row to determine the format
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        first_line = f.readline().strip()

    # Ensembl gene identifiers usually start with ENSG
    if first_line.startswith('ENSG'):
        return 'ensembl'
    else:
        return 'symbol'


def process_gene_list_file(file_path, gene_token_dict, gene_name_id, gene_emb_matrix):
    """Process one gene-list file"""
    print(f"\nProcessing: {os.path.basename(file_path)}")

    # Read the gene list
    gene_list_df = pd.read_csv(file_path, header=None, encoding='utf-8-sig')
    print(f"  Total genes in list: {len(gene_list_df)}")

    # Detect the file format
    file_format = determine_file_format(file_path)
    print(f"  Detected format: {file_format}")

    # Process the gene list
    if file_format == 'ensembl':
        emb_df, not_found = process_ensembl_gene_list(
            gene_list_df, gene_token_dict, gene_emb_matrix
        )
    else:
        emb_df, not_found = process_symbol_gene_list(
            gene_list_df, gene_name_id, gene_token_dict, gene_emb_matrix
        )

    print(f"  Found embeddings: {len(emb_df)}")
    print(f"  Not found: {len(not_found)}")

    return emb_df, not_found


def main():
    parser = argparse.ArgumentParser(
        description='Extract gene embeddings from Geneformer model'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='.',
        help='Directory containing Geneformer model (default: current directory)'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='./gen_lists',
        help='Directory containing gene list files (default: ./gen_lists)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./gene_embeddings',
        help='Directory to save embedding results (default: ./gene_embeddings)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Geneformer-V2-104M',
        help='Model version subfolder name (only process matching gene list file)'
    )

    args = parser.parse_args()

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve the model path
    model_dir = os.path.join(args.model_dir, args.model_name)
    if not os.path.exists(model_dir):
        model_dir = args.model_dir

    # 1. Load the model
    model = load_geneformer_model(model_dir)

    # 2. Load the token dictionary
    gene_token_dict, token_gene_dict, gene_name_id = load_token_dictionary(args.model_dir)

    # 3. Obtain the gene embedding matrix
    gene_emb_matrix = get_gene_embeddings_from_model(model)
    print(f"\nGene embedding matrix shape: {gene_emb_matrix.shape}")

    # 4. Process files in gen_lists that match the model name
    input_dir = Path(args.input_dir)

    # Select gene-list files based on the model name
    # Example: Geneformer-V2-104M -> genelist_Geneformer_*.csv
    model_base_name = args.model_name.split('-')[0]  # Extract the base model name
    model_keyword = f"genelist_{model_base_name}"
    gene_list_files = list(input_dir.glob(f"{model_keyword}_*.csv"))

    if not gene_list_files:
        print(f"No gene list file matching '{model_keyword}' found in {input_dir}")
        return

    print(f"\nFound {len(gene_list_files)} matching gene list file(s) for model: {model_base_name}")

    all_results = {}
    all_not_found = {}

    for file_path in sorted(gene_list_files):
        emb_df, not_found = process_gene_list_file(
            file_path, gene_token_dict, gene_name_id, gene_emb_matrix
        )

        # Save results
        output_name = file_path.stem.replace('genelist_', '') + '_embeddings'
        output_path = os.path.join(args.output_dir, f"{output_name}.csv")
        emb_df.to_csv(output_path, index=False)
        print(f"  Saved embeddings to: {output_path}")

        # Save the list of genes that were not found
        if not_found:
            not_found_path = os.path.join(args.output_dir, f"{output_name}_not_found.txt")
            with open(not_found_path, 'w') as f:
                for gene in not_found:
                    f.write(f"{gene}\n")
            print(f"  Saved not found genes to: {not_found_path}")

        all_results[file_path.name] = len(emb_df)
        all_not_found[file_path.name] = len(not_found)

    # 5. Print the summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'File':<50} {'Found':<10} {'Not Found':<10}")
    print("-"*60)
    for file_name, count in all_results.items():
        not_found_count = all_not_found[file_name]
        print(f"{file_name:<50} {count:<10} {not_found_count:<10}")
    print("="*60)
    print(f"\nAll embeddings saved to: {args.output_dir}")


if __name__ == "__main__":
    main()