"""
TranscriptFormer embedding export from a gene list
Export embedding vectors for the supplied gene list
"""

import os
import logging
import json
import tempfile

# Disable PyTorch JIT before importing torch
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

# Disable torch.compile entirely
import torch._dynamo
torch._dynamo.config.disable = True
torch._dynamo.reset()

# Replace torch.compile with a no-op
_original_compile = torch.compile
def _disabled_compile(*args, **kwargs):
    return args[0] if args else None
torch.compile = _disabled_compile

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import anndata as ad
import numpy as np

from transcriptformer.model.inference import run_inference

# Configuration
CHECKPOINT_PATH = "./checkpoints/tf_sapiens"  # Use tf_sapiens
GENE_LIST_FILE = "./aging_data/TranscriptFormer_genelist.csv"
OUTPUT_PATH = "./embedding_results"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_dummy_h5ad(gene_ids, output_path):
    """Create a synthetic AnnData object from a gene list"""
    n_genes = len(gene_ids)
    # Create a cells x genes count matrix of ones, indicating that each gene is present
    X = np.ones((1, n_genes), dtype=np.float32)

    adata = ad.AnnData(X=X)
    adata.var_names = gene_ids
    adata.var_names_make_unique()
    # Add the ensembl_id column
    adata.var["ensembl_id"] = adata.var_names.tolist()
    # Add the required obs columns
    adata.obs["assay"] = "single-cell RNA sequencing"  # Use an assay type from the vocabulary

    adata.write_h5ad(output_path)
    logging.info(f"Create a synthetic h5ad: {output_path}, gene count: {adata.n_vars}")
    return adata


@hydra.main(
    config_path="src/transcriptformer/cli/conf",
    config_name="inference_config.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    # 0. Configure the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

    # 1. Read the gene list
    gene_df = pd.read_csv(GENE_LIST_FILE, header=None, names=['gene_id'])
    gene_ids = gene_df['gene_id'].tolist()
    logging.info(f"Read the gene list: {len(gene_ids)} genes")

    # 2. Create a synthetic h5ad
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        tmp_h5ad_path = tmp.name
    create_dummy_h5ad(gene_ids, tmp_h5ad_path)

    # 3. Load the model configuration
    config_path = os.path.join(CHECKPOINT_PATH, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    mlflow_cfg = OmegaConf.create(config_dict)

    # Merge configuration settings
    cfg = OmegaConf.merge(mlflow_cfg, cfg)
    OmegaConf.set_struct(cfg, False)

    # Disable compile_block_mask
    if hasattr(cfg.model, 'model_config'):
        cfg.model.model_config.compile_block_mask = False

    cfg.model.inference_config.batch_size = 1
    cfg.model.inference_config.emb_type = "cge"  # Use contextual gene embeddings

    # 4. Set the checkpoint path
    cfg.model.checkpoint_path = CHECKPOINT_PATH
    cfg.model.inference_config.load_checkpoint = os.path.join(CHECKPOINT_PATH, "model_weights.pt")
    cfg.model.data_config.aux_vocab_path = os.path.join(CHECKPOINT_PATH, "vocabs")
    cfg.model.data_config.esm2_mappings_path = os.path.join(CHECKPOINT_PATH, "vocabs")
    # Use the default gene_col_name = "ensembl_id"

    # 5. Run inference
    logging.info("Starting inference...")
    adata_output = run_inference(cfg, data_files=[tmp_h5ad_path])

    # 6. Extract cge embeddings
    if "cge_embeddings" in adata_output.uns and len(adata_output.uns["cge_embeddings"]) > 0:
        embeddings = adata_output.uns["cge_embeddings"]
        gene_names = adata_output.uns["cge_gene_names"]
        cell_indices = adata_output.uns["cge_cell_indices"]

        # Create a DataFrame
        embedding_df = pd.DataFrame(
            embeddings,
            index=gene_names,
            columns=[f"dim_{i}" for i in range(embeddings.shape[1])]
        )

        # 7. Save results
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)
        save_file = os.path.join(OUTPUT_PATH, "gene_embeddings.csv")
        embedding_df.to_csv(save_file)
        logging.info(f"Completed. Results saved to: {save_file}")
        logging.info(f"Embedding dimensions: {embeddings.shape}")
    else:
        logging.error("cge_embeddings were not found")

    # Remove temporary files
    os.unlink(tmp_h5ad_path)


if __name__ == "__main__":
    main()