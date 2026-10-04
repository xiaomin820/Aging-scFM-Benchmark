#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# batch_scvi_gpu_label_task1_fixed.py

import os
import scanpy as sc
import scvi
import pandas as pd
import numpy as np
import torch

# =========================
# Parameter settings
# =========================
data_folder = "."  # Directory path
output_folder = "./scVI_latent_output"
n_latent = 30
max_epochs = 150
early_stopping = True

os.makedirs(output_folder, exist_ok=True)

# =========================
# Check GPU availability
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device in use:", device)

# =========================
# Process all h5ad files in the directory
# =========================
h5ad_files = [f for f in os.listdir(data_folder) if f.endswith(".h5ad")]

for idx, file_name in enumerate(h5ad_files, 1):
    print(f"\n==== [{idx}/{len(h5ad_files)}] Processing file: {file_name} ====")
    file_path = os.path.join(data_folder, file_name)
    
    # 1. Read data
    adata = sc.read_h5ad(file_path)
    
    # 2. QC
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # 3. Identify and discard cells with noninteger counts
    has_decimal_per_cell = np.array([(row.data % 1 != 0).any() for row in adata.X])
    num_decimal_cells = has_decimal_per_cell.sum()
    total_cells = adata.n_obs
    fraction_decimal = num_decimal_cells / total_cells

    print(f"Total cells: {total_cells}")
    print(f"Cells with noninteger counts: {num_decimal_cells}")
    print(f"Fraction of cells with noninteger counts: {fraction_decimal:.4f}")

    adata = adata[~has_decimal_per_cell, :].copy()
    print(f"Cells remaining after removing noninteger counts: {adata.n_obs}")

    if adata.n_obs == 0:
        print("No usable cells; skipping this file")
        continue

    # 4. Configure AnnData for scVI
    scvi.model.SCVI.setup_anndata(adata)

    # 5. Initialize and train the model
    model = scvi.model.SCVI(adata, n_latent=n_latent)
    model.train(max_epochs=max_epochs, early_stopping=early_stopping)

    # 6. Obtain latent embeddings
    latent = model.get_latent_representation()
    latent_df = pd.DataFrame(latent, index=adata.obs_names)

    # Append label_task1 if present without filtering cells
    if "label_task1" in adata.obs.columns:
        latent_df["label_task1"] = adata.obs["label_task1"].values

    # 7. Save CSV
    out_csv = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_scVI_latent.csv")
    latent_df.to_csv(out_csv)
    print(f"Saved: {out_csv}")