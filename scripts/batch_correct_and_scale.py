#!/usr/bin/env python3
"""
batch_correct_and_scale.py

Combine harmonized dataset CSVs, run ComBat batch correction (scanpy),
then perform per-gene z-score scaling and save results (combined + per-dataset).
This script forces sample indices to strings and prefixes them with dataset names
to avoid index-matching errors (fixes AttributeError: 'int' object has no attribute 'startswith').
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

# optional plotting libs are not required for the correction itself
try:
    import scanpy as sc
except Exception as e:
    print("ERROR: scanpy is required. Install via `pip install scanpy` or conda.")
    raise

# --- Config / paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARMONIZED_DIR = PROJECT_ROOT / "data" / "processed" / "harmonized"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
PER_DS_DIR = OUT_DIR / "per_dataset"
COMBINED_RAW = OUT_DIR / "combined_raw_harmonized.csv"
COMBINED_BATCH = OUT_DIR / "combined_batch_corrected_combat.csv"
COMBINED_ZSCORE = OUT_DIR / "combined_batch_corrected_zscore.csv"

print("SCRIPT CWD:", Path.cwd())
print("SCRIPT FILE:", Path(__file__).resolve())
print("PROJECT_ROOT:", PROJECT_ROOT)
print("Looking for harmonized files in:", HARMONIZED_DIR)

# --- find harmonized CSVs ---
if not HARMONIZED_DIR.exists():
    print(f"No harmonized folder found at {HARMONIZED_DIR}")
    sys.exit(1)

csv_files = sorted([p.name for p in HARMONIZED_DIR.glob("*.csv")])
if len(csv_files) == 0:
    print("No harmonized CSV files found in", HARMONIZED_DIR)
    sys.exit(1)

print("Found harmonized CSV files:", csv_files)

# --- load datasets ---
datasets = []  # list of tuples (dataset_name, df)
sample_to_batch = {}  # map sample_id -> dataset_name (after prefixing)

# load common genes list if exists (to enforce order), otherwise infer intersection later
common_genes_path = HARMONIZED_DIR / "common_genes.txt"
common_genes = None
if common_genes_path.exists():
    with open(common_genes_path, "r", encoding="utf-8") as fh:
        common_genes = [line.strip() for line in fh if line.strip()]
    print(f"Loaded {len(common_genes)} common genes from {common_genes_path}")

for fname in csv_files:
    ds_name = Path(fname).stem.replace("_harmonized", "")
    fpath = HARMONIZED_DIR / fname
    print(f"Loading {fname} -> dataset label = '{ds_name}'")
    # assume first column is index (samples)
    df = pd.read_csv(fpath, index_col=0)
    # ensure column names (genes) are strings
    df.columns = df.columns.astype(str)

    # If a common gene list is available, subset & reorder columns to that list
    if common_genes is not None:
        # keep only genes present both places (preserve order from common_genes)
        genes_available = [g for g in common_genes if g in df.columns]
        if len(genes_available) != len(common_genes):
            missing = set(common_genes) - set(df.columns)
            print(f"Warning: dataset {ds_name} missing {len(missing)} common genes (they'll be dropped).")
        df = df.loc[:, genes_available]
    # Guarantee sample index are strings and prefix them for uniqueness
    df.index = df.index.astype(str)
    # If the index values are not already prefixed with ds_name + "__", add it
    if not all(idx.startswith(ds_name + "__") for idx in df.index):
        df.index = [f"{ds_name}__{idx}" for idx in df.index]

    # record sample->batch
    for sample_id in df.index:
        sample_to_batch[sample_id] = ds_name

    datasets.append((ds_name, df))

# Ensure all datasets have the same genes. If we didn't have common_genes initially, compute intersection.
if common_genes is None:
    gene_sets = [set(df.columns) for (_, df) in datasets]
    common = set.intersection(*gene_sets)
    common_genes = sorted(common)
    print(f"No common_genes.txt present — computed intersection size: {len(common_genes)}")
    # subset each dataset
    for i, (ds_name, df) in enumerate(datasets):
        datasets[i] = (ds_name, df.loc[:, common_genes])

# Concatenate datasets (samples x genes)
combined = pd.concat([df for (_, df) in datasets], axis=0)
print("Using", len(common_genes), "common genes")
print("Combined shape (samples x genes):", combined.shape)

OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_DS_DIR.mkdir(parents=True, exist_ok=True)

# Save combined raw
combined.to_csv(COMBINED_RAW)
print("Saved combined raw CSV:", COMBINED_RAW)

# --- Prepare AnnData and run ComBat ---
# scanpy expects anndata with obs (samples) x var (genes)
print("Preparing AnnData for ComBat...")
adata = sc.AnnData(combined)  # adata.obs_names are sample IDs, adata.var_names are genes

# create batch labels
# First try to extract dataset name from sample prefix 'DSNAME__sample'
def extract_batch_from_sample(s):
    if "__" in s:
        return s.split("__", 1)[0]
    return sample_to_batch.get(s, "unknown")

adata.obs["batch"] = [extract_batch_from_sample(s) for s in adata.obs_names]

# run ComBat
print("Running ComBat batch correction (scanpy.pp.combat)...")
try:
    sc.pp.combat(adata, key="batch")
except Exception as e:
    print("ComBat failed:", e)
    raise

# convert back to DataFrame
corrected = pd.DataFrame(adata.X, index=adata.obs_names.astype(str), columns=adata.var_names.astype(str))
corrected.to_csv(COMBINED_BATCH)
print("Saved batch-corrected file:", COMBINED_BATCH)

# --- gene-wise z-score scaling (per gene across samples) ---
# z = (x - mean)/std  (use ddof=0 for population std)
print("Performing z-score scaling (per gene).")
gene_means = corrected.mean(axis=0)
gene_std = corrected.std(axis=0, ddof=0).replace(0, np.nan)  # avoid divide by zero
zscore = (corrected - gene_means) / gene_std
# where gene_std was zero, zscore will be NaN — fill with 0
zscore = zscore.fillna(0.0)
zscore.to_csv(COMBINED_ZSCORE)
print("Saved z-score scaled file:", COMBINED_ZSCORE)

# --- save per-dataset subsets from corrected and scaled combined results ---
print("Saving per-dataset batch-corrected & scaled subsets...")
# Ensure indices are strings
corrected.index = corrected.index.astype(str)
zscore.index = zscore.index.astype(str)

for ds_name, df in datasets:
    # ensure df index is strings and prefixed (we already prefixed earlier)
    df.index = df.index.astype(str)

    # select rows in combined corrected that belong to this ds (match by prefix)
    mask = [idx for idx in corrected.index if idx.startswith(ds_name + "__")]
    if len(mask) == 0:
        # fallback: try intersection by index membership
        mask = [idx for idx in corrected.index if idx in df.index]

    if len(mask) == 0:
        print(f"Warning: couldn't match samples for dataset {ds_name} by index. Skipping per-dataset file.")
        continue

    corrected_subset = corrected.loc[mask]
    zscore_subset = zscore.loc[mask]

    corrected_subset.to_csv(PER_DS_DIR / f"{ds_name}_batch_corrected.csv")
    zscore_subset.to_csv(PER_DS_DIR / f"{ds_name}_batch_corrected_zscore.csv")
    print(f"Saved {ds_name} subset with {len(mask)} samples.")

print("All done.")
