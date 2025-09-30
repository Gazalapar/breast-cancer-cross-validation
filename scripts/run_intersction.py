# scripts/run_intersection.py
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROC_DIR / "harmonized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

datasets = {
    "GSE22820": PROC_DIR / "GSE22820_gene_level.csv",
    "GSE59246": PROC_DIR / "GSE59246_gene_level.csv",
    "GSE42568": PROC_DIR / "GSE42568_gene_level.csv",
    "GSE70947": PROC_DIR / "GSE70947_gene_level.csv",
}

# ---- Load all datasets ----
dfs = {}
print("Loading datasets:")
for name, path in datasets.items():
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Check that mapping step has produced these files.")
    print(f"  - {name}: {path}")
    df = pd.read_csv(path, index_col=0)
    dfs[name] = df
    print(f"    shape: {df.shape}  (samples x genes = rows x columns)")

# ---- Find common genes ----
gene_sets = [set(df.columns) for df in dfs.values()]
common_genes = set.intersection(*gene_sets)
common_genes_sorted = sorted(common_genes)
print(f"\nCommon genes across all datasets: {len(common_genes_sorted)}")

# Print first 10 common genes for quick check
n_show = 10
print(f"First {n_show} common genes (alphabetical):")
for g in common_genes_sorted[:n_show]:
    print("  ", g)

# ---- Save harmonized versions ----
print("\nSaving harmonized datasets:")
for name, df in dfs.items():
    # Keep columns in consistent sorted order
    df_h = df.loc[:, common_genes_sorted]
    out_path = OUT_DIR / f"{name}_harmonized.csv"
    df_h.to_csv(out_path)
    print(f"  - {name}: {df_h.shape} -> {out_path}")

# ---- Save intersection gene list ----
gene_list_path = OUT_DIR / "common_genes.txt"
with open(gene_list_path, "w") as f:
    for g in common_genes_sorted:
        f.write(g + "\n")
print(f"\nSaved common gene list: {gene_list_path}")
