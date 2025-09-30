# scripts/run_mapping.py
import pandas as pd
from pathlib import Path
from src.mapping import read_gpl, map_affy, map_transcript

# ---- Resolve project paths (robust to cwd) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

# auto-detect a GPL-like file
gpl_candidates = [p for p in RAW_DIR.iterdir() if p.is_file() and ('GPL' in p.name.upper() or 'PL570' in p.name.upper())]
if not gpl_candidates:
    raise FileNotFoundError(f"No GPL file found in {RAW_DIR}. Place the platform annotation there.")
GPL_PATH = gpl_candidates[0]
print("Using GPL file:", GPL_PATH)

datasets = {
    "GSE22820": RAW_DIR / "Breast_GSE22820.csv",
    "GSE59246": RAW_DIR / "Breast_GSE59246.csv",
    "GSE42568": RAW_DIR / "Breast_GSE42568.csv",
    "GSE70947": RAW_DIR / "Breast_GSE70947.csv",
}

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load GPL for Affymetrix ----
gpl = read_gpl(str(GPL_PATH))

# ---- Process each dataset ----
for name, path in datasets.items():
    print(f"\n--- {name} ---")
    if not path.exists():
        print(f"  WARNING: dataset file not found: {path}")
        continue
    df = pd.read_csv(path, index_col=0)
    print("Shape:", df.shape)

    # Detect type by checking column names for common Affy suffix
    cols = list(df.columns[:200])
    try:
        if any(isinstance(c, str) and c.endswith("_at") for c in cols):  # Affymetrix probes
            print("Detected Affymetrix probes → mapping with GPL")
            df_gene = map_affy(df, gpl, collapse="mean")
        else:
            print("Detected RefSeq/Ensembl IDs → mapping with mygene")
            df_gene = map_transcript(df, collapse="mean")
    except Exception as e:
        print("Mapping error for", name, ":", str(e))
        continue

    print("Mapped to gene-level:", df_gene.shape)
    out_path = OUT_DIR / f"{name}_gene_level.csv"
    df_gene.to_csv(out_path)
    print("Saved:", out_path)
