# src/mapping.py
import pandas as pd
import mygene
from io import StringIO
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("biothings_client").setLevel(logging.WARNING)
logging.getLogger("mygene").setLevel(logging.WARNING)


# setup simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

mg = mygene.MyGeneInfo()

# PROJECT_ROOT is two levels up from this file (project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_MYG_CACHE_PATH = CACHE_DIR / "mygene_cache.json"

# in-memory caches (will be loaded from/saved to disk)
_MYGENE_POS_CACHE: Dict[str, str] = {}
_MYGENE_NEG_CACHE: set = set()

def _load_cache():
    global _MYGENE_POS_CACHE, _MYGENE_NEG_CACHE
    if _MYG_CACHE_PATH.exists():
        try:
            with open(_MYG_CACHE_PATH, "r", encoding="utf-8") as fh:
                j = json.load(fh)
            pos = j.get("positive", {})
            neg = set(j.get("negative", []))
            # ensure types
            if isinstance(pos, dict):
                _MYGENE_POS_CACHE = pos
            if isinstance(neg, (list, set)):
                _MYGENE_NEG_CACHE = set(neg)
            logging.info(f"Loaded mygene cache: {len(_MYGENE_POS_CACHE)} positives, {len(_MYGENE_NEG_CACHE)} negatives")
        except Exception as e:
            logging.warning(f"Failed to read mygene cache { _MYG_CACHE_PATH }: {e}")
            _MYGENE_POS_CACHE = {}
            _MYGENE_NEG_CACHE = set()
    else:
        # start empty
        _MYGENE_POS_CACHE = {}
        _MYGENE_NEG_CACHE = set()

def _save_cache():
    try:
        obj = {
            "positive": _MYGENE_POS_CACHE,
            "negative": list(_MYGENE_NEG_CACHE),
        }
        with open(_MYG_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2)
        logging.info(f"Saved mygene cache: {len(_MYGENE_POS_CACHE)} positives, {len(_MYGENE_NEG_CACHE)} negatives")
    except Exception as e:
        logging.warning(f"Failed to save mygene cache to {_MYG_CACHE_PATH}: {e}")

# load cache at import time
_load_cache()

# ---------------- Affymetrix (GPL570) mapping ----------------
def read_gpl(gpl_path: str) -> pd.DataFrame:
    """
    Read GPL-style annotation (text/tsv) and return DataFrame with probe->gene mapping.
    The function skips leading comment lines starting with '#'.
    Returns DataFrame with columns ['ID', 'GeneSymbol'].
    """
    with open(gpl_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # find first non-comment header line
    header_idx = 0
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            header_idx = i
            break
    txt = "".join(lines[header_idx:])
    gpl = pd.read_csv(StringIO(txt), sep="\t", dtype=str, low_memory=False)
    # standardize column names
    gpl.columns = [c.strip() for c in gpl.columns]

    # Try to find best gene symbol column
    symbol_col: Optional[str] = None
    # common names
    for c in ["Gene Symbol", "Gene Symbol ", "Gene symbol", "GENE_SYMBOL", "Symbol", "Gene"]:
        if c in gpl.columns:
            symbol_col = c
            break
    if symbol_col is None:
        # fallback: pick first column containing 'symbol' or 'gene' in the name
        for c in gpl.columns:
            if "symbol" in c.lower() or "gene" in c.lower() or "gene_title" in c.lower():
                symbol_col = c
                break
    if symbol_col is None:
        # last resort
        if "Gene Title" in gpl.columns:
            symbol_col = "Gene Title"

    if symbol_col is None:
        raise ValueError("Could not find a Gene Symbol column in GPL file. "
                         "Available columns: " + ", ".join(gpl.columns[:30]))

    # unify: split multi-symbol fields like "BRCA1 /// BRCA2" and take first symbol
    gpl["GeneSymbol"] = gpl[symbol_col].astype(str).str.split("///").str[0].str.strip()

    # ensure ID column exists
    id_col: Optional[str] = None
    for c in ["ID", "ProbeID", "Probe Set ID", "Probe Set ID ", "ID_REF", "Probe Set Name"]:
        if c in gpl.columns:
            id_col = c
            break
    if id_col is None:
        # fallback to the first column
        id_col = gpl.columns[0]

    gpl = gpl[[id_col, "GeneSymbol"]].rename(columns={id_col: "ID"})
    gpl = gpl[gpl["GeneSymbol"].notnull() & (gpl["GeneSymbol"] != "")]
    gpl = gpl.drop_duplicates(subset=["ID"])
    logging.info(f"read_gpl: loaded {len(gpl)} probe->gene mappings from {gpl_path}")
    return gpl.reset_index(drop=True)

def _collapse_mean(df: pd.DataFrame) -> pd.DataFrame:
    # transpose -> groupby -> mean -> transpose back
    return df.T.groupby(level=0).mean().T

def _collapse_maxvar(df: pd.DataFrame) -> pd.DataFrame:
    """
    For genes that have multiple probe columns, pick the probe (column) with maximum variance
    across samples and use that as the gene representative.
    """
    unique_genes = list(pd.Index(df.columns).unique())
    cols = []
    col_names = []
    for gene in unique_genes:
        # boolean mask of columns equal to this gene
        mask = [c == gene for c in df.columns]
        if sum(mask) == 1:
            sel = df.iloc[:, [i for i, m in enumerate(mask) if m][0]]
            cols.append(sel)
            col_names.append(gene)
        else:
            sub = df.loc[:, mask]  # preserves column order
            variances = sub.var(axis=0, ddof=0)
            if variances.isnull().all():
                sel = sub.mean(axis=1)
            else:
                sel_col = variances.idxmax()
                sel = sub[sel_col]
            cols.append(sel)
            col_names.append(gene)
    result = pd.concat(cols, axis=1)
    result.columns = col_names
    return result

def map_affy(expr: pd.DataFrame, gpl: pd.DataFrame, collapse: str = "mean") -> pd.DataFrame:
    """
    Map Affymetrix probe IDs to gene symbols using a GPL DataFrame.
    expr: DataFrame with index = samples, columns = probe IDs.
    collapse: 'mean' or 'maxvar' (for multiple probes -> same gene).
    Returns DataFrame with columns = gene symbols (samples x genes).
    """
    mapping = dict(zip(gpl["ID"], gpl["GeneSymbol"]))
    # find probes present in expr
    present = [c for c in expr.columns if c in mapping]
    missing = [c for c in expr.columns if c not in mapping]
    logging.info(f"map_affy: found {len(present)} probes matching GPL, {len(missing)} probes unmatched")
    if len(present) == 0:
        raise ValueError("No probe IDs in expression match the GPL annotation IDs.")
    df = expr[present].rename(columns=mapping)
    if collapse == "mean":
        return _collapse_mean(df)
    elif collapse == "maxvar":
        return _collapse_maxvar(df)
    else:
        raise ValueError("collapse must be 'mean' or 'maxvar'")

# ---------------- RefSeq / Ensembl mapping (mygene) ----------------

def _clean_id_list(ids: List[str]) -> List[str]:
    """
    Filter out obvious non-mappable tokens and duplicates while preserving order.
    Skips empty strings, the token 'type', array probe IDs that usually start with 'A_' (platform-specific),
    and several other patterns commonly found in GEO files.
    """
    out = []
    seen = set()
    for i in ids:
        if not isinstance(i, str):
            continue
        s = i.strip()
        if s == "":
            continue
        low = s.lower()
        # skip known junk tokens or platform-specific probe names that are not transcript IDs
        if low == "type" or s.startswith("A_") or s.startswith("ERCC") or s.startswith("lincRNA:") or s.startswith("TCONS_") or s.startswith("THC") or (s.startswith("X") and len(s) < 6):
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out

def map_ids_mygene(ids: List[str]) -> Dict[str, str]:
    """
    Map a list of transcript/probe IDs (RefSeq NM_, ENST, etc.) to gene symbols using mygene.
    Returns a mapping dict {query_id: symbol}. Uses on-disk cache to avoid re-querying MyGene for known IDs.
    """
    # use caches loaded at import time
    global _MYGENE_POS_CACHE, _MYGENE_NEG_CACHE
    ids_clean = _clean_id_list(ids)

    # start from cached positives
    mapping: Dict[str, str] = {k: v for k, v in _MYGENE_POS_CACHE.items() if k in ids_clean}

    # determine which IDs still need queries
    to_query = [i for i in ids_clean if (i not in mapping) and (i not in _MYGENE_NEG_CACHE)]

    batch = 500
    logging.info(f"map_ids_mygene: mapping {len(ids_clean)} ids (using cache; need to query {len(to_query)})")
    for i in range(0, len(to_query), batch):
        chunk = to_query[i : i + batch]
        try:
            out = mg.querymany(chunk,
                               scopes=["refseq", "ensembl.transcript", "ensembl.gene", "uniprot", "symbol"],
                               fields="symbol",
                               species="human",
                               as_dataframe=False)
        except Exception as e:
            logging.warning(f"mygene querymany failed on {len(chunk)} ids: {e}. Falling back to single queries.")
            out = []
            for q in chunk:
                try:
                    r = mg.query(q,
                                 scopes="refseq,ensembl.transcript,ensembl.gene,uniprot,symbol",
                                 fields="symbol",
                                 species="human")
                    # mygene.query returns dict for single; normalize to list-like entry
                    if isinstance(r, dict):
                        out.append(r)
                except Exception:
                    out.append({"query": q, "notfound": True})

        # Process results
        for d in out:
            q = d.get("query")
            if not q:
                continue
            if d.get("notfound"):
                _MYGENE_NEG_CACHE.add(q)
                continue
            sym = d.get("symbol")
            if sym:
                mapping[q] = sym
                _MYGENE_POS_CACHE[q] = sym
            else:
                _MYGENE_NEG_CACHE.add(q)

        # save cache progressively to disk so partial progress persists
        _save_cache()

    logging.info(f"map_ids_mygene: total mapped in this call: {len(mapping)} (positives cached now {len(_MYGENE_POS_CACHE)}); negatives cached: {len(_MYGENE_NEG_CACHE)}")
    return mapping

def map_transcript(expr: pd.DataFrame, collapse: str = "mean") -> pd.DataFrame:
    """
    Map transcript/probe-like column names (NM_, ENST, etc.) to gene symbols with mygene.
    expr: DataFrame with samples x transcript columns.
    collapse: 'mean' or 'maxvar'
    Returns DataFrame with samples x genes (columns are gene symbols).
    """
    ids = list(expr.columns)
    mapping = map_ids_mygene(ids)
    mapped_cols = [c for c in expr.columns if c in mapping]
    logging.info(f"map_transcript: {len(mapped_cols)} / {len(ids)} columns mapped to gene symbols")
    if len(mapped_cols) == 0:
        raise ValueError("No transcript IDs were mapped to gene symbols by mygene.")
    df = expr[mapped_cols].rename(columns=mapping)

    if collapse == "mean":
        return _collapse_mean(df)
    elif collapse == "maxvar":
        return _collapse_maxvar(df)
    else:
        raise ValueError("collapse must be 'mean' or 'maxvar'")
