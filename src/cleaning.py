"""
cleaning.py
Phase 2 - Data Cleaning Pipeline
CS 4/5630 Project 2

Cleans the UCI HIGGS CSV dataset.
Tools: pandas (required), re, pathlib, argparse

Usage:
    python cleaning.py --all
    python cleaning.py --file higgs
    python cleaning.py --file higgs --chunked

Use --chunked for large files (such as HIGGS.csv) to avoid
loading the entire file into memory at once. Processes in batches
of 100,000 rows, writing incrementally to the output CSV.
"""

import argparse
import re
import json
from pathlib import Path
import pandas as pd


# CONFIGURATION
# Default file paths — edit these if your directory structure differs

RAW_DIR = Path("data/higgs")
PROCESSED_DIR = Path("data/processed")
CHUNK_SIZE = 100_000  
HIGGS_FEATURE_COLS = [f"feature_{i}" for i in range(1, 29)]
HIGGS_COLUMNS = ["label"] + HIGGS_FEATURE_COLS
FILE_CONFIG = {
    "higgs": {
        "input":     RAW_DIR / "HIGGS.csv",
        "output":    PROCESSED_DIR / "higgs_cleaned.csv",
        "id_col":    None,
        "drop_rows": ["label"],
        "text_fill": [],
        "impute":    {f"feature_{i}": "median" for i in range(1, 29)},
    },
}


# FILE LOADING

def openfile(file_path: Path) -> pd.DataFrame:
    """Load a CSV or JSON file into a DataFrame."""
    p = Path(file_path)
    if not p.exists():
        print(f"ERROR: File not found: {p}")
        return None

    print(f"Loading {p.name}...")
    try:
        if p.suffix == ".csv":
            return pd.read_csv(p, low_memory=False)
        elif p.suffix == ".json":
            try:
                return pd.read_json(p, lines=True)
            except ValueError:
                rows = []
                with open(p, "r", encoding="utf-8-sig") as f:
                    for i, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"  Skipping bad JSON on line {i}")
                return pd.DataFrame(rows)
    except Exception as e:
        print(f"ERROR loading {p}: {e}")
        return None


def iter_json_chunks(file_path: Path, chunk_size: int):
    """Read a JSON lines file in chunks without loading it all into memory.

    Yields DataFrames of up to chunk_size rows at a time.
    This allows processing of files larger than available RAM.
    """
    p = Path(file_path)
    rows = []
    chunk_num = 0
    with open(p, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  Skipping bad JSON on line {i}")
                continue
            if len(rows) >= chunk_size:
                chunk_num += 1
                print(f"  Processing chunk {chunk_num} "
                      f"(rows {(chunk_num-1)*chunk_size+1:,} - {chunk_num*chunk_size:,})...")
                yield pd.DataFrame(rows)
                rows = []
    if rows:
        chunk_num += 1
        print(f"  Processing final chunk {chunk_num} ({len(rows):,} rows)...")
        yield pd.DataFrame(rows)


def iter_csv_chunks(file_path: Path, chunk_size: int):
    """Read a CSV file in chunks using pandas for large datasets."""
    for chunk_num, chunk in enumerate(pd.read_csv(file_path, header=None, chunksize=chunk_size), 1):
        print(f"  Processing chunk {chunk_num} ({len(chunk):,} rows)...")
        yield chunk


# STANDARDIZE COLUMN NAMES
# Required: consistent snake_case names

def to_snake_case(name: str) -> str:
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(name))
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_").lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to snake_case and resolve common aliases."""
    df = df.copy()
    df.columns = [to_snake_case(c) for c in df.columns]
    text_cols = [
        col for col in df.columns
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
    ]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().replace("nan", pd.NA)
    return df


def enforce_higgs_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all HIGGS feature columns are present and stored as float64."""
    df = df.copy()
    missing = [c for c in HIGGS_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected HIGGS feature columns: {missing}")

    coerced_total = 0
    for col in HIGGS_FEATURE_COLS:
        was_non_null = df[col].notna()
        converted = pd.to_numeric(df[col], errors="coerce").astype("float64")
        coerced_total += int((was_non_null & converted.isna()).sum())
        df[col] = converted

    for col in HIGGS_FEATURE_COLS:
        assert pd.api.types.is_float_dtype(df[col]), f"Expected float dtype for {col}"

    if coerced_total:
        print(f"  Coerced {coerced_total:,} non-numeric HIGGS feature values to NaN")
    return df


def flag_higgs_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag outliers in HIGGS features using IQR fences per feature."""
    df = df.copy()
    outlier_mask = pd.DataFrame(False, index=df.index, columns=HIGGS_FEATURE_COLS)

    for col in HIGGS_FEATURE_COLS:
        s = df[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)

        if pd.isna(q1) or pd.isna(q3):
            continue

        iqr = q3 - q1
        if iqr == 0:
            mean = s.mean()
            std = s.std()
            if pd.isna(std) or std == 0:
                continue
            outlier_mask[col] = (s - mean).abs() > (3 * std)
        else:
            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)
            outlier_mask[col] = (s < lower) | (s > upper)

    df["outlier_feature_count"] = outlier_mask.sum(axis=1).astype("int16")
    df["has_outlier"] = df["outlier_feature_count"] > 0
    flagged_rows = int(df["has_outlier"].sum())
    print(f"  Flagged outliers in {flagged_rows:,} rows")
    return df

# MISSING VALUES
# Required: documented drop and impute strategies

def handle_missing_values(df: pd.DataFrame, drop_rows: list = None,
                          impute: dict = None) -> pd.DataFrame:
    """Handle missing values with documented strategies.

    Strategy:
    - Drop rows missing critical ID or timestamp columns (unusable without them)
    - Drop columns with >= 80% missing values (too sparse to be useful)
    - Impute numeric fields with median where specified
    """
    df = df.copy()
    original_rows = len(df)

    if drop_rows:
        valid = [c for c in drop_rows if c in df.columns]
        if valid:
            df = df.dropna(subset=valid)
            print(f"  Dropped {original_rows - len(df):,} rows missing {valid}")

    sparse_cols = df.columns[df.isna().mean() >= 0.8].tolist()
    if sparse_cols:
        df = df.drop(columns=sparse_cols)
        print(f"  Dropped {len(sparse_cols)} sparse columns (>=80% missing): {sparse_cols}")

    if impute:
        for col, strategy in impute.items():
            if col in df.columns and df[col].isna().any():
                fill = df[col].median() if strategy == "median" else df[col].mean()
                df[col] = df[col].fillna(fill)
                print(f"  Imputed '{col}' with {strategy} ({fill:.2f})")

    print(f"  Rows: {original_rows:,} -> {len(df):,}")
    return df


# DEDUPLICATION
# Required: identify and remove rows representing the same real-world event

def deduplicate_records(df: pd.DataFrame, subset: list) -> pd.DataFrame:
   
    original = len(df)
    if not subset or not all(c in df.columns for c in subset):
        print(f"  Skipping dedup — key columns not found: {subset}")
        return df

    df = df.copy()
    df["_missing"] = df.isna().sum(axis=1)
    df = (df.sort_values(subset + ["_missing"])
            .drop_duplicates(subset=subset, keep="first")
            .drop(columns=["_missing"]))

    print(f"  Dedup on {subset}: {original:,} -> {len(df):,} "
          f"({original - len(df):,} removed)")
    return df


# CHUNK CLEANING  (for large files)

def _clean_chunk(df: pd.DataFrame, name: str, cfg: dict) -> pd.DataFrame:
    """Apply all cleaning steps to a single chunk. Used by clean_file_chunked."""
    if name == "higgs" and len(df.columns) == len(HIGGS_COLUMNS):
        df = df.copy()
        df.columns = HIGGS_COLUMNS

    df = standardize_columns(df)

    if name == "higgs":
        df = enforce_higgs_numeric_features(df)

    # Drop rows missing critical columns
    if cfg["drop_rows"]:
        valid = [c for c in cfg["drop_rows"] if c in df.columns]
        if valid:
            df = df.dropna(subset=valid)

    # Fill text columns
    for col in cfg["text_fill"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype("string")

    if name == "higgs":
        df = flag_higgs_outliers(df)

    # Within-chunk dedup only (cross-chunk dedup not feasible without full load)
    id_col = cfg["id_col"]
    if id_col and id_col in df.columns:
        df["_missing"] = df.isna().sum(axis=1)
        df = (df.sort_values([id_col, "_missing"])
                .drop_duplicates(subset=[id_col], keep="first")
                .drop(columns=["_missing"]))

    return df


def clean_file_chunked(name: str) -> None:

    if name not in FILE_CONFIG:
        print(f"ERROR: Unknown file '{name}'.")
        return

    cfg = FILE_CONFIG[name]
    p = cfg["input"]

    if not p.exists():
        print(f"ERROR: File not found: {p}")
        return

    print(f"\n{'='*50}\nCleaning (chunked): {name}\n{'='*50}")
    print(f"  Source: {p}  ({p.stat().st_size / 1e9:.2f} GB)")
    print(f"  Chunk size: {CHUNK_SIZE:,} rows")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg["output"]

    total_in = 0
    total_out = 0
    first_chunk = True

    chunk_iter = iter_csv_chunks(p, CHUNK_SIZE) if p.suffix == ".csv" else iter_json_chunks(p, CHUNK_SIZE)

    for chunk in chunk_iter:
        total_in += len(chunk)
        cleaned = _clean_chunk(chunk, name, cfg)
        total_out += len(cleaned)

        # Write header only on first chunk
        cleaned.to_csv(out, mode="w" if first_chunk else "a",
                       header=first_chunk, index=False, encoding="utf-8")
        first_chunk = False

    print(f"\n  Done. {total_in:,} rows read -> {total_out:,} rows saved.")
    print(f"  Saved -> {out}")


# MAIN CLEANING PIPELINE  (standard, full-file)

def clean_file(name: str) -> pd.DataFrame:
    """Run the full cleaning pipeline for a named dataset."""
    if name not in FILE_CONFIG:
        print(f"ERROR: Unknown file '{name}'. Options: {list(FILE_CONFIG.keys())}")
        return None

    cfg = FILE_CONFIG[name]
    print(f"\n{'='*50}\nCleaning: {name}\n{'='*50}")

    df = openfile(cfg["input"])
    if df is None:
        return None

    if name == "higgs" and len(df.columns) == len(HIGGS_COLUMNS):
        df.columns = HIGGS_COLUMNS

    # 1. Standardize column names
    df = standardize_columns(df)

    if name == "higgs":
        df = enforce_higgs_numeric_features(df)


    # 3. Handle missing values
    df = handle_missing_values(df, drop_rows=cfg["drop_rows"], impute=cfg["impute"])

    if name == "higgs":
        df = flag_higgs_outliers(df)

    # 4. Fill text columns so downstream NLP steps don't break
    for col in cfg["text_fill"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype("string")

    # 5. Deduplicate
    dedup_subset = [cfg["id_col"]] if cfg["id_col"] else []
    df = deduplicate_records(df, subset=dedup_subset)

    # 6. Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg["output"], index=False, encoding="utf-8")
    print(f"  Saved -> {cfg['output']}  ({len(df):,} rows, {df.shape[1]} cols)")

    return df


def run_all(chunked_names: list = None) -> dict:
    """Clean every raw file. Optionally specify which files to run chunked."""
    chunked_names = chunked_names or []
    results = {}
    for name in FILE_CONFIG:
        if name in chunked_names:
            clean_file_chunked(name)
            results[name] = None  # chunked mode doesn't return a df
        else:
            results[name] = clean_file(name)
    print("\n========== All Cleaning Complete ==========")
    for name, df in results.items():
        status = f"{len(df):,} rows" if df is not None else "done (chunked)"
        print(f"  {name}: {status}")
    return results


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleaning Pipeline - CS 4/5630 Project 2")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",  action="store_true", help="Clean all raw files")
    group.add_argument("--file", choices=list(FILE_CONFIG.keys()),
                       help="Clean a single file by name")
    parser.add_argument("--chunked", action="store_true",
                        help="Process in chunks (recommended for HIGGS.csv)")
    args = parser.parse_args()

    if args.all:
        # Auto-use chunked mode for HIGGS when running --all
        run_all(chunked_names=["higgs"])
    elif args.chunked:
        clean_file_chunked(args.file)
    else:
        clean_file(args.file)
