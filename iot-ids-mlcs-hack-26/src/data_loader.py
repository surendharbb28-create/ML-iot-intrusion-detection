"""
data_loader.py — Dataset Loading & Auto-Detection
===================================================
Loads CSV data from data/raw/, auto-detects columns,
handles missing/duplicate/infinite values, and identifies
the label column through configurable heuristics.
"""

import os
import pandas as pd
import numpy as np
import yaml
import logging

logger = logging.getLogger(__name__)

# ── Path helpers ─────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _load_config() -> dict:
    """Load the central YAML configuration."""
    cfg_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ── Core loader ──────────────────────────────────────────────

def load_csv(filepath: str, nrows: int | None = None) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV.
    nrows : int | None
        Optional row limit (useful for large datasets).

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in (".csv", ".txt", ".tsv"):
        raise ValueError(f"Unsupported file extension '{ext}'. Use .csv, .tsv, or .txt")

    sep = "\t" if ext in (".tsv", ".txt") else ","

    # First, try to sniff separator from the first line
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
            if "\t" in first_line and first_line.count("\t") > first_line.count(","):
                sep = "\t"
    except Exception:
        pass

    df = pd.read_csv(
        filepath,
        sep=sep,
        nrows=nrows,
        low_memory=False,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
    )

    logger.info("Loaded %d rows × %d columns from %s", len(df), len(df.columns), filepath)
    return df


def discover_datasets(directory: str | None = None) -> list[str]:
    """
    Scan a directory for CSV/TSV/TXT files.

    Parameters
    ----------
    directory : str | None
        Directory to scan.  Defaults to ``data/raw/``.

    Returns
    -------
    list[str]  — list of absolute paths.
    """
    if directory is None:
        directory = os.path.join(PROJECT_ROOT, "data", "raw")

    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
        return []

    files = []
    for fname in os.listdir(directory):
        if fname.lower().endswith((".csv", ".tsv", ".txt")):
            files.append(os.path.join(directory, fname))
    return sorted(files)


# ── Column inspection ────────────────────────────────────────

def inspect_columns(df: pd.DataFrame) -> dict:
    """
    Return a summary of the DataFrame's columns.

    Returns
    -------
    dict with keys: columns, dtypes, missing, duplicates, shape.
    """
    return {
        "columns": list(df.columns),
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "shape": df.shape,
        "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(df.select_dtypes(include=["object", "category"]).columns),
    }


# ── Label detection ──────────────────────────────────────────

def detect_label_column(df: pd.DataFrame, config: dict | None = None) -> str | None:
    """
    Attempt to auto-detect the label/target column.

    Strategy
    --------
    1. Use ``config['dataset']['label_column']`` if explicitly set.
    2. Walk through ``config['dataset']['label_candidates']`` in order.
    3. Fall back to any column whose name contains "label" or "class" (case-insensitive).

    Returns
    -------
    str | None — column name, or None if detection fails.
    """
    if config is None:
        config = _load_config()

    ds_cfg = config.get("dataset", {})

    # Explicit override
    explicit = ds_cfg.get("label_column")
    if explicit and explicit in df.columns:
        logger.info("Label column (explicit config): %s", explicit)
        return explicit

    # Candidate list
    for cand in ds_cfg.get("label_candidates", []):
        if cand in df.columns:
            logger.info("Label column (auto-detected): %s", cand)
            return cand

    # Fuzzy fallback
    for col in df.columns:
        if any(kw in col.lower() for kw in ("label", "class", "attack", "target")):
            logger.info("Label column (fuzzy match): %s", col)
            return col

    logger.warning("Could not auto-detect a label column.")
    return None


# ── Cleaning ─────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning:
    1. Strip whitespace from string columns.
    2. Drop fully-empty rows.
    3. Drop duplicate rows.
    4. Replace inf / -inf with NaN.
    5. Fill remaining NaN in numeric columns with 0.
    6. Fill remaining NaN in string columns with "unknown".
    """
    df = df.copy()

    # Strip whitespace on object columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Drop all-NaN rows
    df.dropna(how="all", inplace=True)

    # Drop duplicates
    n_dup = df.duplicated().sum()
    if n_dup:
        logger.info("Dropping %d duplicate rows", n_dup)
        df.drop_duplicates(inplace=True)

    # Inf → NaN → fill
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(0)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("unknown")

    df.reset_index(drop=True, inplace=True)
    return df


# ── Label mapping ────────────────────────────────────────────

def map_labels(
    series: pd.Series,
    config: dict | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Map raw labels → binary (NORMAL / ATTACK) and keep originals
    as ``attack_category``.

    Returns
    -------
    (binary_labels, original_labels)
    """
    if config is None:
        config = _load_config()

    lm = config.get("label_mapping", {})
    normal_kw = [k.lower() for k in lm.get("normal_keywords", ["normal", "benign"])]
    # Don't need attack_kw — everything not normal is ATTACK.

    raw = series.astype(str).str.strip().str.lower()
    binary = raw.apply(
        lambda x: "NORMAL" if any(kw in x for kw in normal_kw) else "ATTACK"
    )
    return binary, series.astype(str).str.strip()


# ── Full pipeline convenience ────────────────────────────────

def load_and_prepare(
    filepath: str,
    label_col: str | None = None,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    End-to-end: load → clean → detect label.

    Returns
    -------
    (cleaned DataFrame, label_column_name)

    Raises
    ------
    ValueError if label column cannot be determined.
    """
    config = _load_config()
    df = load_csv(filepath, nrows=nrows)
    df = clean_dataframe(df)

    if label_col is None:
        label_col = detect_label_column(df, config)
    if label_col is None:
        raise ValueError(
            "Unable to auto-detect label column. "
            "Please set 'dataset.label_column' in config/config.yaml or pass label_col explicitly."
        )
    if label_col not in df.columns:
        raise ValueError(f"Configured label column '{label_col}' not found in dataset.")

    return df, label_col
