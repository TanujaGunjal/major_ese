"""
Data loading module for the Healthcare Claims Fraud Detection System.
Provides thread-safe connections to DuckDB and cached loading of Parquet/JSON artifacts.
"""

import duckdb
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, Any
from config.settings import (
    DUCKDB_PATH,
    CLAIMS_PARQUET,
    BENEFICIARIES_PARQUET,
    PROVIDERS_PARQUET,
    NFHS_SAMPLE_PARQUET,
    NFHS_SUMMARY_JSON,
    MODEL_METADATA_PATH,
    EVAL_PREDICTIONS_PARQUET
)


def get_duckdb_connection(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """
    Returns a DuckDB connection pointing to the indexed claims database.
    If the database file does not exist, connects to in-memory DuckDB reading Parquet directly.
    """
    if Path(DUCKDB_PATH).exists():
        return duckdb.connect(str(DUCKDB_PATH), read_only=read_only)
    else:
        # Fallback to direct parquet reading
        con = duckdb.connect()
        if CLAIMS_PARQUET.exists():
            con.execute(f"CREATE VIEW IF NOT EXISTS claims AS SELECT * FROM '{CLAIMS_PARQUET.as_posix()}'")
        if BENEFICIARIES_PARQUET.exists():
            con.execute(f"CREATE VIEW IF NOT EXISTS beneficiaries AS SELECT * FROM '{BENEFICIARIES_PARQUET.as_posix()}'")
        if PROVIDERS_PARQUET.exists():
            con.execute(f"CREATE VIEW IF NOT EXISTS providers AS SELECT * FROM '{PROVIDERS_PARQUET.as_posix()}'")
        return con


def load_model_metadata() -> Dict[str, Any]:
    """Loads model training metadata and cross-validation performance."""
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_providers_df() -> pd.DataFrame:
    """Loads the precomputed provider risk table."""
    if PROVIDERS_PARQUET.exists():
        return pd.read_parquet(PROVIDERS_PARQUET)
    return pd.DataFrame()


def load_nfhs_summary() -> Dict[str, Any]:
    """Loads NFHS-5 full benchmark screening metrics."""
    if NFHS_SUMMARY_JSON.exists():
        with open(NFHS_SUMMARY_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_nfhs_sample_df() -> pd.DataFrame:
    """Loads the screened NFHS-5 sample dataset."""
    if NFHS_SAMPLE_PARQUET.exists():
        return pd.read_parquet(NFHS_SAMPLE_PARQUET)
    return pd.DataFrame()


def load_eval_predictions() -> pd.DataFrame:
    """Loads the holdout test evaluation predictions for dynamic threshold analysis."""
    if EVAL_PREDICTIONS_PARQUET.exists():
        return pd.read_parquet(EVAL_PREDICTIONS_PARQUET)
    return pd.DataFrame()
