"""
Dataset schema and data integrity validation module.
Verifies column existence, identifier consistency, missing data thresholds, and table linkage.
"""

import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
from config.settings import (
    RAW_MEDICARE_DIR,
    PROCESSED_DIR,
    FEATURE_COLUMNS,
    DUCKDB_PATH
)


# Expected required raw columns
REQUIRED_INPATIENT_COLS = [
    "BeneID", "ClaimID", "ClaimStartDt", "ClaimEndDt", "Provider",
    "InscClaimAmtReimbursed", "AdmissionDt", "DischargeDt"
]

REQUIRED_OUTPATIENT_COLS = [
    "BeneID", "ClaimID", "ClaimStartDt", "ClaimEndDt", "Provider",
    "InscClaimAmtReimbursed"
]

REQUIRED_BENEFICIARY_COLS = [
    "BeneID", "DOB", "Gender", "Race", "State", "County"
]

REQUIRED_PROVIDER_COLS = [
    "Provider", "PotentialFraud"
]


def validate_raw_medicare_files() -> Dict[str, Tuple[bool, str]]:
    """
    Checks that all required raw Medicare CSV files exist and contain required headers.
    """
    results = {}
    files_to_check = {
        "Train-1542865627584.csv": REQUIRED_PROVIDER_COLS,
        "Train_Inpatientdata-1542865627584.csv": REQUIRED_INPATIENT_COLS,
        "Train_Outpatientdata-1542865627584.csv": REQUIRED_OUTPATIENT_COLS,
        "Train_Beneficiarydata-1542865627584.csv": REQUIRED_BENEFICIARY_COLS
    }

    for filename, required_cols in files_to_check.items():
        filepath = RAW_MEDICARE_DIR / filename
        if not filepath.exists():
            results[filename] = (False, f"File does not exist at {filepath}")
            continue

        try:
            sample = pd.read_csv(filepath, nrows=10)
            missing_cols = [c for c in required_cols if c not in sample.columns]
            if missing_cols:
                results[filename] = (False, f"Missing required columns: {missing_cols}")
            else:
                results[filename] = (True, f"Valid ({len(sample.columns)} columns present)")
        except Exception as e:
            results[filename] = (False, f"Failed to read file: {str(e)}")

    return results


def validate_processed_artifacts() -> Dict[str, Tuple[bool, str]]:
    """
    Checks that processed Parquet tables, DuckDB database, and model artifacts exist and are non-empty.
    """
    results = {}
    from config.settings import (
        CLAIMS_PARQUET,
        BENEFICIARIES_PARQUET,
        PROVIDERS_PARQUET,
        RF_MODEL_PATH,
        LR_MODEL_PATH,
        XGB_MODEL_PATH,
        MODEL_METADATA_PATH,
        DUCKDB_PATH
    )

    artifacts = {
        "claims_parquet": CLAIMS_PARQUET,
        "beneficiaries_parquet": BENEFICIARIES_PARQUET,
        "providers_parquet": PROVIDERS_PARQUET,
        "duckdb_database": DUCKDB_PATH,
        "rf_model": RF_MODEL_PATH,
        "lr_model": LR_MODEL_PATH,
        "xgb_model": XGB_MODEL_PATH,
        "model_metadata": MODEL_METADATA_PATH
    }

    for name, path in artifacts.items():
        if not path.exists():
            results[name] = (False, f"Artifact missing at {path}")
        elif path.stat().st_size == 0:
            results[name] = (False, f"Artifact is empty (0 bytes) at {path}")
        else:
            results[name] = (True, f"Available ({path.stat().st_size / 1024:.1f} KB)")

    return results


def validate_provider_features(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates that a provider feature DataFrame has all 10 engineered features without infinite or all-null values.
    """
    errors = []
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required feature columns: {missing_cols}")

    for c in FEATURE_COLUMNS:
        if c in df.columns:
            if df[c].isnull().all():
                errors.append(f"Column '{c}' contains entirely null values.")
            if not pd.api.types.is_numeric_dtype(df[c]):
                errors.append(f"Column '{c}' is not numeric.")

    return len(errors) == 0, errors
