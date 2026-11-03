"""
Automated unit tests for data validation, schema integrity, and file availability.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.validation import (
    validate_raw_medicare_files,
    validate_processed_artifacts,
    validate_provider_features
)
from config.settings import (
    DUCKDB_PATH,
    CLAIMS_PARQUET,
    BENEFICIARIES_PARQUET,
    PROVIDERS_PARQUET,
    FEATURE_COLUMNS
)


def test_processed_artifacts_exist():
    """Verify that all processed Parquet files and DuckDB databases exist and are valid."""
    results = validate_processed_artifacts()
    for name, (valid, message) in results.items():
        assert valid is True, f"Artifact check failed for {name}: {message}"


def test_provider_feature_schema():
    """Verify that providers table contains all 10 engineered features."""
    assert PROVIDERS_PARQUET.exists(), "providers.parquet does not exist"
    df = pd.read_parquet(PROVIDERS_PARQUET)
    valid, errors = validate_provider_features(df)
    assert valid is True, f"Feature validation errors: {errors}"
    assert len(df) == 5410, f"Expected 5,410 providers, got {len(df)}"


def test_claims_parquet_schema():
    """Verify that claims table contains essential claim, clinical, and screening columns."""
    assert CLAIMS_PARQUET.exists(), "claims.parquet does not exist"
    df = pd.read_parquet(CLAIMS_PARQUET)
    assert len(df) == 558211, f"Expected 558,211 claims, got {len(df)}"

    expected_cols = [
        "ClaimID", "BeneID", "Provider", "ClaimType", "ClaimStartDt",
        "InscClaimAmtReimbursed", "ClaimDurationDays", "ScreeningFlagsCount",
        "ProviderFraudProbability", "ProviderRiskLevel", "AIReviewRecommendation"
    ]
    for c in expected_cols:
        assert c in df.columns, f"Missing expected column in claims.parquet: {c}"


def test_beneficiaries_parquet_schema():
    """Verify beneficiary table records and demographics."""
    assert BENEFICIARIES_PARQUET.exists(), "beneficiaries.parquet does not exist"
    df = pd.read_parquet(BENEFICIARIES_PARQUET)
    assert len(df) == 138556, f"Expected 138,556 beneficiaries, got {len(df)}"
    assert "BeneID" in df.columns
    assert "Age" in df.columns
    assert "Gender_Desc" in df.columns
