"""
Automated unit tests for DuckDB claims repository queries, filters, and pagination.
"""

import pytest
from src.claims_repository import (
    get_claim_by_id,
    get_beneficiary_profile,
    get_provider_profile,
    get_claims_by_beneficiary,
    get_claims_by_provider,
    search_claims,
    get_executive_metrics
)


def test_get_claim_by_id_existing():
    """Verify lookup of a known valid claim."""
    claim = get_claim_by_id("CLM46614")
    assert claim is not None, "Failed to retrieve existing claim CLM46614"
    assert claim["ClaimID"] == "CLM46614"
    assert claim["BeneID"] == "BENE11001"
    assert claim["Provider"] == "PRV55912"
    assert claim["ClaimType"] == "Inpatient"
    assert claim["InscClaimAmtReimbursed"] == 26000.0


def test_get_claim_by_id_nonexistent():
    """Verify behavior when requesting non-existent claim ID."""
    claim = get_claim_by_id("NONEXISTENT_CLAIM_999999")
    assert claim is None


def test_get_beneficiary_profile():
    """Verify lookup of a known beneficiary."""
    profile = get_beneficiary_profile("BENE11001")
    assert profile is not None, "Failed to retrieve beneficiary BENE11001"
    assert profile["BeneID"] == "BENE11001"
    assert "Age" in profile
    assert profile["Gender"] in [1, 2]


def test_get_provider_profile():
    """Verify lookup of a known provider and check 10 feature values."""
    provider = get_provider_profile("PRV51001")
    assert provider is not None, "Failed to retrieve provider PRV51001"
    assert provider["Provider"] == "PRV51001"
    assert "TotalClaims" in provider
    assert "TotalReimbursement" in provider
    assert "FraudProbability" in provider
    assert provider["RiskLevel"] in ["Low", "Medium", "High"]


def test_get_claims_by_beneficiary():
    """Verify retrieving chronological claims for a beneficiary."""
    claims = get_claims_by_beneficiary("BENE11001")
    assert not claims.empty, "BENE11001 should have claims"
    assert len(claims) >= 3


def test_claims_search_filters():
    """Verify search filtering by type and reimbursement range."""
    res = search_claims(claim_type="Inpatient", min_reimb=5000.0, limit=10)
    assert res["total"] > 0
    assert len(res["records"]) <= 10
    for _, row in res["records"].iterrows():
        assert row["ClaimType"] == "Inpatient"
        assert row["InscClaimAmtReimbursed"] >= 5000.0


def test_executive_metrics():
    """Verify executive summary metrics calculations."""
    m = get_executive_metrics()
    assert m["TotalClaims"] == 558211
    assert m["UniqueProviders"] == 5410
    assert m["UniqueBeneficiaries"] == 138556
    assert m["TotalReimbursement"] > 500000000.0
