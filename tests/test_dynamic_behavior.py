"""
Unit tests verifying dynamic filtering, threshold recalculation, and referential join integrity.
"""

import pytest
import numpy as np
from src.claims_repository import get_executive_metrics
from src.data_loader import load_eval_predictions, get_duckdb_connection
from sklearn.metrics import confusion_matrix, precision_score, recall_score


def test_dynamic_executive_metrics():
    """Verify that executive metrics recalculate dynamically when filters are applied."""
    unfiltered = get_executive_metrics()
    assert unfiltered["TotalClaims"] == 558211

    # Filter by Inpatient only
    inp_only = get_executive_metrics(claim_type="Inpatient")
    assert inp_only["TotalClaims"] == 40474
    assert inp_only["TotalReimbursement"] < unfiltered["TotalReimbursement"]

    # Filter by High Risk provider level
    high_risk = get_executive_metrics(risk_level="High")
    assert high_risk["TotalClaims"] < unfiltered["TotalClaims"]
    assert high_risk["TotalClaims"] > 0


def test_dynamic_threshold_recalculation():
    """Verify that moving the decision threshold updates TP, FP, Precision, and Recall."""
    eval_df = load_eval_predictions()
    assert not eval_df.empty

    y_true = eval_df["ActualLabel"].values
    scores = eval_df["RF_Probability"].values

    # Threshold 0.20
    pred_02 = (scores >= 0.20).astype(int)
    rec_02 = recall_score(y_true, pred_02)
    prec_02 = precision_score(y_true, pred_02)

    # Threshold 0.50
    pred_05 = (scores >= 0.50).astype(int)
    rec_05 = recall_score(y_true, pred_05)
    prec_05 = precision_score(y_true, pred_05)

    # Recall must decrease or stay equal as threshold increases
    assert rec_02 >= rec_05
    # Precision must increase or stay equal as threshold increases
    assert prec_05 >= prec_02


def test_dynamic_screening_thresholds():
    """Verify that changing screening dollar thresholds changes flagged count in DuckDB."""
    con = get_duckdb_connection(read_only=True)

    # Low threshold ($15,000)
    c_low = con.execute("SELECT COUNT(*) FROM claims WHERE ClaimType = 'Inpatient' AND InscClaimAmtReimbursed >= 15000").fetchone()[0]
    # High threshold ($35,000)
    c_high = con.execute("SELECT COUNT(*) FROM claims WHERE ClaimType = 'Inpatient' AND InscClaimAmtReimbursed >= 35000").fetchone()[0]

    assert c_low > c_high
    assert c_high > 0


def test_referential_join_integrity():
    """Verify 100% foreign key match rate with zero orphaned claims."""
    con = get_duckdb_connection(read_only=True)
    res = con.execute("""
        SELECT 
            COUNT(*) - COUNT(p.Provider) AS UnmatchedProviders,
            COUNT(*) - COUNT(b.BeneID) AS UnmatchedBeneficiaries
        FROM claims c
        LEFT JOIN providers p ON c.Provider = p.Provider
        LEFT JOIN beneficiaries b ON c.BeneID = b.BeneID
    """).fetchone()

    assert res[0] == 0, "Found orphaned provider foreign keys"
    assert res[1] == 0, "Found orphaned beneficiary foreign keys"
