"""
Automated unit tests for investigation workspace persistence, screening indicators, and NFHS screening.
"""

import pytest
import pandas as pd
from uuid import uuid4
import src.investigation as investigation
from src.investigation import (
    create_or_get_case,
    update_case_status,
    add_case_note,
    list_cases,
    get_case_full_history,
    get_sqlite_conn,
)
from src.feature_engineering import screen_individual_claim
from src.preprocessing import (
    convert_currency,
    format_currency,
    format_date,
    format_usd,
    usd_to_inr,
)
from src.data_loader import load_nfhs_summary, load_nfhs_sample_df
from src.visualizations import format_inr_axis
import plotly.graph_objects as go


def test_investigation_case_lifecycle():
    """Verify complete case workflow: creation, note addition, status update, and history."""
    test_claim_id = f"TEST_CLAIM_{uuid4().hex}"

    # 1. Create case
    case = create_or_get_case(
        target_type="CLAIM",
        target_id=test_claim_id,
        priority="High",
        reason="Automated test case",
        summary="Unit test mock claim summary"
    )
    assert case["target_id"] == test_claim_id.upper()
    case_id = case["case_id"]

    # 2. Add note
    ok_note = add_case_note(case_id, author="Test Agent", note_text="Initial audit commenced.")
    assert ok_note is True

    # 3. Update status
    ok_status = update_case_status(case_id, new_status="Under Review", investigator="Lead Tester", note="Escalated for medical record review")
    assert ok_status is True

    # 4. Fetch full history
    history = get_case_full_history(case_id)
    assert history["case"]["status"] == "Under Review"
    assert len(history["notes"]) >= 2
    assert len(history["audit"]) >= 2

    conn = get_sqlite_conn()
    with conn:
        conn.execute("DELETE FROM case_notes WHERE case_id = ?", [case_id])
        conn.execute("DELETE FROM audit_log WHERE case_id = ?", [case_id])
        conn.execute("DELETE FROM cases WHERE case_id = ?", [case_id])
    conn.close()


def test_screening_indicator_outlier():
    """Verify reimbursement outlier screening logic."""
    high_claim = {
        "ClaimType": "Inpatient",
        "InscClaimAmtReimbursed": 45000.0,
        "ClaimDurationDays": 5,
        "LengthOfStayDays": 4,
        "ProcedureCount": 2,
        "ProviderRiskLevel": "High"
    }
    screen = screen_individual_claim(high_claim, historical_stats={"reimbursement_p99": 30000})
    assert screen["is_flagged"] is True
    flag_codes = [f["code"] for f in screen["flags"]]
    assert "OUTLIER_REIMBURSEMENT" in flag_codes
    assert "HIGH_RISK_PROVIDER" in flag_codes
    assert screen["flag_count"] == 1


def test_nfhs_sample_and_benchmark():
    """Verify that NFHS-5 sample records and benchmark summary are present and well-structured."""
    summary = load_nfhs_summary()
    assert summary["total_records_screened"] == 724115
    assert summary["total_flagged_records"] == 20514
    assert summary["clean_percentage"] == 97.17

    sample_df = load_nfhs_sample_df()
    assert not sample_df.empty
    assert "risk_category" in sample_df.columns
    assert "risk_score" in sample_df.columns
    assert "R1_diabetes_no_tx" in sample_df.columns


def test_formatting_helpers():
    """Verify source USD remains traceable and INR uses Indian grouping."""
    assert format_currency(12345.67) == "₹10,24,690.61"
    assert format_currency(0) == "₹0.00"
    assert format_usd(12345.67) == "USD 12,345.67"
    assert usd_to_inr(100) == 8300
    assert format_date("2009-08-15") == "2009-08-15"
    assert format_date(None) == "N/A"
    assert format_currency(None) == "Not available"
    assert format_usd(float("nan")) == "Not available"
    assert usd_to_inr(float("nan")) is None


def test_case_uniqueness_priority_updates_and_history_persist(tmp_path, monkeypatch):
    database_path = tmp_path / "investigation.db"
    monkeypatch.setattr(investigation, "SQLITE_DB_PATH", database_path)
    investigation.init_investigation_db()

    created = investigation.create_or_get_case(
        target_type="claim",
        target_id="test_claim_123",
        priority="High",
        reason="Review reimbursement",
        summary="Observed source values require follow-up.",
    )
    reopened = investigation.create_or_get_case(
        target_type="CLAIM",
        target_id="TEST_CLAIM_123",
        priority="Low",
        reason="A different reason should not overwrite the saved case",
    )

    assert created["was_created"] is True
    assert reopened["was_created"] is False
    assert reopened["case_id"] == created["case_id"]
    assert reopened["priority"] == "High"
    assert reopened["reason_for_review"] == "Review reimbursement"

    assert investigation.update_case_status(
        created["case_id"],
        "Under Review",
        investigator="Test Investigator",
        priority="Critical",
        reason="Continue reimbursement review",
        summary="Compare the source documentation with the recorded claim.",
        note="Request supporting records.",
    )
    assert investigation.add_case_note(
        created["case_id"], "Test Investigator", "Follow-up completed."
    )

    persisted = investigation.get_case_full_history(created["case_id"])
    assert persisted["case"]["status"] == "Under Review"
    assert persisted["case"]["priority"] == "Critical"
    assert persisted["case"]["reason_for_review"] == "Continue reimbursement review"
    assert [note["note_text"] for note in persisted["notes"]] == [
        "Request supporting records.",
        "Follow-up completed.",
    ]
    assert len(persisted["audit"]) >= 3

    with investigation.get_sqlite_conn() as connection:
        duplicate_count = connection.execute(
            "SELECT COUNT(*) FROM cases WHERE target_type = ? AND target_id = ?",
            ["CLAIM", "TEST_CLAIM_123"],
        ).fetchone()[0]
    assert duplicate_count == 1


def test_inr_chart_ticks_use_indian_grouping():
    figure = format_inr_axis(
        go.Figure(),
        pd.Series([0, 1234567.89]),
    )
    assert "₹12,34,567.89" in figure.layout.yaxis.ticktext


def test_currency_conversion_uses_configurable_rate(monkeypatch):
    import src.preprocessing as preprocessing

    monkeypatch.setattr(preprocessing, "USD_TO_INR_RATE", 90.0)
    assert convert_currency(2, "USD", "INR") == 180
    assert convert_currency(180, "INR", "USD") == 2
    assert format_currency(2) == "₹180.00"
