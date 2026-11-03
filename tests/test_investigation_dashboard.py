from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.claims_repository import (
    get_claim_by_id,
    get_claims_by_beneficiary,
    get_claims_by_provider,
    get_claim_screening_stats,
    get_screening_claims,
    get_screening_rule_audit,
    search_claims,
)
from src.data_loader import load_eval_predictions, load_model_metadata, load_providers_df
from src.feature_engineering import screen_individual_claim
from src.investigation import list_cases
import src.investigation as investigation
from src.preprocessing import extract_codes_list


def test_beneficiary_history_is_chronological_and_includes_diagnosis_fields():
    history = get_claims_by_beneficiary("BENE11001")

    assert history["ClaimStartDt"].is_monotonic_increasing
    assert history["ClaimID"].tolist() == ["CLM46614", "CLM66048", "CLM68358"]
    assert "ClmDiagnosisCode_10" in history.columns
    assert history["InscClaimAmtReimbursed"].sum() == 36000


def test_provider_claim_search_is_exact_and_paginates():
    first_page = get_claims_by_provider("PRV51459", limit=30, offset=0)
    second_page = get_claims_by_provider("PRV51459", limit=30, offset=30)

    assert len(first_page) == 30
    assert len(second_page) == 30
    assert not set(first_page["ClaimID"]) & set(second_page["ClaimID"])
    assert "PRV51459" not in first_page["ClaimID"].tolist()


def test_provider_threshold_metadata_and_random_forest_score_are_available():
    providers = load_providers_df().set_index("Provider")
    provider = providers.loc["PRV55912"]
    thresholds = load_model_metadata()["thresholds"]

    assert provider["FraudProbability"] == provider["RF_FraudProbability"]
    assert thresholds == {"fraud_threshold": 0.3, "high_risk_threshold": 0.7}
    assert provider["TotalClaims"] == (
        provider["InpatientClaims"] + provider["OutpatientClaims"]
    )


def test_stored_screening_results_separate_unique_claims_from_rule_triggers():
    results = get_screening_claims(limit=100)

    assert results["total"] == 34602
    assert results["rule_trigger_occurrences"] == 35635
    assert len(results["records"]) == 100
    assert results["records"]["RuleCode"].notna().all()
    assert results["records"]["ClaimID"].nunique() <= len(results["records"])


def test_screening_filters_by_investigation_status():
    cases = list_cases(status="New", target_type="CLAIM")
    matching = get_screening_claims(case_ids=[case["target_id"] for case in cases], limit=20)
    assert matching["total"] == len(cases) == 4
    assert set(matching["records"]["ClaimID"]) == {case["target_id"] for case in cases}

    outside_workspace = get_screening_claims(
        exclude_case_ids=[case["target_id"] for case in cases],
        limit=20,
    )
    assert outside_workspace["total"] == 34598


def test_beneficiary_page_renders_actual_search_results_and_pattern_review():
    page_path = Path(__file__).resolve().parents[1] / "pages" / "3_Beneficiary_History.py"
    app = AppTest.from_file(str(page_path), default_timeout=45)
    app.session_state["selected_bene_id"] = "BENE118316"
    app.run()

    assert not app.exception
    assert any(item.value == "29" for item in app.metric if item.label == "Historical claims")
    assert any("Patterns to review" in item.value for item in app.subheader)
    assert any(button.label == "Open selected claim in Claim Investigation" for button in app.button)


def test_provider_page_renders_score_thresholds_shap_and_claims():
    page_path = Path(__file__).resolve().parents[1] / "pages" / "4_Provider_Risk_Analysis.py"
    app = AppTest.from_file(str(page_path), default_timeout=60)
    app.session_state["selected_provider_id"] = "PRV55912"
    app.run()

    assert not app.exception
    assert any(item.label == "Provider model score" and item.value == "90.0%" for item in app.metric)
    assert any(item.label == "Configured provider review threshold" and item.value == "30%" for item in app.metric)
    assert any("Main contributing features (SHAP)" in item.value for item in app.subheader)
    assert any("Provider's actual claims" in item.value for item in app.subheader)
    assert any(button.label == "Open selected claim in Claim Investigation" for button in app.button)


def test_screening_page_renders_distinct_claim_and_trigger_counts():
    page_path = Path(__file__).resolve().parents[1] / "pages" / "5_Suspicious_Claim_Screening.py"
    app = AppTest.from_file(str(page_path), default_timeout=45).run()

    assert not app.exception
    assert any(
        item.label == "Unique claims with selected rule(s)" and item.value == "34,602"
        for item in app.metric
    )
    assert any(
        item.label == "Recorded rule triggers" and item.value == "35,635"
        for item in app.metric
    )
    assert any(button.label == "Investigate selected claim" for button in app.button)
    assert "Investigation status" in [widget.label for widget in app.selectbox]


def test_workspace_page_creates_and_reopens_a_real_claim_case(tmp_path, monkeypatch):
    monkeypatch.setattr(investigation, "SQLITE_DB_PATH", tmp_path / "workspace.db")
    investigation.init_investigation_db()
    page_path = Path(__file__).resolve().parents[1] / "pages" / "8_Investigation_Workspace.py"
    app = AppTest.from_file(str(page_path), default_timeout=45)
    app.session_state["selected_claim_id"] = "CLM46614"
    app.session_state["workspace_target_type"] = "CLAIM"
    app.session_state["workspace_target_id"] = "CLM46614"
    app.run()

    assert not app.exception
    assert any(item.value == "CLM46614" for item in app.text_input)
    reason = next(item for item in app.text_input if item.label == "Reason for investigation")
    reason.set_value("Review actual Medicare claim")
    observations = next(
        item for item in app.text_area if item.label == "Supporting observations"
    )
    observations.set_value("Review the recorded screening evidence.")
    next(item for item in app.button if item.label == "Create or open case").click()
    app.run()

    assert not app.exception
    cases = investigation.list_cases(target_type="CLAIM", search_query="CLM46614")
    assert len(cases) == 1
    case = cases[0]
    assert case["target_id"] == "CLM46614"
    assert case["reason_for_review"] == "Review actual Medicare claim"
    history = investigation.get_case_full_history(case["case_id"])
    assert history["case"]["summary_facts"] == "Review the recorded screening evidence."
    assert any(item.label == "Save case updates" for item in app.button)
    assert any(item.label == "Add dated note" for item in app.button)
    assert any(item.label == "Export case summary (JSON)" for item in app.get("download_button"))

    next(item for item in app.selectbox if item.label == "Review status").set_value("Under Review")
    [item for item in app.selectbox if item.label == "Priority"][-1].set_value("Critical")
    [item for item in app.text_input if item.label == "Reason for investigation"][-1].set_value(
        "Review actual claim documentation"
    )
    [item for item in app.text_area if item.label == "Supporting observations"][-1].set_value(
        "The source record has been selected for document review."
    )
    next(
        item for item in app.text_area
        if item.label == "Dated note for this update (optional)"
    ).set_value("Requested the source documentation.")
    next(item for item in app.button if item.label == "Save case updates").click()
    app.run()
    assert not app.exception

    [item for item in app.text_area if item.label == "Investigator note"][-1].set_value(
        "Documentation request sent."
    )
    next(item for item in app.button if item.label == "Add dated note").click()
    app.run()

    persisted = investigation.get_case_full_history(case["case_id"])
    assert persisted["case"]["status"] == "Under Review"
    assert persisted["case"]["priority"] == "Critical"
    assert persisted["case"]["reason_for_review"] == "Review actual claim documentation"
    assert len(persisted["notes"]) == 2
    app.run()
    assert not app.exception


def test_workspace_creates_case_for_selected_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(investigation, "SQLITE_DB_PATH", tmp_path / "provider_workspace.db")
    investigation.init_investigation_db()
    page_path = Path(__file__).resolve().parents[1] / "pages" / "8_Investigation_Workspace.py"
    app = AppTest.from_file(str(page_path), default_timeout=45)
    app.session_state["workspace_target_type"] = "PROVIDER"
    app.session_state["workspace_target_id"] = "PRV55912"
    app.session_state["selected_provider_id"] = "PRV55912"
    app.run()

    assert not app.exception
    assert any(item.value == "PRV55912" for item in app.text_input)
    reason = next(item for item in app.text_input if item.label == "Reason for investigation")
    reason.set_value("Review provider-level risk context")
    next(item for item in app.button if item.label == "Create or open case").click()
    app.run()

    provider_cases = investigation.list_cases(target_type="PROVIDER")
    assert len(provider_cases) == 1
    assert provider_cases[0]["target_id"] == "PRV55912"
    assert provider_cases[0]["reason_for_review"] == "Review provider-level risk context"
    assert not app.exception


def test_claim_investigation_page_searches_real_claims_and_reports_alerts():
    page_path = Path(__file__).resolve().parents[1] / "pages" / "2_Claim_Investigation.py"
    app = AppTest.from_file(str(page_path), default_timeout=30).run()
    assert not app.exception
    assert any("No screening rule triggered" in item.value for item in app.success)
    assert any(
        "not a fraud probability for this individual claim" in item.value
        for item in app.caption
    )
    provider_score = next(item for item in app.metric if item.label == "Provider model score")
    assert provider_score.value == "90.0%"

    app.text_input[0].set_value("CLM439269")
    app.button[0].click().run()

    assert not app.exception
    assert any("Risk Assessment" in item.value for item in app.subheader)
    assert any("Screening alert recorded" in item.value for item in app.warning)


def test_claim_investigation_page_shows_not_found_for_unknown_claim():
    page_path = Path(__file__).resolve().parents[1] / "pages" / "2_Claim_Investigation.py"
    app = AppTest.from_file(str(page_path), default_timeout=30).run()

    app.text_input[0].set_value("NOT_A_REAL_CLAIM")
    app.button[0].click().run()

    assert not app.exception
    assert any("No claim with ID NOT_A_REAL_CLAIM was found." == item.value for item in app.error)


def test_claim_date_search_returns_real_matching_record():
    results = search_claims(
        claim_id="CLM46614",
        date_from="2009-04-12",
        date_to="2009-04-12",
        limit=10,
    )
    assert results["total"] == 1
    assert results["records"].iloc[0]["ClaimID"] == "CLM46614"


def test_stored_screening_flags_take_precedence_over_generic_cutoffs():
    claim = get_claim_by_id("CLM46614")
    assert claim is not None
    evidence = screen_individual_claim(claim)
    assert not claim["Flag_ReimbursementOutlier"]
    assert evidence["flag_count"] == 0
    assert evidence["is_flagged"] is False
    assert any(flag.get("context_only") for flag in evidence["flags"])


def test_stored_reimbursement_alert_is_explained_at_current_p99_equality():
    claim = get_claim_by_id("CLM439269")
    assert claim is not None
    assert claim["Flag_ReimbursementOutlier"] is True

    stats = get_claim_screening_stats(claim)
    evidence = screen_individual_claim(claim, historical_stats=stats)
    alert = next(
        flag for flag in evidence["flags"]
        if flag["code"] == "OUTLIER_REIMBURSEMENT"
    )

    assert claim["InscClaimAmtReimbursed"] == stats["reimbursement_p99"] == 3300
    assert "equal to" in alert["description"]
    assert "original stored threshold is not available" in alert["description"]
    assert "USD 3,300.00" in alert["baseline"]


def test_flagged_claim_search_uses_stored_screening_alerts():
    results = search_claims(claim_id="CLM439269", flagged_only=True, limit=10)

    assert results["total"] == 1
    assert results["records"].iloc[0]["ClaimID"] == "CLM439269"
    assert results["records"].iloc[0]["StoredScreeningAlert"]


def test_claim_codes_keep_their_source_code_format():
    claim = get_claim_by_id("CLM66048")
    assert claim is not None

    assert extract_codes_list(pd.Series(claim), "ClmProcedureCode", 6) == ["7092"]
    assert extract_codes_list(pd.Series({"ClmDiagnosisCode_1": "042"}), "ClmDiagnosisCode", 1) == ["042"]


def test_screening_queue_uses_recomputed_rule_flags():
    results = get_screening_claims(rule="Reimbursement outlier", limit=20)
    assert results["total"] > 0
    assert not results["records"].empty
    assert results["records"]["Flag_ReimbursementOutlier"].all()
    assert (
        results["records"]["InscClaimAmtReimbursed"]
        >= results["records"]["ClaimTypeP99"]
    ).all()
    assert results["rule_trigger_occurrences"] >= results["total"]


def test_stored_screening_flags_reconciliation():
    audit = get_screening_rule_audit()
    outlier = audit.loc[audit["Rule"] == "Reimbursement outlier"].iloc[0]
    assert int(outlier["RecomputedClaims"]) < int(outlier["StoredClaims"])
    assert int(outlier["Disagreements"]) > 0


def test_holdout_labels_match_actual_provider_records():
    evaluation = load_eval_predictions()
    providers = load_providers_df().set_index("Provider")
    assert len(evaluation) == 1082
    expected_labels = providers.loc[evaluation["Provider"], "FraudLabel"].to_numpy()
    assert (evaluation["ActualLabel"].to_numpy() == expected_labels).all()
    for column in ["RF_Probability", "LR_Probability", "XGB_Probability"]:
        assert evaluation[column].between(0, 1).all()
