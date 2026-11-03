"""Claim lookup, source-backed screening evidence, and investigation actions."""

from datetime import datetime
import math

import pandas as pd
import streamlit as st

from src.claims_repository import (
    get_claim_by_id,
    get_claim_screening_stats,
    get_claims_by_beneficiary,
    get_provider_profile,
    search_claims,
)
from src.data_loader import get_duckdb_connection
from src.feature_engineering import screen_individual_claim
from src.investigation import create_or_get_case, list_cases
from src.preprocessing import (
    convert_currency,
    currency_configuration,
    currency_context_text,
    extract_codes_list,
    format_currency,
    format_date,
    format_money_trace,
    format_usd,
    usd_to_inr,
)
from src.styles import apply_custom_styles, render_header
from src.visualizations import plot_beneficiary_claim_timeline


def _is_available(value):
    if value is None:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _display_value(value):
    return str(value) if _is_available(value) and str(value).strip() else "Not available"


def _money_value(value):
    return format_money_trace(value) if _is_available(value) else "Not available"


def _reset_advanced_search():
    defaults = {
        "advanced_bene_id": "",
        "advanced_provider_id": "",
        "advanced_claim_type": "All",
        "advanced_risk_level": "All",
        "advanced_flagged_only": False,
        "advanced_status": "All",
        "advanced_min_reimbursement": 0.0,
        "advanced_max_reimbursement": convert_currency(125000.0, "USD", "INR"),
        "advanced_sort": "InscClaimAmtReimbursed",
        "advanced_page": 1,
    }
    st.session_state.update(defaults)


apply_custom_styles()
render_header(
    title="Claim Investigation",
    subtitle="Find a claim, review its recorded screening evidence, and choose an investigation action."
)

if "selected_claim_id" not in st.session_state:
    st.session_state.selected_claim_id = "CLM46614"

with st.form("claim_id_search", clear_on_submit=False):
    search_col, button_col = st.columns([5, 1])
    with search_col:
        claim_search_input = st.text_input(
            "Search by Claim ID",
            value=st.session_state.get("selected_claim_id") or "",
            placeholder="Enter a Claim ID, for example CLM46614",
            help="Enter the complete Claim ID from the source records.",
        )
    with button_col:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        search_submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

if search_submitted:
    requested_claim_id = claim_search_input.strip().upper()
    st.session_state.selected_claim_id = requested_claim_id or None
    if not requested_claim_id:
        st.session_state.claim_search_error = "Enter a Claim ID to search."
    elif get_claim_by_id(requested_claim_id) is None:
        st.session_state.claim_search_error = f"No claim with ID {requested_claim_id} was found."
    else:
        st.session_state.selected_claim_id = requested_claim_id
        st.session_state.pop("claim_search_error", None)

if st.session_state.get("claim_search_error"):
    st.error(st.session_state.claim_search_error)

current_claim_id = st.session_state.get("selected_claim_id")
claim_record = get_claim_by_id(current_claim_id) if current_claim_id else None

if not claim_record:
    st.info("Search for a Claim ID to view its record.")
else:
    bene_id = claim_record.get("BeneID")
    provider_id = claim_record.get("Provider")
    provider_record = get_provider_profile(provider_id) if _is_available(provider_id) else None
    screening_stats = get_claim_screening_stats(claim_record)
    screening_info = screen_individual_claim(claim_record, historical_stats=screening_stats)
    claim_flags = [flag for flag in screening_info["flags"] if not flag.get("context_only")]

    st.subheader("Claim details")
    detail_rows = [
        ("Claim ID", claim_record.get("ClaimID")),
        ("Beneficiary ID", bene_id),
        ("Provider ID", provider_id),
        ("Claim type", claim_record.get("ClaimType")),
        ("Claim start date", format_date(claim_record.get("ClaimStartDt")) if _is_available(claim_record.get("ClaimStartDt")) else None),
        ("Claim end date", format_date(claim_record.get("ClaimEndDt")) if _is_available(claim_record.get("ClaimEndDt")) else None),
        ("Reimbursement", _money_value(claim_record.get("InscClaimAmtReimbursed"))),
        ("Deductible", _money_value(claim_record.get("DeductibleAmtPaid"))),
        ("Claim duration", f"{claim_record['ClaimDurationDays']} day(s)" if _is_available(claim_record.get("ClaimDurationDays")) else None),
        ("Diagnosis codes", ", ".join(extract_codes_list(pd.Series(claim_record), "ClmDiagnosisCode", 10)) or (
            "None recorded" if _is_available(claim_record.get("DiagnosisCount")) and claim_record.get("DiagnosisCount") == 0 else None
        )),
        ("Admitting diagnosis code", claim_record.get("ClmAdmitDiagnosisCode")),
        ("Procedure codes", ", ".join(extract_codes_list(pd.Series(claim_record), "ClmProcedureCode", 6)) or (
            "None recorded" if _is_available(claim_record.get("ProcedureCount")) and claim_record.get("ProcedureCount") == 0 else None
        )),
    ]
    st.dataframe(
        pd.DataFrame(
            [{"Detail": label, "Value": _display_value(value)} for label, value in detail_rows]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Risk Assessment")
    flag_columns = [
        "Flag_ReimbursementOutlier",
        "Flag_DuplicateBilling",
        "Flag_DurationAnomaly",
        "Flag_ComplexProcedure",
        "Flag_HighFreqBeneficiary",
    ]
    known_flag_columns = [
        column for column in flag_columns
        if column in claim_record and _is_available(claim_record[column])
    ]
    if claim_flags:
        st.warning(f"Screening alert recorded: {len(claim_flags)} claim-level rule(s) are flagged in the source database.")
    elif len(known_flag_columns) != len(flag_columns):
        st.info("Insufficient information: one or more source screening flags are unavailable.")
    else:
        st.success("No screening rule triggered for this claim in the source database.")
    st.caption(
        "Screening indicators are review leads from the source data. They are not a finding of fraud."
    )

    st.subheader("Why is this claim being reviewed?")
    if claim_flags:
        reason_rows = [
            {
                "Rule": flag["title"],
                "Observed value": _display_value(flag.get("observed")),
                "Comparison baseline": _display_value(flag.get("baseline")),
                "Why it may need review": _display_value(flag.get("description")),
            }
            for flag in claim_flags
        ]
        st.dataframe(pd.DataFrame(reason_rows), use_container_width=True, hide_index=True)
    elif len(known_flag_columns) == len(flag_columns):
        st.write("No claim-level screening rule is recorded as triggered.")
    else:
        st.write("The source database does not contain enough screening information to explain a rule finding.")

    st.subheader("Provider-level model context")
    provider_score = None
    provider_risk = None
    if provider_record:
        raw_score = provider_record.get("FraudProbability")
        try:
            parsed_score = float(raw_score) if _is_available(raw_score) else None
        except (TypeError, ValueError):
            parsed_score = None
        if parsed_score is not None and math.isfinite(parsed_score) and 0 <= parsed_score <= 1:
            provider_score = parsed_score
        provider_risk = provider_record.get("RiskLevel")

    score_col, tier_col = st.columns(2)
    score_col.metric(
        "Provider model score",
        f"{provider_score:.1%}" if provider_score is not None else "Not available",
    )
    tier_col.metric("Provider risk tier", _display_value(provider_risk))
    st.caption(
        "This score describes the provider-level model assessment. It is not a fraud probability for this individual claim."
    )

    if _is_available(bene_id):
        with st.expander("Beneficiary claim history"):
            bene_history = get_claims_by_beneficiary(bene_id)
            if bene_history.empty:
                st.info("No beneficiary claim history is available.")
            else:
                st.metric("Claims in available history", f"{len(bene_history):,}")
                timeline = bene_history.copy()
                timeline["ClaimStartDt"] = pd.to_datetime(timeline["ClaimStartDt"], errors="coerce")
                st.plotly_chart(plot_beneficiary_claim_timeline(timeline), use_container_width=True)
                history_display = pd.DataFrame({
                    "Claim ID": bene_history["ClaimID"],
                    "Claim date": bene_history["ClaimStartDt"].map(format_date),
                    "Claim type": bene_history["ClaimType"],
                    "Provider ID": bene_history["Provider"],
                    "Reimbursement (INR)": bene_history["InscClaimAmtReimbursed"].map(format_currency),
                    "Source reimbursement (USD)": bene_history["InscClaimAmtReimbursed"].map(format_usd),
                    "Duration (days)": bene_history["ClaimDurationDays"],
                })
                st.dataframe(history_display, use_container_width=True, hide_index=True)

    with st.expander("Raw source record"):
        st.json(claim_record)

    st.subheader("Investigation actions")
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Open beneficiary history", disabled=not _is_available(bene_id), use_container_width=True):
            st.session_state.selected_bene_id = bene_id
            st.switch_page("pages/3_Beneficiary_History.py")
    with action_col2:
        if st.button("Open provider risk page", disabled=not _is_available(provider_id), use_container_width=True):
            st.session_state.selected_provider_id = provider_id
            st.switch_page("pages/4_Provider_Risk_Analysis.py")

    workspace_case = next(
        (
            item for item in list_cases(target_type="CLAIM")
            if item["target_id"] == claim_record["ClaimID"]
        ),
        None,
    )
    case_col, report_col = st.columns(2)
    with case_col:
        if st.button("Add claim to Investigation Workspace", type="primary", use_container_width=True):
            workspace_case = create_or_get_case(
                target_type="CLAIM",
                target_id=claim_record["ClaimID"],
                priority="High" if provider_risk == "High" else "Medium",
                reason="Recorded claim screening review",
                summary=(
                    f"Claim {claim_record['ClaimID']}; beneficiary {_display_value(bene_id)}; "
                    f"provider {_display_value(provider_id)}; "
                    f"{len(claim_flags)} recorded claim screening alert(s)."
                ),
            )
            st.session_state.active_case_id = workspace_case["case_id"]
            st.session_state.workspace_target_type = "CLAIM"
            st.session_state.workspace_target_id = claim_record["ClaimID"]
            st.success(f"Case active in Workspace: {workspace_case['case_id']}")

    report_lines = [
        "# Healthcare Claim Investigation",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        currency_context_text(),
        "",
        "## Claim details",
    ]
    report_lines.extend(f"- {label}: {_display_value(value)}" for label, value in detail_rows)
    report_lines.extend(["", "## Risk Assessment"])
    if claim_flags:
        report_lines.extend(
            f"- Screening alert recorded: {flag['title']}; observed: {_display_value(flag.get('observed'))}; "
            f"baseline: {_display_value(flag.get('baseline'))}; "
            f"explanation: {_display_value(flag.get('description'))}"
            for flag in claim_flags
        )
    elif len(known_flag_columns) == len(flag_columns):
        report_lines.append("- No screening rule triggered in the source database.")
    else:
        report_lines.append("- Insufficient information to determine screening status.")
    report_lines.extend([
        "",
        "## Provider-level model context",
        f"- Provider risk tier: {_display_value(provider_risk)}",
        f"- Provider model score: {f'{provider_score:.1%}' if provider_score is not None else 'Not available'}",
        "- This is a provider-level score, not a claim-level fraud probability.",
    ])
    with report_col:
        if workspace_case and st.button(
            "Open this case in Investigation Workspace",
            use_container_width=True,
        ):
            st.session_state.active_case_id = workspace_case["case_id"]
            st.session_state.workspace_target_type = "CLAIM"
            st.session_state.workspace_target_id = claim_record["ClaimID"]
            st.switch_page("pages/8_Investigation_Workspace.py")
        st.download_button(
            "Download claim investigation report",
            data="\n".join(report_lines),
            file_name=f"claim_investigation_{claim_record['ClaimID']}.txt",
            mime="text/plain",
            use_container_width=True,
        )

with st.expander("Advanced claim search and page export"):
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        advanced_bene_id = st.text_input(
            "Beneficiary ID contains", key="advanced_bene_id", value=st.session_state.get("advanced_bene_id", "")
        )
        advanced_provider_id = st.text_input(
            "Provider ID contains", key="advanced_provider_id", value=st.session_state.get("advanced_provider_id", "")
        )
    with filter_col2:
        advanced_claim_type = st.selectbox(
            "Claim setting", ["All", "Inpatient", "Outpatient"],
            index=["All", "Inpatient", "Outpatient"].index(st.session_state.get("advanced_claim_type", "All")),
            key="advanced_claim_type",
        )
        advanced_risk_level = st.selectbox(
            "Provider risk level", ["All", "High", "Medium", "Low"],
            index=["All", "High", "Medium", "Low"].index(st.session_state.get("advanced_risk_level", "All")),
            key="advanced_risk_level",
        )
    with filter_col3:
        advanced_flagged = st.checkbox(
            "Only show claims with stored screening alerts",
            value=st.session_state.get("advanced_flagged_only", False),
            key="advanced_flagged_only",
        )
        advanced_status = st.selectbox(
            "Investigation status",
            ["All", "No case", "New", "Under Review", "Escalated", "Reviewed/Closed",
             "Closed - False Positive", "Closed - Suspected Fraud"],
            key="advanced_status",
        )

    amount_col1, amount_col2, amount_col3 = st.columns(3)
    with amount_col1:
        advanced_min_reimbursement = st.number_input(
            "Minimum reimbursement (INR)", min_value=0.0,
            value=st.session_state.get("advanced_min_reimbursement", 0.0),
            step=5000.0, key="advanced_min_reimbursement",
        )
    with amount_col2:
        advanced_max_reimbursement = st.number_input(
            "Maximum reimbursement (INR)", min_value=0.0,
            value=st.session_state.get(
                "advanced_max_reimbursement",
                convert_currency(125000.0, "USD", "INR"),
            ),
            step=25000.0, key="advanced_max_reimbursement",
        )
    with amount_col3:
        advanced_sort = st.selectbox(
            "Sort results by",
            ["InscClaimAmtReimbursed", "ClaimStartDt", "ScreeningFlagsCount", "ClaimDurationDays"],
            key="advanced_sort",
        )

    date_con = get_duckdb_connection(read_only=True)
    try:
        date_min, date_max = date_con.execute(
            "SELECT MIN(ClaimStartDt), MAX(ClaimStartDt) FROM claims"
        ).fetchone()
    finally:
        date_con.close()
    date_min = pd.to_datetime(date_min, errors="coerce")
    date_max = pd.to_datetime(date_max, errors="coerce")
    date_range = None
    if pd.notna(date_min) and pd.notna(date_max):
        date_range = st.date_input(
            "Claim start date range",
            value=(date_min.date(), date_max.date()),
            min_value=date_min.date(),
            max_value=date_max.date(),
            key="advanced_date_range",
        )

    if st.button("Reset advanced search", on_click=_reset_advanced_search):
        st.rerun()

    date_from = date_range[0].isoformat() if isinstance(date_range, (tuple, list)) and date_range else None
    date_to = (
        date_range[1].isoformat()
        if isinstance(date_range, (tuple, list)) and len(date_range) > 1
        else None
    )
    search_signature = (
        advanced_bene_id, advanced_provider_id, advanced_claim_type, advanced_risk_level,
        advanced_flagged, advanced_status, advanced_min_reimbursement,
        advanced_max_reimbursement, advanced_sort, date_from, date_to,
    )
    if st.session_state.get("advanced_search_signature") != search_signature:
        st.session_state.advanced_page = 1
        st.session_state.advanced_search_signature = search_signature
    page_size = 15
    advanced_page = st.session_state.get("advanced_page", 1)

    case_filter_args = {}
    if advanced_status not in ("All", "No case"):
        case_filter_args["claim_ids"] = [
            case["target_id"]
            for case in list_cases(status=advanced_status, target_type="CLAIM")
        ]
    elif advanced_status == "No case":
        case_filter_args["exclude_claim_ids"] = [
            case["target_id"] for case in list_cases(target_type="CLAIM")
        ]

    results = search_claims(
        claim_id=claim_search_input.strip() or None,
        bene_id=advanced_bene_id.strip() or None,
        provider_id=advanced_provider_id.strip() or None,
        claim_type=advanced_claim_type,
        min_reimb=(
            convert_currency(advanced_min_reimbursement, "INR", "USD")
            if advanced_min_reimbursement > 0
            else None
        ),
        max_reimb=(
            convert_currency(advanced_max_reimbursement, "INR", "USD")
            if advanced_max_reimbursement < convert_currency(125000.0, "USD", "INR")
            else None
        ),
        risk_level=advanced_risk_level,
        flagged_only=advanced_flagged,
        date_from=date_from,
        date_to=date_to,
        sort_by=advanced_sort,
        limit=page_size,
        offset=(advanced_page - 1) * page_size,
        **case_filter_args,
    )
    if advanced_page > results["pages"]:
        st.session_state.advanced_page = 1
        st.rerun()

    st.caption(f"{results['total']:,} matching claims")
    search_records = results["records"]
    if search_records.empty:
        st.info("No claims match these filters.")
    else:
        search_display = pd.DataFrame({
            "Claim ID": search_records["ClaimID"],
            "Beneficiary ID": search_records["BeneID"],
            "Provider ID": search_records["Provider"],
            "Claim type": search_records["ClaimType"],
            "Start date": search_records["ClaimStartDt"].map(format_date),
            "Reimbursement (INR)": search_records["InscClaimAmtReimbursed"].map(format_currency),
            "Source reimbursement (USD)": search_records["InscClaimAmtReimbursed"].map(format_usd),
            "Deductible (INR)": search_records["DeductibleAmtPaid"].map(format_currency),
            "Source deductible (USD)": search_records["DeductibleAmtPaid"].map(format_usd),
            "Duration (days)": search_records["ClaimDurationDays"],
            "Screening flags": search_records["ScreeningFlagsCount"],
            "Provider risk tier": search_records["ProviderRiskLevel"],
        })
        st.dataframe(
            search_display,
            use_container_width=True,
            hide_index=True,
        )
        claim_options = search_records["ClaimID"].astype(str).tolist()
        selected_index = (
            claim_options.index(current_claim_id)
            if current_claim_id in claim_options
            else 0
        )
        selected_result = st.selectbox(
            "Select a claim from these results",
            claim_options,
            index=selected_index,
            key="advanced_result_claim",
        )
        open_col, prev_col, next_col, export_col = st.columns([2, 1, 1, 2])
        with open_col:
            if st.button("Open selected claim", use_container_width=True):
                st.session_state.selected_claim_id = selected_result
                st.session_state.pop("claim_search_error", None)
                st.rerun()
        with prev_col:
            if st.button("Previous", disabled=advanced_page <= 1):
                st.session_state.advanced_page -= 1
                st.rerun()
        with next_col:
            if st.button("Next", disabled=advanced_page >= results["pages"]):
                st.session_state.advanced_page += 1
                st.rerun()
        with export_col:
            export_df = search_records.copy()
            export_df["ReimbursementINR"] = export_df["InscClaimAmtReimbursed"].map(usd_to_inr)
            export_df["DeductibleINR"] = export_df["DeductibleAmtPaid"].map(usd_to_inr)
            export_df["USDToINRRate"] = currency_configuration()["rate"]
            export_df["FXEffectiveDate"] = currency_configuration()["effective_date"]
            export_df["FXRateIllustrative"] = currency_configuration()["illustrative"]
            export_df["CurrencyNote"] = currency_context_text()
            st.download_button(
                "Export results page to CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name=f"claims_search_page_{advanced_page}.csv",
                mime="text/csv",
                use_container_width=True,
            )
