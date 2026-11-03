"""Search source-screened Medicare claims and open a claim investigation."""

import pandas as pd
import streamlit as st

from config.settings import CASE_STATUS_OPTIONS
from src.claims_repository import get_screening_claims
from src.investigation import list_cases
from src.preprocessing import (
    format_currency,
    format_date,
    format_usd,
)
from src.styles import apply_custom_styles, render_header


RULES = [
    "All",
    "Reimbursement outlier",
    "Repeated beneficiary/provider/date cluster",
    "Date/duration anomaly",
    "Complex outpatient procedure",
    "High-frequency beneficiary",
]


def _not_available(value):
    return value is None or pd.isna(value)


def _evidence_for_rule(row):
    if row["RuleCode"] == "OUTLIER_REIMBURSEMENT":
        amount = row["InscClaimAmtReimbursed"]
        baseline = row["ClaimTypeP99"]
        observed = (
            f"{format_currency(amount)} (source {format_usd(amount)})"
            if not _not_available(amount)
            else "Not available"
        )
        comparison = (
            f"{format_currency(baseline)} (source {format_usd(baseline)}) "
            f"{row['ClaimType']} p99 in the current dataset; original stored threshold not available"
            if not _not_available(baseline)
            else "Not available"
        )
        return observed, comparison

    if row["RuleCode"] == "DUPLICATE_BILLING":
        return (
            f"{int(row['MatchingKeyCount'])} claim records share this beneficiary/provider/start-date key",
            "At most one record for this key; repeated records are a review lead, not proof of duplicate billing",
        )
    if row["RuleCode"] == "DURATION_INCONSISTENCY":
        dates = (
            f"Start {format_date(row['ClaimStartDt'])}; end {format_date(row['ClaimEndDt'])}"
            if not _not_available(row["ClaimStartDt"]) or not _not_available(row["ClaimEndDt"])
            else "Claim dates not available"
        )
        duration = (
            f"{row['ClaimDurationDays']} day(s)"
            if not _not_available(row["ClaimDurationDays"])
            else "Not available"
        )
        stay = (
            f"{row['LengthOfStayDays']} day(s)"
            if not _not_available(row["LengthOfStayDays"])
            else "Not available"
        )
        return (
            f"{dates}; claim duration {duration}; inpatient stay {stay}",
            "End date on/after start; duration and inpatient stay are non-negative",
        )
    if row["RuleCode"] == "COMPLEX_PROCEDURES":
        count = row["ProcedureCount"]
        observed = f"{int(count)} procedures" if not _not_available(count) else "Not available"
        return observed, "At least 3 procedure codes on an outpatient claim"

    count = row["BeneficiaryClaimCount"]
    observed = f"{int(count)} claims" if not _not_available(count) else "Not available"
    return observed, "At least 15 beneficiary claims in the available dataset window"


apply_custom_styles()
render_header(
    "Suspicious Claim Screening",
    "Search Medicare claims with recorded claim-level screening flags. A flag is a review lead, not a fraud finding.",
)
st.caption(
    "This page uses the screening flags stored with Medicare claim records. NFHS-5 survey consistency rules are separate and are not included here."
)

with st.form("claim_screening_filters"):
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_rule = st.selectbox("Screening rule", RULES, key="screening_rule")
        selected_type = st.selectbox(
            "Claim type",
            ["All", "Inpatient", "Outpatient"],
            key="screening_claim_type",
        )
        provider_search = st.text_input(
            "Provider ID contains",
            placeholder="Optional provider filter",
            key="screening_provider",
        )
    with filter_col2:
        status_options = ["All", "Not in workspace", *CASE_STATUS_OPTIONS]
        selected_status = st.selectbox(
            "Investigation status",
            status_options,
            key="screening_status",
        )
        claim_search = st.text_input(
            "Claim ID contains",
            placeholder="Optional claim filter",
            key="screening_claim_id",
        )
        beneficiary_search = st.text_input(
            "Beneficiary ID contains",
            placeholder="Optional beneficiary filter",
            key="screening_beneficiary",
        )
    apply_filters = st.form_submit_button("Apply filters", type="primary")

case_statuses = list_cases(target_type="CLAIM")
case_by_claim = {case["target_id"]: case for case in case_statuses}
case_filter_args = {}
if selected_status == "Not in workspace":
    case_filter_args["exclude_case_ids"] = list(case_by_claim)
elif selected_status != "All":
    case_filter_args["case_ids"] = [
        case["target_id"]
        for case in case_statuses
        if case["status"] == selected_status
    ]

filter_signature = (
    selected_rule,
    selected_type,
    provider_search.strip(),
    selected_status,
    claim_search.strip(),
    beneficiary_search.strip(),
)
if st.session_state.get("screening_filter_signature") != filter_signature:
    st.session_state.screening_page = 1
    st.session_state.screening_filter_signature = filter_signature

page_size = 100
page = st.session_state.get("screening_page", 1)
results = get_screening_claims(
    rule=selected_rule,
    claim_type=selected_type,
    provider_id=provider_search.strip() or None,
    claim_id=claim_search.strip() or None,
    bene_id=beneficiary_search.strip() or None,
    limit=page_size,
    offset=(page - 1) * page_size,
    **case_filter_args,
)
if page > results["pages"]:
    st.session_state.screening_page = 1
    st.rerun()

unique_col, trigger_col = st.columns(2)
unique_col.metric("Unique claims with selected rule(s)", f"{results['total']:,}")
trigger_col.metric("Recorded rule triggers", f"{results['rule_trigger_occurrences']:,}")
st.caption(
    "The table has one row per claim and triggered rule. A claim with multiple rules appears more than once; "
    "unique claims and rule-trigger rows are counted separately."
)

rows = results["records"]
if rows.empty:
    st.info("No claims with recorded screening flags match these filters.")
else:
    evidence = [_evidence_for_rule(row) for _, row in rows.iterrows()]
    display = pd.DataFrame({
        "Claim ID": rows["ClaimID"].astype(str),
        "Beneficiary ID": rows["BeneID"].astype(str),
        "Provider ID": rows["Provider"].astype(str),
        "Date": rows["ClaimStartDt"].map(format_date),
        "Reimbursement": rows["InscClaimAmtReimbursed"].map(
            lambda value: f"{format_currency(value)} (source {format_usd(value)})"
            if not _not_available(value)
            else "Not available"
        ),
        "Triggered rule": rows["Rule"],
        "Observed value": [item[0] for item in evidence],
        "Comparison baseline": [item[1] for item in evidence],
        "Investigation status": rows["ClaimID"].map(
            lambda claim_id: case_by_claim.get(claim_id, {}).get("status", "Not in workspace")
        ),
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

    action_col, previous_col, next_col = st.columns([2, 1, 1])
    claim_options = rows["ClaimID"].astype(str).drop_duplicates().tolist()
    with action_col:
        claim_to_open = st.selectbox("Claim to investigate", claim_options, key=f"screening_claim_page_{page}")
        if st.button("Investigate selected claim", type="primary", use_container_width=True):
            st.session_state.selected_claim_id = claim_to_open
            st.switch_page("pages/2_Claim_Investigation.py")
    with previous_col:
        if st.button("Previous", disabled=page <= 1):
            st.session_state.screening_page = page - 1
            st.rerun()
    with next_col:
        if st.button("Next", disabled=page >= results["pages"]):
            st.session_state.screening_page = page + 1
            st.rerun()
    st.caption(f"Showing rule-trigger rows page {page} of {results['pages']}.")
