"""Chronological source claims and review context for one beneficiary."""

import pandas as pd
import streamlit as st

from src.claims_repository import get_beneficiary_profile, get_claims_by_beneficiary
from src.preprocessing import (
    extract_codes_list,
    extract_active_conditions,
    format_currency,
    format_date,
    format_usd,
)
from src.styles import apply_custom_styles, render_header


def _overlap_patterns(history):
    pairs = []
    records = history.to_dict("records")
    for left_index, left in enumerate(records):
        left_start = pd.to_datetime(left.get("ClaimStartDt"), errors="coerce")
        left_end = pd.to_datetime(left.get("ClaimEndDt"), errors="coerce")
        if pd.isna(left_start) or pd.isna(left_end):
            continue
        for right in records[left_index + 1:]:
            right_start = pd.to_datetime(right.get("ClaimStartDt"), errors="coerce")
            right_end = pd.to_datetime(right.get("ClaimEndDt"), errors="coerce")
            if pd.isna(right_start) or pd.isna(right_end):
                continue
            if left_start <= right_end and right_start <= left_end:
                pairs.append({
                    "Claim ID": f"{left['ClaimID']} / {right['ClaimID']}",
                    "Date range": (
                        f"{format_date(max(left_start, right_start))} to "
                        f"{format_date(min(left_end, right_end))}"
                    ),
                    "Provider IDs": f"{left['Provider']} / {right['Provider']}",
                    "Claim types": f"{left['ClaimType']} / {right['ClaimType']}",
                })
    return pairs


apply_custom_styles()
render_header(
    "Beneficiary History",
    "Search a real Beneficiary ID and review its linked Medicare claims in date order.",
)

default_id = st.session_state.get("selected_bene_id", "")
with st.form("beneficiary_search"):
    search_col, button_col = st.columns([5, 1])
    with search_col:
        bene_search = st.text_input(
            "Beneficiary ID",
            value=default_id,
            placeholder="Enter an exact ID, for example BENE11001",
        )
    with button_col:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

if submitted:
    requested_id = bene_search.strip().upper()
    st.session_state.selected_bene_id = requested_id
    if not requested_id:
        st.session_state.beneficiary_search_error = "Enter a Beneficiary ID to search."
    else:
        st.session_state.pop("beneficiary_search_error", None)

if st.session_state.get("beneficiary_search_error"):
    st.error(st.session_state.beneficiary_search_error)

bene_id = st.session_state.get("selected_bene_id", "").strip()
if not bene_id:
    st.info("Enter a Beneficiary ID to load its claims.")
    st.stop()

history = get_claims_by_beneficiary(bene_id)
if history.empty:
    st.warning(f"No claims were found for Beneficiary ID {bene_id}.")
    st.stop()

profile = get_beneficiary_profile(bene_id)
history["ClaimStartDt"] = pd.to_datetime(history["ClaimStartDt"], errors="coerce")
history["ClaimEndDt"] = pd.to_datetime(history["ClaimEndDt"], errors="coerce")
history["InscClaimAmtReimbursed"] = pd.to_numeric(
    history["InscClaimAmtReimbursed"], errors="coerce"
)
history = history.sort_values(["ClaimStartDt", "ClaimID"], na_position="last").reset_index(drop=True)

reimbursement_total = history["InscClaimAmtReimbursed"].sum(min_count=1)
summary_col1, summary_col2, summary_col3 = st.columns(3)
summary_col1.metric("Beneficiary ID", bene_id)
summary_col2.metric("Historical claims", f"{len(history):,}")
summary_col3.metric(
    "Total reimbursement (INR)",
    format_currency(reimbursement_total) if pd.notna(reimbursement_total) else "Not available",
    help=(
        f"Source total: {format_usd(reimbursement_total)}"
        if pd.notna(reimbursement_total)
        else "No reimbursement amounts are available."
    ),
)

if profile:
    conditions = extract_active_conditions(pd.Series(profile))
    available_context = [
        f"Age: {profile.get('Age') if pd.notna(profile.get('Age')) else 'Not available'}",
        f"Gender: {profile.get('Gender_Desc') if pd.notna(profile.get('Gender_Desc')) else 'Not available'}",
        f"Conditions recorded: {', '.join(conditions) if conditions else 'None recorded'}",
    ]
    st.caption(" · ".join(available_context))

st.subheader("Claim history")
display_history = history.copy()
display_history["Date"] = display_history["ClaimStartDt"].map(format_date)
display_history["Reimbursement (INR)"] = display_history["InscClaimAmtReimbursed"].map(
    lambda amount: format_currency(amount) if pd.notna(amount) else "Not available"
)
display_history["Source reimbursement (USD)"] = display_history["InscClaimAmtReimbursed"].map(
    lambda amount: format_usd(amount) if pd.notna(amount) else "Not available"
)
display_history["Duration"] = display_history["ClaimDurationDays"].map(
    lambda days: f"{days} day(s)" if pd.notna(days) else "Not available"
)
display_history["Diagnosis information"] = display_history.apply(
    lambda row: ", ".join(
        ([str(row["ClmAdmitDiagnosisCode"])] if pd.notna(row["ClmAdmitDiagnosisCode"]) else [])
        + extract_codes_list(pd.Series(row), "ClmDiagnosisCode", 10)
    ) or "Not available",
    axis=1,
)
st.dataframe(
    display_history[[
        "ClaimID", "Date", "ClaimType", "Provider", "Reimbursement (INR)",
        "Source reimbursement (USD)", "Duration", "Diagnosis information",
    ]].rename(columns={
        "ClaimID": "Claim ID",
        "ClaimType": "Claim type",
        "Provider": "Provider ID",
    }),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Patterns to review")
st.caption(
    "Patterns below are based only on the linked records shown. Repeated or overlapping claims can be legitimate and are not findings of fraud."
)
repeat_groups = (
    history.dropna(subset=["ClaimStartDt"])
    .groupby(["Provider", "ClaimStartDt"], as_index=False)
    .agg(Claims=("ClaimID", lambda values: ", ".join(values.astype(str))),
         RecordCount=("ClaimID", "count"))
)
repeat_groups = repeat_groups[repeat_groups["RecordCount"] > 1].copy()
overlap_pairs = _overlap_patterns(history)
if repeat_groups.empty and not overlap_pairs:
    st.info("No repeated provider/date keys or overlapping date ranges were found in this beneficiary's available claim history.")
else:
    if not repeat_groups.empty:
        repeated_display = repeat_groups.rename(columns={
            "Provider": "Provider ID",
            "ClaimStartDt": "Shared claim start date",
            "Claims": "Claim IDs",
            "RecordCount": "Records with this provider/date key",
        })
        st.markdown("**Repeated provider and claim-start-date keys**")
        st.dataframe(repeated_display, use_container_width=True, hide_index=True)
    if overlap_pairs:
        st.markdown("**Claims with overlapping recorded date ranges**")
        st.dataframe(pd.DataFrame(overlap_pairs), use_container_width=True, hide_index=True)

st.subheader("Investigation actions")
claim_ids = history["ClaimID"].astype(str).tolist()
selected_claim_id = st.selectbox("Select a historical claim", claim_ids, key="beneficiary_history_claim")
if st.button("Open selected claim in Claim Investigation", type="primary"):
    st.session_state.selected_claim_id = selected_claim_id
    st.switch_page("pages/2_Claim_Investigation.py")

providers = sorted(history["Provider"].dropna().astype(str).unique())
if providers:
    selected_provider_id = st.selectbox(
        "Open a provider from this history",
        providers,
        key="beneficiary_history_provider",
    )
    if st.button("Open provider risk analysis"):
        st.session_state.selected_provider_id = selected_provider_id
        st.switch_page("pages/4_Provider_Risk_Analysis.py")
