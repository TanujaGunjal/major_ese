"""Provider-level model context, SHAP explanation, and linked source claims."""

import math

import pandas as pd
import streamlit as st

from config.settings import (
    DEFAULT_FRAUD_THRESHOLD,
    DEFAULT_HIGH_RISK_THRESHOLD,
    FEATURE_COLUMNS,
)
from src.claims_repository import get_claims_by_provider, get_provider_profile
from src.data_loader import load_model_metadata
from src.explainability import explain_provider_prediction
from src.investigation import list_cases
from src.preprocessing import format_currency, format_usd
from src.styles import apply_custom_styles, render_header


def _number(value):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if math.isfinite(numeric_value) else None


def _display_count(value):
    numeric_value = _number(value)
    return f"{int(numeric_value):,}" if numeric_value is not None else "Not available"


def _display_money(value):
    numeric_value = _number(value)
    return format_currency(numeric_value) if numeric_value is not None else "Not available"


apply_custom_styles()
render_header(
    "Provider Risk Analysis",
    "Review the provider-level model assessment and its linked source claims. This is not a claim-level finding.",
)

default_provider = st.session_state.get("selected_provider_id", "")
with st.form("provider_search"):
    search_col, button_col = st.columns([5, 1])
    with search_col:
        provider_search = st.text_input(
            "Provider ID",
            value=default_provider,
            placeholder="Enter an exact ID, for example PRV51001",
        )
    with button_col:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

if submitted:
    requested_provider = provider_search.strip().upper()
    st.session_state.selected_provider_id = requested_provider
    if not requested_provider:
        st.session_state.provider_search_error = "Enter a Provider ID to search."
    else:
        st.session_state.pop("provider_search_error", None)

if st.session_state.get("provider_search_error"):
    st.error(st.session_state.provider_search_error)

provider_id = st.session_state.get("selected_provider_id", "").strip()
if not provider_id:
    st.info("Enter a Provider ID to view its model assessment and linked claims.")
    st.stop()

provider = get_provider_profile(provider_id)
if provider is None:
    st.warning(f"No provider record was found for Provider ID {provider_id}.")
    st.stop()

metadata = load_model_metadata()
thresholds = metadata.get("thresholds", {})
metadata_fraud_threshold = _number(thresholds.get("fraud_threshold"))
metadata_high_risk_threshold = _number(thresholds.get("high_risk_threshold"))
fraud_threshold = (
    metadata_fraud_threshold
    if metadata_fraud_threshold is not None
    else DEFAULT_FRAUD_THRESHOLD
)
high_risk_threshold = (
    metadata_high_risk_threshold
    if metadata_high_risk_threshold is not None
    else DEFAULT_HIGH_RISK_THRESHOLD
)
score = _number(provider.get("FraudProbability"))
if score is not None and not 0 <= score <= 1:
    score = None

st.subheader("Provider model assessment")
score_col, tier_col, count_col, bene_col = st.columns(4)
score_col.metric("Provider model score", f"{score:.1%}" if score is not None else "Not available")
tier_col.metric(
    "Provider risk tier",
    str(provider.get("RiskLevel")) if pd.notna(provider.get("RiskLevel")) else "Not available",
)
count_col.metric("Claims in provider record", _display_count(provider.get("TotalClaims")))
bene_col.metric("Unique beneficiaries", _display_count(provider.get("UniquePatients")))

inpatient_count = _number(provider.get("InpatientClaims"))
outpatient_count = _number(provider.get("OutpatientClaims"))
inpatient_total = _number(provider.get("InpatientReimbursement"))
outpatient_total = _number(provider.get("OutpatientReimbursement"))
average = _number(provider.get("AverageReimbursement"))
total_reimbursement = _number(provider.get("TotalReimbursement"))

type_col1, type_col2, reimb_col1, reimb_col2 = st.columns(4)
type_col1.metric("Inpatient claims", _display_count(inpatient_count))
type_col2.metric("Outpatient claims", _display_count(outpatient_count))
reimb_col1.metric(
    "Total reimbursement (INR)",
    _display_money(total_reimbursement),
    help=f"Source amount: {format_usd(total_reimbursement)}" if total_reimbursement is not None else None,
)
reimb_col2.metric(
    "Average reimbursement (INR)",
    _display_money(average),
    help=f"Source amount: {format_usd(average)}" if average is not None else None,
)

threshold_col1, threshold_col2 = st.columns(2)
threshold_col1.metric("Configured provider review threshold", f"{fraud_threshold:.0%}")
threshold_col2.metric("Configured high-risk threshold", f"{high_risk_threshold:.0%}")
st.caption(
    f"Model: Random Forest. The stored primary provider score uses the Random Forest probability. "
    f"Thresholds are from model_metadata.json (fallback to configured defaults if metadata is unavailable). "
    "A provider-level score does not classify every linked claim as fraudulent."
)
if score is not None:
    if score >= high_risk_threshold:
        st.warning("This provider score meets the configured high-risk review threshold.")
    elif score >= fraud_threshold:
        st.info("This provider score meets the configured provider review threshold.")
    else:
        st.success("This provider score is below the configured provider review threshold.")

st.markdown("#### Reimbursement by claim setting")
reimbursement_summary = pd.DataFrame([
    {
        "Claim type": "All",
        "Claims": _display_count(provider.get("TotalClaims")),
        "Total reimbursement (INR)": _display_money(total_reimbursement),
        "Average reimbursement (INR)": _display_money(average),
    },
    {
        "Claim type": "Inpatient",
        "Claims": _display_count(inpatient_count),
        "Total reimbursement (INR)": _display_money(inpatient_total),
        "Average reimbursement (INR)": _display_money(provider.get("InpatientAvgReimbursement")),
    },
    {
        "Claim type": "Outpatient",
        "Claims": _display_count(outpatient_count),
        "Total reimbursement (INR)": _display_money(outpatient_total),
        "Average reimbursement (INR)": _display_money(provider.get("OutpatientAvgReimbursement")),
    },
])
st.dataframe(reimbursement_summary, use_container_width=True, hide_index=True)
st.caption("Amounts above are converted for display from the USD values stored in provider features.")

feature_values = {column: _number(provider.get(column)) for column in FEATURE_COLUMNS}
st.subheader("Main contributing features (SHAP)")
if any(value is None for value in feature_values.values()):
    st.info("SHAP explanation is unavailable because one or more provider model features are missing.")
else:
    try:
        explanation = explain_provider_prediction(feature_values)
    except Exception as exc:
        st.warning(f"The provider SHAP explanation could not be generated: {exc}")
    else:
        if explanation.get("error"):
            st.info(explanation["error"])
        else:
            st.caption(
                "Local SHAP contributions for the Random Forest provider score—not for an individual claim."
            )
            impact = pd.DataFrame(explanation.get("contributions", []))
            if impact.empty:
                st.info("No SHAP contributions were returned for this provider.")
            else:
                top_impact = impact.sort_values("abs_shap", ascending=False).head(5).copy()
                top_impact["Provider feature value"] = top_impact.apply(
                    lambda row: (
                        f"{format_currency(row['value'])} (source {format_usd(row['value'])})"
                        if "Reimbursement" in row["feature"]
                        else f"{row['value']:,.2f}"
                    ),
                    axis=1,
                )
                st.bar_chart(top_impact.set_index("display_name")["shap_value"])
                st.dataframe(
                    top_impact[[
                        "feature", "Provider feature value", "shap_value",
                        "direction", "description",
                    ]].rename(columns={
                        "feature": "Feature",
                        "shap_value": "SHAP contribution",
                        "direction": "Contribution direction",
                        "description": "What the feature represents",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

provider_cases = list_cases(target_type="PROVIDER", search_query=provider_id)
if provider_cases:
    with st.expander("Provider investigation cases"):
        st.dataframe(provider_cases, use_container_width=True, hide_index=True)

if st.button("Open this provider in Investigation Workspace"):
    st.session_state.selected_provider_id = provider_id
    st.session_state.workspace_target_type = "PROVIDER"
    st.session_state.workspace_target_id = provider_id
    st.switch_page("pages/8_Investigation_Workspace.py")

st.subheader("Provider's actual claims")
st.markdown('<a href="#provider-claims">View linked actual claims</a>', unsafe_allow_html=True)
st.markdown('<div id="provider-claims"></div>', unsafe_allow_html=True)
page_size = 30
total_claims = int(_number(provider.get("TotalClaims")) or 0)
total_pages = max(1, (total_claims + page_size - 1) // page_size)
claims_page_key = f"provider_claims_page_input_{provider_id}"
if st.session_state.get("provider_claims_owner") != provider_id:
    st.session_state.provider_claims_owner = provider_id
    st.session_state.provider_claims_page = 1
    st.session_state[claims_page_key] = 1
if claims_page_key not in st.session_state:
    st.session_state[claims_page_key] = min(
        st.session_state.get("provider_claims_page", 1),
        total_pages,
    )
page = st.number_input(
    "Claims page",
    min_value=1,
    max_value=total_pages,
    step=1,
    key=claims_page_key,
)
st.session_state.provider_claims_page = int(page)
claims = get_claims_by_provider(
    provider_id,
    limit=page_size,
    offset=(int(page) - 1) * page_size,
)
if claims.empty:
    st.info("No source claim records are available for this provider.")
else:
    claim_display = claims.copy()
    claim_display["Claim date"] = pd.to_datetime(
        claim_display["ClaimStartDt"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    claim_display["Reimbursement (INR)"] = claim_display["InscClaimAmtReimbursed"].map(
        lambda value: _display_money(value)
    )
    claim_display["Duration"] = claim_display["ClaimDurationDays"].map(
        lambda value: f"{value} day(s)" if pd.notna(value) else "Not available"
    )
    st.dataframe(
        claim_display[[
            "ClaimID", "BeneID", "ClaimType", "Claim date",
            "Reimbursement (INR)", "Duration",
        ]].rename(columns={
            "ClaimID": "Claim ID",
            "BeneID": "Beneficiary ID",
            "ClaimType": "Claim type",
        }),
        use_container_width=True,
        hide_index=True,
    )
    if total_pages > 1:
        previous_col, page_col, next_col = st.columns([1, 2, 1])
        if previous_col.button("Previous claims", disabled=page <= 1):
            st.session_state[claims_page_key] = int(page) - 1
            st.rerun()
        page_col.caption(f"Page {int(page)} of {total_pages}; {total_claims:,} source claims.")
        if next_col.button("Next claims", disabled=page >= total_pages):
            st.session_state[claims_page_key] = int(page) + 1
            st.rerun()

    claim_ids = claims["ClaimID"].astype(str).tolist()
    selected_claim_id = st.selectbox(
        "Select a linked claim to investigate",
        claim_ids,
        key=f"provider_claim_selection_{provider_id}_{int(page)}",
    )
    if st.button("Open selected claim in Claim Investigation", type="primary"):
        st.session_state.selected_claim_id = selected_claim_id
        st.switch_page("pages/2_Claim_Investigation.py")
