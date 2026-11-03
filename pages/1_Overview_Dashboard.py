import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import get_duckdb_connection
from src.investigation import list_cases
from src.preprocessing import (
    currency_configuration,
    format_currency,
    usd_to_inr,
)
from src.styles import apply_custom_styles, render_header
from src.visualizations import format_inr_axis

apply_custom_styles()
render_header(
    "Overview Dashboard",
    "Current claims activity, reimbursement exposure, provider risk context, and investigation workload.",
)
con = get_duckdb_connection(read_only=True)
date_min, date_max = con.execute("SELECT MIN(ClaimStartDt), MAX(ClaimStartDt) FROM claims").fetchone()
date_range = st.sidebar.date_input(
    "Claim start date range",
    value=(pd.Timestamp(date_min).date(), pd.Timestamp(date_max).date()),
    min_value=pd.Timestamp(date_min).date(),
    max_value=pd.Timestamp(date_max).date(),
    key="overview_dates",
)
claim_type = st.sidebar.selectbox("Claim setting", ["All", "Inpatient", "Outpatient"], key="overview_type")
risk_level = st.sidebar.selectbox("Provider risk tier", ["All", "High", "Medium", "Low"], key="overview_risk")
provider_search = st.sidebar.text_input("Provider ID contains", key="overview_provider")
case_statuses = ["All", "No case", "New", "Under Review", "Escalated", "Reviewed/Closed", "Closed - False Positive", "Closed - Suspected Fraud"]
case_status = st.sidebar.selectbox("Investigation status", case_statuses, key="overview_case_status")

date_from = date_range[0] if isinstance(date_range, (tuple, list)) and date_range else pd.Timestamp(date_min).date()
date_to = date_range[1] if isinstance(date_range, (tuple, list)) and len(date_range) > 1 else pd.Timestamp(date_max).date()
conditions = ["ClaimStartDt >= CAST(? AS DATE)", "ClaimStartDt <= CAST(? AS DATE)"]
params = [date_from.isoformat(), date_to.isoformat()]
if claim_type != "All":
    conditions.append("ClaimType = ?")
    params.append(claim_type)
if risk_level != "All":
    conditions.append("ProviderRiskLevel = ?")
    params.append(risk_level)
if provider_search.strip():
    conditions.append("Provider ILIKE ?")
    params.append(f"%{provider_search.strip()}%")

cases = list_cases(target_type="CLAIM")
status_map = {case["target_id"]: case["status"] for case in cases}
case_ids = [case_id for case_id, status in status_map.items() if status == case_status]
if case_status == "No case":
    known_claim_ids = list(status_map)
    if known_claim_ids:
        conditions.append("ClaimID NOT IN (" + ",".join("?" for _ in known_claim_ids) + ")")
        params.extend(known_claim_ids)
elif case_status != "All":
    if case_ids:
        conditions.append("ClaimID IN (" + ",".join("?" for _ in case_ids) + ")")
        params.extend(case_ids)
    else:
        conditions.append("1 = 0")

where_clause = " WHERE " + " AND ".join(conditions)
summary = con.execute(f"""
    SELECT COUNT(*) AS TotalClaims,
        COUNT(DISTINCT BeneID) AS Beneficiaries,
        COUNT(DISTINCT Provider) AS Providers,
        COALESCE(SUM(InscClaimAmtReimbursed), 0) AS Reimbursement,
        COALESCE(AVG(InscClaimAmtReimbursed), 0) AS AverageReimbursement,
        COUNT(*) FILTER (WHERE ClaimType = 'Inpatient') AS InpatientClaims,
        COUNT(*) FILTER (WHERE ClaimType = 'Outpatient') AS OutpatientClaims,
        COUNT(DISTINCT Provider) FILTER (WHERE ProviderFraudProbability >= 0.30) AS ModelFlaggedProviders
    FROM claims{where_clause}
""", params).fetchdf().iloc[0].to_dict()
screened_summary = con.execute(f"""
    WITH prepared AS (
        SELECT c.*,
            quantile_cont(InscClaimAmtReimbursed, 0.99) OVER (PARTITION BY ClaimType) AS ClaimTypeP99,
            COUNT(*) OVER (PARTITION BY BeneID, Provider, ClaimStartDt) AS MatchingKeyCount,
            COUNT(*) OVER (PARTITION BY BeneID) AS BeneficiaryClaimCount
        FROM claims c
    ),
    screened AS (
        SELECT *,
            (InscClaimAmtReimbursed > ClaimTypeP99) AS ReimbursementOutlier,
            (MatchingKeyCount > 1) AS DuplicateKeyCluster,
            (ClaimDurationDays < 0 OR LengthOfStayDays < 0
                OR (ClaimStartDt IS NOT NULL AND ClaimEndDt IS NOT NULL AND ClaimStartDt > ClaimEndDt)
                OR (AdmissionDt IS NOT NULL AND DischargeDt IS NOT NULL AND AdmissionDt > DischargeDt)) AS DateAnomaly,
            (ClaimType = 'Outpatient' AND ProcedureCount >= 3) AS ComplexProcedure,
            (BeneficiaryClaimCount >= 15) AS HighFrequencyBeneficiary
        FROM prepared
    )
    SELECT COUNT(*) FILTER (WHERE ReimbursementOutlier OR DuplicateKeyCluster OR DateAnomaly OR ComplexProcedure OR HighFrequencyBeneficiary) AS UniqueFlaggedClaims,
        SUM(CAST(ReimbursementOutlier AS INTEGER) + CAST(DuplicateKeyCluster AS INTEGER)
            + CAST(DateAnomaly AS INTEGER) + CAST(ComplexProcedure AS INTEGER)
            + CAST(HighFrequencyBeneficiary AS INTEGER)) AS RuleTriggerOccurrences
    FROM screened{where_clause}
""", params).fetchdf().iloc[0].to_dict()
summary.update(screened_summary)
monthly = con.execute(f"""
    SELECT date_trunc('month', ClaimStartDt) AS Month,
        COUNT(*) AS Claims,
        SUM(InscClaimAmtReimbursed) AS Reimbursement
    FROM claims{where_clause}
    GROUP BY 1 ORDER BY 1
""", params).fetchdf()
distribution = con.execute(f"""
    SELECT ClaimType, COUNT(*) AS Claims
    FROM claims{where_clause} GROUP BY ClaimType
""", params).fetchdf()
risk_distribution = con.execute(f"""
    SELECT ProviderRiskLevel AS RiskLevel, COUNT(DISTINCT Provider) AS Providers
    FROM claims{where_clause} GROUP BY ProviderRiskLevel
""", params).fetchdf()
top_volume = con.execute(f"""
    SELECT Provider, COUNT(*) AS Claims, SUM(InscClaimAmtReimbursed) AS Reimbursement
    FROM claims{where_clause} GROUP BY Provider ORDER BY Claims DESC LIMIT 10
""", params).fetchdf()
top_reimbursement = con.execute(f"""
    SELECT Provider, COUNT(*) AS Claims, SUM(InscClaimAmtReimbursed) AS Reimbursement
    FROM claims{where_clause} GROUP BY Provider ORDER BY Reimbursement DESC LIMIT 10
""", params).fetchdf()
alerts = con.execute(f"""
    WITH prepared AS (
        SELECT c.*,
            quantile_cont(InscClaimAmtReimbursed, 0.99) OVER (PARTITION BY ClaimType) AS ClaimTypeP99,
            COUNT(*) OVER (PARTITION BY BeneID, Provider, ClaimStartDt) AS MatchingKeyCount,
            COUNT(*) OVER (PARTITION BY BeneID) AS BeneficiaryClaimCount
        FROM claims c
    ),
    screened AS (
        SELECT *,
            (InscClaimAmtReimbursed > ClaimTypeP99) AS ReimbursementOutlier,
            (MatchingKeyCount > 1) AS DuplicateKeyCluster,
            (ClaimDurationDays < 0 OR LengthOfStayDays < 0
                OR (ClaimStartDt IS NOT NULL AND ClaimEndDt IS NOT NULL AND ClaimStartDt > ClaimEndDt)
                OR (AdmissionDt IS NOT NULL AND DischargeDt IS NOT NULL AND AdmissionDt > DischargeDt)) AS DateAnomaly,
            (ClaimType = 'Outpatient' AND ProcedureCount >= 3) AS ComplexProcedure,
            (BeneficiaryClaimCount >= 15) AS HighFrequencyBeneficiary
        FROM prepared
    )
    SELECT SUM(CAST(ReimbursementOutlier AS INTEGER)) AS ReimbursementOutliers,
        SUM(CAST(DuplicateKeyCluster AS INTEGER)) AS PossibleDuplicateKeyClaims,
        SUM(CAST(DateAnomaly AS INTEGER)) AS DateAnomalies,
        SUM(CAST(ComplexProcedure AS INTEGER)) AS ComplexProcedures,
        SUM(CAST(HighFrequencyBeneficiary AS INTEGER)) AS HighFrequencyBeneficiaries
    FROM screened{where_clause}
""", params).fetchdf().fillna(0).iloc[0].to_dict()
con.close()

all_cases = list_cases()
cards = st.columns(4)
cards[0].metric("Claims", f"{int(summary['TotalClaims']):,}")
cards[1].metric("Beneficiaries", f"{int(summary['Beneficiaries']):,}")
cards[2].metric("Providers", f"{int(summary['Providers']):,}")
cards[3].metric(
    "Reimbursement (INR)",
    format_currency(summary["Reimbursement"]),
    help=f"Average: {format_currency(summary['AverageReimbursement'])}; converted from source USD.",
)
cards = st.columns(4)
cards[0].metric("Inpatient claims", f"{int(summary['InpatientClaims']):,}")
cards[1].metric("Outpatient claims", f"{int(summary['OutpatientClaims']):,}")
cards[2].metric("ML-flagged providers", f"{int(summary['ModelFlaggedProviders']):,}", help="Provider-level score >= 0.30, matching the stored model screening threshold.")
cards[3].metric("Unique screening-flagged claims", f"{int(summary['UniqueFlaggedClaims']):,}")

monthly["Reimbursement (INR)"] = monthly["Reimbursement"].map(usd_to_inr)
top_volume["Reimbursement (INR)"] = top_volume["Reimbursement"].map(usd_to_inr)
top_reimbursement["Reimbursement (INR)"] = top_reimbursement["Reimbursement"].map(usd_to_inr)

left, right = st.columns(2)
with left:
    if not monthly.empty:
        st.plotly_chart(px.line(monthly, x="Month", y="Claims", markers=True, title="Claims over time"), use_container_width=True)
    if not distribution.empty:
        claim_selection = st.plotly_chart(
            px.pie(distribution, names="ClaimType", values="Claims", hole=0.58, title="Inpatient / outpatient mix"),
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="overview_claim_type_chart",
        )
        selected_points = claim_selection.selection.points
        if selected_points:
            selected_type = selected_points[0].get("label")
            if selected_type in {"Inpatient", "Outpatient"}:
                st.session_state["overview_selected_claim_type"] = selected_type
                st.info(f"Selected **{selected_type}** in the chart. Use the claim drill-down below to open matching records.")
    if not top_volume.empty:
        volume_selection = st.plotly_chart(
            px.bar(top_volume, x="Claims", y="Provider", orientation="h", title="Top providers by claim volume"),
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="overview_provider_volume_chart",
        )
        selected_points = volume_selection.selection.points
        if selected_points:
            st.session_state["overview_selected_provider"] = selected_points[0].get("y")
with right:
    if not monthly.empty:
        reimbursement_chart = px.line(
            monthly,
            x="Month",
            y="Reimbursement (INR)",
            markers=True,
            title="Reimbursement over time (INR, converted from USD)",
        )
        format_inr_axis(reimbursement_chart, monthly["Reimbursement (INR)"])
        st.plotly_chart(reimbursement_chart, use_container_width=True)
    if not risk_distribution.empty:
        risk_selection = st.plotly_chart(
            px.bar(risk_distribution, x="RiskLevel", y="Providers", title="Provider risk distribution"),
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="overview_provider_risk_chart",
        )
        selected_points = risk_selection.selection.points
        if selected_points:
            st.session_state["overview_selected_risk"] = selected_points[0].get("x")
    if not top_reimbursement.empty:
        reimbursement_chart = px.bar(
            top_reimbursement,
            x="Reimbursement (INR)",
            y="Provider",
            orientation="h",
            title="Top providers by reimbursement (INR, converted from USD)",
        )
        format_inr_axis(reimbursement_chart, top_reimbursement["Reimbursement (INR)"], axis="x")
        st.plotly_chart(reimbursement_chart, use_container_width=True)

st.markdown("#### Recomputed screening-rule alerts")
st.caption("Counts are recomputed from source claim fields with the rule definitions shown on the screening page; these are triage indicators, not fraud findings.")
alert_table = pd.DataFrame([
    {"Screening rule": name, "Unique claims": int(value)}
    for name, value in alerts.items()
])
st.dataframe(alert_table, use_container_width=True, hide_index=True)

st.markdown("#### Investigation status")
status_frame = pd.DataFrame(all_cases)
if status_frame.empty:
    st.info("There are no saved investigation cases yet.")
else:
    status_counts = status_frame["status"].value_counts().rename_axis("Status").reset_index(name="Cases")
    st.plotly_chart(px.bar(status_counts, x="Status", y="Cases", title="Workspace case status"), use_container_width=True)

st.markdown("#### Drill into source records")
st.caption("Select a provider or claim setting above, or choose one below, then open its detailed source records.")
drill_provider_options = top_volume["Provider"].astype(str).tolist() if not top_volume.empty else []
selected_provider_default = st.session_state.get("overview_selected_provider")
provider_index = drill_provider_options.index(selected_provider_default) if selected_provider_default in drill_provider_options else 0
drill_type_options = ["All", "Inpatient", "Outpatient"]
selected_type_default = st.session_state.get("overview_selected_claim_type", "All")
selected_type_default = selected_type_default if selected_type_default in drill_type_options else "All"
drill_risk_options = ["All", "High", "Medium", "Low"]
selected_risk_default = st.session_state.get("overview_selected_risk", "All")
selected_risk_default = selected_risk_default if selected_risk_default in drill_risk_options else "All"
drill_col1, drill_col2, drill_col3 = st.columns([2, 1, 1])
with drill_col1:
    if drill_provider_options:
        provider_id = st.selectbox("Provider", drill_provider_options, index=provider_index)
    else:
        provider_id = None
with drill_col2:
    drill_type = st.selectbox("Claim setting", drill_type_options, index=drill_type_options.index(selected_type_default))
with drill_col3:
    drill_risk = st.selectbox("Provider risk", drill_risk_options, index=drill_risk_options.index(selected_risk_default))
st.caption("Selected chart values populate these drill-down controls.")
go_col, spacer = st.columns([1, 3])
with go_col:
    if st.button("Open matching claims", type="primary", use_container_width=True):
        st.session_state.filter_provider_id = provider_id or ""
        st.session_state.filter_claim_type = drill_type
        st.session_state.filter_risk_level = drill_risk
        st.session_state.in_prov_id = provider_id or ""
        st.session_state.in_claim_type = drill_type
        st.session_state.in_risk_level = drill_risk
        st.session_state.page_number = 1
        st.switch_page("pages/2_Claim_Investigation.py")

with st.expander("How this dashboard identifies items for investigation"):
    st.markdown("""
    **There is no validated claim-level fraud target in these project data.** The system does not train a model that calls an individual claim fraudulent.

    1. **Provider-level supervised model:** Random Forest, Logistic Regression, and XGBoost use provider aggregates (claim volume, reimbursement, beneficiaries, and inpatient/outpatient totals) to predict the dataset's provider-level `PotentialFraud` label. The score is a provider risk signal only.
    2. **Claim-level rules:** the current screening view recomputes reimbursement strictly above each claim-type 99th percentile, repeated beneficiary/provider/start-date combinations, invalid date or duration sequences, outpatient claims with at least three procedures, and beneficiaries with at least 15 claims. The stored legacy reimbursement flags used an inclusive cutoff and are reconciled on the screening page.
    3. **Investigator decision:** flags are triage evidence. They are not confirmed fraud, and should not be used by themselves to deny payment or accuse a person/provider.
    """)

st.markdown("#### Continue investigation")
nav1, nav2, nav3, nav4 = st.columns(4)
with nav1:
    if st.button("Claim Investigation", use_container_width=True):
        st.switch_page("pages/2_Claim_Investigation.py")
with nav2:
    if st.button("Screening Alerts", use_container_width=True):
        st.switch_page("pages/5_Suspicious_Claim_Screening.py")
with nav3:
    if st.button("Model Evidence", use_container_width=True):
        st.switch_page("pages/6_ML_Model_Performance.py")
with nav4:
    if st.button("Investigation Workspace", use_container_width=True):
        st.switch_page("pages/8_Investigation_Workspace.py")
st.download_button(
    "Export filtered top-provider records",
    top_reimbursement.assign(
        ReimbursementSourceUSD=top_reimbursement["Reimbursement"],
        USDToINRRate=currency_configuration()["rate"],
        FXEffectiveDate=currency_configuration()["effective_date"],
        FXRateIllustrative=currency_configuration()["illustrative"],
    ).to_csv(index=False).encode("utf-8"),
    file_name="overview_top_providers.csv",
    mime="text/csv",
)
