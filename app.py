"""
AI-Based Healthcare Claims Fraud Detection System — Phase 4
Main Entry Point using Streamlit Native Navigation.
"""

import streamlit as st
from src.styles import apply_custom_styles

# Page configuration
st.set_page_config(
    page_title="Healthcare Claims Fraud Investigation System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global styling
apply_custom_styles()

# Initialize global investigation state
if "selected_claim_id" not in st.session_state:
    st.session_state.selected_claim_id = "CLM46614"
if "selected_bene_id" not in st.session_state:
    st.session_state.selected_bene_id = "BENE11001"
if "selected_provider_id" not in st.session_state:
    st.session_state.selected_provider_id = "PRV51003"

# Sidebar Branding
with st.sidebar:
    st.markdown("### 🛡️ Claims Fraud AI")
    st.caption("AI-Assisted Healthcare Investigation & Decision Support Platform")
    with st.expander("Page guide", expanded=False):
        st.markdown("""
        - **Overview:** dataset KPIs, trends, filters, and drill-downs.
        - **Claim Investigation:** search real claim IDs and inspect source details, evidence, and related history.
        - **Beneficiary History:** claims timeline, reimbursements, and providers for a beneficiary.
        - **Provider Risk Analysis:** provider score, peer context, SHAP, and paginated source claims.
        - **Suspicious Claim Screening:** source-pipeline claim rules and alert evidence.
        - **ML Model Performance:** provider-level holdout metrics, thresholds, confusion matrix, and SHAP.
        - **NFHS-5:** separate survey consistency rules and sample; never joined to Medicare.
        - **Investigation Workspace:** persistent case status, notes, audit trail, and report export.
        - **Data Explorer:** source availability, schemas, completeness, duplicates, and linkage.
        """)

# Define Navigation Pages (Exact Master Prompt Structure)
pages = [
    st.Page("pages/1_Overview_Dashboard.py", title="Overview Dashboard", icon="📊", default=True),
    st.Page("pages/2_Claim_Investigation.py", title="Claim Investigation", icon="🔍"),
    st.Page("pages/3_Beneficiary_History.py", title="Beneficiary History", icon="👤"),
    st.Page("pages/4_Provider_Risk_Analysis.py", title="Provider Risk Analysis", icon="🏥"),
    st.Page("pages/5_Suspicious_Claim_Screening.py", title="Suspicious Claim Screening", icon="⚠️"),
    st.Page("pages/6_ML_Model_Performance.py", title="ML Model Performance", icon="🔬"),
    st.Page("pages/7_NFHS_Medical_Consistency.py", title="NFHS Medical Consistency", icon="🩺"),
    st.Page("pages/8_Investigation_Workspace.py", title="Investigation Workspace", icon="📋"),
    st.Page("pages/9_Data_Explorer.py", title="Data Explorer", icon="🗄️")
]

pg = st.navigation(pages)

pg.run()
