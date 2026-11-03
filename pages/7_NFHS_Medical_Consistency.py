import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_nfhs_sample_df, load_nfhs_summary
from src.styles import apply_custom_styles, render_header

RULES = {
    "R1_diabetes_no_tx": "Diabetes reported without treatment",
    "R1_htn_no_tx": "Hypertension reported without treatment",
    "R2_tx_without_diabetes": "Diabetes treatment without reported diagnosis",
    "R2_tx_without_htn": "Hypertension treatment without reported diagnosis",
    "R3_high_glucose_no_dx": "High glucose (>=200) without reported diabetes diagnosis",
    "R4_diabetes_no_meds": "Diabetes reported without prescribed medication",
}
RULE_WEIGHTS = {
    "R1_diabetes_no_tx": 3,
    "R1_htn_no_tx": 3,
    "R2_tx_without_diabetes": 5,
    "R2_tx_without_htn": 5,
    "R3_high_glucose_no_dx": 4,
    "R4_diabetes_no_meds": 2,
}

apply_custom_styles()
render_header("NFHS-5 Medical Consistency", "Explore the independent NFHS-5 sample screened by six weighted clinical consistency rules.")
data = load_nfhs_sample_df()
summary = load_nfhs_summary()
if data.empty:
    st.error("NFHS-5 screened sample is unavailable. Check data/nfhs/nfhs5_screened_sample.parquet.")
    st.stop()
missing_rule_columns = sorted(set(RULES) - set(data.columns))
if missing_rule_columns or not {"risk_score", "risk_category"}.issubset(data.columns):
    st.error(f"NFHS screened sample is missing required rule/score columns: {missing_rule_columns or ['risk_score/risk_category']}")
    st.stop()

recomputed_score = data[list(RULE_WEIGHTS)].mul(pd.Series(RULE_WEIGHTS)).sum(axis=1)
recomputed_category = recomputed_score.map(
    lambda score: "Clean" if score == 0 else "Low" if score <= 2 else "Medium" if score <= 4 else "High" if score <= 6 else "Critical"
)
score_mismatches = int((recomputed_score != data["risk_score"]).sum())
category_mismatches = int((recomputed_category != data["risk_category"]).sum())

dist = data["risk_category"].value_counts().reindex(["Clean", "Low", "Medium", "High", "Critical"], fill_value=0)
metrics = st.columns(4)
metrics[0].metric("Sample records", f"{len(data):,}")
metrics[1].metric("Sample flagged", f"{int((data['risk_score'] > 0).sum()):,}")
metrics[2].metric("Sample high / critical", f"{int(data['risk_category'].isin(['High', 'Critical']).sum()):,}")
metrics[3].metric("Recorded benchmark rows†", f"{int(summary.get('total_records_screened', 0)):,}")
if score_mismatches or category_mismatches:
    st.error(f"NFHS sample validation found {score_mismatches:,} weighted-score mismatches and {category_mismatches:,} risk-category mismatches.")
else:
    st.success("All available sample risk scores and categories match the documented six-rule weighted calculation.")
st.caption("†The full-benchmark count is present only in the supplied summary JSON; the full raw benchmark is unavailable here, so that count cannot be independently revalidated. Detailed records and filters below refer to the available screened sample.")

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.bar(dist.rename_axis("Risk category").reset_index(name="Records"), x="Risk category", y="Records", title="Sample risk distribution"), use_container_width=True)
with c2:
    rule_counts = data[list(RULES)].sum().rename(index=RULES).sort_values()
    st.plotly_chart(px.bar(rule_counts.rename_axis("Rule").reset_index(name="Records"), x="Records", y="Rule", orientation="h", title="Sample rule-wise trigger counts"), use_container_width=True)

with st.expander("Rule explanations and weights"):
    st.dataframe(pd.DataFrame([
        {"Rule": name, "Weight": weight, "Interpretation": RULES[column]}
        for column, name, weight in [
            ("R1_diabetes_no_tx", "R1 diabetes without treatment", RULE_WEIGHTS["R1_diabetes_no_tx"]),
            ("R1_htn_no_tx", "R1 hypertension without treatment", RULE_WEIGHTS["R1_htn_no_tx"]),
            ("R2_tx_without_diabetes", "R2 treatment without diabetes", RULE_WEIGHTS["R2_tx_without_diabetes"]),
            ("R2_tx_without_htn", "R2 treatment without hypertension", RULE_WEIGHTS["R2_tx_without_htn"]),
            ("R3_high_glucose_no_dx", "R3 high glucose without diagnosis", RULE_WEIGHTS["R3_high_glucose_no_dx"]),
            ("R4_diabetes_no_meds", "R4 diabetes without medication", RULE_WEIGHTS["R4_diabetes_no_meds"]),
        ]
    ]), use_container_width=True, hide_index=True)

col1, col2, col3 = st.columns([1, 1, 2])
risk_choice = col1.selectbox("Risk category", ["All", *dist.index.tolist()])
rule_choice = col2.selectbox("Triggered rule", ["Any", *RULES.keys()], format_func=lambda value: "Any triggered rule" if value == "Any" else RULES[value])
case_search = col3.text_input("Search CASEID")
filtered = data
if risk_choice != "All":
    filtered = filtered[filtered["risk_category"] == risk_choice]
if rule_choice != "Any":
    filtered = filtered[filtered[rule_choice].astype(bool)]
if case_search.strip():
    filtered = filtered[filtered["CASEID"].astype(str).str.contains(case_search.strip(), case=False, na=False)]
st.metric("Matching sample records", f"{len(filtered):,}")
st.dataframe(filtered, use_container_width=True, hide_index=True)
st.download_button(
    "Export filtered NFHS records",
    filtered.to_csv(index=False).encode("utf-8"),
    file_name="nfhs5_filtered_screening_records.csv",
    mime="text/csv",
)
