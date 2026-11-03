import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_loader import load_eval_predictions, load_model_metadata, load_providers_df
from src.explainability import explain_provider_prediction, get_global_feature_importance
from src.styles import apply_custom_styles, render_header
from config.settings import BASE_DIR, FEATURE_COLUMNS

apply_custom_styles()
render_header("ML Model Performance & Explainability", "Evaluate provider-level models on the saved labeled evaluation predictions.")
evaluation = load_eval_predictions()
if evaluation.empty or "ActualLabel" not in evaluation:
    st.error("The evaluation prediction dataset or ActualLabel column is unavailable.")
    st.stop()

score_columns = {
    "Random Forest": "RF_Probability",
    "Logistic Regression": "LR_Probability",
    "XGBoost": "XGB_Probability",
}
model_name = st.selectbox("Evaluation model", list(score_columns))
score_col = score_columns[model_name]
if score_col not in evaluation:
    st.error(f"Required prediction score `{score_col}` is missing from the evaluation dataset.")
    st.stop()

y_true = pd.to_numeric(evaluation["ActualLabel"], errors="coerce")
scores = pd.to_numeric(evaluation[score_col], errors="coerce")
valid = y_true.notna() & scores.notna()
y_true = y_true[valid].astype(int).to_numpy()
scores = scores[valid].astype(float).to_numpy()
if len(y_true) == 0 or len(np.unique(y_true)) < 2:
    st.error("Evaluation requires valid prediction scores and both target classes.")
    st.stop()

threshold = st.slider("Decision threshold", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
y_pred = (scores >= threshold).astype(int)
metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "Precision": precision_score(y_true, y_pred, zero_division=0),
    "Recall": recall_score(y_true, y_pred, zero_division=0),
    "F1": f1_score(y_true, y_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_true, scores),
    "Average Precision": average_precision_score(y_true, scores),
}
columns = st.columns(6)
for column, (name, value) in zip(columns, metrics.items()):
    column.metric(name, f"{value:.3f}")

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
fig = px.imshow(
    cm,
    text_auto=True,
    x=["Predicted negative", "Predicted positive"],
    y=["Actual negative", "Actual positive"],
    color_continuous_scale="Blues",
    title=f"{model_name} confusion matrix at threshold {threshold:.2f}",
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"Metrics recalculated from {len(y_true):,} independent 80/20 stratified holdout rows with non-missing labels and scores. "
    "ROC-AUC and Average Precision are threshold-independent."
)

metadata = load_model_metadata()
cv_results = metadata.get("cv_results", {})
comparison_path = BASE_DIR / "data" / "medicare" / "final_model_comparison.csv"
if cv_results and comparison_path.exists():
    comparison_csv = pd.read_csv(comparison_path).set_index("Model")
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Average Precision"]
    comparisons = []
    for model_key, stored in cv_results.items():
        if model_key not in comparison_csv.index:
            continue
        csv_row = comparison_csv.loc[model_key]
        for metric in metric_names:
            stored_value = float(stored[metric])
            csv_value = float(csv_row[metric])
            comparisons.append({
                "Model": model_key,
                "Metric": metric,
                "model_metadata.json": stored_value,
                "final_model_comparison.csv": csv_value,
                "Difference": round(stored_value - csv_value, 4),
                "Match within 0.001": abs(stored_value - csv_value) <= 0.001,
            })
    comparison_frame = pd.DataFrame(comparisons)
    st.markdown("#### Stored cross-validation results cross-check")
    st.caption("The dashboard compares both existing artifacts; their values are not silently merged or hardcoded.")
    st.dataframe(comparison_frame, use_container_width=True, hide_index=True)
    if not comparison_frame["Match within 0.001"].all():
        st.warning(
            "The supplied CV metadata and comparison CSV disagree on one or more metrics. "
            "Treat these historical CV summaries as separate recorded runs; the threshold metrics above use the regenerated, independent holdout predictions."
        )
elif cv_results:
    st.warning(f"Cross-validation CSV is unavailable: {comparison_path}")

st.markdown("#### Feature importance")
importance = get_global_feature_importance()
if not importance.empty:
    st.bar_chart(importance.set_index("Feature")["Importance"])
else:
    st.info("Random Forest feature importance is unavailable.")

st.markdown("#### Provider-level SHAP explanation")
providers = load_providers_df()
if providers.empty or "Provider" not in providers:
    st.warning("Provider feature records are unavailable; model evaluation above remains available, but provider-level SHAP cannot be shown.")
    st.stop()
provider_id = st.selectbox("Select provider for local explanation", providers["Provider"].astype(str).tolist())
selected = providers.loc[providers["Provider"].astype(str) == provider_id].iloc[0]
explanation = explain_provider_prediction({feature: float(selected.get(feature, 0) or 0) for feature in FEATURE_COLUMNS})
if explanation.get("error"):
    st.info(explanation["error"])
else:
    st.caption("Current SHAP integration explains the Random Forest provider score; it is not claim-level attribution.")
    contributions = pd.DataFrame(explanation["contributions"])
    st.bar_chart(contributions.set_index("display_name")["shap_value"])
    st.dataframe(contributions[["feature", "value", "shap_value", "direction", "description"]], use_container_width=True, hide_index=True)
