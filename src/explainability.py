"""
Explainable AI (XAI) module providing SHAP explanations and human-readable investigation reasons.
Explains why a provider received an elevated risk score and how features contributed.
"""

import pandas as pd
import numpy as np
import shap
from typing import Dict, Any, List, Optional
from src.predict import predictor
from config.settings import FEATURE_COLUMNS


# Human-friendly descriptions for each engineered feature
FEATURE_EXPLANATIONS = {
    "TotalClaims": "Total number of claims submitted across inpatient and outpatient care.",
    "TotalReimbursement": "Cumulative insurance reimbursement requested across all claims.",
    "AverageReimbursement": "Mean reimbursement amount per claim submitted by this provider.",
    "UniquePatients": "Number of unique beneficiaries treated by this provider.",
    "InpatientClaims": "Volume of acute hospital inpatient admissions billed.",
    "InpatientReimbursement": "Total reimbursement collected for inpatient admissions.",
    "InpatientAvgReimbursement": "Average reimbursement collected per inpatient stay.",
    "OutpatientClaims": "Volume of outpatient visits, procedures, or consultations billed.",
    "OutpatientReimbursement": "Total reimbursement collected for outpatient services.",
    "OutpatientAvgReimbursement": "Average reimbursement collected per outpatient encounter."
}


def explain_provider_prediction(features_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes local feature contributions for an individual provider using SHAP TreeExplainer on Random Forest.
    Returns sorted positive and negative contributors with human-readable rationale.
    """
    if predictor.rf_model is None:
        return {"error": "Random Forest model not loaded"}

    # Format input
    row = [float(features_dict.get(col, 0.0)) for col in FEATURE_COLUMNS]
    X_single = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(predictor.rf_model)
    shap_vals = explainer.shap_values(X_single)

    if isinstance(shap_vals, list):
        # Class 1 (Potential Fraud)
        class_shap = shap_vals[1][0]
    elif shap_vals.ndim == 3:
        class_shap = shap_vals[0, :, 1]
    else:
        class_shap = shap_vals[0]

    # Combine with feature values and descriptions
    contributions = []
    for i, col in enumerate(FEATURE_COLUMNS):
        shap_val = float(class_shap[i])
        val = float(row[i])
        contributions.append({
            "feature": col,
            "display_name": col.replace("Reimbursement", " Reimb"),
            "value": val,
            "shap_value": shap_val,
            "abs_shap": abs(shap_val),
            "direction": "Pushes Risk Higher" if shap_val > 0 else "Lowers Risk Score",
            "description": FEATURE_EXPLANATIONS.get(col, "")
        })

    # Sort by absolute impact
    contributions.sort(key=lambda x: x["abs_shap"], reverse=True)

    # Synthesize plain-English investigation findings
    top_elevating = [c for c in contributions if c["shap_value"] > 0][:3]
    top_reducing = [c for c in contributions if c["shap_value"] < 0][:2]

    reasons = []
    for c in top_elevating:
        feat = c["feature"]
        val = c["value"]
        if "Reimbursement" in feat:
            reasons.append(f"Elevated {feat} (source USD {val:,.2f}) substantially increased the model's suspicion index.")
        else:
            reasons.append(f"High volume for {feat} ({val:,.0f}) drove the model toward a higher risk tier.")

    if not reasons:
        reasons.append("Provider metrics remain within ordinary baseline distribution ranges.")

    return {
        "contributions": contributions,
        "primary_reasons": reasons,
        "base_value": float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value),
        "prediction_probability": float(predictor.rf_model.predict_proba(X_single)[0, 1])
    }


def get_global_feature_importance() -> pd.DataFrame:
    """
    Returns global Gini importance from the validated Random Forest model.
    """
    if predictor.rf_model is None:
        return pd.DataFrame()

    importances = predictor.rf_model.feature_importances_
    df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": importances,
        "Description": [FEATURE_EXPLANATIONS.get(f, "") for f in FEATURE_COLUMNS]
    }).sort_values("Importance", ascending=False)

    return df
