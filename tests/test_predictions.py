"""
Automated unit tests for ML models, prediction consistency, thresholds, and SHAP explainability.
"""

import pytest
from src.predict import predictor
from src.explainability import explain_provider_prediction, get_global_feature_importance
from config.settings import FEATURE_COLUMNS


def test_models_loaded():
    """Verify that all three trained models (RF, LR, XGBoost) are loaded."""
    assert predictor.rf_model is not None, "Random Forest model is not loaded"
    assert predictor.lr_model is not None, "Logistic Regression model is not loaded"
    assert predictor.xgb_model is not None, "XGBoost model is not loaded"


def test_prediction_output_format():
    """Verify prediction output structure and probability bounds."""
    sample_features = {
        "TotalClaims": 100,
        "TotalReimbursement": 50000.0,
        "AverageReimbursement": 500.0,
        "UniquePatients": 80,
        "InpatientClaims": 10,
        "InpatientReimbursement": 30000.0,
        "InpatientAvgReimbursement": 3000.0,
        "OutpatientClaims": 90,
        "OutpatientReimbursement": 20000.0,
        "OutpatientAvgReimbursement": 222.0
    }

    pred = predictor.predict_provider(sample_features, model_name="Random Forest")
    assert "probability" in pred
    assert 0.0 <= pred["probability"] <= 1.0
    assert pred["risk_tier"] in ["Low", "Medium", "High"]
    assert isinstance(pred["is_flagged"], bool)


def test_shap_explanation_generation():
    """Verify local SHAP contributions computation."""
    sample_features = {
        "TotalClaims": 250,
        "TotalReimbursement": 200000.0,
        "AverageReimbursement": 800.0,
        "UniquePatients": 180,
        "InpatientClaims": 25,
        "InpatientReimbursement": 150000.0,
        "InpatientAvgReimbursement": 6000.0,
        "OutpatientClaims": 225,
        "OutpatientReimbursement": 50000.0,
        "OutpatientAvgReimbursement": 222.0
    }

    explanation = explain_provider_prediction(sample_features)
    assert "contributions" in explanation
    assert len(explanation["contributions"]) == len(FEATURE_COLUMNS)
    assert "primary_reasons" in explanation
    assert len(explanation["primary_reasons"]) > 0


def test_global_feature_importance():
    """Verify global feature importance ordering."""
    imp_df = get_global_feature_importance()
    assert not imp_df.empty
    assert len(imp_df) == len(FEATURE_COLUMNS)
    assert "TotalReimbursement" in imp_df["Feature"].values
