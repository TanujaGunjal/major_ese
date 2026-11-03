"""
Machine learning inference module for healthcare fraud risk prediction.
Loads trained models (Random Forest, Logistic Regression, XGBoost) and computes probabilities.
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from config.settings import (
    RF_MODEL_PATH,
    LR_MODEL_PATH,
    XGB_MODEL_PATH,
    FEATURE_COLUMNS,
    DEFAULT_FRAUD_THRESHOLD,
    DEFAULT_HIGH_RISK_THRESHOLD
)


class ProviderRiskPredictor:
    """Wrapper class managing trained provider-level models and probability inference."""

    def __init__(self):
        self.rf_model = None
        self.lr_model = None
        self.xgb_model = None
        self._load_models()

    def _load_models(self):
        """Loads available model artifacts safely."""
        if RF_MODEL_PATH.exists():
            self.rf_model = joblib.load(RF_MODEL_PATH)
        if LR_MODEL_PATH.exists():
            self.lr_model = joblib.load(LR_MODEL_PATH)
        if XGB_MODEL_PATH.exists():
            self.xgb_model = joblib.load(XGB_MODEL_PATH)

    def predict_provider(
        self,
        features_dict: Dict[str, float],
        model_name: str = "Random Forest",
        threshold: float = DEFAULT_FRAUD_THRESHOLD,
        high_risk_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Predicts fraud risk probability for an individual provider's feature dict.
        """
        # Build DataFrame with exact feature columns
        row = [float(features_dict.get(col, 0.0)) for col in FEATURE_COLUMNS]
        X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        prob = self._get_probability(X, model_name)

        if prob >= high_risk_threshold:
            risk_tier = "High"
        elif prob >= threshold:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"

        is_flagged = bool(prob >= threshold)

        return {
            "model_used": model_name,
            "probability": float(prob),
            "percentage": float(np.round(prob * 100, 2)),
            "risk_tier": risk_tier,
            "is_flagged": is_flagged,
            "threshold_used": threshold,
            "features_input": dict(zip(FEATURE_COLUMNS, row))
        }

    def _get_probability(self, X: pd.DataFrame, model_name: str) -> float:
        """Helper to get probability from chosen model."""
        if model_name == "Logistic Regression" and self.lr_model is not None:
            return self.lr_model.predict_proba(X)[0, 1]
        elif model_name == "XGBoost" and self.xgb_model is not None:
            return self.xgb_model.predict_proba(X)[0, 1]
        elif self.rf_model is not None:
            return self.rf_model.predict_proba(X)[0, 1]
        else:
            return 0.0


# Global singleton instance
predictor = ProviderRiskPredictor()
