"""
Configuration settings for the AI-Based Healthcare Claims Fraud Detection System.
Centralizes paths, database configurations, model parameters, and visual theming.
"""

from pathlib import Path
import os

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_MEDICARE_DIR = DATA_DIR / "medicare"
PROCESSED_DIR = DATA_DIR / "processed"
NFHS_DIR = DATA_DIR / "nfhs"
MODELS_DIR = BASE_DIR / "models"

# Database files
DUCKDB_PATH = DATA_DIR / "claims_fraud.duckdb"
SQLITE_DB_PATH = DATA_DIR / "investigation_workspace.db"

# Parquet files
CLAIMS_PARQUET = PROCESSED_DIR / "claims.parquet"
BENEFICIARIES_PARQUET = PROCESSED_DIR / "beneficiaries.parquet"
PROVIDERS_PARQUET = PROCESSED_DIR / "providers.parquet"
EVAL_PREDICTIONS_PARQUET = PROCESSED_DIR / "eval_predictions.parquet"
NFHS_SAMPLE_PARQUET = NFHS_DIR / "nfhs5_screened_sample.parquet"
NFHS_SUMMARY_JSON = NFHS_DIR / "nfhs5_rule_summary.json"

# Model paths
RF_MODEL_PATH = MODELS_DIR / "rf_provider_model.joblib"
LR_MODEL_PATH = MODELS_DIR / "lr_provider_model.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgb_provider_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Decision thresholds
DEFAULT_FRAUD_THRESHOLD = 0.30  # Screening cutoff for potential fraud risk
DEFAULT_HIGH_RISK_THRESHOLD = 0.70  # Prioritized manual investigation cutoff

# Medicare source reimbursements are USD; INR displays are illustrative only.
USD_TO_INR_RATE = float(os.getenv("CLAIMS_DASHBOARD_USD_TO_INR", "83.0"))
USD_TO_INR_EFFECTIVE_DATE = os.getenv(
    "CLAIMS_DASHBOARD_FX_EFFECTIVE_DATE",
    "2026-10-28 (illustrative demonstration setting; not a current market quote)"
)
if USD_TO_INR_RATE <= 0:
    raise ValueError("CLAIMS_DASHBOARD_USD_TO_INR must be greater than zero.")

# Feature columns (10 provider-level features validated in Phase 3)
FEATURE_COLUMNS = [
    "TotalClaims",
    "TotalReimbursement",
    "AverageReimbursement",
    "UniquePatients",
    "InpatientClaims",
    "InpatientReimbursement",
    "InpatientAvgReimbursement",
    "OutpatientClaims",
    "OutpatientReimbursement",
    "OutpatientAvgReimbursement"
]

# Case status options for investigator workspace
CASE_STATUS_OPTIONS = ["New", "Under Review", "Escalated", "Reviewed/Closed", "Closed - False Positive", "Closed - Suspected Fraud"]
CASE_PRIORITY_OPTIONS = ["Low", "Medium", "High", "Critical"]

# Custom UI Theme
THEME = {
    "primary": "#0D5C75",       # Deep Teal
    "secondary": "#1E293B",     # Charcoal Navy
    "accent": "#0284C7",        # Medical Blue
    "warning": "#F59E0B",       # Amber
    "danger": "#EF4444",        # Crimson Red
    "success": "#10B981",       # Emerald Green
    "background": "#F8FAFC",    # Clean Light Grey
    "card_bg": "#FFFFFF",       # Pure White
    "border": "#E2E8F0"         # Border Grey
}
