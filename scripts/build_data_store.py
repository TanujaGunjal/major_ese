import os
import json
import duckdb
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import shap

print("=== STARTING FULL DATA STORE & MODEL PIPELINE BUILD ===")

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

raw_dir = "data/medicare"
train_labels_file = os.path.join(raw_dir, "Train-1542865627584.csv")
inpatient_file = os.path.join(raw_dir, "Train_Inpatientdata-1542865627584.csv")
outpatient_file = os.path.join(raw_dir, "Train_Outpatientdata-1542865627584.csv")
beneficiary_file = os.path.join(raw_dir, "Train_Beneficiarydata-1542865627584.csv")

# 1. LOAD RAW DATA
print("1. Loading raw Medicare tables...")
labels_df = pd.read_csv(train_labels_file)
inpatient_df = pd.read_csv(inpatient_file)
outpatient_df = pd.read_csv(outpatient_file)
beneficiary_df = pd.read_csv(beneficiary_file)

print(f"Loaded {len(labels_df):,} providers, {len(inpatient_df):,} inpatient claims, {len(outpatient_df):,} outpatient claims, {len(beneficiary_df):,} beneficiaries.")

# 2. FEATURE ENGINEERING AT PROVIDER LEVEL (10 FEATURES)
print("\n2. Engineering provider-level features...")
all_claims_light = pd.concat([
    inpatient_df[['Provider', 'ClaimID', 'BeneID', 'InscClaimAmtReimbursed']],
    outpatient_df[['Provider', 'ClaimID', 'BeneID', 'InscClaimAmtReimbursed']]
], ignore_index=True)

claims_per_provider = all_claims_light.groupby("Provider").agg(
    TotalClaims=("ClaimID", "count"),
    TotalReimbursement=("InscClaimAmtReimbursed", "sum"),
    AverageReimbursement=("InscClaimAmtReimbursed", "mean"),
    UniquePatients=("BeneID", "nunique")
).reset_index()

inpatient_prov = inpatient_df.groupby("Provider").agg(
    InpatientClaims=("ClaimID", "count"),
    InpatientReimbursement=("InscClaimAmtReimbursed", "sum"),
    InpatientAvgReimbursement=("InscClaimAmtReimbursed", "mean")
).reset_index()

outpatient_prov = outpatient_df.groupby("Provider").agg(
    OutpatientClaims=("ClaimID", "count"),
    OutpatientReimbursement=("InscClaimAmtReimbursed", "sum"),
    OutpatientAvgReimbursement=("InscClaimAmtReimbursed", "mean")
).reset_index()

provider_features = labels_df.merge(claims_per_provider, on="Provider", how="left")
provider_features = provider_features.merge(inpatient_prov, on="Provider", how="left")
provider_features = provider_features.merge(outpatient_prov, on="Provider", how="left")
provider_features = provider_features.fillna(0)
provider_features["FraudLabel"] = provider_features["PotentialFraud"].map({"No": 0, "Yes": 1})

feature_cols = [
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

X = provider_features[feature_cols]
y = provider_features["FraudLabel"]

# 3. TRAIN & EVALUATE MODELS (RF, LR, XGBoost)
print("\n3. Training & cross-validating models (RF, LR, XGBoost)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "ROC_AUC": "roc_auc",
    "Average_Precision": "average_precision"
}

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
])
scale_pos = float((y == 0).sum() / (y == 1).sum())
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric="logloss",
    random_state=42
)

cv_comparison = {}
for name, model in [("Random Forest", rf_model), ("Logistic Regression", lr_pipe), ("XGBoost", xgb_model)]:
    res = cross_validate(model, X, y, cv=cv, scoring=scoring)
    cv_comparison[name] = {
        "Accuracy": float(np.round(res["test_Accuracy"].mean(), 4)),
        "Precision": float(np.round(res["test_Precision"].mean(), 4)),
        "Recall": float(np.round(res["test_Recall"].mean(), 4)),
        "F1 Score": float(np.round(res["test_F1"].mean(), 4)),
        "ROC-AUC": float(np.round(res["test_ROC_AUC"].mean(), 4)),
        "Average Precision": float(np.round(res["test_Average_Precision"].mean(), 4))
    }
    print(f"  {name}: {cv_comparison[name]}")

# Train models on full dataset for authoritative dashboard predictions
print("\nFitting final production models...")
rf_model.fit(X, y)
lr_pipe.fit(X, y)
xgb_model.fit(X, y)

# Compute probabilities
provider_features["RF_FraudProbability"] = rf_model.predict_proba(X)[:, 1]
provider_features["LR_FraudProbability"] = lr_pipe.predict_proba(X)[:, 1]
provider_features["XGB_FraudProbability"] = xgb_model.predict_proba(X)[:, 1]

# Primary model: Random Forest (consistent with Phase 3 prototype)
provider_features["FraudProbability"] = provider_features["RF_FraudProbability"]

def assign_risk_tier(prob):
    if prob >= 0.70:
        return "High"
    elif prob >= 0.30:
        return "Medium"
    else:
        return "Low"

provider_features["RiskLevel"] = provider_features["FraudProbability"].apply(assign_risk_tier)
provider_features["FraudPrediction"] = (provider_features["FraudProbability"] >= 0.30).map({True: "Fraud Risk", False: "Not Flagged"})

# Save models
joblib.dump(rf_model, "models/rf_provider_model.joblib")
joblib.dump(lr_pipe, "models/lr_provider_model.joblib")
joblib.dump(xgb_model, "models/xgb_provider_model.joblib")

# Compute SHAP explainer on a background sample for instant local explanations
print("Pre-computing SHAP TreeExplainer on Random Forest...")
rf_explainer = shap.TreeExplainer(rf_model)
# Sample 200 background rows for fast summary
background_sample = X.sample(n=min(300, len(X)), random_state=42)
sample_shap_values = rf_explainer.shap_values(background_sample)
if isinstance(sample_shap_values, list):
    sample_shap_for_class1 = sample_shap_values[1]
elif sample_shap_values.ndim == 3:
    sample_shap_for_class1 = sample_shap_values[:, :, 1]
else:
    sample_shap_for_class1 = sample_shap_values

# Save model metadata
meta = {
    "feature_columns": feature_cols,
    "cv_results": cv_comparison,
    "thresholds": {"fraud_threshold": 0.30, "high_risk_threshold": 0.70},
    "total_providers": len(provider_features),
    "fraud_positive_count": int(y.sum()),
    "fraud_positive_pct": float(np.round(y.mean() * 100, 2)),
    "risk_level_counts": provider_features["RiskLevel"].value_counts().to_dict(),
    "rf_feature_importances": dict(zip(feature_cols, [float(v) for v in rf_model.feature_importances_]))
}

with open("models/model_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved models and metadata.")

# 4. PREPARE BENEFICIARY PARQUET
print("\n4. Formatting Beneficiaries...")
beneficiary_df["DOB"] = pd.to_datetime(beneficiary_df["DOB"], errors="coerce")
beneficiary_df["DOD"] = pd.to_datetime(beneficiary_df["DOD"], errors="coerce")
ref_date = pd.to_datetime("2009-12-31") # Dataset reference year
beneficiary_df["Age"] = ((ref_date - beneficiary_df["DOB"]).dt.days / 365.25).round().astype(int)
beneficiary_df["Gender_Desc"] = beneficiary_df["Gender"].map({1: "Male", 2: "Female"}).fillna("Unknown")
beneficiary_df.to_parquet("data/processed/beneficiaries.parquet", index=False)

# 5. PREPARE COMBINED CLAIMS WITH SCREENING INDICATORS
print("\n5. Combining claims & engineering claim-level screening indicators...")
inpatient_df["ClaimType"] = "Inpatient"
outpatient_df["ClaimType"] = "Outpatient"

# Standardize date formats
for df in [inpatient_df, outpatient_df]:
    df["ClaimStartDt"] = pd.to_datetime(df["ClaimStartDt"], errors="coerce")
    df["ClaimEndDt"] = pd.to_datetime(df["ClaimEndDt"], errors="coerce")
    df["InscClaimAmtReimbursed"] = pd.to_numeric(df["InscClaimAmtReimbursed"], errors="coerce").fillna(0.0)
    df["DeductibleAmtPaid"] = pd.to_numeric(df["DeductibleAmtPaid"], errors="coerce").fillna(0.0)
    df["ClaimDurationDays"] = (df["ClaimEndDt"] - df["ClaimStartDt"]).dt.days + 1

inpatient_df["AdmissionDt"] = pd.to_datetime(inpatient_df["AdmissionDt"], errors="coerce")
inpatient_df["DischargeDt"] = pd.to_datetime(inpatient_df["DischargeDt"], errors="coerce")
inpatient_df["LengthOfStayDays"] = (inpatient_df["DischargeDt"] - inpatient_df["AdmissionDt"]).dt.days
outpatient_df["AdmissionDt"] = pd.NaT
outpatient_df["DischargeDt"] = pd.NaT
outpatient_df["DiagnosisGroupCode"] = None
outpatient_df["ClmAdmitDiagnosisCode"] = None
outpatient_df["LengthOfStayDays"] = 0

common_cols = [
    "ClaimID", "BeneID", "ClaimType", "Provider", "ClaimStartDt", "ClaimEndDt",
    "InscClaimAmtReimbursed", "DeductibleAmtPaid", "ClaimDurationDays",
    "AdmissionDt", "DischargeDt", "LengthOfStayDays",
    "DiagnosisGroupCode", "ClmAdmitDiagnosisCode",
    "AttendingPhysician", "OperatingPhysician", "OtherPhysician",
    "ClmDiagnosisCode_1", "ClmDiagnosisCode_2", "ClmDiagnosisCode_3", "ClmDiagnosisCode_4", "ClmDiagnosisCode_5",
    "ClmDiagnosisCode_6", "ClmDiagnosisCode_7", "ClmDiagnosisCode_8", "ClmDiagnosisCode_9", "ClmDiagnosisCode_10",
    "ClmProcedureCode_1", "ClmProcedureCode_2", "ClmProcedureCode_3", "ClmProcedureCode_4", "ClmProcedureCode_5", "ClmProcedureCode_6"
]

all_claims = pd.concat([inpatient_df[common_cols], outpatient_df[common_cols]], ignore_index=True)

# Add diagnosis count and procedure count
diag_cols = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
proc_cols = [f"ClmProcedureCode_{i}" for i in range(1, 7)]

all_claims["DiagnosisCount"] = all_claims[diag_cols].notna().sum(axis=1)
all_claims["ProcedureCount"] = all_claims[proc_cols].notna().sum(axis=1)

# Mechanism A: Transparent Claim-Level Screening Indicators
print("Calculating screening indicators...")
# Indicator 1: Reimbursement Outlier (>99th percentile for claim type)
inpatient_p99 = float(all_claims[all_claims["ClaimType"] == "Inpatient"]["InscClaimAmtReimbursed"].quantile(0.99))
outpatient_p99 = float(all_claims[all_claims["ClaimType"] == "Outpatient"]["InscClaimAmtReimbursed"].quantile(0.99))

all_claims["Flag_ReimbursementOutlier"] = False
all_claims.loc[(all_claims["ClaimType"] == "Inpatient") & (all_claims["InscClaimAmtReimbursed"] >= inpatient_p99), "Flag_ReimbursementOutlier"] = True
all_claims.loc[(all_claims["ClaimType"] == "Outpatient") & (all_claims["InscClaimAmtReimbursed"] >= outpatient_p99), "Flag_ReimbursementOutlier"] = True

# Indicator 2: Duplicate billing flag (same BeneID, same Provider, same ClaimStartDt, multiple claims)
dup_keys = all_claims.duplicated(subset=["BeneID", "Provider", "ClaimStartDt"], keep=False)
all_claims["Flag_DuplicateBilling"] = dup_keys

# Indicator 3: Duration inconsistency (Discharge date < Admission date OR ClaimDurationDays < 0)
all_claims["Flag_DurationAnomaly"] = (all_claims["ClaimDurationDays"] < 0) | (all_claims["LengthOfStayDays"] < 0)

# Indicator 4: High procedure complexity (>3 procedures on outpatient claim)
all_claims["Flag_ComplexProcedure"] = (all_claims["ClaimType"] == "Outpatient") & (all_claims["ProcedureCount"] >= 3)

# Indicator 5: High frequency billing (>10 claims for same beneficiary)
bene_counts = all_claims["BeneID"].value_counts()
high_freq_benes = set(bene_counts[bene_counts >= 15].index)
all_claims["Flag_HighFreqBeneficiary"] = all_claims["BeneID"].isin(high_freq_benes)

# Sum total screening flags
screening_flag_cols = ["Flag_ReimbursementOutlier", "Flag_DuplicateBilling", "Flag_DurationAnomaly", "Flag_ComplexProcedure", "Flag_HighFreqBeneficiary"]
all_claims["ScreeningFlagsCount"] = all_claims[screening_flag_cols].sum(axis=1)

# Merge provider risk context
prov_risk_map = provider_features.set_index("Provider")
all_claims["ProviderFraudProbability"] = all_claims["Provider"].map(prov_risk_map["FraudProbability"]).fillna(0.0)
all_claims["ProviderRiskLevel"] = all_claims["Provider"].map(prov_risk_map["RiskLevel"]).fillna("Low")
all_claims["ProviderPotentialFraud"] = all_claims["Provider"].map(prov_risk_map["PotentialFraud"]).fillna("No")

# AI Review Recommendation (Decision Support Rule)
def assign_review_recommendation(row):
    if row["ProviderRiskLevel"] == "High" and row["InscClaimAmtReimbursed"] > 0:
        return "Manual Investigation"
    elif row["ProviderRiskLevel"] == "High" or row["ScreeningFlagsCount"] >= 2:
        return "Risk Review"
    elif row["ScreeningFlagsCount"] == 1 or row["ProviderRiskLevel"] == "Medium":
        return "Priority Review"
    else:
        return "Routine Review"

all_claims["AIReviewRecommendation"] = all_claims.apply(assign_review_recommendation, axis=1)

print(f"Total claims processed: {len(all_claims):,}")
print("Screening flag counts distribution:")
print(all_claims["ScreeningFlagsCount"].value_counts())
print("\nAI Review Recommendations distribution:")
print(all_claims["AIReviewRecommendation"].value_counts())

# Save Parquet files
print("\nWriting Parquet tables...")
provider_features.to_parquet("data/processed/providers.parquet", index=False)
all_claims.to_parquet("data/processed/claims.parquet", index=False)

# 6. BUILD DUCKDB DATABASE FOR ULTRA-FAST INDEXED SEARCH
print("\n6. Initializing DuckDB database with indexes...")
db_path = "data/claims_fraud.duckdb"
if os.path.exists(db_path):
    os.remove(db_path)

con = duckdb.connect(db_path)
con.execute("CREATE TABLE providers AS SELECT * FROM 'data/processed/providers.parquet'")
con.execute("CREATE TABLE beneficiaries AS SELECT * FROM 'data/processed/beneficiaries.parquet'")
con.execute("CREATE TABLE claims AS SELECT * FROM 'data/processed/claims.parquet'")

# Create indexes
con.execute("CREATE INDEX idx_claims_claimid ON claims (ClaimID)")
con.execute("CREATE INDEX idx_claims_beneid ON claims (BeneID)")
con.execute("CREATE INDEX idx_claims_provider ON claims (Provider)")
con.execute("CREATE INDEX idx_beneficiaries_beneid ON beneficiaries (BeneID)")
con.execute("CREATE INDEX idx_providers_provider ON providers (Provider)")

con.close()
print("DuckDB database created and indexed successfully at data/claims_fraud.duckdb")

print("\n=== DATA STORE & MODEL PIPELINE BUILD COMPLETE! ===")
