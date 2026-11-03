import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

train_labels = pd.read_csv("data/medicare/Train-1542865627584.csv")
train_inpatient = pd.read_csv("data/medicare/Train_Inpatientdata-1542865627584.csv")
train_outpatient = pd.read_csv("data/medicare/Train_Outpatientdata-1542865627584.csv")

all_claims = pd.concat([
    train_inpatient[['Provider', 'ClaimID', 'BeneID', 'InscClaimAmtReimbursed']],
    train_outpatient[['Provider', 'ClaimID', 'BeneID', 'InscClaimAmtReimbursed']]
], ignore_index=True)

claims_per_provider = all_claims.groupby("Provider").agg(
    TotalClaims=("ClaimID", "count"),
    TotalReimbursement=("InscClaimAmtReimbursed", "sum"),
    AverageReimbursement=("InscClaimAmtReimbursed", "mean"),
    UniquePatients=("BeneID", "nunique")
).reset_index()

inpatient_features = train_inpatient.groupby("Provider").agg(
    InpatientClaims=("ClaimID", "count"),
    InpatientReimbursement=("InscClaimAmtReimbursed", "sum"),
    InpatientAvgReimbursement=("InscClaimAmtReimbursed", "mean")
).reset_index()

outpatient_features = train_outpatient.groupby("Provider").agg(
    OutpatientClaims=("ClaimID", "count"),
    OutpatientReimbursement=("InscClaimAmtReimbursed", "sum"),
    OutpatientAvgReimbursement=("InscClaimAmtReimbursed", "mean")
).reset_index()

provider_data = train_labels.merge(claims_per_provider, on="Provider", how="left")
provider_data = provider_data.merge(inpatient_features, on="Provider", how="left")
provider_data = provider_data.merge(outpatient_features, on="Provider", how="left")
provider_data = provider_data.fillna(0)
provider_data["FraudLabel"] = provider_data["PotentialFraud"].map({"No": 0, "Yes": 1})

feature_columns = [
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

X = provider_data[feature_columns]
y = provider_data["FraudLabel"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "ROC_AUC": "roc_auc",
    "Average_Precision": "average_precision"
}

rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
])
scale_pos = (y == 0).sum() / (y == 1).sum()
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

models = {
    "Random Forest": rf_model,
    "Logistic Regression": lr_pipe,
    "XGBoost": xgb_model
}

results = []
for name, model in models.items():
    cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring)
    results.append({
        "Model": name,
        "Accuracy": round(cv_res["test_Accuracy"].mean(), 4),
        "Precision": round(cv_res["test_Precision"].mean(), 4),
        "Recall": round(cv_res["test_Recall"].mean(), 4),
        "F1 Score": round(cv_res["test_F1"].mean(), 4),
        "ROC-AUC": round(cv_res["test_ROC_AUC"].mean(), 4),
        "Average Precision": round(cv_res["test_Average_Precision"].mean(), 4)
    })

res_df = pd.DataFrame(results)
print("=== 5-FOLD CROSS VALIDATION RESULTS ===")
print(res_df.to_string(index=False))

# Now let's train hold-out test (80:20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print("\nHold-out train/test split:", X_train.shape, X_test.shape)
rf_model.fit(X_train, y_train)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n--- THRESHOLD ANALYSIS (RF on Holdout Test) ---")
for t in [0.20, 0.30, 0.40, 0.50]:
    y_pred_t = (y_prob >= t).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score
    p = precision_score(y_test, y_pred_t, zero_division=0)
    r = recall_score(y_test, y_pred_t, zero_division=0)
    f = f1_score(y_test, y_pred_t, zero_division=0)
    print(f"Threshold: {t:.2f} | Precision: {p*100:.1f}% | Recall: {r*100:.1f}% | F1: {f*100:.1f}%")
