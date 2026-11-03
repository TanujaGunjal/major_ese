import pandas as pd
import numpy as np

train_labels = pd.read_csv("data/medicare/Train-1542865627584.csv")
train_inpatient = pd.read_csv("data/medicare/Train_Inpatientdata-1542865627584.csv")
train_outpatient = pd.read_csv("data/medicare/Train_Outpatientdata-1542865627584.csv")

print(f"Train labels shape: {train_labels.shape}")
print(f"Train inpatient shape: {train_inpatient.shape}")
print(f"Train outpatient shape: {train_outpatient.shape}")

# Combined claims
all_claims = pd.concat([
    train_inpatient[['Provider', 'ClaimID', 'BeneID', 'InscClaimAmtReimbursed']],
    train_outpatient[['Provider', 'ClaimID', 'BeneID', 'InscClaimAmtReimbursed']]
], ignore_index=True)

# Provider-level aggregations
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

# Fill providers with 0 for missing claim types
provider_data = provider_data.fillna(0)

provider_data["FraudLabel"] = provider_data["PotentialFraud"].map({"No": 0, "Yes": 1})

print("Engineered provider dataset shape:", provider_data.shape)
print("Columns:", provider_data.columns.tolist())
print(provider_data.head())
