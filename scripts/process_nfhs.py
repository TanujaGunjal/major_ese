import os
import pyreadstat
import pandas as pd
import numpy as np

sav_path = r"c:\Users\Rutuja J Kokane\Downloads\IAIR7ESV\IAIR7EFL.SAV"
os.makedirs("data/nfhs", exist_ok=True)
output_sample_path = "data/nfhs/nfhs5_screened_sample.parquet"
output_summary_path = "data/nfhs/nfhs5_rule_summary.json"

required_variables = [
    "CASEID",
    "S728A",
    "S728AB",
    "S728B",
    "S728BB",
    "SB55",
    "SB56",
    "SB57",
    "SB74"
]

print("Reading NFHS-5 sample (30,000 rows)...")
df, meta = pyreadstat.read_sav(
    sav_path,
    usecols=required_variables,
    row_limit=35000
)

print(f"Loaded {len(df)} records. Applying 6 medical consistency screening rules...")

# Rule 1: Diagnosis without treatment
df["R1_diabetes_no_tx"] = ((df["S728A"] == 1) & (df["S728AB"] == 0)).astype(int)
df["R1_htn_no_tx"] = ((df["S728B"] == 1) & (df["S728BB"] == 0)).astype(int)

# Rule 2: Treatment without diagnosis
df["R2_tx_without_diabetes"] = ((df["S728AB"] == 1) & (df["S728A"] == 0)).astype(int)
df["R2_tx_without_htn"] = ((df["S728BB"] == 1) & (df["S728B"] == 0)).astype(int)

# Rule 3: High glucose without diabetes diagnosis
df["R3_high_glucose_no_dx"] = (
    (df["SB74"].notna()) &
    (~df["SB74"].isin([995, 996, 998])) &
    (df["SB74"] >= 200) &
    (df["S728A"] == 0)
).astype(int)

# Rule 4: Diabetes without prescribed medication
df["R4_diabetes_no_meds"] = ((df["S728A"] == 1) & (df["SB57"] == 0)).astype(int)

# Weighted screening risk score
df["risk_score"] = (
    df["R1_diabetes_no_tx"] * 3 +
    df["R1_htn_no_tx"] * 3 +
    df["R2_tx_without_diabetes"] * 5 +
    df["R2_tx_without_htn"] * 5 +
    df["R3_high_glucose_no_dx"] * 4 +
    df["R4_diabetes_no_meds"] * 2
)

def assign_risk_category(score):
    if score == 0:
        return "Clean"
    elif score <= 2:
        return "Low"
    elif score <= 4:
        return "Medium"
    elif score <= 6:
        return "High"
    else:
        return "Critical"

df["risk_category"] = df["risk_score"].apply(assign_risk_category)

rule_columns = [
    "R1_diabetes_no_tx",
    "R1_htn_no_tx",
    "R2_tx_without_diabetes",
    "R2_tx_without_htn",
    "R3_high_glucose_no_dx",
    "R4_diabetes_no_meds"
]

df["total_rules_triggered"] = df[rule_columns].sum(axis=1)

print("\n--- SAMPLE SCREENING SUMMARY ---")
print("Total records:", len(df))
print("Flagged records (score > 0):", (df["risk_score"] > 0).sum())
print("Risk category distribution:")
print(df["risk_category"].value_counts())

for r in rule_columns:
    print(f"{r}: {df[r].sum()}")

# Save sample to parquet
df.to_parquet(output_sample_path, index=False)
print(f"\nSaved sample to {output_sample_path}")

# Also record the full notebook benchmark statistics from Phase 3 presentation:
full_benchmark = {
    "total_records_screened": 724115,
    "total_flagged_records": 20514,
    "flagged_percentage": 2.83,
    "high_and_critical_cases": 2799,
    "clean_count": 703601,
    "clean_percentage": 97.17,
    "sample_records": len(df),
    "sample_flagged": int((df["risk_score"] > 0).sum()),
    "sample_distribution": df["risk_category"].value_counts().to_dict(),
    "sample_rule_counts": {r: int(df[r].sum()) for r in rule_columns}
}

import json
with open(output_summary_path, "w") as f:
    json.dump(full_benchmark, f, indent=2)
print(f"Saved benchmark summary to {output_summary_path}")
