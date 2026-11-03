# AI-Based Healthcare Claims Fraud Detection System
### Phase 4 Enterprise Decision-Support & Investigation Platform

**Final-Year Major Project | Department of Computer Engineering**  
**Bharatiya Vidya Bhavan's Sardar Patel Institute of Technology (SPIT), Mumbai**  

---

## 1. Project Overview & Phase 4 Objectives

Healthcare insurance fraud, improper billing, and clinical inconsistencies cost health systems hundreds of billions of dollars annually. In real-world claims adjudication, confirmed fraud labels exist primarily at the **provider level** (via Office of Inspector General exclusion lists, regulatory audits, and licensing sanctions) rather than as definitive, verified labels on each individual transaction.

This system bridges that gap by delivering an interactive, explainable, and multi-layered **healthcare claims investigation dashboard** designed to assist insurance investigators, auditors, and clinical reviewers in triaging, exploring, and adjudicating suspicious patterns.

### Key Capabilities
1. **Interactive Claim Explorer (Primary View)**: Sub-second search across **558,211 real Medicare claims** with granular inspection of admitting diagnoses, secondary ICD codes, surgical procedures, physician IDs, and hospital lengths of stay.
2. **Transparent Claim Screening Indicators (Mechanism A)**: Rule-based screening detecting reimbursement outliers (top 1% percentile), duplicate billing submissions, date sequence anomalies, and high-frequency billing.
3. **Provider-Level Machine Learning Risk Intelligence**: Supervised models (**Random Forest, Logistic Regression, XGBoost**) evaluated with 5-Fold Stratified Cross-Validation on 10 engineered features.
4. **Explainable AI (SHAP)**: Local TreeExplainer attribution quantifying how specific features drive suspicion scores, coupled with plain-English investigator findings.
5. **Longitudinal Beneficiary Timelines**: Complete claims chronologies for individual patients across multiple providers, combined with chronic disease profiles.
6. **Persistent Investigation Workspace**: SQLite-backed case review queue with one case per claim/provider, editable status and priority, reasons, supporting observations, dated notes, audit history, and JSON exports. Case status is not proof of fraud.
7. **Separate NFHS-5 Clinical Consistency Module**: Reuses six weighted rules on the available 35,000-record screened sample without invalid record linkage to Medicare claims. A JSON summary records nationwide benchmark figures, but the full raw benchmark is not present to independently revalidate them.

---

## 2. System Architecture & Modular Structure

```text
Major Project Dashboard/
├── app.py                          # Authoritative Streamlit application entry point
├── requirements.txt                # Production dependency specification
├── README.md                       # Comprehensive system documentation
├── .gitignore                      # Git exclusion rules
│
├── config/
│   └── settings.py                 # Paths, thresholds, color theme & database settings
│
├── src/
│   ├── data_loader.py              # Parquet, DuckDB, and JSON cached data loaders
│   ├── validation.py               # Dataset schema & integrity validation suite
│   ├── preprocessing.py            # Code parsing, currency/date formatters, condition mappings
│   ├── feature_engineering.py      # 10 provider aggregate features & claim screening flags
│   ├── claims_repository.py        # High-performance DuckDB query engine (<10ms lookups)
│   ├── predict.py                  # Model inference wrapper for RF, LR, and XGBoost
│   ├── explainability.py           # SHAP local/global feature attribution & plain-English reasons
│   ├── investigation.py            # Persistent SQLite case management, notes, & audit logs
│   ├── data_loader.py              # DuckDB, Parquet, metadata, evaluation, and NFHS loaders
│   ├── visualizations.py           # Standardized Plotly chart generators
│   └── styles.py                   # Healthcare analytics CSS theme & KPI metric cards
│
├── pages/
│   ├── 1_Overview_Dashboard.py
│   ├── 2_Claim_Investigation.py
│   ├── 3_Beneficiary_History.py
│   ├── 4_Provider_Risk_Analysis.py
│   ├── 5_Suspicious_Claim_Screening.py
│   ├── 6_ML_Model_Performance.py
│   ├── 7_NFHS_Medical_Consistency.py
│   ├── 8_Investigation_Workspace.py
│   └── 9_Data_Explorer.py
│
├── models/
│   ├── rf_provider_model.joblib    # Production Random Forest Classifier
│   ├── lr_provider_model.joblib    # StandardScaler + Logistic Regression Pipeline
│   ├── xgb_provider_model.joblib   # Tuned XGBoost Classifier (scale_pos_weight=9.69)
│   ├── model_metadata.json         # CV metrics, feature importances & thresholds
│   └── README.md                   # Model artifact documentation
│
├── data/
│   ├── medicare/                   # Raw Medicare CSV files (inpatient, outpatient, beneficiary)
│   ├── processed/                  # Optimized Parquet tables (claims, beneficiaries, providers)
│   ├── nfhs/                       # NFHS-5 screened sample & nationwide benchmark summary
│   ├── claims_fraud.duckdb         # Columnar DuckDB indexed database for rapid search
│   ├── investigation_workspace.db  # SQLite database for persistent investigator cases
│   └── README.md                   # Dataset documentation & ethical segregation rules
│
├── scripts/
│   ├── build_data_store.py         # Full pipeline runner: builds DuckDB, trains models, saves Parquet
│   └── process_nfhs.py             # Extracts & screens real NFHS-5 sample records
│
└── tests/
    ├── conftest.py                 # Pytest environment & path configuration
    ├── test_data_validation.py     # Schema, required columns & artifact integrity tests
    ├── test_claim_queries.py       # DuckDB lookup, filter, and pagination unit tests
    ├── test_predictions.py         # Model loading, probability inference & SHAP tests
    ├── test_dashboard_logic.py     # Workspace persistence, screening logic & formatting tests
    └── test_investigation_dashboard.py # Date filters and stored screening evidence
```

---

## 3. Installation & Setup Instructions

### Prerequisites
- Python 3.11, 3.12, or 3.13 (64-bit)
- 4 GB RAM minimum (8 GB recommended)
- Windows PowerShell, Linux Bash, or macOS Terminal

### A. Windows Setup (Local)

1. Open PowerShell and navigate to the project directory:
   ```powershell
   cd "C:\Users\Rutuja J Kokane\Downloads\Major Project Dashboard"
   ```

2. Create a virtual environment (if not already present):
   ```powershell
   python -m venv .venv
   ```

3. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

5. Run automated tests to verify system integrity:
   ```powershell
   python -m pytest -v tests
   ```

6. Launch the Streamlit dashboard:
   ```powershell
   streamlit run app.py
   ```
   *The application will open automatically in your browser at `http://localhost:8501`.*

   From the workspace used for this implementation, the exact command is:
   ```powershell
   cd "C:\Users\Rutuja J Kokane\Downloads\Major Project Dashboard"
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

---

### B. Linux / macOS Setup

1. Open Terminal and navigate to the folder:
   ```bash
   cd "Major Project Dashboard"
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python -m pytest -v tests
   streamlit run app.py
   ```

---

### C. Google Colab Execution Instructions

To run the complete dashboard in Google Colab with a live public URL:

```python
# Cell 1: Clone or upload project folder
!pip install -q streamlit duckdb plotly pyreadstat xgboost shap

# Cell 2: Verify datasets and build data store
!python scripts/build_data_store.py

# Cell 3: Run automated tests
!python -m pytest -v tests

# Cell 4: Launch Streamlit with public tunnel (using localtunnel or ngrok)
!npm install -g localtunnel -q
!streamlit run app.py --server.headless true --server.port 8501 &>/content/streamlit.log &
!npx localtunnel --port 8501
```

---

## 4. Machine Learning Pipeline & Validated Benchmarks

### Provider-Level Feature Engineering (10 Features)
The Medicare claims pipeline derives 10 aggregate numerical features describing each healthcare provider's clinical billing footprint:

| # | Feature Name | Description |
|---|--------------|-------------|
| 1 | `TotalClaims` | Total claims billed across inpatient and outpatient care |
| 2 | `TotalReimbursement` | Cumulative insurance reimbursement collected ($) |
| 3 | `AverageReimbursement` | Mean reimbursement amount per claim ($) |
| 4 | `UniquePatients` | Number of distinct Medicare beneficiaries treated |
| 5 | `InpatientClaims` | Volume of acute inpatient hospital admissions billed |
| 6 | `InpatientReimbursement` | Cumulative reimbursement for inpatient care ($) |
| 7 | `InpatientAvgReimbursement` | Mean reimbursement per inpatient admission ($) |
| 8 | `OutpatientClaims` | Volume of outpatient clinic visits / procedures billed |
| 9 | `OutpatientReimbursement` | Cumulative reimbursement for outpatient care ($) |
| 10 | `OutpatientAvgReimbursement` | Mean reimbursement per outpatient visit ($) |

### Model evaluation and result provenance

The project contains two historical 5-fold CV summaries: `models/model_metadata.json` and `data/medicare/final_model_comparison.csv`. They disagree on several Random Forest and XGBoost metrics, reflecting different recorded training runs/configurations. The ML page displays both sources and their per-metric differences rather than selecting one or hardcoding a merged result.

The interactive threshold metrics are calculated from `data/processed/eval_predictions.parquet`. This file is generated by `scripts/generate_eval_data.py` using an 80/20 stratified provider split; each evaluation estimator is fit on the training partition only and scores only the held-out providers. Regenerate it with:

```powershell
python scripts/generate_eval_data.py
```

The threshold slider recalculates accuracy, precision, recall, F1, and confusion-matrix counts from those holdout scores and labels. ROC-AUC and Average Precision are calculated from scores and do not depend on the selected threshold. None of these provider-level metrics describes claim-level fraud detection.

---

## 5. Claim-Level Fraud Dataset Research & Findings (For Co-Guide)

### Ground Truth Dilemma in Healthcare Fraud
A common inquiry from academic advisors is why claim-level machine learning models are not trained directly on individual claims. Research into open health data reveals a structural bottleneck:
1. **HIPAA & Privacy Regulations**: True confirmed claim-level fraud records contain confidential investigative records and patient protected health information (PHI) that cannot be published openly.
2. **Investigation Reality**: Audits and enforcement actions by the HHS-OIG and CMS target **providers** (via billing exclusions and corporate integrity agreements), meaning ground truth labels in public datasets (like Kaggle Medicare Fraud Detection and CMS SynPUF) exist **strictly at the provider level**.
3. **The Weak Supervision Risk**: Labeling every individual claim from a flagged provider as "fraud" represents methodological malpractice (label noise), because even an excluded provider bills hundreds of legitimate, standard claims.

### Real & Synthetic Dataset Comparison for Further Research

| Dataset | Origin | Schema & Granularity | Target Definition | License / Availability | Linkable to Medicare? |
|---|---|---|---|---|:---:|
| **CMS Synthetic Medicare (SynPUF)** *(Current)* | CMS Public Use Files | Inpatient, Outpatient, Beneficiary, Carrier claims | Provider-level `PotentialFraud` (Yes/No) | Public Domain (U.S. Govt) | Native (Current Project) |
| **NHIS Healthcare Claims & Fraud Dataset** | Kaggle Community | Patient demographics, admission, procedure, financial | Synthetic binary claim flag (`is_fraud: 0/1`) | CC0: Public Domain | ❌ No (Different patient IDs & format) |
| **MIMIC-IV Clinical Database** | MIT / PhysioNet | Intensive care clinical notes, vitals, ICD-10 | Clinical outcomes (Mortality, LOS) — No fraud labels | PhysioNet Credentialed Access | ❌ No |
| **OIG List of Excluded Individuals/Entities (LEIE)** | HHS-OIG | Excluded healthcare providers & NPI numbers | Exclusion reasons (Default, Kickbacks, Patient Abuse) | Public Domain (Updated Monthly) | ✅ Yes (Via Provider NPI) |

### System Implementation: Two-Tiered Architecture
To provide robust claim-level analysis without inventing false labels, this Phase 4 system implements:
- **Mechanism A (Claim-Level Screening)**: Transparent clinical and statistical anomaly screening (reimbursement outlier percentiles, duplicate billing flags, date sequence contradictions).
- **Mechanism B (Provider Risk Context)**: Provider ML probability scores paired with each claim to prioritize manual review.

---

## 6. NFHS-5 Medical Consistency Screening Module

The National Family Health Survey (NFHS-5) module functions as an independent clinical consistency validation engine:

### Six Clinical Consistency Rules
1. **R1A — Diabetes without Treatment**: Prior diagnosis self-reported, but no lifestyle advice or clinical management.
2. **R1B — Hypertension without Treatment**: High blood pressure self-reported, but no active monitoring or treatment.
3. **R2A — Treatment without Diabetes Diagnosis**: Receiving diabetes medical treatment despite reporting no prior diagnosis.
4. **R2B — Treatment without Hypertension Diagnosis**: Taking antihypertensive therapy despite reporting no prior hypertension.
5. **R3 — High Glucose without Diagnosis**: Random blood glucose >= 200 mg/dL without an established diabetes diagnosis.
6. **R4 — Diabetes without Prescribed Medication**: Diagnosed diabetes without prescribed pharmacological medication.

### Available sample and benchmark provenance
- **Available screened sample**: 35,000 records, including 1,213 sample records with a non-zero rule score.
- **Recorded nationwide benchmark summary**: 724,115 records, 20,514 flagged, and 2,799 high/critical. These figures are preserved in `data/nfhs/nfhs5_rule_summary.json`; the full raw benchmark is unavailable in this project folder, so the figures cannot currently be independently verified.
- The dashboard validates every available sample record's weighted score and risk category against the six documented rules. NFHS consistency indicators are not Medicare claims or confirmed fraud findings.

---

## 7. Automated Testing Suite

The test suite includes unit and integration coverage for data integrity, database querying, ML inference, screening evidence, case management, and currency formatting.

To execute tests:
```bash
python -m pytest -v tests
```

### Verified Test Coverage:
- `tests/test_data_validation.py`: Verifies existence and non-empty size of processed Parquet files, DuckDB database, and model artifacts; validates all 10 engineered feature columns.
- `tests/test_claim_queries.py`: Tests DuckDB lookups for specific claims (`CLM46614`), non-existent IDs, beneficiary histories, provider profiles, and dynamic multi-column search filters.
- `tests/test_predictions.py`: Validates model loading (RF, LR, XGBoost), probability bounds (`0.0 <= p <= 1.0`), risk tier assignments, and local SHAP TreeExplainer contribution calculations.
- `tests/test_dashboard_logic.py`: Tests SQLite transaction safety, case creation, note addition, status updates, audit log persistence, screening rule trigger logic, and NFHS benchmark loading.

---

## 8. Regulatory & Ethical Disclaimers

> **IMPORTANT REGULATORY NOTICE**  
> 1. **Decision Support**: This software is intended strictly as an investigation prioritization and decision-support tool for trained healthcare claim auditors.  
> 2. **No Automated Fraud Determinations**: An elevated machine-learning score or triggered screening indicator does not constitute legal proof of fraud, an assertion of criminal liability, or grounds for automatic claim denial.  
> 3. **Medical Necessity**: Administrative claims data alone cannot establish medical necessity or clinical malpractice without full review of physician clinical notes, signed orders, and operative records.  
> 4. **Dataset Segregation**: NFHS-5 public health survey records and Medicare claims are maintained in separate analytical modules and are not linked at the record level.
