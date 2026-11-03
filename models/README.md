# Machine Learning Model Artifacts

This directory stores the trained model pipelines, feature definitions, and cross-validation metadata for the healthcare fraud detection system.

## Available Artifacts

1. `rf_provider_model.joblib`: Production Random Forest Classifier (n_estimators=100, random_state=42) trained on 10 engineered provider-level features.
2. `lr_provider_model.joblib`: Scikit-learn Pipeline with `StandardScaler` and `LogisticRegression` (class_weight='balanced', max_iter=1000).
3. `xgb_provider_model.joblib`: Tuned XGBoost Classifier (scale_pos_weight=9.69, max_depth=4, learning_rate=0.1, n_estimators=100).
4. `model_metadata.json`: Contains the exact feature order, 5-fold stratified cross-validation benchmark metrics, risk thresholds (0.30 and 0.70), and Gini feature importances.

## Retraining Models

To regenerate or retrain all models from raw Medicare tables, execute:

```bash
python scripts/build_data_store.py
```
