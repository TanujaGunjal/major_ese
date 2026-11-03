"""Generate leakage-free stratified holdout probabilities for dashboard evaluation."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import FEATURE_COLUMNS, PROCESSED_DIR


def main() -> None:
    providers = pd.read_parquet(PROCESSED_DIR / "providers.parquet")
    features = providers[FEATURE_COLUMNS]
    labels = providers["FraudLabel"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    if positive_count == 0:
        raise ValueError("The training split contains no positive provider labels.")
    models = {
        "RF_Probability": RandomForestClassifier(n_estimators=100, random_state=42),
        "LR_Probability": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]),
        "XGB_Probability": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=negative_count / positive_count,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    evaluation = x_test.copy()
    evaluation["Provider"] = providers.loc[x_test.index, "Provider"].values
    evaluation["ActualLabel"] = y_test.values
    for column, model in models.items():
        model.fit(x_train, y_train)
        evaluation[column] = model.predict_proba(x_test)[:, 1]

    output = PROCESSED_DIR / "eval_predictions.parquet"
    evaluation.to_parquet(output, index=False)
    print(f"Saved independent 80/20 holdout predictions ({len(evaluation)} providers) to {output}")


if __name__ == "__main__":
    main()
