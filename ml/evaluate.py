"""
ml/evaluate.py
=================================================================
Loads the ALREADY-SAVED pipeline from backend/model/health_model.pkl
and re-evaluates it on the holdout split saved by train.py.

This is deliberately separate from train.py: it proves that the
.pkl file on disk is the same model reported on during training,
which is exactly what backend/ml_predictor.py will load at runtime.

Run:
    python ml/evaluate.py
Requires:
    backend/model/health_model.pkl  (created by train.py)
    ml/test_split.csv               (created by train.py)
=================================================================
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from scoring import CATEGORY_NAMES

THIS_DIR = Path(__file__).parent
MODEL_PATH = THIS_DIR.parent / "backend" / "model" / "health_model.pkl"
TEST_SPLIT_PATH = THIS_DIR / "test_split.csv"
TARGET_COLUMN = "health_status"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run `python ml/train.py` first.")
    if not TEST_SPLIT_PATH.exists():
        raise FileNotFoundError(f"{TEST_SPLIT_PATH} not found. Run `python ml/train.py` first.")

    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]

    test_df = pd.read_csv(TEST_SPLIT_PATH)
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    y_pred = pipeline.predict(X_test)

    print("=== Re-evaluating saved model file (backend/model/health_model.pkl) ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall (macro):    {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1-score (macro):  {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")

    labels_sorted = sorted(CATEGORY_NAMES.keys())
    target_names = [CATEGORY_NAMES[i] for i in labels_sorted]

    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, labels=labels_sorted, target_names=target_names, zero_division=0
    ))

    print("Confusion matrix (rows = actual, cols = predicted):")
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    print(pd.DataFrame(cm, index=target_names, columns=target_names))


if __name__ == "__main__":
    main()
