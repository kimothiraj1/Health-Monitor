"""
ml/train.py
=================================================================
Trains the RandomForestClassifier on ml/dataset.csv and saves a
complete scikit-learn Pipeline (preprocessing + model) to
backend/model/health_model.pkl.

Why a Pipeline (not just a bare model)?
  Bundling preprocessing with the model means backend/ml_predictor.py
  only has to call `pipeline.predict(...)`. There is no risk of the
  API preprocessing inputs differently than training did, because
  there is only ONE copy of that logic, saved inside the pickle.

Run:
    python ml/train.py
Requires:
    ml/dataset.csv (generate it first with generate_dataset.py)
Output:
    backend/model/health_model.pkl
    ml/test_split.csv          (holdout rows, used by evaluate.py)
=================================================================
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scoring import CATEGORY_NAMES

# The exact feature order used for training. backend/ml_predictor.py
# MUST build its input row using this same order.
FEATURE_COLUMNS = [
    "age",
    "sleep_hours",
    "steps",
    "exercise_minutes",
    "water_intake",
    "stress_level",
    "heart_rate",
    "calories",
    "screen_time",
    "weight",
    "mood",
]
TARGET_COLUMN = "health_status"

RANDOM_STATE = 42
TEST_SIZE = 0.2

THIS_DIR = Path(__file__).parent
DATASET_PATH = THIS_DIR / "dataset.csv"
TEST_SPLIT_PATH = THIS_DIR / "test_split.csv"
MODEL_OUTPUT_PATH = THIS_DIR.parent / "backend" / "model" / "health_model.pkl"


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run `python ml/generate_dataset.py` first."
        )
    df = pd.read_csv(DATASET_PATH)

    # Basic data-quality checks (this is the "preprocessing" step
    # required for a real ML pipeline, even on synthetic data).
    missing = df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum().sum()
    if missing > 0:
        print(f"Found {missing} missing values -- dropping affected rows.")
        df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1, 2}):
        raise ValueError("health_status must only contain values 0, 1, 2.")

    return df


def build_pipeline() -> Pipeline:
    """
    Preprocessing + model pipeline.

    StandardScaler is not strictly required for a Random Forest
    (trees are not distance-based), but it is included here so the
    pipeline demonstrates a real preprocessing step and so the same
    code would keep working if the model were swapped for something
    scale-sensitive later.
    """
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n=== Evaluation on held-out test set ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro):    {recall:.4f}")
    print(f"F1-score (macro):  {f1:.4f}")

    print("\nClassification report:")
    labels_sorted = sorted(CATEGORY_NAMES.keys())
    target_names = [CATEGORY_NAMES[i] for i in labels_sorted]
    print(classification_report(
        y_test, y_pred, labels=labels_sorted, target_names=target_names, zero_division=0
    ))

    print("Confusion matrix (rows = actual, cols = predicted):")
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)

    model: RandomForestClassifier = pipeline.named_steps["model"]
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print("\nFeature importances (sorted):")
    print(importances.sort_values(ascending=False).round(4))


def main():
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    evaluate(pipeline, X_test, y_test)

    # Save the holdout split so evaluate.py can re-check the SAVED
    # model file later without retraining.
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test
    test_df.to_csv(TEST_SPLIT_PATH, index=False)

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_columns": FEATURE_COLUMNS}, MODEL_OUTPUT_PATH)
    print(f"\nSaved trained pipeline -> {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
