"""
backend/ml_predictor.py
=================================================================
ML PREDICTION LAYER.

This module is the only part of the backend that:
  - loads the trained model file (backend/model/health_model.pkl)
  - prepares a single day's input into the exact row shape the
    model was trained on
  - calls pipeline.predict() / predict_proba()
  - computes the separate rule-based wellness score

backend/main.py never imports joblib, never touches the .pkl file,
and never calls scikit-learn directly. It only calls
`predict_wellness(...)` from this module. That is the architectural
boundary the college presentation should point to:

    main.py  --calls-->  ml_predictor.py  --loads-->  health_model.pkl
=================================================================
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from ml.scoring import CATEGORY_NAMES, category_label, compute_wellness_score

MODEL_PATH = Path(__file__).parent / "model" / "health_model.pkl"

_bundle = None  # lazy-loaded singleton: {"pipeline": ..., "feature_columns": [...]}


class ModelNotLoadedError(RuntimeError):
    """Raised when the .pkl model file is missing or fails to load."""


def _load_bundle():
    global _bundle
    if _bundle is not None:
        return _bundle

    if not MODEL_PATH.exists():
        raise ModelNotLoadedError(
            f"Model file not found at {MODEL_PATH}. "
            "Run `python ml/generate_dataset.py` then `python ml/train.py` first."
        )

    try:
        _bundle = joblib.load(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - surface a clear message either way
        raise ModelNotLoadedError(f"Failed to load model file: {exc}") from exc

    if "pipeline" not in _bundle or "feature_columns" not in _bundle:
        raise ModelNotLoadedError(
            "Model file is missing expected keys ('pipeline', 'feature_columns'). "
            "Re-run `python ml/train.py` to regenerate it."
        )
    return _bundle


def is_model_ready() -> bool:
    try:
        _load_bundle()
        return True
    except ModelNotLoadedError:
        return False


@dataclass
class PredictionResult:
    status: str            # e.g. "Good"
    status_code: int       # 0, 1, 2
    wellness_score: float  # 0-100, from the rule-based formula
    confidence: float      # RF's predict_proba for the predicted class
    class_probabilities: Dict[str, float]


def predict_wellness(input_data: dict) -> PredictionResult:
    """
    input_data must contain exactly the raw feature values (same
    keys as ml/train.py's FEATURE_COLUMNS, any order):
        age, sleep_hours, steps, exercise_minutes, water_intake,
        stress_level, heart_rate, calories, screen_time, weight, mood
    """
    bundle = _load_bundle()
    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]

    # Build the row in EXACTLY the order the model was trained on.
    # This ordering guarantee is why feature_columns is saved inside
    # the same .pkl file rather than duplicated as a hard-coded list
    # in the backend.
    try:
        row = pd.DataFrame([[input_data[col] for col in feature_columns]], columns=feature_columns)
    except KeyError as exc:
        raise ValueError(f"Missing required feature: {exc}") from exc

    # --- 1. ML prediction (comes from the trained Random Forest) ---
    predicted_class = int(pipeline.predict(row)[0])
    probabilities = pipeline.predict_proba(row)[0]
    class_probabilities = {
        category_label(i): round(float(p), 4) for i, p in enumerate(probabilities)
    }
    confidence = round(float(probabilities[predicted_class]), 4)

    # --- 2. Wellness score (rule-based, independent of the model) ---
    wellness_score = compute_wellness_score(
        sleep_hours=input_data["sleep_hours"],
        steps=input_data["steps"],
        exercise_minutes=input_data["exercise_minutes"],
        water_intake=input_data["water_intake"],
        stress_level=input_data["stress_level"],
        heart_rate=input_data["heart_rate"],
        calories=input_data["calories"],
        screen_time=input_data["screen_time"],
        mood=input_data["mood"],
    )

    return PredictionResult(
        status=category_label(predicted_class),
        status_code=predicted_class,
        wellness_score=wellness_score,
        confidence=confidence,
        class_probabilities=class_probabilities,
    )
