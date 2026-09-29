"""
ml/scoring.py
=================================================================
Single source of truth for the RULE-BASED "Wellness Score".

IMPORTANT CONCEPTUAL POINT (explain this in your presentation):

  There are TWO different numbers in this project and they come
  from TWO different places:

  1. "ML prediction" (health_status: Needs Improvement / Moderate /
     Good) -> produced by the trained RandomForestClassifier in
     backend/ml_predictor.py. This is the only place a real ML
     model is used.

  2. "Wellness Score" (0-100) -> produced by the plain, transparent
     arithmetic formula in THIS file. It is a rule-based scoring
     mechanism, not a model output and not a medical score.

  We keep these separate on purpose:
    - It makes it obvious, for grading/demo purposes, exactly what
      the ML model is and is not responsible for.
    - It avoids the common mistake of quietly turning a classifier's
      confidence/probability into a "score" and calling it something
      it isn't.

  This file is imported by:
    - ml/generate_dataset.py   -> to create noisy training labels
    - backend/ml_predictor.py  -> to compute the score shown live
=================================================================
"""

from dataclasses import dataclass

# ---------------------------------------------------------------
# Category labels used everywhere in the project. Keep these three
# in sync with ml/train.py (target encoding) and the frontend.
# ---------------------------------------------------------------
CATEGORY_NAMES = {
    0: "Needs Improvement",
    1: "Moderate",
    2: "Good",
}

# Score thresholds that turn a 0-100 score into a category.
# These are also used to build synthetic training labels.
GOOD_THRESHOLD = 75
MODERATE_THRESHOLD = 50


def score_to_category(score: float) -> int:
    """Map a 0-100 wellness score to a category id (0, 1, 2)."""
    if score >= GOOD_THRESHOLD:
        return 2
    if score >= MODERATE_THRESHOLD:
        return 1
    return 0


def category_label(category_id: int) -> str:
    return CATEGORY_NAMES.get(category_id, "Unknown")


# ---------------------------------------------------------------
# Component scoring helpers.
# Each returns a value between 0.0 and 1.0.
# ---------------------------------------------------------------
def _higher_is_better(value: float, floor: float, target: float) -> float:
    """0.0 at/under `floor`, 1.0 at/above `target`, linear between."""
    if target == floor:
        return 1.0
    if value <= floor:
        return 0.0
    if value >= target:
        return 1.0
    return (value - floor) / (target - floor)


def _lower_is_better(value: float, target: float, ceiling: float) -> float:
    """1.0 at/under `target`, 0.0 at/above `ceiling`, linear between."""
    if ceiling == target:
        return 1.0
    if value <= target:
        return 1.0
    if value >= ceiling:
        return 0.0
    return (ceiling - value) / (ceiling - target)


def _ideal_band(value: float, low: float, high: float, floor: float, ceiling: float) -> float:
    """1.0 inside [low, high], decaying linearly to 0.0 at floor/ceiling."""
    if low <= value <= high:
        return 1.0
    if value < low:
        if value <= floor:
            return 0.0
        return (value - floor) / (low - floor)
    if value >= ceiling:
        return 0.0
    return (ceiling - value) / (ceiling - high)


@dataclass
class WellnessComponents:
    """Per-feature sub-scores (0-1) — handy for debugging/demo purposes."""
    sleep: float
    steps: float
    exercise: float
    water: float
    stress: float
    heart_rate: float
    calories: float
    screen_time: float
    mood: float


# Weights must sum to 100. Age and weight are intentionally NOT part
# of this formula (see note at the bottom of this file) but they are
# still passed to the Random Forest as raw features.
WEIGHTS = {
    "sleep": 15,
    "steps": 15,
    "exercise": 15,
    "water": 8,
    "stress": 12,
    "heart_rate": 10,
    "calories": 8,
    "screen_time": 7,
    "mood": 10,
}
assert sum(WEIGHTS.values()) == 100


def compute_components(
    sleep_hours: float,
    steps: float,
    exercise_minutes: float,
    water_intake: float,
    stress_level: float,
    heart_rate: float,
    calories: float,
    screen_time: float,
    mood: float,
) -> WellnessComponents:
    return WellnessComponents(
        sleep=_ideal_band(sleep_hours, low=7, high=9, floor=0, ceiling=14),
        steps=_higher_is_better(steps, floor=0, target=10000),
        exercise=_higher_is_better(exercise_minutes, floor=0, target=45),
        water=_higher_is_better(water_intake, floor=0, target=2.5),
        stress=_lower_is_better(stress_level, target=3, ceiling=10),
        heart_rate=_ideal_band(heart_rate, low=60, high=80, floor=40, ceiling=110),
        calories=_ideal_band(calories, low=1800, high=2600, floor=1000, ceiling=4000),
        screen_time=_lower_is_better(screen_time, target=2, ceiling=12),
        mood=_higher_is_better(mood, floor=1, target=8),
    )


def compute_wellness_score(
    sleep_hours: float,
    steps: float,
    exercise_minutes: float,
    water_intake: float,
    stress_level: float,
    heart_rate: float,
    calories: float,
    screen_time: float,
    mood: float,
) -> float:
    """
    Deterministic, transparent 0-100 wellness score. Does NOT use
    age or weight, and does NOT use the Random Forest model at all.
    """
    c = compute_components(
        sleep_hours, steps, exercise_minutes, water_intake, stress_level,
        heart_rate, calories, screen_time, mood,
    )
    total = (
        c.sleep * WEIGHTS["sleep"]
        + c.steps * WEIGHTS["steps"]
        + c.exercise * WEIGHTS["exercise"]
        + c.water * WEIGHTS["water"]
        + c.stress * WEIGHTS["stress"]
        + c.heart_rate * WEIGHTS["heart_rate"]
        + c.calories * WEIGHTS["calories"]
        + c.screen_time * WEIGHTS["screen_time"]
        + c.mood * WEIGHTS["mood"]
    )
    return round(total, 2)


# -----------------------------------------------------------------
# NOTE on age & weight:
# A single day's `weight` value and a person's `age` don't tell you
# much about *daily lifestyle* wellness on their own (unlike, say,
# sleep or steps) without extra context such as height or a
# baseline/trend. Rather than inventing an arbitrary penalty for
# them, we exclude them from the score formula and instead pass them
# to the Random Forest as ordinary features. If you check
# `feature_importances_` after training (see ml/train.py), you
# should see age and weight ranked low — that's expected, and it's
# a good talking point for a presentation on feature importance.
# -----------------------------------------------------------------
