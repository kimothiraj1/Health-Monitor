"""
ml/generate_dataset.py
=================================================================
Generates a SYNTHETIC lifestyle dataset for training the Random
Forest classifier.

Why synthetic data?
  Public "health" datasets almost always describe clinical/medical
  measurements (blood pressure, glucose, disease diagnosis) which
  would make this look like a medical diagnosis tool -- exactly
  what this project must NOT be. Instead we generate a lifestyle
  dataset with clearly-labeled, realistic ranges for each feature.

How the target label is created (read this before your presentation):
  1. Each row's 11 feature values are sampled from realistic ranges
     using numpy's random distributions (not simple uniform noise on
     the same easy rule every time -- see below).
  2. `ml/scoring.py` turns 9 of those features into a 0-100
     "wellness score" using a fixed, transparent weighted formula.
  3. We ADD random Gaussian noise to that score and, on top of that,
     randomly FLIP a small percentage of labels to a neighboring
     category. This simulates the fact that real self-reported
     wellness assessments are never perfectly deterministic.
  4. The noisy score is bucketed into 3 classes to become the
     `health_status` target column that the Random Forest is
     actually trained on.

  The Random Forest never sees the score formula -- it only sees the
  11 raw features and the final noisy category label, so it has to
  learn a genuine decision boundary rather than reverse-engineer
  arithmetic. `age` and `weight` are included as features but are
  NOT used by the scoring formula, so they act as a mild test of
  whether the model correctly learns to weight them less.

Run:
    python ml/generate_dataset.py
Output:
    ml/dataset.csv
=================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path

from scoring import compute_wellness_score, score_to_category, category_label

SEED = 42
N_ROWS = 4000
LABEL_FLIP_PROBABILITY = 0.06   # simulates messy real-world labels
SCORE_NOISE_STD = 6.0           # gaussian noise added to the score


def generate_raw_features(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Sample each feature independently from a realistic distribution."""
    age = rng.integers(16, 70, size=n)
    sleep_hours = np.clip(rng.normal(6.4, 2.0, size=n), 0, 14)
    steps = np.clip(rng.normal(6200, 4600, size=n), 0, 25000).astype(int)
    exercise_minutes = np.clip(rng.exponential(20, size=n), 0, 150).astype(int)
    water_intake = np.clip(rng.normal(1.8, 1.0, size=n), 0, 6).round(2)
    stress_level = np.clip(rng.normal(5.6, 2.4, size=n), 1, 10).round().astype(int)
    heart_rate = np.clip(rng.normal(77, 14, size=n), 45, 115).astype(int)
    calories = np.clip(rng.normal(2350, 680, size=n), 1000, 4000).astype(int)
    screen_time = np.clip(rng.normal(6.2, 3.2, size=n), 0, 16).round(1)
    weight = np.clip(rng.normal(70, 15, size=n), 40, 130).round(1)
    mood = np.clip(rng.normal(5.7, 2.3, size=n), 1, 10).round().astype(int)

    return pd.DataFrame({
        "age": age,
        "sleep_hours": sleep_hours.round(1),
        "steps": steps,
        "exercise_minutes": exercise_minutes,
        "water_intake": water_intake,
        "stress_level": stress_level,
        "heart_rate": heart_rate,
        "calories": calories,
        "screen_time": screen_time,
        "weight": weight,
        "mood": mood,
    })


def build_noisy_labels(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    clean_scores = df.apply(
        lambda r: compute_wellness_score(
            sleep_hours=r["sleep_hours"],
            steps=r["steps"],
            exercise_minutes=r["exercise_minutes"],
            water_intake=r["water_intake"],
            stress_level=r["stress_level"],
            heart_rate=r["heart_rate"],
            calories=r["calories"],
            screen_time=r["screen_time"],
            mood=r["mood"],
        ),
        axis=1,
    )

    noisy_scores = np.clip(
        clean_scores + rng.normal(0, SCORE_NOISE_STD, size=len(df)), 0, 100
    )
    labels = noisy_scores.apply(score_to_category)

    # Randomly flip a small percentage of labels to a neighboring
    # category to mimic imperfect real-world labeling.
    flip_mask = rng.random(len(df)) < LABEL_FLIP_PROBABILITY
    flip_direction = rng.choice([-1, 1], size=len(df))
    flipped_labels = np.clip(labels.to_numpy() + flip_direction, 0, 2)
    labels = np.where(flip_mask, flipped_labels, labels.to_numpy())

    df = df.copy()
    df["wellness_score_reference"] = clean_scores.round(2)
    df["health_status"] = labels.astype(int)
    df["health_status_label"] = [category_label(c) for c in df["health_status"]]
    return df


def main():
    rng = np.random.default_rng(SEED)
    df = generate_raw_features(rng, N_ROWS)
    df = build_noisy_labels(df, rng)

    out_path = Path(__file__).parent / "dataset.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} rows -> {out_path}")
    print("\nClass balance:")
    print(df["health_status_label"].value_counts())
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()
