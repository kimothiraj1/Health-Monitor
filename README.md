# Daily Health & Wellness Monitor

A locally-run web app that takes a day's lifestyle inputs (sleep, steps,
exercise, stress, etc.) and returns a lifestyle-based wellness indicator
using a trained Random Forest classifier, plus a chart of your trend over
time.

**This is a portfolio/college project, not a medical device.** It does not
diagnose any disease and should not be used for medical decisions.

---

## 1. Project overview

You fill in ~11 numbers about your day on a single dashboard page. The
backend validates the input, asks a trained `RandomForestClassifier` to
classify the day as **Good / Moderate / Needs Improvement**, computes a
separate 0-100 wellness score, saves the entry to a local SQLite database,
and returns everything to the browser. Every past entry is plotted on a
line chart so you can see your trend over time.

## 2. Features

- 11-field daily input form with client-side and server-side validation
- Real prediction from a trained Random Forest model (no hard-coded logic)
- Transparent, separately-computed 0-100 wellness score
- SQLite storage — every submission is saved permanently
- Line chart (Chart.js) of wellness score over time, built from real stored records
- FastAPI auto-generated docs at `/docs`
- 100% local — no cloud, no Docker, no accounts

## 3. Architecture

```
FRONTEND (HTML/CSS/JS)
      |  fetch() -> POST /predict, GET /history
      v
BACKEND / API  (backend/main.py)
      |  validates input (Pydantic), then calls...
      v
ML PREDICTION LAYER  (backend/ml_predictor.py)
      |  loads once, calls pipeline.predict() / predict_proba()
      v
TRAINED MODEL  (backend/model/health_model.pkl)
      |
      v  (result flows back up)
ML PREDICTION LAYER -> BACKEND / API -> JSON -> FRONTEND
```

`backend/main.py` never imports `joblib` or `scikit-learn` and never opens
the `.pkl` file — it only calls `predict_wellness()` from
`ml_predictor.py`. That boundary is deliberate: it's the line you'd draw
on a whiteboard in a presentation to show "backend" vs. "ML layer".

Database access is similarly isolated: `main.py` calls `crud.py`, which
calls `database.py` — `main.py` never writes raw SQL/ORM queries itself.

**Model training is a completely separate, offline process** (the `ml/`
folder). It has nothing to do with the live API — it only produces the
`.pkl` file that `ml_predictor.py` later loads.

```
ml/generate_dataset.py -> ml/dataset.csv -> ml/train.py -> backend/model/health_model.pkl
```

## 4. Tech stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, vanilla JavaScript, Chart.js |
| Backend / API | FastAPI, Pydantic |
| ML | scikit-learn (RandomForestClassifier), pandas, numpy, joblib |
| Database | SQLite via SQLAlchemy |

## 5. Folder structure

```
health-monitor/
├── backend/
│   ├── main.py            # FastAPI app, routes, serves frontend
│   ├── ml_predictor.py    # ML PREDICTION LAYER — loads .pkl, runs inference
│   ├── schemas.py         # Pydantic request/response models
│   ├── database.py        # SQLAlchemy engine/session/table
│   ├── crud.py            # Database read/write functions
│   ├── requirements.txt
│   └── model/
│       └── health_model.pkl   # created by ml/train.py
│
├── ml/
│   ├── scoring.py          # shared wellness-score formula (used by both ml/ and backend/)
│   ├── generate_dataset.py # builds the synthetic dataset
│   ├── dataset.csv         # created by generate_dataset.py
│   ├── train.py            # trains + evaluates + saves the model
│   └── evaluate.py         # re-evaluates the SAVED .pkl file
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── README.md
└── .gitignore
```

## 6. How the ML model works

1. **Features (11):** age, sleep_hours, steps, exercise_minutes,
   water_intake, stress_level, heart_rate, calories, screen_time, weight, mood.
2. **Target (`health_status`):** 0 = Needs Improvement, 1 = Moderate, 2 = Good.
3. **Model:** `RandomForestClassifier` inside a scikit-learn `Pipeline`
   (with a `StandardScaler` preprocessing step), saved as one object with
   `joblib` so the API never has to reimplement preprocessing.
4. **Two different numbers, on purpose:**
   - **ML prediction** (`Good` / `Moderate` / `Needs Improvement`) — the
     actual output of the trained Random Forest.
   - **Wellness score** (0-100) — a transparent, rule-based formula in
     `ml/scoring.py`, independent of the model. It is not a medically
     validated score and is not derived from the model's confidence.
   - `age` and `weight` feed the model but are intentionally excluded
     from the score formula — see the note at the bottom of
     `ml/scoring.py` for why, and check `feature_importances_` printed
     by `train.py` to see the model largely agrees.

## 7. Dataset explanation

No public dataset is used, because public "health" datasets are almost
always clinical (diagnosis-oriented), which is exactly what this project
avoids. Instead, `ml/generate_dataset.py` **synthesizes** ~4,000 rows:

1. Each feature is sampled from a realistic distribution (e.g. sleep
   hours ~ Normal(6.4, 2.0), clipped to 0–14).
2. `ml/scoring.py`'s formula turns 9 of the features into a 0-100 score.
3. Gaussian noise is added to that score, and ~6% of the resulting labels
   are randomly flipped to a neighboring category — this stops the
   Random Forest from simply "memorizing" the formula and forces it to
   learn a real, imperfect decision boundary, similar to noisy
   real-world self-reported data.
4. The noisy score is bucketed into the 3 classes to make `health_status`.

## 8. How to train the model

```
python ml/generate_dataset.py
python ml/train.py
```

`train.py` prints accuracy, macro precision/recall/F1, a full
classification report, a confusion matrix, and feature importances, then
saves the pipeline to `backend/model/health_model.pkl`.

Optionally, re-check the saved file independently:

```
python ml/evaluate.py
```

## 9. How to run FastAPI

See section 11 below for the full step-by-step with exact commands.

## 10. How to use the dashboard

1. Open `http://127.0.0.1:8000` in your browser.
2. Fill in today's numbers across the three sections (rest & activity,
   body & vitals, mind & screen).
3. Click **Analyze My Day**.
4. Your status, wellness score, and a short message appear immediately.
5. The chart below updates with your full history.

## 11. Running the project locally (Windows)

```
cd health-monitor

python -m venv venv
venv\Scripts\activate

pip install -r backend\requirements.txt

python ml\generate_dataset.py
python ml\train.py

uvicorn backend.main:app --reload
```

Then open your browser to:

```
http://127.0.0.1:8000
```

FastAPI serves the frontend itself, so this single URL is all you need —
there is no separate frontend server to run.

Interactive API docs:

```
http://127.0.0.1:8000/docs
```

(macOS/Linux: replace `venv\Scripts\activate` with `source venv/bin/activate`
and use forward slashes in the `python` commands, e.g. `python ml/train.py`.)

## 12. API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check + whether the model loaded |
| POST | `/predict` | Submit today's data, get a prediction |
| GET | `/history?limit=60` | Recent saved records, oldest first |

## 13. Example API request

```
POST /predict
Content-Type: application/json

{
  "age": 28,
  "sleep_hours": 7.5,
  "steps": 9000,
  "exercise_minutes": 40,
  "water_intake": 2.2,
  "stress_level": 4,
  "heart_rate": 68,
  "calories": 2200,
  "screen_time": 3.5,
  "weight": 68,
  "mood": 7
}
```

## 14. Example API response

```json
{
  "status": "Good",
  "wellness_score": 82.4,
  "confidence": 0.87,
  "class_probabilities": {
    "Needs Improvement": 0.02,
    "Moderate": 0.11,
    "Good": 0.87
  },
  "message": "Your lifestyle indicators look positive today. Keep it up."
}
```

## 15. Screenshots

_Add screenshots of the dashboard, result card, and chart here before
submitting your project._

## 16. Future improvements

- Let the user edit/delete a past entry
- Export history to CSV
- Add a "streak" or weekly-summary view
- Support multiple user profiles (currently single-user/local)
- Add unit tests for `ml_predictor.py` and the API routes

## 17. Disclaimer

This application classifies **lifestyle input patterns only**. It does
not diagnose, treat, or assess any medical condition, and the wellness
score is a self-defined heuristic, not a clinically validated measure.
Consult a qualified professional for any health concerns.
