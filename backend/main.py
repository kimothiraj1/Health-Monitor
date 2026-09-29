"""
backend/main.py
=================================================================
BACKEND / API LAYER.

This file is intentionally "thin": it validates requests, calls
backend/ml_predictor.py for anything ML-related, calls backend/crud.py
for anything database-related, and returns JSON. It never loads the
.pkl model and never calls scikit-learn directly -- that separation
is the whole point of this architecture.

    Frontend --HTTP--> main.py --calls--> ml_predictor.py --loads--> health_model.pkl
                          |
                          `--calls--> crud.py --> database.py --> health.db

Run from the PROJECT ROOT (health-monitor/), not from inside backend/:
    uvicorn backend.main:app --reload
Then open:
    http://127.0.0.1:8000
=================================================================
"""

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from backend import crud
from backend.database import get_db, init_db
from backend.ml_predictor import ModelNotLoadedError, is_model_ready, predict_wellness
from backend.schemas import HealthCheckResponse, HealthInput, HistoryItem, PredictionResponse

PROJECT_ROOT = Path(__file__).parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(
    title="Daily Health & Wellness Monitor",
    description=(
        "Tracks daily lifestyle inputs and returns a lifestyle-based wellness "
        "indicator. This is NOT a medical device and does not diagnose disease."
    ),
    version="1.0.0",
)

# Allow the frontend (even if served from a different port/origin during
# development) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


def _friendly_message(status: str) -> str:
    messages = {
        "Good": "Your lifestyle indicators look positive today. Keep it up.",
        "Moderate": "Your lifestyle indicators are okay, with some room to improve.",
        "Needs Improvement": "Several lifestyle indicators suggest today was a tough day for your habits.",
    }
    return messages.get(status, "Here is today's lifestyle-based wellness indicator.")


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
def health_check():
    """Simple liveness check. Also reports whether the trained model loaded."""
    return HealthCheckResponse(status="ok", model_loaded=is_model_ready())


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(payload: HealthInput, db: Session = Depends(get_db)):
    """
    Validates today's input, asks the ML prediction layer for a
    result, saves the record, and returns the prediction.
    """
    input_data = payload.model_dump()

    try:
        result = predict_wellness(input_data)
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The trained model is not available yet. "
                f"{exc} "
                "Run `python ml/generate_dataset.py` and `python ml/train.py`, then restart the server."
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    crud.create_record(
        db,
        input_data=input_data,
        prediction=result.status,
        wellness_score=result.wellness_score,
    )

    return PredictionResponse(
        status=result.status,
        wellness_score=result.wellness_score,
        confidence=result.confidence,
        class_probabilities=result.class_probabilities,
        message=_friendly_message(result.status),
    )


@app.get("/history", response_model=list[HistoryItem], tags=["History"])
def history(limit: int = 60, db: Session = Depends(get_db)):
    """Returns saved daily records, oldest first, for the trend chart."""
    records = crud.get_history(db, limit=limit)
    return records


# ---------------------------------------------------------------
# Serve the frontend. Mounted LAST and at "/" so it acts as a
# catch-all for static files (index.html, style.css, script.js)
# without shadowing the API routes registered above.
# ---------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
