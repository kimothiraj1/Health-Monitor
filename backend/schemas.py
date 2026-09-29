"""
backend/schemas.py
=================================================================
Pydantic models used by FastAPI for request validation and
response typing. FastAPI uses these to auto-generate /docs.
=================================================================
"""

from datetime import datetime
from typing import Dict

from pydantic import BaseModel, Field


class HealthInput(BaseModel):
    age: int = Field(..., ge=10, le=100, description="Age in years")
    sleep_hours: float = Field(..., ge=0, le=24, description="Hours of sleep last night")
    steps: int = Field(..., ge=0, le=100000, description="Daily step count")
    exercise_minutes: int = Field(..., ge=0, le=600, description="Minutes of exercise today")
    water_intake: float = Field(..., ge=0, le=15, description="Water intake in liters")
    stress_level: int = Field(..., ge=1, le=10, description="Self-rated stress, 1 (low) - 10 (high)")
    heart_rate: int = Field(..., ge=30, le=220, description="Resting heart rate (bpm)")
    calories: int = Field(..., ge=0, le=10000, description="Calories consumed today")
    screen_time: float = Field(..., ge=0, le=24, description="Screen time in hours")
    weight: float = Field(..., ge=20, le=300, description="Weight in kilograms")
    mood: int = Field(..., ge=1, le=10, description="Self-rated mood, 1 (low) - 10 (high)")

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "mood": 7,
            }
        }
    }


class PredictionResponse(BaseModel):
    status: str = Field(..., description="Predicted category from the Random Forest model")
    wellness_score: float = Field(..., description="0-100 rule-based wellness score (not from the ML model)")
    confidence: float = Field(..., description="Random Forest's predicted probability for `status`")
    class_probabilities: Dict[str, float] = Field(..., description="Predicted probability for every class")
    message: str = Field(..., description="Short human-readable summary")


class HistoryItem(BaseModel):
    id: int
    date: datetime
    prediction: str
    wellness_score: float

    model_config = {"from_attributes": True}


class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
