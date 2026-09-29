"""
backend/crud.py
=================================================================
Database operations. main.py calls these functions instead of
writing raw SQLAlchemy queries inline in the API route handlers.
=================================================================
"""

from sqlalchemy.orm import Session

from backend.database import HealthRecord


def create_record(db: Session, input_data: dict, prediction: str, wellness_score: float) -> HealthRecord:
    record = HealthRecord(
        age=input_data["age"],
        sleep_hours=input_data["sleep_hours"],
        steps=input_data["steps"],
        exercise_minutes=input_data["exercise_minutes"],
        water_intake=input_data["water_intake"],
        stress_level=input_data["stress_level"],
        heart_rate=input_data["heart_rate"],
        calories=input_data["calories"],
        screen_time=input_data["screen_time"],
        weight=input_data["weight"],
        mood=input_data["mood"],
        prediction=prediction,
        wellness_score=wellness_score,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_history(db: Session, limit: int = 60):
    """Most recent `limit` records, returned in chronological order (oldest first)
    so the frontend can plot them left-to-right on the chart."""
    records = (
        db.query(HealthRecord)
        .order_by(HealthRecord.date.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(records))
