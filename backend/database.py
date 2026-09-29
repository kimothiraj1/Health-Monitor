"""
backend/database.py
=================================================================
SQLite setup using SQLAlchemy. The database file (health.db) is
created automatically the first time the backend starts -- there
is no manual setup step.
=================================================================
"""

from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "health.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# check_same_thread=False is required for SQLite + FastAPI, since
# FastAPI can use the connection from more than one worker thread.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class HealthRecord(Base):
    __tablename__ = "health_records"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    age = Column(Integer, nullable=False)
    sleep_hours = Column(Float, nullable=False)
    steps = Column(Integer, nullable=False)
    exercise_minutes = Column(Integer, nullable=False)
    water_intake = Column(Float, nullable=False)
    stress_level = Column(Integer, nullable=False)
    heart_rate = Column(Integer, nullable=False)
    calories = Column(Integer, nullable=False)
    screen_time = Column(Float, nullable=False)
    weight = Column(Float, nullable=False)
    mood = Column(Integer, nullable=False)

    prediction = Column(String, nullable=False)       # e.g. "Good"
    wellness_score = Column(Float, nullable=False)     # 0-100


def init_db() -> None:
    """Create tables if they don't already exist. Safe to call every startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a session, always closes it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
