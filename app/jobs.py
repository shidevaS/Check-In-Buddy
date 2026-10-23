"""End-of-day job: run when simulated time crosses midnight.

Marks unlogged meds/meals for the finished day as missed, then runs the
deterministic escalation rules. Alerts are only ever produced by
app.escalation - never hand-written.
"""
from datetime import timedelta
from typing import List, Dict, Any

from app.database import SessionLocal
from app.models import User, Medication, MedicationLog, MealLog
from app.escalation import run_escalation


def _mark_missed_for(db, user_id: int, day: str) -> None:
    """Materialize 'missed' rows for a finished day (idempotent)."""
    meds = db.query(Medication).filter(Medication.user_id == user_id).all()
    for med in meds:
        existing = (
            db.query(MedicationLog)
            .filter(
                MedicationLog.user_id == user_id,
                MedicationLog.medication_name == med.name,
                MedicationLog.date == day,
            )
            .first()
        )
        if existing is None:
            db.add(
                MedicationLog(
                    user_id=user_id,
                    medication_name=med.name,
                    date=day,
                    taken=False,
                    time_taken="",
                    source="end_of_day",
                )
            )

    for meal_type in ("breakfast", "lunch", "dinner"):
        existing = (
            db.query(MealLog)
            .filter(
                MealLog.user_id == user_id,
                MealLog.date == day,
                MealLog.meal_type == meal_type,
            )
            .first()
        )
        if existing is None:
            db.add(
                MealLog(
                    user_id=user_id,
                    date=day,
                    meal_type=meal_type,
                    eaten=False,
                    notes="not logged by end of day",
                )
            )


def end_of_day_job(user_id: int = None) -> List[Dict[str, Any]]:
    """Run the end-of-day job for the day that just finished."""
    from app.clock import now

    finished_day = (now().date() - timedelta(days=1)).isoformat()
    db = SessionLocal()
    try:
        users = (
            db.query(User).filter(User.id == user_id).all()
            if user_id is not None
            else db.query(User).all()
        )
        all_alerts: List[Dict[str, Any]] = []
        for user in users:
            _mark_missed_for(db, user.id, finished_day)
            db.flush()
            all_alerts.extend(run_escalation(db, user.id))
        db.commit()
        return all_alerts
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()