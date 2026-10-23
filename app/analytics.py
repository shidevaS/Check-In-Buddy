"""Deterministic, non-diagnostic summaries of recent check-in activity."""

from datetime import timedelta

from app.clock import now
from app.database import SessionLocal
from app.models import CheckIn, MealLog, Medication, MedicationLog, MoodLog


def wellness_report(user_id: int) -> dict:
    """Summarize the last seven days without making medical conclusions."""
    db = SessionLocal()
    try:
        today = now().date()
        start = (today - timedelta(days=6)).isoformat()
        end = today.isoformat()
        medication_count = db.query(Medication).filter(Medication.user_id == user_id).count()
        medication_logs = (
            db.query(MedicationLog)
            .filter(
                MedicationLog.user_id == user_id,
                MedicationLog.date >= start,
                MedicationLog.date <= end,
            )
            .all()
        )
        meals = (
            db.query(MealLog)
            .filter(
                MealLog.user_id == user_id,
                MealLog.date >= start,
                MealLog.date <= end,
            )
            .all()
        )
        moods = (
            db.query(MoodLog)
            .filter(
                MoodLog.user_id == user_id,
                MoodLog.date >= start,
                MoodLog.date <= end,
            )
            .all()
        )
        checkins = (
            db.query(CheckIn)
            .filter(
                CheckIn.user_id == user_id,
                CheckIn.date >= start,
                CheckIn.date <= end,
            )
            .all()
        )

        med_total = medication_count * 7
        med_score = min(100, round(
            100 * sum(1 for log in medication_logs if log.taken) / med_total
        )) if med_total else 100
        meal_score = min(100, round(100 * sum(1 for meal in meals if meal.eaten) / 21))
        checkin_score = min(100, round(100 * sum(1 for checkin in checkins if checkin.completed) / 7))
        mood_score = round(100 * sum(mood.mood for mood in moods) / (5 * len(moods))) if moods else 100
        components = {
            "medication_logging": med_score,
            "meal_logging": meal_score,
            "check_ins": checkin_score,
            "mood_logging": mood_score,
        }
        score = round(sum(components.values()) / len(components))
        insights = []
        if checkin_score < 70:
            insights.append("A regular check-in can help you keep track of your routine.")
        if meal_score < 70:
            insights.append("Some meals were not logged this week.")
        if med_score < 70:
            insights.append("Some medication entries are missing; check your log if helpful.")
        if not insights:
            insights.append("Your recent check-in log looks consistent.")
        return {
            "score": score,
            "period": {"from": start, "through": end},
            "components": components,
            "insights": insights,
            "disclaimer": "This summary reflects logged activity only and is not medical advice.",
        }
    finally:
        db.close()
