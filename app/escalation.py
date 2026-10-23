"""Deterministic escalation rules - all trust-critical decisions live here.

The LLM never decides whether to alert the family. These rules do.

R1: a medication not taken 2 consecutive days (warning)
R2: 2+ meals skipped (false or unlogged by end of day) in one day (warning)
R3: mood <= 2 three consecutive days (warning)
R4: emergency language from the safety checker (urgent)  [raised by safety pre-check]
R5: no completed check-in in 24 simulated hours (info)

Mid-day, an unlogged med/meal on today is "not yet logged", not "missed";
only at end-of-day does it count as missed.

Zero alerts on a normal day. Dedup by stable key. Each alert stores an explanation.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from app.models import (
    User,
    Medication,
    MedicationLog,
    MealLog,
    MoodLog,
    CheckIn,
    Alert,
)


def _recent_dates(days: int) -> List[str]:
    """Return the last `days` dates as YYYY-MM-DD, oldest first."""
    from app.clock import now

    today = now().date()
    return [(today - timedelta(days=days - 1 - i)).isoformat() for i in range(days)]


def _upsert_alert(
    db,
    user_id: int,
    rule_id: str,
    severity: str,
    message: str,
    explanation: str,
    dedup_key: str,
    now: Optional[datetime] = None,
) -> Optional[Alert]:
    """Create an alert unless an identical dedup_key already exists."""
    existing = (
        db.query(Alert)
        .filter(Alert.user_id == user_id, Alert.dedup_key == dedup_key)
        .first()
    )
    if existing:
        return None

    alert = Alert(
        user_id=user_id,
        created_at=(now or datetime.now()).isoformat(),
        severity=severity,
        rule_id=rule_id,
        message=message,
        status="new",
        explanation=explanation,
        dedup_key=dedup_key,
    )
    db.add(alert)
    db.flush()
    return alert


def check_r1_missed_medication(db, user_id: int, now: Optional[datetime] = None) -> List[Alert]:
    """R1: a medication not taken 2 consecutive days (warning)."""
    raised = []
    meds = db.query(Medication).filter(Medication.user_id == user_id).all()
    days = _recent_dates(2)

    for med in meds:
        logs = {
            log.date: log
            for log in db.query(MedicationLog)
            .filter(
                MedicationLog.user_id == user_id,
                MedicationLog.medication_name == med.name,
                MedicationLog.date.in_(days),
            )
            .all()
        }
        missed_both = all(
            (d in logs and not logs[d].taken) or (d not in logs) for d in days
        )
        if missed_both:
            alert = _upsert_alert(
                db,
                user_id,
                rule_id="R1",
                severity="warning",
                message=f"{med.name} has not been taken for 2 days in a row.",
                explanation=(
                    f"MedicationLog shows no 'taken' entry for {med.name} on "
                    f"{days[0]} or {days[1]} (scheduled {med.scheduled_time})."
                ),
                dedup_key=f"R1:{med.name}:{days[-1]}",
                now=now,
            )
            if alert:
                raised.append(alert)
    return raised


def check_r2_skipped_meals(db, user_id: int, now: Optional[datetime] = None) -> List[Alert]:
    """R2: 2+ meals skipped in one day (warning)."""
    raised = []
    for d in _recent_dates(1):
        meals = (
            db.query(MealLog)
            .filter(MealLog.user_id == user_id, MealLog.date == d)
            .all()
        )
        skipped = [m for m in meals if not m.eaten]
        if len(skipped) >= 2:
            types = ", ".join(sorted(m.meal_type for m in skipped))
            alert = _upsert_alert(
                db,
                user_id,
                rule_id="R2",
                severity="warning",
                message=f"{len(skipped)} meals were skipped on {d}.",
                explanation=f"MealLog rows with eaten=False on {d}: {types}.",
                dedup_key=f"R2:{d}",
                now=now,
            )
            if alert:
                raised.append(alert)
    return raised


def check_r3_low_mood(db, user_id: int, now: Optional[datetime] = None) -> List[Alert]:
    """R3: mood <= 2 three consecutive days (warning)."""
    raised = []
    days = _recent_dates(3)
    logs = {
        log.date: log
        for log in db.query(MoodLog)
        .filter(MoodLog.user_id == user_id, MoodLog.date.in_(days))
        .all()
    }
    if all(d in logs and logs[d].mood <= 2 for d in days):
        values = [logs[d].mood for d in days]
        alert = _upsert_alert(
            db,
            user_id,
            rule_id="R3",
            severity="warning",
            message=f"Low mood reported for 3 days in a row ({days[0]} to {days[2]}).",
            explanation=f"MoodLog values on {days}: {values} (all <= 2).",
            dedup_key=f"R3:{days[-1]}",
            now=now,
        )
        if alert:
            raised.append(alert)
    return raised


def check_r5_no_checkin(db, user_id: int, now: Optional[datetime] = None) -> List[Alert]:
    """R5: no completed check-in in 24 simulated hours (info)."""
    raised = []
    from app.clock import now as simulated_now

    today_date = (now or simulated_now()).date()
    today = today_date.isoformat()
    yesterday = (today_date - timedelta(days=1)).isoformat()

    recent = (
        db.query(CheckIn)
        .filter(
            CheckIn.user_id == user_id,
            CheckIn.date.in_([today, yesterday]),
            CheckIn.completed.is_(True),
        )
        .first()
    )
    if not recent:
        alert = _upsert_alert(
            db,
            user_id,
            rule_id="R5",
            severity="info",
            message="No completed check-in in the last 24 hours.",
            explanation=f"No CheckIn row with completed=True on {yesterday} or {today}.",
            dedup_key=f"R5:{today}",
            now=now,
        )
        if alert:
            raised.append(alert)
    return raised


def run_escalation(db, user_id: int, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Run all deterministic escalation rules after a turn."""
    raised: List[Alert] = []
    raised.extend(check_r1_missed_medication(db, user_id, now))
    raised.extend(check_r2_skipped_meals(db, user_id, now))
    raised.extend(check_r3_low_mood(db, user_id, now))
    raised.extend(check_r5_no_checkin(db, user_id, now))
    db.flush()
    return [
        {
            "id": a.id,
            "rule_id": a.rule_id,
            "severity": a.severity,
            "message": a.message,
            "explanation": a.explanation,
        }
        for a in raised
    ]