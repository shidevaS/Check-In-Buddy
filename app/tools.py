"""Tools - the single source of truth.

Each tool is a plain Python function (user_id, **kwargs) -> {"ok","data","error"}.
The same JSON schemas feed the LLM providers and the MCP server, so a tool can
never drift between the two.
"""
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional

from app.clock import now
from app.database import SessionLocal
from app.models import (
    Alert,
    MealLog,
    Medication,
    MedicationLog,
    MoodLog,
    Reminder,
)


# --------------------------------------------------------------------------- #
# Core tools
# --------------------------------------------------------------------------- #
def log_medication(user_id: int, medication_name: str = "", time_taken: str = "", **_) -> Dict[str, Any]:
    """Log medication intake. Rejects unknown meds, upserts one row per day."""
    if not medication_name:
        return {"ok": False, "error": "medication_name is required"}
    db = SessionLocal()
    try:
        med = (
            db.query(Medication)
            .filter(
                Medication.user_id == user_id,
                Medication.name.ilike(medication_name),
            )
            .first()
        )
        if not med:
            known = [m.name for m in db.query(Medication).filter(Medication.user_id == user_id).all()]
            return {
                "ok": False,
                "error": f"Unknown medication '{medication_name}'. Known: {', '.join(known) or 'none'}",
            }

        day = now().date().isoformat()
        log = (
            db.query(MedicationLog)
            .filter(
                MedicationLog.user_id == user_id,
                MedicationLog.medication_name == med.name,
                MedicationLog.date == day,
            )
            .first()
        )
        if log:
            log.taken = True
            log.time_taken = time_taken or log.time_taken or now().strftime("%H:%M")
            log.source = "llm"
        else:
            log = MedicationLog(
                user_id=user_id,
                medication_name=med.name,
                date=day,
                taken=True,
                time_taken=time_taken or now().strftime("%H:%M"),
                source="llm",
            )
            db.add(log)
        db.commit()
        return {"ok": True, "data": {"medication": med.name, "date": day, "taken": True}}
    finally:
        db.close()


def log_meal(user_id: int, meal_type: str = "", notes: str = "", **_) -> Dict[str, Any]:
    """Log a meal (breakfast|lunch|dinner)."""
    if meal_type not in ("breakfast", "lunch", "dinner"):
        return {"ok": False, "error": "meal_type must be breakfast, lunch, or dinner"}
    db = SessionLocal()
    try:
        day = now().date().isoformat()
        log = (
            db.query(MealLog)
            .filter(
                MealLog.user_id == user_id,
                MealLog.date == day,
                MealLog.meal_type == meal_type,
            )
            .first()
        )
        if log:
            log.eaten = True
            log.notes = notes or log.notes
        else:
            log = MealLog(user_id=user_id, date=day, meal_type=meal_type, eaten=True, notes=notes)
            db.add(log)
        db.commit()
        return {"ok": True, "data": {"meal_type": meal_type, "date": day, "eaten": True}}
    finally:
        db.close()


def log_mood(user_id: int, mood: int = 0, notes: str = "", **_) -> Dict[str, Any]:
    """Log mood on a 1-5 scale."""
    try:
        mood = int(mood)
    except (TypeError, ValueError):
        return {"ok": False, "error": "mood must be an integer 1-5"}
    if not 1 <= mood <= 5:
        return {"ok": False, "error": "mood must be between 1 and 5"}
    db = SessionLocal()
    try:
        day = now().date().isoformat()
        log = MoodLog(user_id=user_id, date=day, mood=mood, notes=notes)
        db.add(log)
        db.commit()
        return {"ok": True, "data": {"mood": mood, "date": day}}
    finally:
        db.close()


def set_reminder(user_id: int, text: str = "", remind_at: str = "", **_) -> Dict[str, Any]:
    """Set a reminder. Accepts HH:MM (today) or an ISO timestamp; must be future."""
    if not text or not remind_at:
        return {"ok": False, "error": "text and remind_at are required"}
    try:
        if len(remind_at) <= 5 and ":" in remind_at:
            parsed = datetime.combine(now().date(), time.fromisoformat(remind_at))
        else:
            parsed = datetime.fromisoformat(remind_at)
    except ValueError:
        return {"ok": False, "error": "remind_at must be HH:MM or an ISO timestamp"}

    current = now()
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    if parsed <= current:
        return {"ok": False, "error": "remind_at must be in the future"}

    db = SessionLocal()
    try:
        reminder = Reminder(
            user_id=user_id,
            text=text,
            remind_at=parsed.isoformat(),
            status="pending",
        )
        db.add(reminder)
        db.commit()
        return {"ok": True, "data": {"reminder_id": reminder.id, "remind_at": reminder.remind_at}}
    finally:
        db.close()


def notify_family(user_id: int, reason: str = "", severity: str = "info", message: str = "", **_) -> Dict[str, Any]:
    """Raise an alert for the family contact (deduped by stable key)."""
    if severity not in ("info", "warning", "urgent"):
        return {"ok": False, "error": "severity must be info, warning, or urgent"}
    if not reason:
        return {"ok": False, "error": "reason is required"}

    from app.escalation import _upsert_alert

    db = SessionLocal()
    try:
        day = now().date().isoformat()
        alert = _upsert_alert(
            db,
            user_id,
            rule_id=reason,
            severity=severity,
            message=message or f"Family notified: {reason}",
            explanation=f"notify_family tool called with reason={reason}",
            dedup_key=f"{reason}:{day}",
            now=now(),
        )
        db.commit()
        if alert is None:
            return {"ok": True, "data": {"deduped": True}}
        return {"ok": True, "data": {"alert_id": alert.id, "severity": severity}}
    finally:
        db.close()


def get_recent_history(user_id: int, limit: int = 10, **_) -> Dict[str, Any]:
    """Recent check-ins, medications, meals, and moods."""
    db = SessionLocal()
    try:
        from app.models import CheckIn

        limit = max(1, min(int(limit or 10), 28))
        check_ins = (
            db.query(CheckIn).filter(CheckIn.user_id == user_id).order_by(CheckIn.date.desc()).limit(limit).all()
        )
        meds = (
            db.query(MedicationLog)
            .filter(MedicationLog.user_id == user_id)
            .order_by(MedicationLog.date.desc())
            .limit(limit * 2)
            .all()
        )
        moods = (
            db.query(MoodLog).filter(MoodLog.user_id == user_id).order_by(MoodLog.date.desc()).limit(limit).all()
        )
        meals = (
            db.query(MealLog).filter(MealLog.user_id == user_id).order_by(MealLog.date.desc()).limit(limit * 3).all()
        )
        return {
            "ok": True,
            "data": {
                "check_ins": [{"date": c.date, "completed": c.completed} for c in check_ins],
                "medications": [
                    {"date": m.date, "name": m.medication_name, "taken": m.taken, "time": m.time_taken}
                    for m in meds
                ],
                "moods": [{"date": m.date, "mood": m.mood} for m in moods],
                "meals": [{"date": m.date, "meal_type": m.meal_type, "eaten": m.eaten} for m in meals],
            },
        }
    finally:
        db.close()


def get_wellness_insights(user_id: int, **_) -> Dict[str, Any]:
    """Deterministic wellness score 0-100 with component breakdown and insights."""
    from app.analytics import wellness_report

    return {"ok": True, "data": wellness_report(user_id)}


# --------------------------------------------------------------------------- #
# Registry + schemas (single source of truth)
# --------------------------------------------------------------------------- #
TOOL_FUNCTIONS = [
    log_medication,
    log_meal,
    log_mood,
    set_reminder,
    notify_family,
    get_recent_history,
    get_wellness_insights,
]

TOOL_REGISTRY = {fn.__name__: (lambda args, _fn=fn: _fn(**args)) for fn in TOOL_FUNCTIONS}

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "log_medication",
            "description": "Log that a medication was taken. Rejects medications that are not on the user's list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "medication_name": {"type": "string", "description": "e.g. Metformin"},
                    "time_taken": {"type": "string", "description": "HH:MM, optional"},
                },
                "required": ["medication_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_meal",
            "description": "Log a meal the user ate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "meal_type": {"type": "string", "enum": ["breakfast", "lunch", "dinner"]},
                    "notes": {"type": "string"},
                },
                "required": ["meal_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_mood",
            "description": "Log the user's mood on a 1-5 scale (1 = very low, 5 = very good).",
            "parameters": {
                "type": "object",
                "properties": {
                    "mood": {"type": "integer", "minimum": 1, "maximum": 5},
                    "notes": {"type": "string"},
                },
                "required": ["mood"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a future reminder for the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "remind_at": {"type": "string", "description": "HH:MM or ISO timestamp"},
                },
                "required": ["text", "remind_at"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notify_family",
            "description": "Raise an alert for the family contact. Use sparingly and explain why.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "severity": {"type": "string", "enum": ["info", "warning", "urgent"]},
                    "message": {"type": "string"},
                },
                "required": ["reason", "severity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_history",
            "description": "Fetch the user's recent check-ins, medications, meals, and moods.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 10}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_wellness_insights",
            "description": "Get the deterministic wellness score, components, and insights.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def get_tool_schemas() -> List[Dict[str, Any]]:
    return TOOL_SCHEMAS