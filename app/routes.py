"""HTTP API for check-ins, activity, reminders, and the family dashboard."""

import os

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.agent import agent_loop
from app.clock import now
from app.database import SessionLocal
from app.emotion_model import emotion_status
from app.escalation import run_escalation
from app.llm import ProviderError, provider_status
from app.models import (
    Alert,
    CheckIn,
    Message,
    MealLog,
    Medication,
    MedicationLog,
    MoodLog,
    Reminder,
    User,
)
from app.tools import log_meal, log_medication, log_mood, set_reminder, notify_family

router = APIRouter()


class ChatRequest(BaseModel):
    user_id: int = 1
    message: str = Field(min_length=1, max_length=2000)


class UserActivityRequest(BaseModel):
    user_id: int = 1


class MedicationRequest(UserActivityRequest):
    medication_name: str = Field(min_length=1, max_length=100)
    time_taken: str = Field(default="", max_length=30)


class MealRequest(UserActivityRequest):
    meal_type: str
    notes: str = Field(default="", max_length=500)


class MoodRequest(UserActivityRequest):
    mood: int = Field(ge=1, le=5)


class ReminderRequest(UserActivityRequest):
    text: str = Field(min_length=1, max_length=250)
    remind_at: str = Field(min_length=1, max_length=40)


class MedicationRequest(UserActivityRequest):
    medication_name: str = Field(min_length=1, max_length=100)
    time_taken: str = Field(default="", max_length=30)


class MealRequest(UserActivityRequest):
    meal_type: str
    notes: str = Field(default="", max_length=500)


class MoodRequest(UserActivityRequest):
    mood: int = Field(ge=1, le=5)
    notes: str = Field(default="", max_length=500)


class ReminderRequest(UserActivityRequest):
    text: str = Field(min_length=1, max_length=250)
    remind_at: str = Field(min_length=1, max_length=40)


def _require_user(db, user_id: int) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found. Seed the demo data first.")
    return user


def _dashboard_data(db, user_id: int) -> dict:
    user = _require_user(db, user_id)
    today = now().date().isoformat()
    medications = db.query(Medication).filter(Medication.user_id == user_id).order_by(Medication.scheduled_time).all()
    medication_logs = (
        db.query(MedicationLog)
        .filter(MedicationLog.user_id == user_id, MedicationLog.date == today)
        .all()
    )
    meal_logs = (
        db.query(MealLog)
        .filter(MealLog.user_id == user_id, MealLog.date == today)
        .all()
    )
    mood = (
        db.query(MoodLog)
        .filter(MoodLog.user_id == user_id, MoodLog.date == today)
        .order_by(MoodLog.id.desc())
        .first()
    )
    checkin = (
        db.query(CheckIn)
        .filter(CheckIn.user_id == user_id, CheckIn.date == today)
        .order_by(CheckIn.id.desc())
        .first()
    )
    reminders = (
        db.query(Reminder)
        .filter(Reminder.user_id == user_id, Reminder.status == "pending")
        .order_by(Reminder.remind_at)
        .all()
    )
    alerts = (
        db.query(Alert)
        .filter(Alert.user_id == user_id)
        .order_by(Alert.created_at.desc())
        .limit(20)
        .all()
    )
    log_by_name = {entry.medication_name.lower(): entry for entry in medication_logs}
    meal_by_type = {entry.meal_type: entry for entry in meal_logs}
    return {
        "user": {
            "id": user.id,
            "name": user.name,
            "age": user.age,
            "family_contact_name": user.family_contact_name,
        },
        "date": today,
        "medications": [
            {
                "name": medication.name,
                "scheduled_time": medication.scheduled_time,
                "taken": log_by_name.get(medication.name.lower()).taken
                if medication.name.lower() in log_by_name
                else False,
                "time_taken": log_by_name.get(medication.name.lower()).time_taken
                if medication.name.lower() in log_by_name
                else None,
            }
            for medication in medications
        ],
        "meals": [
            {
                "meal_type": meal_type,
                "eaten": meal_by_type[meal_type].eaten if meal_type in meal_by_type else False,
            }
            for meal_type in ("breakfast", "lunch", "dinner")
        ],
        "mood": mood.mood if mood else None,
        "check_in_completed": bool(checkin and checkin.completed),
        "reminders": [
            {"id": reminder.id, "text": reminder.text, "remind_at": reminder.remind_at}
            for reminder in reminders
        ],
        "alerts": [
            {
                "id": alert.id,
                "created_at": alert.created_at,
                "severity": alert.severity,
                "rule_id": alert.rule_id,
                "message": alert.message,
                "status": alert.status,
            }
            for alert in alerts
        ],
    }


@router.get("/status")
def get_status():
    status = provider_status()
    model_status = emotion_status()
    status.update({
        "status": "ok",
        "api_key_configured": (
            bool(os.getenv("NVIDIA_API_KEY"))
            if status["provider"] == "nvidia_nim"
            else bool(os.getenv("ANTHROPIC_API_KEY"))
            if status["provider"] == "anthropic"
            else True
        ),
        "emotion_model": model_status,
    })
    return status


@router.get("/chat/history/{user_id}")
def chat_history(user_id: int, limit: int = Query(default=30, ge=1, le=100)):
    db = SessionLocal()
    try:
        _require_user(db, user_id)
        messages = (
            db.query(Message)
            .filter(
                Message.user_id == user_id,
                Message.role.in_(("user", "assistant")),
            )
            .order_by(Message.id.desc())
            .limit(limit)
            .all()
        )
        return {
            "messages": [
                {"role": message.role, "content": message.content}
                for message in reversed(messages)
            ]
        }
    finally:
        db.close()


@router.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be blank.")
    try:
        result = agent_loop(request.user_id, message)
        # Enhance response with additional metadata for hackathon demo
        return {
            "reply": result.get("reply", "I'm here whenever you're ready."),
            "model": result.get("model", "unknown"),
            "fallback": result.get("fallback"),
            "actions_taken": result.get("actions_taken", []),
            "alerts": result.get("alerts", []),
        }
    except ProviderError as exc:
        code = 503 if "NVIDIA_API_KEY is not configured" in str(exc) else 502
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/chat/test")
def chat_test(request: ChatRequest):
    """Test endpoint that works in deterministic mode without NVIDIA API key."""
    from app.llm import get_provider, LLM_PROVIDER, DeterministicMockProvider, ProviderError, _provider_registry
    from app.agent import agent_loop
    
    # Save and override provider to deterministic mock for test endpoint
    original_provider = _provider_registry.get("nvidia_nim")
    _provider_registry["nvidia_nim"] = DeterministicMockProvider()
    
    try:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=422, detail="Message cannot be blank.")
        result = agent_loop(request.user_id, message)
        return {
            "reply": result.get("reply", "I'm here whenever you're ready."),
            "model": result.get("model", "deterministic-fallback"),
            "fallback": result.get("fallback"),
            "actions_taken": result.get("actions_taken", []),
            "alerts": result.get("alerts", []),
            "provider": "deterministic-mock",
        }
    except ProviderError as exc:
        code = 503 if "NVIDIA_API_KEY is not configured" in str(exc) else 502
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        # Restore original provider
        _provider_registry["nvidia_nim"] = original_provider


@router.get("/dashboard/{user_id}")
def dashboard(user_id: int):
    db = SessionLocal()
    try:
        return _dashboard_data(db, user_id)
    finally:
        db.close()


@router.get("/family/{user_id}")
def family_dashboard(user_id: int):
    db = SessionLocal()
    try:
        return _dashboard_data(db, user_id)
    finally:
        db.close()


@router.post("/check-ins")
def complete_check_in(request: UserActivityRequest):
    db = SessionLocal()
    try:
        _require_user(db, request.user_id)
        today = now().date().isoformat()
        checkin = (
            db.query(CheckIn)
            .filter(CheckIn.user_id == request.user_id, CheckIn.date == today)
            .first()
        )
        if checkin is None:
            checkin = CheckIn(user_id=request.user_id, date=today, completed=True)
            db.add(checkin)
        else:
            checkin.completed = True
        db.commit()
        alerts = run_escalation(db, request.user_id, now())
        db.commit()
        return {"ok": True, "date": today, "completed": True, "alerts": alerts}
    finally:
        db.close()


@router.post("/logs/medication")
def medication_log(request: MedicationRequest):
    db = SessionLocal()
    try:
        _require_user(db, request.user_id)
    finally:
        db.close()
    result = log_medication(
        request.user_id,
        medication_name=request.medication_name,
        time_taken=request.time_taken,
    )
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/logs/meal")
def meal_log(request: MealRequest):
    db = SessionLocal()
    try:
        _require_user(db, request.user_id)
    finally:
        db.close()
    result = log_meal(
        request.user_id,
        meal_type=request.meal_type,
        notes=request.notes,
    )
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/logs/mood")
def mood_log(request: MoodRequest):
    db = SessionLocal()
    try:
        _require_user(db, request.user_id)
    finally:
        db.close()
    result = log_mood(
        request.user_id,
        mood=request.mood,
        notes=request.notes,
    )
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/reminders")
def create_reminder(request: ReminderRequest):
    db = SessionLocal()
    try:
        _require_user(db, request.user_id)
    finally:
        db.close()
    result = set_reminder(
        request.user_id,
        text=request.text,
        remind_at=request.remind_at,
    )
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: int, user_id: int = Query(default=1)):
    db = SessionLocal()
    try:
        _require_user(db, user_id)
        alert = (
            db.query(Alert)
            .filter(Alert.id == alert_id, Alert.user_id == user_id)
            .first()
        )
        if alert is None:
            raise HTTPException(status_code=404, detail="Alert not found.")
        alert.status = "acknowledged"
        db.commit()
        return {"ok": True, "alert_id": alert.id, "status": alert.status}
    finally:
        db.close()
