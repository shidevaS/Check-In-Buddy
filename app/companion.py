from app.database import SessionLocal
from app.models import User, PersonalFact
import re
from typing import Dict, Optional


_FACT_PATTERNS = (
    (re.compile(r"\bI love\s+(.+)", re.IGNORECASE), "hobby", "likes"),
    (re.compile(r"\bI like\s+(.+)", re.IGNORECASE), "preference", "likes"),
    (re.compile(r"\bI prefer\s+(.+)", re.IGNORECASE), "preference", "prefers"),
    (re.compile(r"\bmy favorite\s+([a-z ]+?)\s+is\s+(.+)", re.IGNORECASE), "preference", None),
)


def extract_fact(message: str) -> Optional[Dict[str, str]]:
    """Extract only explicit likes/preferences; never infer health information."""
    text = " ".join(message.strip().split())
    if not text or len(text) > 300:
        return None
    for pattern, category, key in _FACT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if key is None:
            fact_key = "favorite_" + re.sub(r"\W+", "_", match.group(1).strip().lower()).strip("_")
            value = match.group(2).strip(" .!?")
        else:
            value = match.group(1).strip(" .!?")
            if not value or len(value) > 100:
                return None
            fact_key = key + "_" + re.sub(r"\W+", "_", value.lower()).strip("_")[:60]
        if value:
            return {"category": category, "key": fact_key, "value": value[:100]}
    return None


def get_personal_facts(user_id: int) -> str:
    """Get personal facts for a user, formatted for system prompt."""
    db = SessionLocal()
    try:
        facts = db.query(PersonalFact).filter(PersonalFact.user_id == user_id).all()
        if not facts:
            return "No personal facts stored yet."

        formatted = []
        for f in facts:
            category_map = {
                "person": "Person",
                "hobby": "Hobby",
                "preference": "Preference",
                "date": "Important Date",
                "other": "Other",
            }
            cat = category_map.get(f.category, f.category)
            formatted.append(f"{cat}: {f.key} = {f.value}")

        return "; ".join(formatted)
    finally:
        db.close()


def get_companion_profile(user_id: int) -> str:
    """Get companion profile info."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return "User not found."

        return f"Name: {user.name}, Age: {user.age}, Family contact: {user.family_contact_name}, Timezone: {user.timezone}"
    finally:
        db.close()