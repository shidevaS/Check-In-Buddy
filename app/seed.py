"""Database seeding for Check-in Buddy.

Persona: Asha Nair, 72, family contact Rahul.
Meds: Metformin 08:00, Amlodipine 20:00 (synthetic labels only).

Variants: normal, demo, missed_meds, low_mood, declining.
CLI: python -m app.seed [--normal] [--variant demo]
"""
import argparse
import random
from datetime import date, datetime, timedelta

from app.database import engine, SessionLocal
from app.models import (
    Base,
    User,
    Medication,
    MedicationLog,
    MealLog,
    MoodLog,
    CheckIn,
    PersonalFact,
)

PERSONA = {
    "name": "Asha Nair",
    "age": 72,
    "family_contact_name": "Rahul",
    "timezone": "UTC",
}

MEDS = [
    {"name": "Metformin", "scheduled_time": "08:00"},
    {"name": "Amlodipine", "scheduled_time": "20:00"},
]

FACTS = [
    {"category": "person", "key": "son", "value": "Rahul"},
    {"category": "person", "key": "granddaughter", "value": "Meera"},
    {"category": "hobby", "key": "gardening", "value": "loves gardening"},
    {"category": "hobby", "key": "crosswords", "value": "does crosswords daily"},
    {"category": "preference", "key": "morning_tea", "value": "morning tea"},
    {
        "category": "date",
        "key": "upcoming_visit",
        "value": "family visit",
        "date_md": (date.today() + timedelta(days=3)).strftime("%m-%d"),
    },
]


def _clear(db):
    for table in reversed(Base.metadata.sorted_tables):
        db.execute(table.delete())
    db.commit()


def _create_user(db) -> User:
    user = User(**PERSONA)
    db.add(user)
    db.commit()
    db.refresh(user)
    for m in MEDS:
        db.add(Medication(user_id=user.id, **m))
    for f in FACTS:
        db.add(PersonalFact(user_id=user.id, created_at=datetime.now().isoformat(), **f))
    db.commit()
    return user


def _day(user_id, offset, *, meds_taken=None, meals_eaten=None, mood=4, completed=True):
    """Build a day's worth of rows. offset=0 is today."""
    d = (date.today() - timedelta(days=offset)).isoformat()
    rows = [CheckIn(user_id=user_id, date=d, completed=completed)]
    for med in MEDS:
        taken = True if meds_taken is None else meds_taken.get(med["name"], True)
        rows.append(
            MedicationLog(
                user_id=user_id,
                medication_name=med["name"],
                date=d,
                taken=taken,
                time_taken=med["scheduled_time"] if taken else "",
                source="manual",
            )
        )
    for meal_type in ("breakfast", "lunch", "dinner"):
        eaten = True if meals_eaten is None else meals_eaten.get(meal_type, True)
        rows.append(
            MealLog(
                user_id=user_id,
                date=d,
                meal_type=meal_type,
                eaten=eaten,
                notes="" if eaten else "skipped",
            )
        )
    rows.append(MoodLog(user_id=user_id, date=d, mood=mood, notes=""))
    return rows, d


def seed_database(normal: bool = False, variant: str = "demo", force: bool = False):
    """Seed the database. Idempotent unless force=True."""
    Base.metadata.create_all(bind=engine)
    if normal:
        variant = "normal"

    db = SessionLocal()
    try:
        existing = db.query(User).first()
        if existing and not force:
            return {"status": "already_seeded"}
        if existing:
            _clear(db)

        user = _create_user(db)
        rows = []
        rnd = random.Random(1234)

        if variant == "normal":
            for offset in range(7):
                day_rows, _ = _day(user.id, offset, mood=rnd.choice([4, 5]))
                rows.extend(day_rows)

        elif variant == "missed_meds":
            for offset in range(7):
                meds = {"Metformin": offset > 3, "Amlodipine": offset > 3}
                day_rows, _ = _day(user.id, offset, meds_taken=meds, mood=3)
                rows.extend(day_rows)

        elif variant == "low_mood":
            for offset in range(7):
                day_rows, _ = _day(user.id, offset, mood=2 if offset < 4 else 3)
                rows.extend(day_rows)

        elif variant == "declining":
            for offset in range(7):
                meds = {"Metformin": offset < 4, "Amlodipine": offset < 2}
                meals = {"dinner": offset >= 3, "lunch": offset >= 5}
                mood = max(1, 5 - (6 - offset) // 2)
                day_rows, _ = _day(
                    user.id, offset, meds_taken=meds, meals_eaten=meals, mood=mood
                )
                rows.extend(day_rows)

        else:  # demo (default): missed-medication pattern -> "Needs attention"
                # This variant is designed to trigger R1 (missed medication 2 consecutive days)
                # and R5 (no completed check-in) alerts for hackathon demonstration.
                # Metformin missed on days 4,5,6 (offset >= 4); Amlodipine missed on days 0,1,2 (offset < 3)
                for offset in range(7):
                    meds = {"Metformin": offset >= 4, "Amlodipine": offset < 3}
                    meals = {"breakfast": offset != 0}
                    day_rows, _ = _day(
                        user.id,
                        offset,
                        meds_taken=meds,
                        meals_eaten=meals,
                        mood=rnd.choice([2, 3, 4]),
                        completed=offset != 0,  # No check-in on day 0 to trigger R5
                    )
                    rows.extend(day_rows)

                db.add_all(rows)
                db.commit()

                # Run the real escalation engine over seeded history
                from app.escalation import run_escalation

                alerts = run_escalation(db, user.id)
                db.commit()

                return {
                    "status": "seeded",
                    "variant": variant,
                    "user_id": user.id,
                    "alerts": len(alerts),
                }
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def reset_demo(user_id: int = 1):
    """Reset demo state to the default 'demo' variant."""
    return seed_database(variant="demo", force=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Check-in Buddy")
    parser.add_argument("--normal", action="store_true", help="seed a normal week (zero alerts)")
    parser.add_argument(
        "--variant",
        choices=["normal", "demo", "missed_meds", "low_mood", "declining"],
        default="demo",
    )
    parser.add_argument("--force", action="store_true", help="wipe and reseed")
    args = parser.parse_args()
    print(seed_database(normal=args.normal, variant=args.variant, force=args.force))