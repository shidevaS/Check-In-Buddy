"""Simulated clock.

All time reads in the app go through here so a demo can jump hours/days
instantly. State is persisted to a JSON file; tests can use an in-memory
override for isolation.
"""
import json
import os
from datetime import datetime, date, time, timedelta
from typing import Optional

CLOCK_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "clock_state.json")

_override: Optional[str] = None


def _read() -> dict:
    if os.path.exists(CLOCK_FILE):
        try:
            with open(CLOCK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"simulated_time": datetime.now().isoformat()}


def _write(state: dict) -> None:
    with open(CLOCK_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f)


def now() -> datetime:
    """Current simulated time (override wins over persisted state)."""
    if _override:
        return datetime.fromisoformat(_override)
    return datetime.fromisoformat(_read()["simulated_time"])


def set_time(iso_str: str) -> datetime:
    """Persist a new simulated time."""
    dt = datetime.fromisoformat(iso_str)
    _write({"simulated_time": dt.isoformat()})
    return dt


def set_override(iso_str: Optional[str]) -> None:
    """Set (or clear) an in-memory override for test isolation."""
    global _override
    _override = iso_str


def reset() -> datetime:
    """Reset the simulated clock to real now."""
    return set_time(datetime.now().isoformat())


def advance(hours: int = 0, days: int = 0) -> datetime:
    """Advance the simulated clock; runs the end-of-day job per midnight crossed."""
    current = now()
    target = current + timedelta(hours=hours, days=days)
    crossed = target.date() - current.date()
    set_time(target.isoformat())
    if crossed.days > 0:
        from app.jobs import end_of_day_job

        end_of_day_job()
    return target


def advance_to_evening() -> datetime:
    """Jump to 18:00 today."""
    current = now()
    return set_time(datetime.combine(current.date(), time(18, 0)).isoformat())


def advance_to_next_day() -> datetime:
    """Jump to 00:00 tomorrow, running the end-of-day job."""
    current = now()
    target = datetime.combine(current.date() + timedelta(days=1), time(0, 0))
    set_time(target.isoformat())
    from app.jobs import end_of_day_job

    end_of_day_job()
    return target


def today_str() -> str:
    return now().date().isoformat()