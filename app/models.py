from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    family_contact_name = Column(String, nullable=False)
    timezone = Column(String, nullable=False, default="UTC")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    medications = relationship("Medication", back_populates="user")
    medication_logs = relationship("MedicationLog", back_populates="user")
    meal_logs = relationship("MealLog", back_populates="user")
    mood_logs = relationship("MoodLog", back_populates="user")
    reminders = relationship("Reminder", back_populates="user")
    alerts = relationship("Alert", back_populates="user")
    messages = relationship("Message", back_populates="user")
    check_ins = relationship("CheckIn", back_populates="user")
    personal_facts = relationship("PersonalFact", back_populates="user")


class Medication(Base):
    __tablename__ = "medication"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    name = Column(String, nullable=False)
    scheduled_time = Column(String, nullable=False)  # HH:MM

    user = relationship("User", back_populates="medications")


class MedicationLog(Base):
    __tablename__ = "medication_log"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    medication_name = Column(String, nullable=False)
    date = Column(String, nullable=False)  # YYYY-MM-DD
    taken = Column(Boolean, default=False)
    time_taken = Column(String, nullable=True)  # HH:MM or ISO
    source = Column(String, nullable=True)  # llm, manual, reminder

    user = relationship("User", back_populates="medication_logs")


class MealLog(Base):
    __tablename__ = "meal_log"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    date = Column(String, nullable=False)  # YYYY-MM-DD
    meal_type = Column(String, nullable=False)  # breakfast | lunch | dinner
    eaten = Column(Boolean, default=False)
    notes = Column(String, nullable=True)

    user = relationship("User", back_populates="meal_logs")


class MoodLog(Base):
    __tablename__ = "mood_log"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    date = Column(String, nullable=False)  # YYYY-MM-DD
    mood = Column(Integer, nullable=False)  # 1-5
    notes = Column(String, nullable=True)

    user = relationship("User", back_populates="mood_logs")


class Reminder(Base):
    __tablename__ = "reminder"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    text = Column(String, nullable=False)
    remind_at = Column(String, nullable=False)  # ISO timestamp
    status = Column(String, nullable=False, default="pending")  # pending | sent | acknowledged

    user = relationship("User", back_populates="reminders")


class Alert(Base):
    __tablename__ = "alert"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    created_at = Column(String, nullable=False)  # naive ISO
    severity = Column(String, nullable=False)  # info | warning | urgent
    rule_id = Column(String, nullable=False)
    message = Column(String, nullable=False)
    status = Column(String, nullable=False, default="new")  # new | acknowledged
    explanation = Column(String, nullable=True)
    dedup_key = Column(String, nullable=True)

    user = relationship("User", back_populates="alerts")


class Message(Base):
    __tablename__ = "message"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    timestamp = Column(String, nullable=False)  # naive ISO
    role = Column(String, nullable=False)  # user | assistant | tool
    content = Column(String, nullable=False)
    tool_name = Column(String, nullable=True)  # if from tool

    user = relationship("User", back_populates="messages")


class CheckIn(Base):
    __tablename__ = "check_in"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    date = Column(String, nullable=False)  # YYYY-MM-DD
    completed = Column(Boolean, default=False)

    user = relationship("User", back_populates="check_ins")


class PersonalFact(Base):
    __tablename__ = "personal_fact"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    category = Column(String, nullable=False)  # person | hobby | preference | date | other
    key = Column(String, nullable=False)
    value = Column(String, nullable=False)
    date_md = Column(String, nullable=True)  # MM-DD
    created_at = Column(String, nullable=False, default=lambda: datetime.now().isoformat())

    user = relationship("User", back_populates="personal_facts")