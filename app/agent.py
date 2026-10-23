"""Conversation orchestration; safety and escalation remain deterministic."""

import json
from typing import Any, Dict, List, Optional

from app.clock import now
from app.database import SessionLocal
from app.emotion_model import assess_emotion
from app.llm import LLMResponse, get_provider
from app.models import Message, PersonalFact, User
from app.safety import safety_pre_check
from app.tools import TOOL_REGISTRY, get_tool_schemas

MAX_TOOL_ITERATIONS = 4
MEMORY_DAYS = 7


def _save_message(db, user_id: int, role: str, content: str, tool_name: Optional[str] = None) -> None:
    db.add(
        Message(
            user_id=user_id,
            timestamp=now().isoformat(),
            role=role,
            content=content,
            tool_name=tool_name,
        )
    )
    db.commit()


def _memory_block(user_id: int) -> str:
    from app.tools import get_recent_history

    history = get_recent_history(user_id, limit=MEMORY_DAYS * 4)
    return json.dumps(history.get("data", {}), default=str)


def _facts_block(db, user_id: int) -> str:
    facts = db.query(PersonalFact).filter(PersonalFact.user_id == user_id).all()
    if not facts:
        return "No personal facts stored yet."
    return "; ".join(f"{fact.category}:{fact.key}={fact.value}" for fact in facts)


def _grounded_answer(message: str) -> Optional[Dict[str, Any]]:
    from app.rag import answer_wellness_question

    return answer_wellness_question(message)


def _learn_fact(db, user_id: int, message: str) -> Optional[Dict[str, str]]:
    from app.companion import extract_fact

    fact = extract_fact(message)
    if not fact:
        return None
    existing = (
        db.query(PersonalFact)
        .filter(
            PersonalFact.user_id == user_id,
            PersonalFact.category == fact["category"],
            PersonalFact.key == fact["key"],
        )
        .first()
    )
    if existing:
        existing.value = fact["value"]
    else:
        db.add(
            PersonalFact(
                user_id=user_id,
                category=fact["category"],
                key=fact["key"],
                value=fact["value"],
                created_at=now().isoformat(),
            )
        )
    db.commit()
    return fact


def _conversation_context(db, user_id: int) -> List[Dict[str, str]]:
    """Include recent user/assistant turns so follow-up questions have context."""
    history = (
        db.query(Message)
        .filter(
            Message.user_id == user_id,
            Message.role.in_(("user", "assistant")),
        )
        .order_by(Message.id.desc())
        .limit(12)
        .all()
    )
    return [
        {"role": message.role, "content": message.content}
        for message in reversed(history)
    ]


def _system_prompt(db, user_id: int, user_message: str) -> str:
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise ValueError(f"User {user_id} was not found.")
    emotion = assess_emotion(user_message)
    emotion_guidance = (
        f"The local dataset-trained text classifier tentatively suggests "
        f"{emotion['label']} (confidence {emotion['confidence']:.0%}). "
        "This is uncertain; do not state it as fact, diagnose, or reveal a "
        "classifier label. If appropriate, respond with gentle empathy and "
        "let the user describe their own feelings."
        if emotion
        else "Do not infer or label the user's emotions; ask gently if unclear."
    )
    return f"""You are Check-in Buddy, a warm voice companion doing a short daily wellness check-in with {user.name} ({user.age}), who lives alone. Family contact: {user.family_contact_name}.

You are a wellness companion, NOT a medical device. Never diagnose, interpret symptoms, or give medical or dosage advice. Never tell the user they are "fine" or "healthy". Always keep a gentle non-medical disclaimer available.

Things you remember about {user.name}: {_facts_block(db, user_id)}
Recent activity (last {MEMORY_DAYS} days): {_memory_block(user_id)}
Current simulated date: {now().date().isoformat()}
Tentative conversational tone guidance: {emotion_guidance}

Guidelines:
- Keep replies short, kind, and spoken-friendly.
- Use tools when the user wants to log something or asks for their history/insights.
- Only mention family escalation when a tool result says an alert was raised.
- Never invent facts; only use what is in memory above.
"""


def _execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    return tool(args)


def _run_escalation(db, user_id: int) -> List[Dict[str, Any]]:
    from app.escalation import run_escalation

    return run_escalation(db, user_id, now())


def agent_loop(user_id: int, user_message: str) -> Dict[str, Any]:
    """Run one turn, preserving tool calls and provider-fallback metadata."""
    db = SessionLocal()
    try:
        if db.query(User).filter(User.id == user_id).first() is None:
            raise ValueError(f"User {user_id} was not found.")

        _save_message(db, user_id, "user", user_message)

        safety = safety_pre_check(user_message)
        if safety:
            _save_message(db, user_id, "assistant", safety["response"])
            actions = [{"type": "safety_check", "result": safety["action"]}]
            if safety.get("urgent"):
                from app.tools import notify_family

                alert_result = notify_family(
                    user_id,
                    reason=safety["rule_id"],
                    severity="urgent",
                    message=safety["response"],
                )
                actions.append({"type": "alert_raised", "result": alert_result})
            return {
                "reply": safety["response"],
                "actions_taken": actions,
                "alerts": _run_escalation(db, user_id),
            }

        grounded = _grounded_answer(user_message)
        if grounded:
            _save_message(db, user_id, "assistant", grounded["reply"], "rag")
            return {
                "reply": grounded["reply"],
                "actions_taken": [{
                    "type": "grounded_answer",
                    "citations": grounded["citations"],
                }],
                "alerts": _run_escalation(db, user_id),
            }

        actions: List[Dict[str, Any]] = []
        fact = _learn_fact(db, user_id, user_message)
        if fact:
            actions.append({"type": "remembered_fact", "fact": fact})

        provider = get_provider()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _system_prompt(db, user_id, user_message)},
            *_conversation_context(db, user_id),
        ]
        tool_schemas = get_tool_schemas()
        fallback = None
        model = ""

        for _ in range(MAX_TOOL_ITERATIONS):
            response: LLMResponse = provider.chat(messages, tools=tool_schemas)
            model = response.model
            fallback = response.fallback or fallback
            if not response.tool_calls:
                reply = response.content or "I'm here whenever you're ready."
                _save_message(db, user_id, "assistant", reply)
                return {
                    "reply": reply,
                    "actions_taken": actions,
                    "alerts": _run_escalation(db, user_id),
                    "model": model,
                    "fallback": fallback,
                }

            messages.append({
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": response.tool_calls,
            })
            for call in response.tool_calls:
                function = call.get("function") or {}
                name = function.get("name", "")
                raw_arguments = function.get("arguments") or "{}"
                try:
                    args = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                args["user_id"] = user_id
                result = _execute_tool(name, args)
                actions.append({"type": "tool", "tool": name, "result": result})
                _save_message(db, user_id, "tool", json.dumps(result), name)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": json.dumps(result),
                })

        reply = "I've handled what I could. Could you say that another way?"
        _save_message(db, user_id, "assistant", reply)
        return {
            "reply": reply,
            "actions_taken": actions,
            "alerts": _run_escalation(db, user_id),
            "model": model,
            "fallback": fallback,
        }
    finally:
        db.close()
