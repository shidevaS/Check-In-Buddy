"""Tiny keyword retriever over app-owned, general wellness guidance."""

from pathlib import Path
from typing import Any, Dict, Optional

TIPS_FILE = Path(__file__).resolve().parent.parent / "docs" / "care_tips.md"


def answer_wellness_question(message: str) -> Optional[Dict[str, Any]]:
    """Return a cited tip for clearly matching general wellness questions."""
    text = message.lower()
    if not any(
        prompt in text
        for prompt in ("?", "how can", "how do", "any tips", "what helps", "tell me about")
    ):
        return None
    sections = TIPS_FILE.read_text(encoding="utf-8").split("\n## ")
    matches = []
    for section in sections[1:]:
        title, _, body = section.partition("\n")
        topic = title.lower()
        topic_terms = {
            "hydration": ("water", "hydrate", "drink", "fluid"),
            "gentle movement": ("movement", "exercise", "walk", "stretch"),
            "sleep routine": ("sleep", "bedtime", "rest", "wind down"),
            "staying connected": ("friend", "family", "social", "lonely", "connected"),
            "balanced meals": ("meal", "food", "eat", "eating"),
        }.get(topic, (topic,))
        score = sum(1 for term in topic_terms if term in text)
        if score:
            matches.append((score, title.strip(), body.strip()))
    if not matches:
        return None
    score, title, body = max(matches, key=lambda entry: entry[0])
    excerpt = " ".join(body.split())
    return {
        "reply": excerpt,
        "citations": [{"title": title, "source": "docs/care_tips.md"}],
    }
