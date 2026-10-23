"""Load and query the optional classifier trained from the supplied datasets."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from app.database import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "trained_models" / "emotion_classifier.joblib"


@lru_cache(maxsize=1)
def _load_model() -> Optional[Dict[str, Any]]:
    if not MODEL_PATH.is_file():
        return None
    import joblib

    result = joblib.load(MODEL_PATH)
    if not isinstance(result, dict) or "pipeline" not in result or "labels" not in result:
        raise ValueError(f"Invalid emotion classifier artifact: {MODEL_PATH}")
    return result


def emotion_status() -> Dict[str, Any]:
    """Expose model training readiness without loading its artifact."""
    try:
        bundle = _load_model()
    except (OSError, ValueError, ImportError) as exc:
        return {"trained": False, "error": str(exc)}
    if bundle is None:
        return {
            "trained": False,
            "training_command": "python -m app.train_emotion_model",
            "datasets_present": {
                "go_emotions_dataset.csv.zip": (
                    PROJECT_ROOT / "go_emotions_dataset.csv.zip"
                ).is_file(),
                "text.csv.zip": (PROJECT_ROOT / "text.csv.zip").is_file(),
            },
        }
    return {
        "trained": True,
        "labels": bundle["labels"],
        "accuracy": bundle.get("accuracy"),
        "trained_at": bundle.get("trained_at"),
        "examples": bundle.get("examples"),
    }


def assess_emotion(message: str) -> Optional[Dict[str, Any]]:
    """Predict coarse emotional tone; return None when uncertain or untrained."""
    bundle = _load_model()
    if bundle is None or not message.strip():
        return None
    probabilities = bundle["pipeline"].predict_proba([message])[0]
    best_index = int(probabilities.argmax())
    confidence = float(probabilities[best_index])
    if confidence < 0.48:
        return None
    return {
        "label": str(bundle["labels"][best_index]),
        "confidence": round(confidence, 3),
    }
