import re
from typing import Optional, Dict, Any

# Emergency phrases that trigger immediate safety response
EMERGENCY_PHRASES = [
    "chest pain",
    "can't breathe",
    "shortness of breath",
    "fall",
    "fell",
    "black out",
    "passed out",
    "suicide",
    "self-harm",
    "kill myself",
    "end my life",
    "overdose",
    "poisoning",
]

# Medical-advice phrases that trigger refusal + redirect
MEDICAL_ADVICE_PHRASES = [
    "dosage",
    "how much",
    "how many",
    "prescribe",
    "prescription",
    "diagnose",
    "diagnosis",
    "symptoms",
    "symptom",
    "condition",
    "medical advice",
]


def safety_pre_check(message: str) -> Optional[Dict[str, Any]]:
    """Deterministic safety pre-check, independent of LLM.
    
    Returns a safety action dict if a trigger is found, None otherwise.
    """
    msg_lower = message.lower().strip()

    # Check for emergency phrases
    for phrase in EMERGENCY_PHRASES:
        if phrase in msg_lower:
            return {
                "action": "emergency",
                "response": "I'm concerned about your safety. Please contact emergency services or your family member immediately. I've raised an urgent alert.",
                "rule_id": "R4",
                "urgent": True,
            }

    # Check for medical advice phrases
    for phrase in MEDICAL_ADVICE_PHRASES:
        if phrase in msg_lower:
            return {
                "action": "medical_refusal",
                "response": "I cannot provide medical dosage advice or diagnoses. Please consult with a doctor or pharmacist for medical guidance.",
                "rule_id": "R3",
                "urgent": False,
            }

    # Check for "you're fine/healthy" patterns (never say this)
    if re.search(r"\byou're\s+(fine|healthy)\b", msg_lower):
        return {
            "action": "disclaimer",
            "response": "I'm a wellness companion, not a medical service. I cannot confirm or deny any health status.",
            "rule_id": "R3",
            "urgent": False,
        }

    return None