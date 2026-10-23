import requests
import pytest

from app.llm import NVIDIA_NIMProvider, ProviderError


def _completion(model: str) -> dict:
    return {
        "model": model,
        "choices": [{"message": {"content": "I'm here with you.", "tool_calls": []}}],
        "usage": {"total_tokens": 7},
    }


def test_nvidia_timeout_uses_nemotron_and_reports_fallback():
    provider = NVIDIA_NIMProvider()
    provider.model = "deepseek-ai/deepseek-v4.1-flash"
    provider.fallback_model = "nvidia/nemotron-3.5-lightning-30b-a3b"
    models = []

    def request(model, messages, tools, tool_choice):
        models.append(model)
        if model == provider.model:
            raise requests.exceptions.Timeout("primary timed out")
        return _completion(model)

    provider._request = request
    response = provider.chat([{"role": "user", "content": "Hello"}])

    assert models == [provider.model, provider.fallback_model]
    assert response.content == "I'm here with you."
    assert response.model == provider.fallback_model
    assert response.fallback == {
        "reason": "timeout",
        "primary_model": provider.model,
        "model": provider.fallback_model,
        "primary_model_name": "DeepSeek V4.1 Flash",
        "model_name": "Nemotron 3.5 Lightning",
    }


def test_nvidia_timeout_surfaces_fallback_failure():
    provider = NVIDIA_NIMProvider()
    provider.model = "deepseek-ai/deepseek-v4.1-flash"
    provider.fallback_model = "nvidia/nemotron-3.5-lightning-30b-a3b"

    def request(model, messages, tools, tool_choice):
        raise requests.exceptions.Timeout(f"{model} timed out")

    provider._request = request

    with pytest.raises(ProviderError, match="fallback"):
        provider.chat([{"role": "user", "content": "Hello"}])
