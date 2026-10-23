"""Provider-agnostic chat with a primary and fallback NVIDIA NIM model."""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia_nim").strip().lower()
BASE_URL = os.getenv(
    "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
).rstrip("/")
API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.getenv(
    "NVIDIA_NIM_MODEL", "nvidia/nemotron-3-nano-30b-a3b"
)
NVIDIA_FALLBACK_MODEL = os.getenv(
    "NVIDIA_NIM_FALLBACK_MODEL", ""
)
NVIDIA_TIMEOUT = float(os.getenv("NVIDIA_NIM_TIMEOUT", "60"))

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")


def model_display_name(model: str) -> str:
    """Make known model IDs readable in the app without hiding custom IDs."""
    lowered = model.lower()
    if "nemotron-3-nano-30b-a3b" in lowered:
        return "NVIDIA Nemotron 3 Nano 30B A3B"
    if "deepseek" in lowered and "v4.1" in lowered:
        return "DeepSeek V4.1 Flash"
    if "nemotron" in lowered and "3.5" in lowered:
        return "Nemotron 3.5 Lightning"
    return model


class ProviderError(RuntimeError):
    """A configured language-model provider could not return a response."""


class LLMResponse:
    """Normalized provider response, including optional fallback metadata."""

    def __init__(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        model: str = "",
        usage: Optional[Dict[str, Any]] = None,
        fallback: Optional[Dict[str, str]] = None,
    ):
        self.content = content
        self.tool_calls = tool_calls or []
        self.model = model
        self.usage = usage or {}
        self.fallback = fallback


class LLMProvider:
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> LLMResponse:
        raise NotImplementedError


class NVIDIA_NIMProvider(LLMProvider):
    """NVIDIA NIM client; timeouts on the primary model trigger one fallback."""

    def __init__(self):
        self.base_url = BASE_URL
        self.api_key = API_KEY
        self.model = NVIDIA_MODEL
        self.fallback_model = NVIDIA_FALLBACK_MODEL
        self.timeout = NVIDIA_TIMEOUT

    def _request(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
    ) -> Dict[str, Any]:
        import requests

        if not self.api_key:
            raise ProviderError("NVIDIA_API_KEY is not configured.")

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 700,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ProviderError("NVIDIA NIM returned an invalid response.")
        return data

    @staticmethod
    def _normalize_response(
        data: Dict[str, Any],
        requested_model: str,
        fallback: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise ProviderError("NVIDIA NIM returned no completion choices.")
        message = choices[0].get("message") or {}
        normalized_calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments", "{}")
            normalized_calls.append({
                "id": call.get("id", ""),
                "type": "function",
                "function": {
                    "name": function.get("name", ""),
                    "arguments": arguments,
                },
            })
        content = message.get("content") or ""
        if not content and not normalized_calls:
            raise ProviderError("NVIDIA NIM returned an empty completion.")
        return LLMResponse(
            content=content,
            tool_calls=normalized_calls,
            model=data.get("model", requested_model),
            usage=data.get("usage") or {},
            fallback=fallback,
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> LLMResponse:
        import requests

        try:
            data = self._request(self.model, messages, tools, tool_choice)
            return self._normalize_response(data, self.model)
        except requests.exceptions.Timeout as primary_error:
            if not self.fallback_model or self.fallback_model == self.model:
                raise ProviderError(
                    f"NVIDIA NIM primary model {self.model} timed out and no "
                    "distinct fallback model is configured."
                ) from primary_error

            try:
                data = self._request(
                    self.fallback_model, messages, tools, tool_choice
                )
            except requests.exceptions.RequestException as fallback_error:
                raise ProviderError(
                    f"NVIDIA NIM primary model {self.model} timed out; fallback "
                    f"{self.fallback_model} also failed: {fallback_error}"
                ) from fallback_error
            except ProviderError:
                raise

            return self._normalize_response(
                data,
                self.fallback_model,
                fallback={
                    "reason": "timeout",
                    "primary_model": self.model,
                    "model": self.fallback_model,
                    "primary_model_name": model_display_name(self.model),
                    "model_name": model_display_name(self.fallback_model),
                },
            )
        except requests.exceptions.RequestException as exc:
            raise ProviderError(
                f"NVIDIA NIM request failed for {self.model}: {exc}"
            ) from exc


class DeterministicMockProvider(LLMProvider):
    """Simple local provider for offline development and tests."""

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> LLMResponse:
        last_message = str(messages[-1].get("content", "")).lower() if messages else ""
        if any(term in last_message for term in ("chest pain", "can't breathe", "fell", "self-harm")):
            content = (
                "I'm concerned about your safety. Please contact emergency "
                "services or your family member immediately."
            )
        elif "how are you" in last_message or "check-in" in last_message:
            content = "I'm here with you. How are you feeling today?"
        else:
            content = (
                "I'm here to help with your daily wellness check-in. You can "
                "log medication, meals, mood, or set a reminder."
            )
        return LLMResponse(content=content, model="deterministic-mock")


class AnthropicProvider(LLMProvider):
    """Anthropic client for deployments that explicitly select that provider."""

    def __init__(self):
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not configured.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = ANTHROPIC_MODEL

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> LLMResponse:
        import json

        system_message = next(
            (message.get("content", "") for message in messages if message.get("role") == "system"),
            "",
        )
        conversation = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            if role == "tool":
                result = {
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id", ""),
                    "content": message.get("content", ""),
                }
                if (
                    conversation
                    and conversation[-1]["role"] == "user"
                    and isinstance(conversation[-1]["content"], list)
                    and all(block.get("type") == "tool_result" for block in conversation[-1]["content"])
                ):
                    conversation[-1]["content"].append(result)
                else:
                    conversation.append({"role": "user", "content": [result]})
                continue
            tool_calls = message.get("tool_calls") or []
            if role == "assistant" and tool_calls:
                content = []
                if message.get("content"):
                    content.append({"type": "text", "text": message["content"]})
                for call in tool_calls:
                    function = call.get("function") or {}
                    arguments = function.get("arguments") or "{}"
                    try:
                        input_data = json.loads(arguments) if isinstance(arguments, str) else arguments
                    except json.JSONDecodeError:
                        input_data = {}
                    content.append({
                        "type": "tool_use",
                        "id": call.get("id", ""),
                        "name": function.get("name", ""),
                        "input": input_data if isinstance(input_data, dict) else {},
                    })
                conversation.append({"role": "assistant", "content": content})
                continue
            if role in ("user", "assistant"):
                conversation.append({"role": role, "content": message.get("content") or ""})
        anthropic_tools = [
            {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "input_schema": tool["function"]["parameters"],
            }
            for tool in (tools or [])
        ]
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.3,
                system=system_message,
                messages=conversation,
                tools=anthropic_tools or None,
                tool_choice={"type": "auto"} if anthropic_tools and tool_choice != "none" else None,
            )
        except Exception as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        content = ""
        tool_calls = []
        for block in response.content:
            if getattr(block, "type", "") == "text":
                content += block.text
            elif getattr(block, "type", "") == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return LLMResponse(content, tool_calls, self.model, usage)


_provider_registry: Dict[str, LLMProvider] = {
    "nvidia_nim": NVIDIA_NIMProvider(),
    "deterministic": DeterministicMockProvider(),
}


def get_provider() -> LLMProvider:
    """Get the configured provider, creating optional integrations lazily."""
    if LLM_PROVIDER == "anthropic":
        if "anthropic" not in _provider_registry:
            _provider_registry["anthropic"] = AnthropicProvider()
        return _provider_registry["anthropic"]
    try:
        return _provider_registry[LLM_PROVIDER]
    except KeyError:
        # Fall back to deterministic mock if configured provider is not available
        if "deterministic-mock" not in _provider_registry:
            _provider_registry["deterministic-mock"] = DeterministicMockProvider()
        return _provider_registry["deterministic-mock"]


def provider_status() -> Dict[str, str]:
    """Return safe model configuration for the status endpoint."""
    if LLM_PROVIDER == "nvidia_nim":
        return {
            "provider": LLM_PROVIDER,
            "model": NVIDIA_MODEL,
            "model_name": model_display_name(NVIDIA_MODEL),
            "fallback_model": NVIDIA_FALLBACK_MODEL,
            "fallback_model_name": model_display_name(NVIDIA_FALLBACK_MODEL),
        }
    if LLM_PROVIDER == "anthropic":
        return {
            "provider": LLM_PROVIDER,
            "model": ANTHROPIC_MODEL,
            "model_name": ANTHROPIC_MODEL,
        }
    return {"provider": LLM_PROVIDER, "model": "deterministic-mock"}
