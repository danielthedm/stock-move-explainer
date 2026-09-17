"""Minimal chat-completion clients. Plain HTTP keeps the dependency surface
small and makes both vendors the same ~20 lines behind one Protocol."""
from typing import Protocol

import httpx

from app.providers.base import request_json

Message = dict  # {"role": "user" | "assistant", "content": str}


class LLMProvider(Protocol):
    model: str

    def complete(self, system: str, messages: list[Message], max_tokens: int = 900) -> str: ...


class AnthropicLLM:
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str | None = None, client: httpx.Client | None = None):
        self.model = model or "claude-haiku-4-5-20251001"
        self._client = client or httpx.Client(timeout=60)
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def complete(self, system: str, messages: list[Message], max_tokens: int = 900) -> str:
        body = {"model": self.model, "max_tokens": max_tokens, "system": system, "messages": messages}
        data = request_json(self._client, "POST", self.URL, headers=self._headers, json=body)
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()


class OpenAILLM:
    URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str, model: str | None = None, client: httpx.Client | None = None):
        self.model = model or "gpt-4o-mini"
        self._client = client or httpx.Client(timeout=60)
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def complete(self, system: str, messages: list[Message], max_tokens: int = 900) -> str:
        body = {
            "model": self.model,
            "max_completion_tokens": max_tokens,  # accepted by all current chat models
            "messages": [{"role": "system", "content": system}, *messages],
        }
        data = request_json(self._client, "POST", self.URL, headers=self._headers, json=body)
        return (data["choices"][0]["message"].get("content") or "").strip()
