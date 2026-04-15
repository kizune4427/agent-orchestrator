from __future__ import annotations

import os

import httpx

from orchestrator.agents.base import BaseAgentClient

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient(BaseAgentClient):
    """Calls OpenRouter's OpenAI-compatible chat completions endpoint."""

    def __init__(self, model: str, system_prompt: str) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self._api_key = os.environ.get("OPENROUTER_API_KEY", "")

    def run(self, user_message: str) -> str:
        if not self._api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable is not set."
            )
        response = httpx.post(
            _OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
            },
            timeout=120.0,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        if not text:
            raise RuntimeError(
                f"OpenRouterClient received empty response for model {self.model!r}"
            )
        return text
