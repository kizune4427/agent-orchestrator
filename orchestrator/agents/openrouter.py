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
        if "/" not in self.model:
            raise ValueError(
                f"OpenRouter model {self.model!r} looks wrong — "
                "expected provider-prefixed format like 'anthropic/claude-sonnet-4.6' or "
                "'openai/gpt-4o'. "
                "Unprefixed names (e.g. 'claude-sonnet-4-6') are Anthropic SDK IDs "
                "and will be rejected by OpenRouter."
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
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            hint = ""
            if response.status_code == 400:
                hint = (
                    f" Check that {self.model!r} is a valid OpenRouter model ID "
                    "(see https://openrouter.ai/models)."
                )
            raise RuntimeError(
                f"OpenRouter request failed HTTP {response.status_code} "
                f"for model {self.model!r}: {detail}{hint}"
            ) from exc
        text = response.json()["choices"][0]["message"]["content"]
        if not text:
            raise RuntimeError(
                f"OpenRouterClient received empty response for model {self.model!r}"
            )
        return text
