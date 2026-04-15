from __future__ import annotations

import os

from orchestrator.agents.base import BaseAgentClient


class OpenRouterClient(BaseAgentClient):
    """Calls OpenRouter's OpenAI-compatible chat completions endpoint."""

    def __init__(self, model: str, system_prompt: str) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self._api_key = os.environ.get("OPENROUTER_API_KEY", "")

    def run(self, user_message: str) -> str:
        raise NotImplementedError("OpenRouterClient.run() implemented in Task 5")
