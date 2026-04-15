from __future__ import annotations

from typing import Literal

from orchestrator.agents.base import BaseAgentClient
from orchestrator.config import RunConfig


def make_client(
    role: Literal["planner", "evaluator", "advisor", "generator"],
    config: RunConfig,
    system_prompt: str,
) -> BaseAgentClient:
    """Return the appropriate client for this role based on RunConfig."""
    model = getattr(config, f"{role}_model")
    if config.backend == "openrouter":
        from orchestrator.agents.openrouter import OpenRouterClient
        return OpenRouterClient(model=model, system_prompt=system_prompt)
    from orchestrator.agents.client import AgentClient
    return AgentClient(name=role.capitalize(), model=model, system_prompt=system_prompt)
