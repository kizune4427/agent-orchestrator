from unittest.mock import patch, MagicMock
import pytest
from orchestrator.config import RunConfig
from orchestrator.agents.factory import make_client
from orchestrator.agents.base import BaseAgentClient


def test_make_client_returns_agent_client_for_anthropic():
    cfg = RunConfig(run_id="test", backend="anthropic")
    client = make_client("planner", cfg, "system prompt")
    from orchestrator.agents.client import AgentClient
    assert isinstance(client, AgentClient)


def test_make_client_returns_openrouter_client_for_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg = RunConfig(run_id="test", backend="openrouter", planner_model="openai/gpt-4o")
    client = make_client("planner", cfg, "system prompt")
    from orchestrator.agents.openrouter import OpenRouterClient
    assert isinstance(client, OpenRouterClient)


def test_make_client_uses_correct_model_per_role():
    cfg = RunConfig(
        run_id="test",
        backend="anthropic",
        planner_model="claude-opus-4-6",
        evaluator_model="claude-haiku-4-5-20251001",
    )
    planner_client = make_client("planner", cfg, "prompt")
    evaluator_client = make_client("evaluator", cfg, "prompt")
    assert planner_client.model == "claude-opus-4-6"
    assert evaluator_client.model == "claude-haiku-4-5-20251001"


def test_base_agent_client_is_abstract():
    with pytest.raises(TypeError):
        BaseAgentClient()  # type: ignore
