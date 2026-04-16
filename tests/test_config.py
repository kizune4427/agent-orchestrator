import os
import pytest
from orchestrator.config import RunConfig


def test_run_config_defaults():
    cfg = RunConfig(run_id="20260414-120000-abc123")
    assert cfg.backend == "anthropic"
    assert cfg.planner_model == "claude-sonnet-4-6"
    assert cfg.evaluator_model == "claude-sonnet-4-6"
    assert cfg.advisor_model == "claude-opus-4-6"
    assert cfg.generator_model == "claude-sonnet-4-6"
    assert cfg.auto_approve is False
    assert cfg.parallel is False
    assert cfg.from_node is None


def test_run_config_is_frozen():
    cfg = RunConfig(run_id="20260414-120000-abc123")
    with pytest.raises(Exception):
        cfg.backend = "openrouter"  # type: ignore


def test_run_config_from_env(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openrouter")
    monkeypatch.setenv("PLANNER_MODEL", "mistralai/mixtral-8x7b")
    monkeypatch.setenv("EVALUATOR_MODEL", "openai/gpt-4o")
    monkeypatch.setenv("ADVISOR_MODEL", "anthropic/claude-opus-4-6")
    monkeypatch.setenv("GENERATOR_MODEL", "openai/gpt-4o-mini")
    cfg = RunConfig.from_env(run_id="test-id")
    assert cfg.backend == "openrouter"
    assert cfg.planner_model == "mistralai/mixtral-8x7b"
    assert cfg.evaluator_model == "openai/gpt-4o"
    assert cfg.advisor_model == "anthropic/claude-opus-4-6"
    assert cfg.generator_model == "openai/gpt-4o-mini"


def test_run_config_from_env_defaults(monkeypatch):
    for key in ("LLM_BACKEND", "PLANNER_MODEL", "EVALUATOR_MODEL", "ADVISOR_MODEL", "GENERATOR_MODEL"):
        monkeypatch.delenv(key, raising=False)
    cfg = RunConfig.from_env(run_id="test-id")
    assert cfg.backend == "anthropic"
    assert cfg.planner_model == "claude-sonnet-4-6"


def test_run_config_from_env_openrouter_defaults(monkeypatch):
    """Unspecified models fall back to provider-prefixed OR IDs, not Anthropic SDK names."""
    for key in ("PLANNER_MODEL", "EVALUATOR_MODEL", "ADVISOR_MODEL", "GENERATOR_MODEL"):
        monkeypatch.delenv(key, raising=False)
    cfg = RunConfig.from_env(run_id="test-id", backend="openrouter")
    assert cfg.backend == "openrouter"
    assert cfg.planner_model == "anthropic/claude-sonnet-4.6"
    assert cfg.evaluator_model == "anthropic/claude-sonnet-4.6"
    assert cfg.advisor_model == "anthropic/claude-opus-4.6"
    assert cfg.generator_model == "anthropic/claude-sonnet-4.6"


def test_run_config_from_env_openrouter_partial_override(monkeypatch):
    """Specified models kept; unspecified roles get OR-formatted defaults."""
    for key in ("PLANNER_MODEL", "EVALUATOR_MODEL", "ADVISOR_MODEL", "GENERATOR_MODEL"):
        monkeypatch.delenv(key, raising=False)
    cfg = RunConfig.from_env(
        run_id="test-id",
        backend="openrouter",
        planner_model="openai/gpt-4o",
        generator_model="openai/gpt-4o",
    )
    assert cfg.planner_model == "openai/gpt-4o"
    assert cfg.generator_model == "openai/gpt-4o"
    # evaluator and advisor fall back to OR-formatted defaults, not Anthropic SDK names
    assert "/" in cfg.evaluator_model
    assert "/" in cfg.advisor_model


def test_run_config_branches_default():
    cfg = RunConfig(run_id="test")
    assert cfg.branches == 2


def test_run_config_branches_custom():
    cfg = RunConfig(run_id="test", branches=3)
    assert cfg.branches == 3


def test_run_config_from_env_branches():
    cfg = RunConfig.from_env(run_id="test", branches=4)
    assert cfg.branches == 4
