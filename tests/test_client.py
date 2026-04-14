# tests/test_client.py
from unittest.mock import MagicMock, patch
import pytest
from orchestrator.agents.client import AgentClient


def _make_mock_anthropic(agent_text: str = '{"summary": "test"}'):
    """Build a mock anthropic.Anthropic() with agents/sessions wired up."""
    mock_client = MagicMock()

    # agents.create returns an object with .id and .version
    mock_agent = MagicMock()
    mock_agent.id = "agent_test123"
    mock_agent.version = 1
    mock_client.beta.agents.create.return_value = mock_agent

    # environments.create
    mock_env = MagicMock()
    mock_env.id = "env_test123"
    mock_client.beta.environments.create.return_value = mock_env

    # sessions.create
    mock_session = MagicMock()
    mock_session.id = "sesn_test123"
    mock_client.beta.sessions.create.return_value = mock_session

    # sessions.events.send (fire and forget)
    mock_client.beta.sessions.events.send.return_value = None

    # sessions.stream — returns a context manager yielding events
    mock_text_event = MagicMock()
    mock_text_event.type = "agent.message"
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = agent_text
    mock_text_event.content = [mock_text_block]

    mock_idle_event = MagicMock()
    mock_idle_event.type = "session.status_idle"
    mock_stop_reason = MagicMock()
    mock_stop_reason.type = "end_turn"
    mock_idle_event.stop_reason = mock_stop_reason

    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(
        return_value=iter([mock_text_event, mock_idle_event])
    )
    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream_ctx.__exit__ = MagicMock(return_value=False)
    mock_client.beta.sessions.stream.return_value = mock_stream_ctx

    return mock_client


def test_agent_client_run_returns_text():
    mock_anthropic = _make_mock_anthropic('{"verdict": "pass", "blockers": [], "next_actions": []}')
    with patch("orchestrator.agents.client.anthropic.Anthropic", return_value=mock_anthropic):
        client = AgentClient(
            name="TestAgent",
            model="claude-sonnet-4-6",
            system_prompt="You are a test agent.",
        )
        result = client.run("Hello, world")

    assert '{"verdict": "pass"' in result


def test_agent_client_creates_agent_and_session():
    mock_anthropic = _make_mock_anthropic()
    with patch("orchestrator.agents.client.anthropic.Anthropic", return_value=mock_anthropic):
        client = AgentClient(
            name="Planner",
            model="claude-sonnet-4-6",
            system_prompt="You are a planner.",
        )
        client.run("Plan something")

    mock_anthropic.beta.agents.create.assert_called_once()
    create_call_kwargs = mock_anthropic.beta.agents.create.call_args[1]
    assert create_call_kwargs["name"] == "Planner"
    assert create_call_kwargs["model"] == "claude-sonnet-4-6"
    mock_anthropic.beta.sessions.create.assert_called_once()


def test_agent_client_sends_message_and_streams():
    mock_anthropic = _make_mock_anthropic()
    with patch("orchestrator.agents.client.anthropic.Anthropic", return_value=mock_anthropic):
        client = AgentClient(
            name="Evaluator",
            model="claude-sonnet-4-6",
            system_prompt="You are an evaluator.",
        )
        client.run("Evaluate this plan")

    # Message was sent
    mock_anthropic.beta.sessions.events.send.assert_called_once()
    sent_call = mock_anthropic.beta.sessions.events.send.call_args
    events = sent_call[1]["events"]
    assert events[0]["type"] == "user.message"
    assert events[0]["content"][0]["text"] == "Evaluate this plan"

    # Stream was opened
    mock_anthropic.beta.sessions.stream.assert_called_once()


def test_agent_client_with_file_write_tools():
    mock_anthropic = _make_mock_anthropic()
    with patch("orchestrator.agents.client.anthropic.Anthropic", return_value=mock_anthropic):
        client = AgentClient(
            name="Generator",
            model="claude-sonnet-4-6",
            system_prompt="You are a generator.",
            tools=[{"type": "agent_toolset_20260401", "default_config": {"enabled": True}}],
        )
        client.run("Implement this")

    create_call = mock_anthropic.beta.agents.create.call_args[1]
    assert len(create_call["tools"]) == 1
    assert create_call["tools"][0]["type"] == "agent_toolset_20260401"
