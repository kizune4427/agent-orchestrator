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
    mock_anthropic.beta.environments.create.assert_called_once()


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


def test_agent_client_handles_session_terminated():
    mock_anthropic = _make_mock_anthropic()
    mock_terminated_event = MagicMock()
    mock_terminated_event.type = "session.status_terminated"

    mock_text_event = MagicMock()
    mock_text_event.type = "agent.message"
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = "partial"
    mock_text_event.content = [mock_block]

    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(
        return_value=iter([mock_text_event, mock_terminated_event])
    )
    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream_ctx.__exit__ = MagicMock(return_value=False)
    mock_anthropic.beta.sessions.stream.return_value = mock_stream_ctx

    with patch("orchestrator.agents.client.anthropic.Anthropic", return_value=mock_anthropic):
        client = AgentClient(name="X", model="m", system_prompt="s")
        result = client.run("msg")
    assert result == "partial"


def test_agent_client_continues_on_requires_action():
    mock_anthropic = _make_mock_anthropic()

    # Event sequence: text → requires_action idle → more text → clean idle
    text_event_1 = MagicMock()
    text_event_1.type = "agent.message"
    block_1 = MagicMock()
    block_1.type = "text"
    block_1.text = "first"
    text_event_1.content = [block_1]

    requires_action_event = MagicMock()
    requires_action_event.type = "session.status_idle"
    requires_action_stop = MagicMock()
    requires_action_stop.type = "requires_action"
    requires_action_event.stop_reason = requires_action_stop

    text_event_2 = MagicMock()
    text_event_2.type = "agent.message"
    block_2 = MagicMock()
    block_2.type = "text"
    block_2.text = " second"
    text_event_2.content = [block_2]

    clean_idle_event = MagicMock()
    clean_idle_event.type = "session.status_idle"
    clean_stop = MagicMock()
    clean_stop.type = "end_turn"
    clean_idle_event.stop_reason = clean_stop

    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(
        return_value=iter([text_event_1, requires_action_event, text_event_2, clean_idle_event])
    )
    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream_ctx.__exit__ = MagicMock(return_value=False)
    mock_anthropic.beta.sessions.stream.return_value = mock_stream_ctx

    with patch("orchestrator.agents.client.anthropic.Anthropic", return_value=mock_anthropic):
        client = AgentClient(name="X", model="m", system_prompt="s")
        result = client.run("msg")
    assert result == "first second"


def test_agent_client_raises_on_empty_response():
    mock_anthropic = _make_mock_anthropic()

    # Stream that terminates immediately with no text
    mock_terminated = MagicMock()
    mock_terminated.type = "session.status_terminated"

    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter([mock_terminated]))
    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream_ctx.__exit__ = MagicMock(return_value=False)
    mock_anthropic.beta.sessions.stream.return_value = mock_stream_ctx

    with patch("orchestrator.agents.client.anthropic.Anthropic", return_value=mock_anthropic):
        client = AgentClient(name="X", model="m", system_prompt="s")
        with pytest.raises(RuntimeError, match="received no text"):
            client.run("msg")
