import json
from unittest.mock import MagicMock, patch
import pytest
from orchestrator.agents.openrouter import OpenRouterClient


def _mock_httpx_response(text: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": text}}]
    }
    return mock_resp


def test_openrouter_client_run_returns_text(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    client = OpenRouterClient(model="openai/gpt-4o", system_prompt="You are a planner.")

    mock_resp = _mock_httpx_response('{"summary": "test plan"}')
    with patch("orchestrator.agents.openrouter.httpx.post", return_value=mock_resp) as mock_post:
        result = client.run("Plan something")

    assert result == '{"summary": "test plan"}'
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args[1]
    payload = call_kwargs["json"]
    assert payload["model"] == "openai/gpt-4o"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == "You are a planner."
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"] == "Plan something"


def test_openrouter_client_sends_auth_header(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-testkey")
    client = OpenRouterClient(model="openai/gpt-4o", system_prompt="s")

    mock_resp = _mock_httpx_response("response")
    with patch("orchestrator.agents.openrouter.httpx.post", return_value=mock_resp) as mock_post:
        client.run("msg")

    headers = mock_post.call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer sk-or-testkey"


def test_openrouter_client_raises_on_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = OpenRouterClient(model="openai/gpt-4o", system_prompt="s")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        client.run("msg")


def test_openrouter_client_raises_on_empty_response(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    client = OpenRouterClient(model="openai/gpt-4o", system_prompt="s")

    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"choices": [{"message": {"content": ""}}]}
    with patch("orchestrator.agents.openrouter.httpx.post", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="empty response"):
            client.run("msg")
