import pytest
from unittest.mock import patch
from orchestrator.checkpoint import checkpoint


def test_checkpoint_auto_approve_returns_none():
    result = checkpoint(label="planner", summary="Plan summary", auto_approve=True)
    assert result is None


def test_checkpoint_approve_input_returns_none():
    with patch("builtins.input", return_value="a"):
        result = checkpoint(label="planner", summary="Plan summary", auto_approve=False)
    assert result is None


def test_checkpoint_feedback_returns_feedback_string():
    with patch("builtins.input", side_effect=["f", "Please add error handling"]):
        result = checkpoint(label="planner", summary="Plan summary", auto_approve=False)
    assert result == "Please add error handling"


def test_checkpoint_skip_all_returns_sentinel():
    with patch("builtins.input", return_value="s"):
        result = checkpoint(label="planner", summary="Plan summary", auto_approve=False)
    assert result == "SKIP_ALL"


def test_checkpoint_quit_raises_system_exit():
    with patch("builtins.input", return_value="q"):
        with pytest.raises(SystemExit) as exc_info:
            checkpoint(label="planner", summary="Plan summary", auto_approve=False)
    assert exc_info.value.code == 0
