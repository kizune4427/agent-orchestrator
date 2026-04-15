# tests/test_nodes.py
import json
from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError

from orchestrator.state import (
    GraphState,
    Plan,
    EvaluationResult,
    AdvisorMemo,
)


def _make_client_mock(response_text: str) -> MagicMock:
    mock = MagicMock()
    mock.run.return_value = response_text
    return mock


# ---------------------------------------------------------------------------
# Planner node
# ---------------------------------------------------------------------------

def test_planner_node_returns_plan():
    from orchestrator.nodes.planner import planner_node

    valid_plan_json = json.dumps({
        "summary": "Build a REST API for todos",
        "steps": ["Define models", "Write handlers", "Add tests"],
        "open_questions": ["Which DB?"],
    })
    mock_client = _make_client_mock(valid_plan_json)

    state: GraphState = {
        "idea": "Build a REST API for todos",
        "phase": "planning",
        "revision_count": 0,
        "advisor_used": False,
        "plan": None,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": None,
        "advisor_memo": None,
    }

    with patch("orchestrator.nodes.planner.AgentClient", return_value=mock_client):
        result = planner_node(state)

    assert "plan" in result
    assert isinstance(result["plan"], Plan)
    assert result["plan"].summary == "Build a REST API for todos"
    assert result["revision_count"] == 1


def test_planner_node_includes_eval_feedback_in_message():
    from orchestrator.nodes.planner import planner_node

    valid_plan_json = json.dumps({
        "summary": "Revised plan",
        "steps": ["Step 1"],
        "open_questions": [],
    })
    mock_client = _make_client_mock(valid_plan_json)

    eval_result = EvaluationResult(
        verdict="fail",
        blockers=["Missing error handling"],
        next_actions=["Add try/except"],
    )
    state: GraphState = {
        "idea": "Build something",
        "phase": "planning",
        "revision_count": 1,
        "advisor_used": False,
        "plan": None,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": eval_result,
        "advisor_memo": None,
    }

    with patch("orchestrator.nodes.planner.AgentClient", return_value=mock_client):
        planner_node(state)

    call_args = mock_client.run.call_args[0][0]
    assert "Missing error handling" in call_args


def test_planner_node_includes_advisor_memo():
    from orchestrator.nodes.planner import planner_node

    valid_plan_json = json.dumps({
        "summary": "Plan with advisor input",
        "steps": ["Step 1"],
        "open_questions": [],
    })
    mock_client = _make_client_mock(valid_plan_json)

    memo = AdvisorMemo(
        analysis="Root cause: vague requirements",
        recommendations=["Define acceptance criteria"],
        suggested_approach="Start with a use-case diagram",
    )
    state: GraphState = {
        "idea": "Build something",
        "phase": "planning",
        "revision_count": 0,
        "advisor_used": True,
        "plan": None,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": None,
        "advisor_memo": memo,
    }

    with patch("orchestrator.nodes.planner.AgentClient", return_value=mock_client):
        planner_node(state)

    call_args = mock_client.run.call_args[0][0]
    assert "Root cause: vague requirements" in call_args


def test_planner_node_retries_on_bad_json():
    from orchestrator.nodes.planner import planner_node

    valid_plan_json = json.dumps({
        "summary": "Recovered plan",
        "steps": ["Step 1"],
        "open_questions": [],
    })
    mock_client = MagicMock()
    # First call returns invalid JSON, second returns valid
    mock_client.run.side_effect = ["not json at all", valid_plan_json]

    state: GraphState = {
        "idea": "Build something",
        "phase": "planning",
        "revision_count": 0,
        "advisor_used": False,
        "plan": None,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": None,
        "advisor_memo": None,
    }

    with patch("orchestrator.nodes.planner.AgentClient", return_value=mock_client):
        result = planner_node(state)

    assert mock_client.run.call_count == 2
    assert result["plan"].summary == "Recovered plan"


def test_planner_node_raises_after_two_bad_json():
    from orchestrator.nodes.planner import planner_node

    mock_client = _make_client_mock("still not json")
    mock_client.run.side_effect = ["bad json 1", "bad json 2"]

    state: GraphState = {
        "idea": "Build something",
        "phase": "planning",
        "revision_count": 0,
        "advisor_used": False,
        "plan": None,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": None,
        "advisor_memo": None,
    }

    with patch("orchestrator.nodes.planner.AgentClient", return_value=mock_client):
        with pytest.raises(ValueError, match="Failed to parse planner response"):
            planner_node(state)


# ---------------------------------------------------------------------------
# Evaluator node
# ---------------------------------------------------------------------------

def test_evaluator_node_planning_phase_pass():
    from orchestrator.nodes.evaluator import evaluator_node

    valid_eval_json = json.dumps({
        "verdict": "pass",
        "blockers": [],
        "next_actions": [],
    })
    mock_client = _make_client_mock(valid_eval_json)

    plan = Plan(
        summary="Build a REST API",
        steps=["Step 1"],
        open_questions=[],
    )
    state: GraphState = {
        "idea": "Build a REST API",
        "phase": "planning",
        "revision_count": 1,
        "advisor_used": False,
        "plan": plan,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": None,
        "advisor_memo": None,
    }

    with patch("orchestrator.nodes.evaluator.AgentClient", return_value=mock_client):
        result = evaluator_node(state)

    assert result["evaluation"].verdict == "pass"
    # When planning passes, evaluator should set sprint_contract
    assert result.get("sprint_contract") is not None


def test_evaluator_node_planning_phase_fail():
    from orchestrator.nodes.evaluator import evaluator_node

    valid_eval_json = json.dumps({
        "verdict": "fail",
        "blockers": ["No acceptance criteria"],
        "next_actions": ["Add acceptance criteria to each step"],
    })
    mock_client = _make_client_mock(valid_eval_json)

    plan = Plan(summary="Vague plan", steps=["Do stuff"], open_questions=[])
    state: GraphState = {
        "idea": "Do something",
        "phase": "planning",
        "revision_count": 1,
        "advisor_used": False,
        "plan": plan,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": None,
        "advisor_memo": None,
    }

    with patch("orchestrator.nodes.evaluator.AgentClient", return_value=mock_client):
        result = evaluator_node(state)

    assert result["evaluation"].verdict == "fail"
    assert "No acceptance criteria" in result["evaluation"].blockers


def test_evaluator_node_implementation_phase():
    from orchestrator.nodes.evaluator import evaluator_node
    from orchestrator.state import Implementation, SprintContract, Task

    valid_eval_json = json.dumps({
        "verdict": "pass",
        "blockers": [],
        "next_actions": [],
    })
    mock_client = _make_client_mock(valid_eval_json)

    impl = Implementation(files_written=["src/main.py"], summary="Implemented main module")
    contract = SprintContract(
        goal="Build main module",
        tasks=[Task(id="T1", description="Write main.py", acceptance_criteria=["File exists"])],
        constraints=[],
    )
    state: GraphState = {
        "idea": "Build something",
        "phase": "implementation",
        "revision_count": 1,
        "advisor_used": False,
        "plan": None,
        "sprint_contract": contract,
        "implementation": impl,
        "evaluation": None,
        "advisor_memo": None,
    }

    with patch("orchestrator.nodes.evaluator.AgentClient", return_value=mock_client):
        result = evaluator_node(state)

    assert result["evaluation"].verdict == "pass"
    # Phase should not be updated by evaluator — routing handles that
    assert "phase" not in result
