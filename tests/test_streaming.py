import io
import sys
import pytest
from orchestrator.state import Plan, EvaluationResult, Implementation, AdvisorMemo
from orchestrator.streaming import print_node_summary


def _capture(node_name: str, state_delta: dict) -> str:
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        print_node_summary(node_name, state_delta)
    finally:
        sys.stdout = old_stdout
    return buf.getvalue()


def test_planner_summary_shows_goal_and_steps():
    plan = Plan(summary="Build a REST API", steps=["Step 1", "Step 2"], open_questions=[])
    output = _capture("planner", {"plan": plan})
    assert "planner" in output
    assert "Build a REST API" in output
    assert "Step 1" in output
    assert "Step 2" in output


def test_evaluator_pass_summary():
    result = EvaluationResult(verdict="pass", blockers=[], next_actions=[])
    output = _capture("evaluator", {"evaluation": result})
    assert "evaluator" in output
    assert "PASS" in output


def test_evaluator_fail_summary_shows_blockers():
    result = EvaluationResult(
        verdict="fail",
        blockers=["Missing error handling"],
        next_actions=["Add try/except"],
    )
    output = _capture("evaluator", {"evaluation": result})
    assert "FAIL" in output
    assert "Missing error handling" in output


def test_generator_summary_shows_files():
    impl = Implementation(files_written=["generated/main.py", "generated/test_main.py"], summary="Done")
    output = _capture("generator", {"implementation": impl})
    assert "generator" in output
    assert "main.py" in output


def test_advisor_summary_shown():
    memo = AdvisorMemo(analysis="Root cause found", recommendations=["Fix it"], suggested_approach=None)
    output = _capture("advisor", {"advisor_memo": memo})
    assert "advisor" in output
    assert "Root cause found" in output


def test_internal_node_prints_nothing():
    output = _capture("mark_done", {"phase": "done"})
    assert output.strip() == ""
