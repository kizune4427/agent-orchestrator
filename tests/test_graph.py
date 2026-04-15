# tests/test_graph.py
import pytest
from orchestrator.state import EvaluationResult, GraphState


def _state_with_eval(verdict: str, revision: int, advisor_used: bool, phase: str = "planning") -> GraphState:
    return {
        "idea": "test",
        "phase": phase,
        "revision_count": revision,
        "advisor_used": advisor_used,
        "plan": None,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": EvaluationResult(
            verdict=verdict,
            blockers=["issue"] if verdict == "fail" else [],
            next_actions=["fix it"] if verdict == "fail" else [],
        ),
        "advisor_memo": None,
    }


# ---------------------------------------------------------------------------
# route_after_evaluator — pure function tests
# ---------------------------------------------------------------------------

def test_route_planning_pass_goes_to_generator():
    from orchestrator.graph import route_after_evaluator

    state = _state_with_eval("pass", revision=1, advisor_used=False, phase="planning")
    assert route_after_evaluator(state) == "to_generator"


def test_route_implementation_pass_is_done():
    from orchestrator.graph import route_after_evaluator

    state = _state_with_eval("pass", revision=1, advisor_used=False, phase="implementation")
    assert route_after_evaluator(state) == "done"


def test_route_fail_low_count_revises():
    from orchestrator.graph import route_after_evaluator

    state = _state_with_eval("fail", revision=1, advisor_used=False)
    assert route_after_evaluator(state) == "revise"


def test_route_fail_count_2_revises():
    from orchestrator.graph import route_after_evaluator

    state = _state_with_eval("fail", revision=2, advisor_used=False)
    assert route_after_evaluator(state) == "revise"


def test_route_fail_count_3_no_advisor_routes_to_advisor():
    from orchestrator.graph import route_after_evaluator

    state = _state_with_eval("fail", revision=3, advisor_used=False)
    assert route_after_evaluator(state) == "advisor"


def test_route_fail_count_3_advisor_used_hard_stops():
    from orchestrator.graph import route_after_evaluator

    state = _state_with_eval("fail", revision=3, advisor_used=True)
    assert route_after_evaluator(state) == "hard_stop"


def test_route_fail_high_count_advisor_used_hard_stops():
    from orchestrator.graph import route_after_evaluator

    state = _state_with_eval("fail", revision=5, advisor_used=True)
    assert route_after_evaluator(state) == "hard_stop"


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def test_graph_compiles():
    from orchestrator.graph import build_graph

    graph = build_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    from orchestrator.graph import build_graph

    graph = build_graph()
    node_names = set(graph.nodes)
    assert "planner" in node_names
    assert "evaluator" in node_names
    assert "advisor" in node_names
    assert "generator" in node_names
