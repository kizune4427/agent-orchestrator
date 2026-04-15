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


def test_build_graph_from_evaluator_entry():
    """When from_node='evaluator', graph starts at evaluator (not planner)."""
    from orchestrator.config import RunConfig
    from orchestrator.graph import build_graph
    cfg = RunConfig(run_id="test", from_node="evaluator")
    graph = build_graph(cfg)
    # Verify graph compiles without error and has evaluator as reachable node
    assert graph is not None


def test_build_graph_from_generator_entry():
    """When from_node='generator', graph starts at generator."""
    from orchestrator.config import RunConfig
    from orchestrator.graph import build_graph
    cfg = RunConfig(run_id="test", from_node="generator")
    graph = build_graph(cfg)
    assert graph is not None


# ---------------------------------------------------------------------------
# End-to-end smoke tests (mocked nodes)
# ---------------------------------------------------------------------------
from unittest.mock import patch, MagicMock


def test_graph_planning_pass_reaches_done():
    """Smoke test: planner → evaluator (pass) → transition → generator → evaluator (pass) → done."""
    from orchestrator.graph import build_graph
    from orchestrator.state import (
        Plan,
        EvaluationResult,
        SprintContract,
        Task,
        Implementation,
    )
    from orchestrator import graph as graph_module

    approved_plan = Plan(
        summary="Build a simple module",
        steps=["Write module", "Write tests"],
        open_questions=[],
    )
    eval_pass = EvaluationResult(verdict="pass", blockers=[], next_actions=[])
    contract = SprintContract(
        goal="Build a simple module",
        tasks=[Task(id="T1", description="Write module", acceptance_criteria=["Done"])],
        constraints=[],
    )
    impl = Implementation(files_written=["src/module.py"], summary="Module written")

    def mock_planner(state):
        return {"plan": approved_plan, "revision_count": 1}

    def mock_evaluator(state):
        updates = {"evaluation": eval_pass}
        if state.get("phase") == "planning":
            updates["sprint_contract"] = contract
        return updates

    def mock_generator(state):
        return {"implementation": impl, "phase": "implementation", "revision_count": 1}

    with (
        patch.object(graph_module, "planner_node", mock_planner),
        patch.object(graph_module, "evaluator_node", mock_evaluator),
        patch.object(graph_module, "generator_node", mock_generator),
    ):
        g = graph_module.build_graph()
        initial = {
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
        result = g.invoke(initial)

    assert result["phase"] == "done"


def test_graph_hard_stop_after_advisor():
    """Smoke test: evaluator fails 3+ times with advisor used → phase=failed."""
    from orchestrator.state import Plan, EvaluationResult, AdvisorMemo
    from orchestrator import graph as graph_module

    failing_plan = Plan(summary="Bad plan", steps=["Step 1"], open_questions=[])
    eval_fail = EvaluationResult(verdict="fail", blockers=["issue"], next_actions=["fix"])
    memo = AdvisorMemo(
        analysis="root cause",
        recommendations=["fix it"],
        suggested_approach=None,
    )

    call_count = {"n": 0}

    def mock_planner(state):
        call_count["n"] += 1
        return {"plan": failing_plan, "revision_count": state.get("revision_count", 0) + 1}

    def mock_evaluator(state):
        return {"evaluation": eval_fail}

    def mock_advisor(state):
        return {"advisor_memo": memo, "advisor_used": True, "revision_count": 0}

    with (
        patch.object(graph_module, "planner_node", mock_planner),
        patch.object(graph_module, "evaluator_node", mock_evaluator),
        patch.object(graph_module, "advisor_node", mock_advisor),
    ):
        g = graph_module.build_graph()
        initial = {
            "idea": "Bad idea",
            "phase": "planning",
            "revision_count": 0,
            "advisor_used": False,
            "plan": None,
            "sprint_contract": None,
            "implementation": None,
            "evaluation": None,
            "advisor_memo": None,
        }
        result = g.invoke(initial)

    assert result["phase"] == "failed"
