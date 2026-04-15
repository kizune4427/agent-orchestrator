# tests/test_state.py
import pytest
from pydantic import ValidationError
from orchestrator.state import (
    Plan,
    EvaluationResult,
    AdvisorMemo,
    Task,
    SprintContract,
    Implementation,
)


def test_import_orchestrator():
    import orchestrator  # noqa: F401


def test_plan_valid():
    plan = Plan(
        summary="Build a REST API",
        steps=["Define endpoints", "Write handlers"],
        open_questions=["Which framework?"],
    )
    assert plan.summary == "Build a REST API"
    assert len(plan.steps) == 2


def test_evaluation_result_pass():
    result = EvaluationResult(
        verdict="pass",
        blockers=[],
        next_actions=[],
    )
    assert result.verdict == "pass"


def test_evaluation_result_fail():
    result = EvaluationResult(
        verdict="fail",
        blockers=["Missing error handling"],
        next_actions=["Add try/except blocks"],
    )
    assert result.verdict == "fail"
    assert len(result.blockers) == 1


def test_evaluation_result_invalid_verdict():
    with pytest.raises(ValidationError):
        EvaluationResult(verdict="maybe", blockers=[], next_actions=[])


def test_advisor_memo_no_approach():
    memo = AdvisorMemo(
        analysis="The plan lacks specificity",
        recommendations=["Add concrete file paths"],
        suggested_approach=None,
    )
    assert memo.suggested_approach is None


def test_sprint_contract_valid():
    contract = SprintContract(
        goal="Implement auth module",
        tasks=[
            Task(
                id="T1",
                description="Create user model",
                acceptance_criteria=["Model has email field"],
            )
        ],
        constraints=["No external auth libraries"],
    )
    assert len(contract.tasks) == 1
    assert contract.tasks[0].id == "T1"


def test_implementation_valid():
    impl = Implementation(
        files_written=["src/auth.py", "tests/test_auth.py"],
        summary="Implemented basic auth",
    )
    assert len(impl.files_written) == 2


def test_graph_state_minimal():
    from orchestrator.state import GraphState
    state: GraphState = {
        "idea": "build an API",
        "revision_count": 0,
        "advisor_used": False,
    }
    assert state["idea"] == "build an API"
    assert state["revision_count"] == 0


def test_graph_state_accepts_run_config():
    from orchestrator.config import RunConfig
    from orchestrator.state import GraphState

    cfg = RunConfig(run_id="test-id")
    state: GraphState = {
        "idea": "test",
        "revision_count": 0,
        "advisor_used": False,
        "run_config": cfg,
    }
    assert state["run_config"].run_id == "test-id"


def test_path_spec_model():
    from orchestrator.state import PathSpec
    spec = PathSpec(name="event-driven", focus="Use events for decoupling")
    assert spec.name == "event-driven"


def test_branch_result_model():
    from orchestrator.state import BranchResult, Plan, EvaluationResult, PathSpec
    br = BranchResult(
        name="event-driven",
        plan=Plan(summary="s", steps=["s1"], open_questions=[]),
        evaluation=EvaluationResult(verdict="pass", blockers=[], next_actions=[]),
    )
    assert br.name == "event-driven"
