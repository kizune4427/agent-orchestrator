from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from orchestrator.agents.client import AgentClient
from orchestrator.state import EvaluationResult, GraphState, SprintContract, Task

_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"

_PLANNING_SYSTEM_PROMPT = """\
You are an evaluation agent reviewing a software development plan.

Assess whether the plan is complete, actionable, and free of critical blockers.

Criteria for PASS:
- Plan has a clear, concrete summary
- Steps are specific enough to act on
- No fundamental unknowns that block progress

IMPORTANT: Respond ONLY with valid JSON. No prose, no markdown fences.
{
  "verdict": "pass" or "fail",
  "blockers": ["<critical issue 1>"],
  "next_actions": ["<specific improvement 1>"]
}
blockers and next_actions must be empty lists if verdict is "pass".
"""

_IMPLEMENTATION_SYSTEM_PROMPT = """\
You are an evaluation agent reviewing a software implementation against its sprint contract.

Assess whether each task's acceptance criteria are met.

IMPORTANT: Respond ONLY with valid JSON. No prose, no markdown fences.
{
  "verdict": "pass" or "fail",
  "blockers": ["<unmet criterion 1>"],
  "next_actions": ["<specific fix 1>"]
}
blockers and next_actions must be empty lists if verdict is "pass".
"""

_MODEL = "claude-sonnet-4-6"


def _build_planning_message(state: GraphState) -> str:
    plan = state["plan"]
    parts = [
        f"Idea: {state['idea']}",
        f"\nPlan summary: {plan.summary}",
        "\nSteps:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan.steps)),
    ]
    if plan.open_questions:
        parts.append(
            "\nOpen questions:\n" + "\n".join(f"- {q}" for q in plan.open_questions)
        )
    return "\n".join(parts)


def _build_implementation_message(state: GraphState) -> str:
    impl = state["implementation"]
    contract = state["sprint_contract"]
    parts = [
        f"Sprint goal: {contract.goal}",
        "\nTasks and acceptance criteria:",
    ]
    for task in contract.tasks:
        parts.append(f"\n  Task {task.id}: {task.description}")
        for criterion in task.acceptance_criteria:
            parts.append(f"    - {criterion}")
    parts.append(f"\nFiles written: {', '.join(impl.files_written)}")
    parts.append(f"Implementation summary: {impl.summary}")
    return "\n".join(parts)


def _build_sprint_contract(plan) -> SprintContract:
    """Convert an approved plan into a sprint contract."""
    tasks = [
        Task(
            id=f"T{i+1}",
            description=step,
            acceptance_criteria=[f"{step} is complete and tested"],
        )
        for i, step in enumerate(plan.steps)
    ]
    return SprintContract(
        goal=plan.summary,
        tasks=tasks,
        constraints=[],
    )


def _parse_eval(text: str) -> EvaluationResult:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    data = json.loads(text)
    return EvaluationResult.model_validate(data)


def _persist_eval(result: EvaluationResult, phase: str, revision: int) -> None:
    _ARTIFACTS_DIR.mkdir(exist_ok=True)
    path = _ARTIFACTS_DIR / f"eval_{phase}_v{revision}.json"
    path.write_text(result.model_dump_json(indent=2))


def evaluator_node(state: GraphState) -> dict:
    phase = state["phase"]
    if phase not in ("planning", "implementation"):
        raise ValueError(f"evaluator_node received unexpected phase: {phase!r}")
    if phase == "planning":
        system_prompt = _PLANNING_SYSTEM_PROMPT
        message = _build_planning_message(state)
    else:
        system_prompt = _IMPLEMENTATION_SYSTEM_PROMPT
        message = _build_implementation_message(state)

    client = AgentClient(
        name="Evaluator",
        model=_MODEL,
        system_prompt=system_prompt,
    )

    response = client.run(message)
    try:
        result = _parse_eval(response)
    except (json.JSONDecodeError, ValidationError) as exc:
        retry = message + "\n\nRespond ONLY with the raw JSON object, no other text."
        response = client.run(retry)
        try:
            result = _parse_eval(response)
        except (json.JSONDecodeError, ValidationError) as exc2:
            raise ValueError(f"Failed to parse evaluator response after retry: {exc2}") from exc

    _persist_eval(result, phase, state.get("revision_count", 0))

    updates: dict = {"evaluation": result}
    # If the plan passed, generate sprint contract
    if phase == "planning" and result.verdict == "pass":
        updates["sprint_contract"] = _build_sprint_contract(state["plan"])

    return updates
