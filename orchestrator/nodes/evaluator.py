from __future__ import annotations

import json
import re

from pydantic import ValidationError

from orchestrator.agents.factory import make_client
from orchestrator.history import artifact_dir
from orchestrator.state import BranchResult, EvaluationResult, GraphState, SprintContract, Task

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

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        inner = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
        if inner:
            text = inner

    # Fallback: extract first {...} block in case of surrounding prose
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

    if not text:
        raise ValueError("Evaluator returned empty content after parsing")

    data = json.loads(text)
    return EvaluationResult.model_validate(data)


def _persist_eval(result: EvaluationResult, phase: str, revision: int, run_id: str) -> None:
    adir = artifact_dir(run_id)
    adir.mkdir(parents=True, exist_ok=True)
    path = adir / f"eval_{phase}_v{revision}.json"
    path.write_text(result.model_dump_json(indent=2))


_SELECTOR_SYSTEM_PROMPT = """\
You are an evaluator selecting the best implementation plan from multiple candidates.
Given a list of (name, plan, evaluation) tuples, pick the plan most likely to succeed.

IMPORTANT: Respond ONLY with valid JSON:
{"selected": "<name of the chosen path>"}
"""


def selector_node(state: GraphState) -> dict:
    """Select the best branch when parallel mode produced multiple plans."""
    run_config = state["run_config"]
    branches: list[BranchResult] = state.get("branches", [])

    if len(branches) <= 1:
        # Nothing to select — return as-is
        if branches:
            return {"plan": branches[0].plan, "branches": branches}
        return {}

    client = make_client("evaluator", run_config, _SELECTOR_SYSTEM_PROMPT)

    candidates = []
    for br in branches:
        steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(br.plan.steps))
        open_q_text = (
            "\n".join(f"  - {q}" for q in br.plan.open_questions)
            if br.plan.open_questions
            else "  none"
        )
        candidates.append(
            f"Name: {br.name}\n"
            f"Summary: {br.plan.summary}\n"
            f"Steps:\n{steps_text}\n"
            f"Open questions:\n{open_q_text}"
        )
    message = "Candidates:\n\n" + "\n\n---\n\n".join(candidates)

    response = client.run(message)
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    data = json.loads(text)
    selected_name = data["selected"]

    selected_branch = next(
        (br for br in branches if br.name == selected_name),
        None,
    )
    if selected_branch is None:
        valid = [br.name for br in branches]
        raise ValueError(
            f"Selector returned unknown branch name {selected_name!r}. "
            f"Valid names: {valid}. Check selector LLM response."
        )

    # Persist losing branches
    adir = artifact_dir(run_config.run_id)
    adir.mkdir(parents=True, exist_ok=True)
    losers = [br for br in branches if br.name != selected_branch.name]
    branches_log = [
        {
            "name": br.name,
            "plan": br.plan.model_dump(),
            "evaluation": br.evaluation.model_dump(),
        }
        for br in losers
    ]
    (adir / "branches.json").write_text(json.dumps(branches_log, indent=2))

    # Persist the selected plan
    plan = selected_branch.plan
    plan_content = f"# Selected Plan: {selected_branch.name}\n\n**Summary:** {plan.summary}\n\n"
    plan_content += "## Steps\n\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan.steps))
    if plan.open_questions:
        plan_content += "\n\n## Open Questions\n\n" + "\n".join(f"- {q}" for q in plan.open_questions)
    (adir / "plan_selected.md").write_text(plan_content)

    return {"plan": selected_branch.plan, "branches": branches}


def evaluator_node(state: GraphState) -> dict:
    run_config = state["run_config"]
    phase = state["phase"]
    if phase not in ("planning", "implementation"):
        raise ValueError(f"evaluator_node received unexpected phase: {phase!r}")
    if phase == "planning":
        system_prompt = _PLANNING_SYSTEM_PROMPT
        message = _build_planning_message(state)
    else:
        system_prompt = _IMPLEMENTATION_SYSTEM_PROMPT
        message = _build_implementation_message(state)

    client = make_client("evaluator", run_config, system_prompt)

    response = client.run(message)
    try:
        result = _parse_eval(response)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        retry = message + "\n\nRespond ONLY with the raw JSON object, no other text."
        response2 = client.run(retry)
        try:
            result = _parse_eval(response2)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc2:
            raise ValueError(
                f"Failed to parse evaluator response after retry: {exc2}\n"
                f"  Initial response ({len(response)} chars): {response[:300]!r}\n"
                f"  Retry response  ({len(response2)} chars): {response2[:300]!r}"
            ) from exc

    _persist_eval(result, phase, state.get("revision_count", 0), run_config.run_id)

    updates: dict = {"evaluation": result}
    if phase == "planning" and result.verdict == "pass":
        updates["sprint_contract"] = _build_sprint_contract(state["plan"])

    return updates
