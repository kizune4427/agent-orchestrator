from __future__ import annotations

import asyncio
import itertools
import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from orchestrator.agents.factory import make_client
from orchestrator.history import artifact_dir
from orchestrator.state import GraphState, PathSpec, Plan

_SYSTEM_PROMPT = """\
You are a planning agent for a software development workflow.

Given a user's idea (and optionally evaluation feedback and advisor recommendations),
produce a structured plan.

IMPORTANT: Respond ONLY with valid JSON. No prose, no markdown fences, no explanation.
The JSON must match exactly this schema:
{
  "summary": "<one-sentence summary>",
  "steps": ["<step 1>", "<step 2>"],
  "open_questions": ["<question 1>"]
}
"""


def _build_message(state: GraphState) -> str:
    parts = [f"Idea: {state['idea']}"]

    if state.get("evaluation") and state["evaluation"].verdict == "fail":
        eval_ = state["evaluation"]
        parts.append("\nPrevious evaluation FAILED.")
        if eval_.blockers:
            parts.append("Blockers:\n" + "\n".join(f"- {b}" for b in eval_.blockers))
        if eval_.next_actions:
            parts.append(
                "Required next actions:\n"
                + "\n".join(f"- {a}" for a in eval_.next_actions)
            )

    if state.get("advisor_memo"):
        memo = state["advisor_memo"]
        parts.append("\nAdvisor analysis:")
        parts.append(memo.analysis)
        if memo.recommendations:
            parts.append(
                "Advisor recommendations:\n"
                + "\n".join(f"- {r}" for r in memo.recommendations)
            )
        if memo.suggested_approach:
            parts.append(f"Suggested approach: {memo.suggested_approach}")

    return "\n\n".join(parts)


def _parse_plan(text: str) -> Plan:
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    data = json.loads(text)
    return Plan.model_validate(data)


def _persist_plan(plan: Plan, revision: int, run_id: str) -> None:
    artifacts = artifact_dir(run_id)
    artifacts.mkdir(parents=True, exist_ok=True)
    path = artifacts / f"plan_v{revision}.md"
    content = f"# Plan (revision {revision})\n\n**Summary:** {plan.summary}\n\n"
    content += "## Steps\n\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan.steps))
    if plan.open_questions:
        content += "\n\n## Open Questions\n\n" + "\n".join(
            f"- {q}" for q in plan.open_questions
        )
    path.write_text(content)


def plan_paths(state: GraphState, *, personas_path: Path | None = None) -> list[PathSpec]:
    """Load personas from personas.yaml and return the first N as PathSpec list."""
    run_config = state["run_config"]
    n = run_config.branches
    if personas_path is None:
        personas_path = Path(__file__).resolve().parent.parent.parent / "personas.yaml"
    with personas_path.open() as f:
        data = yaml.safe_load(f)
    all_personas = [PathSpec(name=p["name"], focus=p["focus"]) for p in data["personas"]]
    specs = list(itertools.islice(itertools.cycle(all_personas), n))
    return specs


def run_branch(state: GraphState, spec: PathSpec) -> "Plan":
    """Stage 2: Run a single planner branch seeded with a PathSpec focus."""
    run_config = state["run_config"]
    system = _SYSTEM_PROMPT + f"\n\nExploration angle: {spec.name} — {spec.focus}"
    client = make_client("planner", run_config, system)
    message = _build_message(state) + f"\n\nPath focus: {spec.name} — {spec.focus}"
    response = client.run(message)
    return _parse_plan(response)


def planner_node(state: GraphState) -> dict:
    run_config = state["run_config"]
    revision = state.get("revision_count", 0) + 1

    if run_config.parallel and not state.get("branches"):
        # Parallel mode: Stage 1 — get path specs, Stage 2 — run branches concurrently
        specs = plan_paths(state)

        async def _run_all() -> list:
            loop = asyncio.get_running_loop()
            tasks = [
                loop.run_in_executor(None, run_branch, state, spec)
                for spec in specs
            ]
            return await asyncio.gather(*tasks)

        plans = asyncio.run(_run_all())

        from orchestrator.state import BranchResult, EvaluationResult
        # Wrap as BranchResult with placeholder evaluations (selector_node will evaluate)
        branches = [
            BranchResult(
                name=spec.name,
                plan=plan,
                evaluation=EvaluationResult(verdict="pass", blockers=[], next_actions=[]),
            )
            for spec, plan in zip(specs, plans)
        ]
        _persist_plan(plans[0], revision, run_config.run_id)
        return {"plan": plans[0], "revision_count": revision, "branches": branches}

    # Standard single-branch mode
    client = make_client("planner", run_config, _SYSTEM_PROMPT)
    message = _build_message(state)

    response = client.run(message)
    try:
        plan = _parse_plan(response)
    except (json.JSONDecodeError, ValidationError):
        retry_message = (
            message
            + "\n\nIMPORTANT: Your previous response was not valid JSON. "
            "Respond ONLY with the raw JSON object, no other text."
        )
        response = client.run(retry_message)
        try:
            plan = _parse_plan(response)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"Failed to parse planner response after retry: {exc}") from exc

    _persist_plan(plan, revision, run_config.run_id)
    return {"plan": plan, "revision_count": revision}
