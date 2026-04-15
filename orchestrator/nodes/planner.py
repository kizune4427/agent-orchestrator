from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from orchestrator.agents.client import AgentClient
from orchestrator.state import GraphState, Plan

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

_MODEL = "claude-sonnet-4-6"

_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"


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


def _persist_plan(plan: Plan, revision: int) -> None:
    artifacts = _ARTIFACTS_DIR
    artifacts.mkdir(exist_ok=True)
    path = artifacts / f"plan_v{revision}.md"
    content = f"# Plan (revision {revision})\n\n**Summary:** {plan.summary}\n\n"
    content += "## Steps\n\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan.steps))
    if plan.open_questions:
        content += "\n\n## Open Questions\n\n" + "\n".join(
            f"- {q}" for q in plan.open_questions
        )
    path.write_text(content)


def planner_node(state: GraphState) -> dict:
    client = AgentClient(
        name="Planner",
        model=_MODEL,
        system_prompt=_SYSTEM_PROMPT,
    )
    message = _build_message(state)
    revision = state.get("revision_count", 0) + 1

    # First attempt
    response = client.run(message)
    try:
        plan = _parse_plan(response)
    except (json.JSONDecodeError, ValidationError):
        # One retry with explicit JSON instruction
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

    _persist_plan(plan, revision)
    return {"plan": plan, "revision_count": revision}
