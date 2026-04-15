from __future__ import annotations

import json

from pydantic import ValidationError

from orchestrator.agents.factory import make_client
from orchestrator.history import artifact_dir
from orchestrator.state import AdvisorMemo, GraphState

_SYSTEM_PROMPT = """\
You are a senior technical advisor in a software development workflow.
A planning agent has failed 3 consecutive times to produce an acceptable plan.
Your job: identify the root cause of repeated failures and provide concrete,
actionable guidance that will unblock the planner on the next attempt.
IMPORTANT: Respond ONLY with valid JSON. No prose, no markdown fences.
{
  "analysis": "<root cause of repeated failures>",
  "recommendations": ["<concrete recommendation 1>", "<concrete recommendation 2>"],
  "suggested_approach": "<optional: specific approach to try>" or null
}
"""

def _build_message(state: GraphState) -> str:
    parts = [f"Original idea: {state['idea']}"]

    revision_count = state.get("revision_count", 3)
    parts.append(f"This plan has failed evaluation {revision_count} times.")

    if state.get("plan"):
        plan = state["plan"]
        parts.append(f"\nLatest plan summary: {plan.summary}")
        if plan.steps:
            parts.append(
                "Plan steps:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan.steps))
            )

    if state.get("evaluation"):
        eval_ = state["evaluation"]
        if eval_.blockers:
            parts.append(
                "Evaluation blockers:\n" + "\n".join(f"- {b}" for b in eval_.blockers)
            )
        if eval_.next_actions:
            parts.append(
                "Required next actions:\n" + "\n".join(f"- {a}" for a in eval_.next_actions)
            )

    parts.append(
        "\nPlease diagnose the root cause of the repeated failures and provide "
        "concrete recommendations to unblock the planner on the next attempt."
    )

    return "\n\n".join(parts)


def _parse_memo(text: str) -> AdvisorMemo:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    data = json.loads(text)
    return AdvisorMemo.model_validate(data)


def _persist_memo(memo: AdvisorMemo, run_id: str) -> None:
    adir = artifact_dir(run_id)
    adir.mkdir(parents=True, exist_ok=True)
    path = adir / "advisor_memo.md"
    content = "# Advisor Memo\n\n"
    content += f"## Analysis\n\n{memo.analysis}\n\n"
    content += "## Recommendations\n\n"
    content += "\n".join(f"- {r}" for r in memo.recommendations)
    if memo.suggested_approach:
        content += f"\n\n## Suggested Approach\n\n{memo.suggested_approach}\n"
    path.write_text(content)


def advisor_node(state: GraphState) -> dict:
    run_config = state["run_config"]
    client = make_client("advisor", run_config, _SYSTEM_PROMPT)
    message = _build_message(state)

    response = client.run(message)
    try:
        memo = _parse_memo(response)
    except (json.JSONDecodeError, ValidationError) as exc:
        retry_message = message + "\n\nRespond ONLY with the raw JSON object, no other text."
        response = client.run(retry_message)
        try:
            memo = _parse_memo(response)
        except (json.JSONDecodeError, ValidationError) as exc2:
            raise ValueError(
                f"Failed to parse advisor response after retry: {exc2}"
            ) from exc

    _persist_memo(memo, run_config.run_id)
    return {"advisor_memo": memo, "advisor_used": True, "revision_count": 0}
