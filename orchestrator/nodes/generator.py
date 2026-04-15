from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from orchestrator.agents.client import AgentClient
from orchestrator.state import GraphState, Implementation

_SYSTEM_PROMPT = """\
You are a code generation agent for a software development workflow.

Given a sprint contract (goal, tasks, and constraints), write the required
implementation and test files using the write tool.

After you have finished writing all files, respond ONLY with valid JSON.
No prose, no markdown fences, no explanation. The JSON must match exactly:
{
  "files_written": ["<path/to/file1>", "<path/to/file2>"],
  "summary": "<brief description of what was implemented>"
}
"""

_MODEL = "claude-sonnet-4-6"

_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"

_GENERATOR_TOOLS = [
    {
        "type": "agent_toolset_20260401",
        "default_config": {"enabled": False},
        "configs": [
            {"name": "write", "enabled": True},
            {"name": "read", "enabled": True},
            {"name": "edit", "enabled": True},
            {"name": "glob", "enabled": True},
        ],
    }
]


def _build_message(state: GraphState) -> str:
    contract = state["sprint_contract"]
    parts = [f"Sprint Goal: {contract.goal}"]

    parts.append("\nTasks:")
    for task in contract.tasks:
        parts.append(f"- [{task.id}] {task.description}")
        if task.acceptance_criteria:
            parts.append(
                "  Acceptance criteria:\n"
                + "\n".join(f"  - {c}" for c in task.acceptance_criteria)
            )

    if contract.constraints:
        parts.append(
            "\nConstraints:\n" + "\n".join(f"- {c}" for c in contract.constraints)
        )

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

    parts.append(
        "\nAfter writing all files, respond ONLY with the JSON summary. No other text."
    )

    return "\n\n".join(parts)


def _parse_implementation(text: str) -> Implementation:
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # Find the LAST {...} block to skip any preamble prose
    start = text.rfind("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    data = json.loads(text)
    return Implementation.model_validate(data)


def _persist_implementation(impl: Implementation) -> None:
    artifacts = _ARTIFACTS_DIR
    artifacts.mkdir(exist_ok=True)
    path = artifacts / "implementation.md"
    content = "# Implementation\n\n"
    content += "## Files Written\n\n"
    content += "\n".join(f"- `{f}`" for f in impl.files_written)
    content += f"\n\n## Summary\n\n{impl.summary}\n"
    path.write_text(content)


def generator_node(state: GraphState) -> dict:
    client = AgentClient(
        name="Generator",
        model=_MODEL,
        system_prompt=_SYSTEM_PROMPT,
        tools=_GENERATOR_TOOLS,
    )
    message = _build_message(state)

    # First attempt
    response = client.run(message)
    try:
        impl = _parse_implementation(response)
    except (json.JSONDecodeError, ValidationError):
        # One retry with explicit JSON instruction
        retry_message = (
            message
            + "\n\nNow respond ONLY with the JSON summary. No other text."
        )
        response = client.run(retry_message)
        try:
            impl = _parse_implementation(response)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(
                f"Failed to parse generator response after retry: {exc}"
            ) from exc

    _persist_implementation(impl)
    return {
        "implementation": impl,
        "phase": "implementation",
        "revision_count": state.get("revision_count", 0) + 1,
    }
