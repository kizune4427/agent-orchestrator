from __future__ import annotations

import json
from pathlib import Path

from orchestrator.agents.client import AgentClient
from orchestrator.state import GraphState, Implementation

_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "generated"

_SYSTEM_PROMPT = """\
You are a software implementation agent. Given an approved sprint contract,
implement all tasks by providing the complete content of each file to create.

For each task, write clean, well-structured implementation code and a
corresponding test file.

Respond with ONLY this JSON (no prose, no markdown fences):
{
  "files": [
    {
      "path": "<relative file path, e.g. wordcount.py>",
      "content": "<complete file content as a string>"
    }
  ],
  "summary": "<one-paragraph summary of what was implemented>"
}
"""

_MODEL = "claude-sonnet-4-6"


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
        "\nProvide complete file contents in the JSON format specified. No other text."
    )

    return "\n\n".join(parts)


def _parse_response(text: str) -> tuple[list[dict], str]:
    """Extract (files, summary) from the generator's JSON response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    data = json.loads(text)
    return data["files"], data["summary"]


def _write_files_locally(files: list[dict]) -> list[str]:
    """Write generated files under generated/ and return their absolute paths."""
    _OUTPUT_DIR.mkdir(exist_ok=True)
    written: list[str] = []
    for f in files:
        dest = _OUTPUT_DIR / f["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f["content"])
        written.append(str(dest))
    return written


def _persist_implementation(impl: Implementation) -> None:
    _ARTIFACTS_DIR.mkdir(exist_ok=True)
    path = _ARTIFACTS_DIR / "implementation.md"
    content = "# Implementation\n\n"
    content += "## Files Written\n\n"
    content += "\n".join(f"- `{f}`" for f in impl.files_written)
    content += f"\n\n## Summary\n\n{impl.summary}\n"
    path.write_text(content)


def generator_node(state: GraphState) -> dict:
    # No file-write tools — generator returns file contents inline as JSON,
    # and this node writes them to the local filesystem under generated/.
    client = AgentClient(
        name="Generator",
        model=_MODEL,
        system_prompt=_SYSTEM_PROMPT,
    )
    message = _build_message(state)

    response = client.run(message)
    try:
        files, summary = _parse_response(response)
    except (json.JSONDecodeError, KeyError) as exc:
        retry_message = (
            message + "\n\nRespond ONLY with the raw JSON object, no other text."
        )
        response = client.run(retry_message)
        try:
            files, summary = _parse_response(response)
        except (json.JSONDecodeError, KeyError) as exc2:
            raise ValueError(
                f"Failed to parse generator response after retry: {exc2}"
            ) from exc

    written_paths = _write_files_locally(files)
    impl = Implementation(files_written=written_paths, summary=summary)
    _persist_implementation(impl)
    return {
        "implementation": impl,
        "phase": "implementation",
        "revision_count": state.get("revision_count", 0) + 1,
    }
