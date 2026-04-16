from __future__ import annotations

import json
import re
from pathlib import Path

from orchestrator.agents.factory import make_client
from orchestrator.history import artifact_dir
from orchestrator.state import GraphState, Implementation

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
        inner = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
        if inner:
            text = inner

    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

    data = json.loads(text)
    return data["files"], data["summary"]


def _write_files_locally(files: list[dict], run_id: str) -> list[str]:
    """Write generated files under generated/{run_id}/ and return absolute paths."""
    output_dir = Path(__file__).resolve().parent.parent.parent / "generated" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for f in files:
        dest = output_dir / f["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f["content"])
        written.append(str(dest))
    return written


def _persist_implementation(impl: Implementation, run_id: str) -> None:
    adir = artifact_dir(run_id)
    adir.mkdir(parents=True, exist_ok=True)
    path = adir / "implementation.md"
    content = "# Implementation\n\n"
    content += "## Files Written\n\n"
    content += "\n".join(f"- `{f}`" for f in impl.files_written)
    content += f"\n\n## Summary\n\n{impl.summary}\n"
    path.write_text(content)


def generator_node(state: GraphState) -> dict:
    run_config = state["run_config"]
    client = make_client("generator", run_config, _SYSTEM_PROMPT)
    message = _build_message(state)

    revision = state.get("revision_count", 0)
    if revision > 0:
        print(f"\n⚙  Generator running (revision {revision + 1})...", flush=True)
    else:
        print("\n⚙  Generator running — implementing sprint...", flush=True)

    response = client.run(message)
    try:
        files, summary = _parse_response(response)
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print("  ↻ Parse failed, retrying with stricter prompt...", flush=True)
        retry_message = (
            message + "\n\nRespond ONLY with the raw JSON object, no other text."
        )
        response2 = client.run(retry_message)
        try:
            files, summary = _parse_response(response2)
        except (json.JSONDecodeError, KeyError, ValueError) as exc2:
            raise ValueError(
                f"Failed to parse generator response after retry: {exc2}\n"
                f"  Initial response ({len(response)} chars): {response[:300]!r}\n"
                f"  Retry response  ({len(response2)} chars): {response2[:300]!r}"
            ) from exc

    written_paths = _write_files_locally(files, run_config.run_id)
    impl = Implementation(files_written=written_paths, summary=summary)
    _persist_implementation(impl, run_config.run_id)
    return {
        "implementation": impl,
        "phase": "implementation",
        "revision_count": state.get("revision_count", 0) + 1,
    }
