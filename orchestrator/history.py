from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def generate_run_id() -> str:
    """Return a unique run ID: YYYYMMDD-HHMMSS-{rand6}."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    sha6 = uuid.uuid4().hex[:6]
    return f"{ts}-{sha6}"


def artifact_dir(run_id: str, *, base: Path | None = None) -> Path:
    """Return the artifact directory for a given run ID."""
    if base is None:
        base = Path(__file__).resolve().parent.parent / "artifacts"
    return base / run_id


def append_run_index(
    *,
    run_id: str,
    idea: str,
    phase: str,
    index_path: Path | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    last_phase_reached: str | None = None,
) -> None:
    """Append a single JSON line to runs.jsonl.

    When phase == "error", pass error_type / error_message / last_phase_reached
    so failed runs can be diagnosed without digging through stderr.
    """
    if index_path is None:
        index_path = Path(__file__).resolve().parent.parent / "artifacts" / "runs.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    entry: dict = {
        "run_id": run_id,
        "idea": idea,
        "phase": phase,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    if error_type is not None:
        entry["error_type"] = error_type
    if error_message is not None:
        entry["error_message"] = error_message
    if last_phase_reached is not None:
        entry["last_phase_reached"] = last_phase_reached
    with index_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
