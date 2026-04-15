from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def generate_run_id() -> str:
    """Return a unique run ID: YYYYMMDD-HHMMSS-{sha6}."""
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
) -> None:
    """Append a single JSON line to runs.jsonl."""
    if index_path is None:
        index_path = Path(__file__).resolve().parent.parent / "artifacts" / "runs.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "run_id": run_id,
        "idea": idea,
        "phase": phase,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    with index_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
