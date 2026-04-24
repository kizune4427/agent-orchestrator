import json
import re
from pathlib import Path
import pytest
from orchestrator.history import generate_run_id, artifact_dir, append_run_index


def test_generate_run_id_format():
    run_id = generate_run_id()
    # Format: YYYYMMDD-HHMMSS-xxxxxx
    assert re.match(r"^\d{8}-\d{6}-[a-f0-9]{6}$", run_id), f"Bad format: {run_id}"


def test_generate_run_id_unique():
    ids = {generate_run_id() for _ in range(10)}
    assert len(ids) == 10


def test_artifact_dir(tmp_path):
    d = artifact_dir("20260414-120000-abc123", base=tmp_path)
    assert d == tmp_path / "20260414-120000-abc123"


def test_artifact_dir_creates_parents(tmp_path):
    d = artifact_dir("20260414-120000-abc123", base=tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    assert d.exists()


def test_append_run_index(tmp_path):
    index_path = tmp_path / "runs.jsonl"
    append_run_index(
        index_path=index_path,
        run_id="20260414-120000-abc123",
        idea="Build a thing",
        phase="done",
    )
    lines = index_path.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["run_id"] == "20260414-120000-abc123"
    assert entry["idea"] == "Build a thing"
    assert entry["phase"] == "done"
    assert "timestamp" in entry


def test_append_run_index_appends(tmp_path):
    index_path = tmp_path / "runs.jsonl"
    append_run_index(index_path=index_path, run_id="id1", idea="idea1", phase="done")
    append_run_index(index_path=index_path, run_id="id2", idea="idea2", phase="failed")
    lines = index_path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["run_id"] == "id1"
    assert json.loads(lines[1])["run_id"] == "id2"


def test_append_run_index_error_fields(tmp_path):
    index_path = tmp_path / "runs.jsonl"
    append_run_index(
        index_path=index_path,
        run_id="id_err",
        idea="boom",
        phase="error",
        error_type="ValueError",
        error_message="Failed to parse evaluator response",
        last_phase_reached="evaluator",
    )
    entry = json.loads(index_path.read_text().splitlines()[0])
    assert entry["phase"] == "error"
    assert entry["error_type"] == "ValueError"
    assert entry["error_message"] == "Failed to parse evaluator response"
    assert entry["last_phase_reached"] == "evaluator"


def test_append_run_index_omits_error_fields_when_none(tmp_path):
    index_path = tmp_path / "runs.jsonl"
    append_run_index(index_path=index_path, run_id="id_ok", idea="ok", phase="done")
    entry = json.loads(index_path.read_text().splitlines()[0])
    assert "error_type" not in entry
    assert "error_message" not in entry
    assert "last_phase_reached" not in entry
