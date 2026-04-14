# Agent Orchestrator — Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap the agent-orchestrator repo with a working planner/evaluator loop using LangGraph and the Anthropic Managed Agents API.

**Architecture:** Single flat LangGraph graph with phase-tracked state. Five nodes (planner, evaluator, advisor, generator, plus routing). Pydantic schemas validated in-memory; artifacts persisted as files under `artifacts/`. Anthropic Managed Agents API (`/v1/agents`, `/v1/sessions`) drives each role via a thin `AgentClient` wrapper.

**Tech Stack:** Python 3.11, uv, langgraph, anthropic (≥0.92.0), pydantic, typer, pytest, pytest-mock

---

## File Map

| File | Purpose |
|------|---------|
| `pyproject.toml` | uv project config, all dependencies |
| `.python-version` | Pins Python 3.11 |
| `.env.example` | Documents required env vars |
| `main.py` | CLI entry point (`python main.py "idea"`) |
| `orchestrator/__init__.py` | Package init |
| `orchestrator/state.py` | `GraphState` TypedDict + all Pydantic schemas |
| `orchestrator/agents/__init__.py` | Package init |
| `orchestrator/agents/client.py` | `AgentClient` — thin wrapper around Managed Agents API |
| `orchestrator/nodes/__init__.py` | Package init |
| `orchestrator/nodes/planner.py` | Planner node — drafts/revises plan |
| `orchestrator/nodes/evaluator.py` | Evaluator node — pass/fail for plan and implementation |
| `orchestrator/nodes/advisor.py` | Advisor node — deep analysis on repeated failures |
| `orchestrator/nodes/generator.py` | Generator node — implements sprint contract |
| `orchestrator/graph.py` | LangGraph graph + routing function |
| `artifacts/.gitkeep` | Keep artifacts dir in git |
| `tests/__init__.py` | Test package |
| `tests/test_state.py` | Schema validation tests |
| `tests/test_client.py` | AgentClient unit tests (mocked SDK) |
| `tests/test_nodes.py` | Node unit tests (mocked client) |
| `tests/test_graph.py` | Routing logic + graph compilation tests |

---

## Task 1: Bootstrap project with uv

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.env.example`
- Create: `artifacts/.gitkeep`

- [ ] **Step 1: Write the failing test** — verify the environment can import our top-level package

```python
# tests/__init__.py  (empty)
# tests/test_state.py — write this first so the project structure is needed
def test_import_orchestrator():
    import orchestrator  # noqa: F401
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
uv run pytest tests/test_state.py::test_import_orchestrator -v
```

Expected: `ModuleNotFoundError: No module named 'orchestrator'`

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "agent-orchestrator"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.92.0",
    "langgraph>=0.3.0",
    "pydantic>=2.0.0",
    "typer>=0.12.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.12.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Create `.python-version`**

```
3.11
```

- [ ] **Step 5: Create `.env.example`**

```bash
# Copy to .env and fill in values
ANTHROPIC_API_KEY=your_key_here
```

- [ ] **Step 6: Create package skeleton**

```bash
mkdir -p orchestrator/agents orchestrator/nodes artifacts tests
touch orchestrator/__init__.py
touch orchestrator/agents/__init__.py
touch orchestrator/nodes/__init__.py
touch tests/__init__.py
touch artifacts/.gitkeep
```

- [ ] **Step 7: Install dependencies**

```bash
uv sync --extra dev
```

Expected: resolves and installs all packages, creates `uv.lock`

- [ ] **Step 8: Run the test to confirm it passes**

```bash
uv run pytest tests/test_state.py::test_import_orchestrator -v
```

Expected: `PASSED`

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock .python-version .env.example artifacts/.gitkeep \
        orchestrator/__init__.py orchestrator/agents/__init__.py \
        orchestrator/nodes/__init__.py tests/__init__.py
git commit -m "Bootstrap project: uv environment, package skeleton, deps"
```

---

## Task 2: GraphState TypedDict and Pydantic schemas

**Files:**
- Create: `orchestrator/state.py`
- Modify: `tests/test_state.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_state.py
import pytest
from pydantic import ValidationError
from orchestrator.state import (
    Plan,
    EvaluationResult,
    AdvisorMemo,
    Task,
    SprintContract,
    Implementation,
)


def test_import_orchestrator():
    import orchestrator  # noqa: F401


def test_plan_valid():
    plan = Plan(
        summary="Build a REST API",
        steps=["Define endpoints", "Write handlers"],
        open_questions=["Which framework?"],
    )
    assert plan.summary == "Build a REST API"
    assert len(plan.steps) == 2


def test_evaluation_result_pass():
    result = EvaluationResult(
        verdict="pass",
        blockers=[],
        next_actions=[],
    )
    assert result.verdict == "pass"


def test_evaluation_result_fail():
    result = EvaluationResult(
        verdict="fail",
        blockers=["Missing error handling"],
        next_actions=["Add try/except blocks"],
    )
    assert result.verdict == "fail"
    assert len(result.blockers) == 1


def test_evaluation_result_invalid_verdict():
    with pytest.raises(ValidationError):
        EvaluationResult(verdict="maybe", blockers=[], next_actions=[])


def test_advisor_memo_no_approach():
    memo = AdvisorMemo(
        analysis="The plan lacks specificity",
        recommendations=["Add concrete file paths"],
        suggested_approach=None,
    )
    assert memo.suggested_approach is None


def test_sprint_contract_valid():
    contract = SprintContract(
        goal="Implement auth module",
        tasks=[
            Task(
                id="T1",
                description="Create user model",
                acceptance_criteria=["Model has email field"],
            )
        ],
        constraints=["No external auth libraries"],
    )
    assert len(contract.tasks) == 1
    assert contract.tasks[0].id == "T1"


def test_implementation_valid():
    impl = Implementation(
        files_written=["src/auth.py", "tests/test_auth.py"],
        summary="Implemented basic auth",
    )
    assert len(impl.files_written) == 2
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_state.py -v
```

Expected: `ImportError` on `from orchestrator.state import ...`

- [ ] **Step 3: Create `orchestrator/state.py`**

```python
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel
from typing_extensions import TypedDict


class Plan(BaseModel):
    summary: str
    steps: list[str]
    open_questions: list[str]


class EvaluationResult(BaseModel):
    verdict: Literal["pass", "fail"]
    blockers: list[str]
    next_actions: list[str]


class AdvisorMemo(BaseModel):
    analysis: str
    recommendations: list[str]
    suggested_approach: Optional[str] = None


class Task(BaseModel):
    id: str
    description: str
    acceptance_criteria: list[str]


class SprintContract(BaseModel):
    goal: str
    tasks: list[Task]
    constraints: list[str]


class Implementation(BaseModel):
    files_written: list[str]
    summary: str


class GraphState(TypedDict, total=False):
    idea: str
    phase: Literal["planning", "implementation", "done", "failed"]
    plan: Optional[Plan]
    sprint_contract: Optional[SprintContract]
    implementation: Optional[Implementation]
    evaluation: Optional[EvaluationResult]
    advisor_memo: Optional[AdvisorMemo]
    revision_count: int
    advisor_used: bool
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_state.py -v
```

Expected: all 8 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add orchestrator/state.py tests/test_state.py
git commit -m "Add GraphState TypedDict and Pydantic schemas with tests"
```

---

## Task 3: AgentClient wrapper

**Files:**
- Create: `orchestrator/agents/client.py`
- Create: `tests/test_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_client.py
from unittest.mock import MagicMock, patch, call
import pytest
from orchestrator.agents.client import AgentClient


def _make_mock_anthropic(agent_text: str = '{"summary": "test"}'):
    """Build a mock anthropic.Anthropic() with agents/sessions wired up."""
    mock_client = MagicMock()

    # agents.create returns an object with .id and .version
    mock_agent = MagicMock()
    mock_agent.id = "agent_test123"
    mock_agent.version = 1
    mock_client.beta.agents.create.return_value = mock_agent

    # environments.create
    mock_env = MagicMock()
    mock_env.id = "env_test123"
    mock_client.beta.environments.create.return_value = mock_env

    # sessions.create
    mock_session = MagicMock()
    mock_session.id = "sesn_test123"
    mock_client.beta.sessions.create.return_value = mock_session

    # sessions.events.send (fire and forget)
    mock_client.beta.sessions.events.send.return_value = None

    # sessions.stream — returns a context manager yielding events
    mock_text_event = MagicMock()
    mock_text_event.type = "agent.message"
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = agent_text
    mock_text_event.content = [mock_text_block]

    mock_idle_event = MagicMock()
    mock_idle_event.type = "session.status_idle"
    mock_stop_reason = MagicMock()
    mock_stop_reason.type = "end_turn"
    mock_idle_event.stop_reason = mock_stop_reason

    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(
        return_value=iter([mock_text_event, mock_idle_event])
    )
    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream_ctx.__exit__ = MagicMock(return_value=False)
    mock_client.beta.sessions.stream.return_value = mock_stream_ctx

    return mock_client


def test_agent_client_run_returns_text():
    mock_anthropic = _make_mock_anthropic('{"verdict": "pass", "blockers": [], "next_actions": []}')
    with patch("orchestrator.agents.client.anthropic.Anthropic", return_value=mock_anthropic):
        client = AgentClient(
            name="TestAgent",
            model="claude-sonnet-4-6",
            system_prompt="You are a test agent.",
        )
        result = client.run("Hello, world")

    assert '{"verdict": "pass"' in result


def test_agent_client_creates_agent_and_session():
    mock_anthropic = _make_mock_anthropic()
    with patch("orchestrator.agents.client.anthropic.Anthropic", return_value=mock_anthropic):
        client = AgentClient(
            name="Planner",
            model="claude-sonnet-4-6",
            system_prompt="You are a planner.",
        )
        client.run("Plan something")

    mock_anthropic.beta.agents.create.assert_called_once()
    create_call_kwargs = mock_anthropic.beta.agents.create.call_args[1]
    assert create_call_kwargs["name"] == "Planner"
    assert create_call_kwargs["model"] == "claude-sonnet-4-6"
    mock_anthropic.beta.sessions.create.assert_called_once()


def test_agent_client_sends_message_and_streams():
    mock_anthropic = _make_mock_anthropic()
    with patch("orchestrator.agents.client.anthropic.Anthropic", return_value=mock_anthropic):
        client = AgentClient(
            name="Evaluator",
            model="claude-sonnet-4-6",
            system_prompt="You are an evaluator.",
        )
        client.run("Evaluate this plan")

    # Message was sent
    mock_anthropic.beta.sessions.events.send.assert_called_once()
    sent_call = mock_anthropic.beta.sessions.events.send.call_args
    events = sent_call[1]["events"]
    assert events[0]["type"] == "user.message"
    assert events[0]["content"][0]["text"] == "Evaluate this plan"

    # Stream was opened
    mock_anthropic.beta.sessions.stream.assert_called_once()


def test_agent_client_with_file_write_tools():
    mock_anthropic = _make_mock_anthropic()
    with patch("orchestrator.agents.client.anthropic.Anthropic", return_value=mock_anthropic):
        client = AgentClient(
            name="Generator",
            model="claude-sonnet-4-6",
            system_prompt="You are a generator.",
            tools=[{"type": "agent_toolset_20260401", "default_config": {"enabled": True}}],
        )
        client.run("Implement this")

    create_call = mock_anthropic.beta.agents.create.call_args[1]
    assert len(create_call["tools"]) == 1
    assert create_call["tools"][0]["type"] == "agent_toolset_20260401"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_client.py -v
```

Expected: `ImportError` on `from orchestrator.agents.client import AgentClient`

- [ ] **Step 3: Create `orchestrator/agents/client.py`**

```python
from __future__ import annotations

import os
import uuid

import anthropic
from dotenv import load_dotenv

load_dotenv()


class AgentClient:
    """Thin wrapper around the Anthropic Managed Agents API.

    Creates a fresh agent + environment + session per `run()` call.
    Suitable for v1 where each run is independent.
    """

    def __init__(
        self,
        name: str,
        model: str,
        system_prompt: str,
        tools: list[dict] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self._client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    def run(self, user_message: str) -> str:
        """Create a session, send user_message, stream response, return text."""
        # Create a throwaway environment per run (v1 simplicity)
        env = self._client.beta.environments.create(
            name=f"orchestrator-{uuid.uuid4().hex[:8]}",
            config={"type": "cloud", "networking": {"type": "unrestricted"}},
        )

        # Create agent with role-specific config
        agent = self._client.beta.agents.create(
            name=self.name,
            model=self.model,
            system=self.system_prompt,
            tools=self.tools,
        )

        # Create session pinned to this agent version
        session = self._client.beta.sessions.create(
            agent={"type": "agent", "id": agent.id, "version": agent.version},
            environment_id=env.id,
        )

        # Stream-first: open stream, send message, collect response
        text_parts: list[str] = []

        with self._client.beta.sessions.stream(session_id=session.id) as stream:
            self._client.beta.sessions.events.send(
                session_id=session.id,
                events=[
                    {
                        "type": "user.message",
                        "content": [{"type": "text", "text": user_message}],
                    }
                ],
            )

            for event in stream:
                if event.type == "agent.message":
                    for block in event.content:
                        if block.type == "text":
                            text_parts.append(block.text)
                elif event.type == "session.status_idle":
                    if event.stop_reason.type != "requires_action":
                        break
                elif event.type == "session.status_terminated":
                    break

        return "".join(text_parts)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_client.py -v
```

Expected: all 4 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add orchestrator/agents/client.py tests/test_client.py
git commit -m "Add AgentClient wrapper for Managed Agents API with unit tests"
```

---

## Task 4: Planner node

**Files:**
- Create: `orchestrator/nodes/planner.py`
- Modify: `tests/test_nodes.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_nodes.py
import json
from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError

from orchestrator.state import (
    GraphState,
    Plan,
    EvaluationResult,
    AdvisorMemo,
)


def _make_client_mock(response_text: str) -> MagicMock:
    mock = MagicMock()
    mock.run.return_value = response_text
    return mock


# ---------------------------------------------------------------------------
# Planner node
# ---------------------------------------------------------------------------

def test_planner_node_returns_plan():
    from orchestrator.nodes.planner import planner_node

    valid_plan_json = json.dumps({
        "summary": "Build a REST API for todos",
        "steps": ["Define models", "Write handlers", "Add tests"],
        "open_questions": ["Which DB?"],
    })
    mock_client = _make_client_mock(valid_plan_json)

    state: GraphState = {
        "idea": "Build a REST API for todos",
        "phase": "planning",
        "revision_count": 0,
        "advisor_used": False,
        "plan": None,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": None,
        "advisor_memo": None,
    }

    with patch("orchestrator.nodes.planner.AgentClient", return_value=mock_client):
        result = planner_node(state)

    assert "plan" in result
    assert isinstance(result["plan"], Plan)
    assert result["plan"].summary == "Build a REST API for todos"
    assert result["revision_count"] == 1


def test_planner_node_includes_eval_feedback_on_retry():
    from orchestrator.nodes.planner import planner_node

    valid_plan_json = json.dumps({
        "summary": "Revised plan",
        "steps": ["Step 1"],
        "open_questions": [],
    })
    mock_client = _make_client_mock(valid_plan_json)

    eval_result = EvaluationResult(
        verdict="fail",
        blockers=["Missing error handling"],
        next_actions=["Add try/except"],
    )
    state: GraphState = {
        "idea": "Build something",
        "phase": "planning",
        "revision_count": 1,
        "advisor_used": False,
        "plan": None,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": eval_result,
        "advisor_memo": None,
    }

    with patch("orchestrator.nodes.planner.AgentClient", return_value=mock_client):
        planner_node(state)

    call_args = mock_client.run.call_args[0][0]
    assert "Missing error handling" in call_args


def test_planner_node_includes_advisor_memo():
    from orchestrator.nodes.planner import planner_node

    valid_plan_json = json.dumps({
        "summary": "Plan with advisor input",
        "steps": ["Step 1"],
        "open_questions": [],
    })
    mock_client = _make_client_mock(valid_plan_json)

    memo = AdvisorMemo(
        analysis="Root cause: vague requirements",
        recommendations=["Define acceptance criteria"],
        suggested_approach="Start with a use-case diagram",
    )
    state: GraphState = {
        "idea": "Build something",
        "phase": "planning",
        "revision_count": 0,
        "advisor_used": True,
        "plan": None,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": None,
        "advisor_memo": memo,
    }

    with patch("orchestrator.nodes.planner.AgentClient", return_value=mock_client):
        planner_node(state)

    call_args = mock_client.run.call_args[0][0]
    assert "Root cause: vague requirements" in call_args


def test_planner_node_retries_on_bad_json():
    from orchestrator.nodes.planner import planner_node

    valid_plan_json = json.dumps({
        "summary": "Recovered plan",
        "steps": ["Step 1"],
        "open_questions": [],
    })
    mock_client = MagicMock()
    # First call returns invalid JSON, second returns valid
    mock_client.run.side_effect = ["not json at all", valid_plan_json]

    state: GraphState = {
        "idea": "Build something",
        "phase": "planning",
        "revision_count": 0,
        "advisor_used": False,
        "plan": None,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": None,
        "advisor_memo": None,
    }

    with patch("orchestrator.nodes.planner.AgentClient", return_value=mock_client):
        result = planner_node(state)

    assert mock_client.run.call_count == 2
    assert result["plan"].summary == "Recovered plan"


def test_planner_node_raises_after_two_bad_json():
    from orchestrator.nodes.planner import planner_node

    mock_client = _make_client_mock("still not json")
    mock_client.run.side_effect = ["bad json 1", "bad json 2"]

    state: GraphState = {
        "idea": "Build something",
        "phase": "planning",
        "revision_count": 0,
        "advisor_used": False,
        "plan": None,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": None,
        "advisor_memo": None,
    }

    with patch("orchestrator.nodes.planner.AgentClient", return_value=mock_client):
        with pytest.raises(ValueError, match="Failed to parse planner response"):
            planner_node(state)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_nodes.py -v -k "planner"
```

Expected: `ImportError` or `ModuleNotFoundError`

- [ ] **Step 3: Create `orchestrator/nodes/planner.py`**

```python
from __future__ import annotations

import json
import os
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
    artifacts = Path("artifacts")
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
    except (json.JSONDecodeError, ValidationError, KeyError):
        # One retry with explicit JSON instruction
        retry_message = (
            message
            + "\n\nIMPORTANT: Your previous response was not valid JSON. "
            "Respond ONLY with the raw JSON object, no other text."
        )
        response = client.run(retry_message)
        try:
            plan = _parse_plan(response)
        except (json.JSONDecodeError, ValidationError, KeyError) as exc:
            raise ValueError(f"Failed to parse planner response after retry: {exc}") from exc

    _persist_plan(plan, revision)
    return {"plan": plan, "revision_count": revision}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_nodes.py -v -k "planner"
```

Expected: all 5 planner tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add orchestrator/nodes/planner.py tests/test_nodes.py
git commit -m "Add planner node with JSON retry logic and artifact persistence"
```

---

## Task 5: Evaluator node

**Files:**
- Modify: `orchestrator/nodes/evaluator.py`
- Modify: `tests/test_nodes.py`

- [ ] **Step 1: Append evaluator tests to `tests/test_nodes.py`**

```python
# append to tests/test_nodes.py

# ---------------------------------------------------------------------------
# Evaluator node
# ---------------------------------------------------------------------------

def test_evaluator_node_planning_phase_pass():
    from orchestrator.nodes.evaluator import evaluator_node

    valid_eval_json = json.dumps({
        "verdict": "pass",
        "blockers": [],
        "next_actions": [],
    })
    mock_client = _make_client_mock(valid_eval_json)

    plan = Plan(
        summary="Build a REST API",
        steps=["Step 1"],
        open_questions=[],
    )
    state: GraphState = {
        "idea": "Build a REST API",
        "phase": "planning",
        "revision_count": 1,
        "advisor_used": False,
        "plan": plan,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": None,
        "advisor_memo": None,
    }

    with patch("orchestrator.nodes.evaluator.AgentClient", return_value=mock_client):
        result = evaluator_node(state)

    assert result["evaluation"].verdict == "pass"
    # When planning passes, evaluator should set sprint_contract
    assert result.get("sprint_contract") is not None


def test_evaluator_node_planning_phase_fail():
    from orchestrator.nodes.evaluator import evaluator_node

    valid_eval_json = json.dumps({
        "verdict": "fail",
        "blockers": ["No acceptance criteria"],
        "next_actions": ["Add acceptance criteria to each step"],
    })
    mock_client = _make_client_mock(valid_eval_json)

    plan = Plan(summary="Vague plan", steps=["Do stuff"], open_questions=[])
    state: GraphState = {
        "idea": "Do something",
        "phase": "planning",
        "revision_count": 1,
        "advisor_used": False,
        "plan": plan,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": None,
        "advisor_memo": None,
    }

    with patch("orchestrator.nodes.evaluator.AgentClient", return_value=mock_client):
        result = evaluator_node(state)

    assert result["evaluation"].verdict == "fail"
    assert "No acceptance criteria" in result["evaluation"].blockers


def test_evaluator_node_implementation_phase():
    from orchestrator.nodes.evaluator import evaluator_node
    from orchestrator.state import Implementation, SprintContract, Task

    valid_eval_json = json.dumps({
        "verdict": "pass",
        "blockers": [],
        "next_actions": [],
    })
    mock_client = _make_client_mock(valid_eval_json)

    impl = Implementation(files_written=["src/main.py"], summary="Implemented main module")
    contract = SprintContract(
        goal="Build main module",
        tasks=[Task(id="T1", description="Write main.py", acceptance_criteria=["File exists"])],
        constraints=[],
    )
    state: GraphState = {
        "idea": "Build something",
        "phase": "implementation",
        "revision_count": 1,
        "advisor_used": False,
        "plan": None,
        "sprint_contract": contract,
        "implementation": impl,
        "evaluation": None,
        "advisor_memo": None,
    }

    with patch("orchestrator.nodes.evaluator.AgentClient", return_value=mock_client):
        result = evaluator_node(state)

    assert result["evaluation"].verdict == "pass"
    # Phase should not be updated by evaluator — routing handles that
    assert "phase" not in result
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/test_nodes.py -v -k "evaluator"
```

Expected: `ImportError` on `from orchestrator.nodes.evaluator import evaluator_node`

- [ ] **Step 3: Create `orchestrator/nodes/evaluator.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from orchestrator.agents.client import AgentClient
from orchestrator.state import EvaluationResult, GraphState, SprintContract, Task

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

_MODEL = "claude-sonnet-4-6"


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
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    data = json.loads(text)
    return EvaluationResult.model_validate(data)


def _persist_eval(result: EvaluationResult, phase: str, revision: int) -> None:
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    path = artifacts / f"eval_{phase}_v{revision}.json"
    path.write_text(result.model_dump_json(indent=2))


def evaluator_node(state: GraphState) -> dict:
    phase = state["phase"]
    if phase == "planning":
        system_prompt = _PLANNING_SYSTEM_PROMPT
        message = _build_planning_message(state)
    else:
        system_prompt = _IMPLEMENTATION_SYSTEM_PROMPT
        message = _build_implementation_message(state)

    client = AgentClient(
        name="Evaluator",
        model=_MODEL,
        system_prompt=system_prompt,
    )

    response = client.run(message)
    try:
        result = _parse_eval(response)
    except (json.JSONDecodeError, ValidationError) as exc:
        retry = response + "\n\nRespond ONLY with the raw JSON object, no other text."
        response = client.run(retry)
        try:
            result = _parse_eval(response)
        except (json.JSONDecodeError, ValidationError) as exc2:
            raise ValueError(f"Failed to parse evaluator response after retry: {exc2}") from exc2

    _persist_eval(result, phase, state.get("revision_count", 0))

    updates: dict = {"evaluation": result}
    # If the plan passed, generate sprint contract
    if phase == "planning" and result.verdict == "pass":
        updates["sprint_contract"] = _build_sprint_contract(state["plan"])

    return updates
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_nodes.py -v -k "evaluator"
```

Expected: all 3 evaluator tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add orchestrator/nodes/evaluator.py tests/test_nodes.py
git commit -m "Add evaluator node with planning/implementation phase support"
```

---

## Task 6: Advisor node

**Files:**
- Create: `orchestrator/nodes/advisor.py`
- Modify: `tests/test_nodes.py`

- [ ] **Step 1: Append advisor tests to `tests/test_nodes.py`**

```python
# append to tests/test_nodes.py

# ---------------------------------------------------------------------------
# Advisor node
# ---------------------------------------------------------------------------

def test_advisor_node_returns_memo():
    from orchestrator.nodes.advisor import advisor_node
    from orchestrator.state import AdvisorMemo

    valid_memo_json = json.dumps({
        "analysis": "The plan lacks concrete file structure",
        "recommendations": ["Define directory layout", "Specify config format"],
        "suggested_approach": "Start with a monorepo layout",
    })
    mock_client = _make_client_mock(valid_memo_json)

    plan = Plan(
        summary="Build something vague",
        steps=["Do stuff", "More stuff"],
        open_questions=["How?"],
    )
    eval_result = EvaluationResult(
        verdict="fail",
        blockers=["Too vague"],
        next_actions=["Be specific"],
    )
    state: GraphState = {
        "idea": "Build a thing",
        "phase": "planning",
        "revision_count": 3,
        "advisor_used": False,
        "plan": plan,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": eval_result,
        "advisor_memo": None,
    }

    with patch("orchestrator.nodes.advisor.AgentClient", return_value=mock_client):
        result = advisor_node(state)

    assert "advisor_memo" in result
    assert isinstance(result["advisor_memo"], AdvisorMemo)
    assert result["advisor_memo"].analysis == "The plan lacks concrete file structure"
    assert result["advisor_used"] is True
    assert result["revision_count"] == 0  # reset for next planner round


def test_advisor_node_includes_history_in_message():
    from orchestrator.nodes.advisor import advisor_node

    valid_memo_json = json.dumps({
        "analysis": "root cause",
        "recommendations": ["fix it"],
        "suggested_approach": None,
    })
    mock_client = _make_client_mock(valid_memo_json)

    plan = Plan(summary="Plan", steps=["Step 1"], open_questions=[])
    eval_result = EvaluationResult(
        verdict="fail",
        blockers=["Critical blocker"],
        next_actions=["Fix blocker"],
    )
    state: GraphState = {
        "idea": "Original idea",
        "phase": "planning",
        "revision_count": 3,
        "advisor_used": False,
        "plan": plan,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": eval_result,
        "advisor_memo": None,
    }

    with patch("orchestrator.nodes.advisor.AgentClient", return_value=mock_client):
        advisor_node(state)

    call_args = mock_client.run.call_args[0][0]
    assert "Original idea" in call_args
    assert "Critical blocker" in call_args
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/test_nodes.py -v -k "advisor"
```

Expected: `ImportError`

- [ ] **Step 3: Create `orchestrator/nodes/advisor.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from orchestrator.agents.client import AgentClient
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

_MODEL = "claude-opus-4-6"


def _build_message(state: GraphState) -> str:
    plan = state.get("plan")
    eval_ = state.get("evaluation")

    parts = [
        f"Original idea: {state['idea']}",
        f"\nThis plan has failed evaluation {state.get('revision_count', 3)} times.",
    ]

    if plan:
        parts.append(
            f"\nLatest plan summary: {plan.summary}\n"
            f"Steps: {', '.join(plan.steps)}"
        )

    if eval_:
        parts.append("\nLatest evaluation failure:")
        if eval_.blockers:
            parts.append("Blockers:\n" + "\n".join(f"  - {b}" for b in eval_.blockers))
        if eval_.next_actions:
            parts.append(
                "Suggested next actions:\n"
                + "\n".join(f"  - {a}" for a in eval_.next_actions)
            )

    parts.append(
        "\nPlease diagnose the root cause and provide recommendations that "
        "will allow the planner to succeed on its next attempt."
    )
    return "\n".join(parts)


def _parse_memo(text: str) -> AdvisorMemo:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    data = json.loads(text)
    return AdvisorMemo.model_validate(data)


def _persist_memo(memo: AdvisorMemo) -> None:
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    path = artifacts / "advisor_memo.md"
    content = f"# Advisor Memo\n\n## Analysis\n{memo.analysis}\n\n"
    content += "## Recommendations\n\n" + "\n".join(
        f"- {r}" for r in memo.recommendations
    )
    if memo.suggested_approach:
        content += f"\n\n## Suggested Approach\n{memo.suggested_approach}"
    path.write_text(content)


def advisor_node(state: GraphState) -> dict:
    client = AgentClient(
        name="Advisor",
        model=_MODEL,
        system_prompt=_SYSTEM_PROMPT,
    )
    message = _build_message(state)

    response = client.run(message)
    try:
        memo = _parse_memo(response)
    except (json.JSONDecodeError, ValidationError) as exc:
        retry = response + "\n\nRespond ONLY with the raw JSON object, no other text."
        response = client.run(retry)
        try:
            memo = _parse_memo(response)
        except (json.JSONDecodeError, ValidationError) as exc2:
            raise ValueError(f"Failed to parse advisor response after retry: {exc2}") from exc2

    _persist_memo(memo)
    return {
        "advisor_memo": memo,
        "advisor_used": True,
        "revision_count": 0,  # reset revision counter for next planner pass
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_nodes.py -v -k "advisor"
```

Expected: all 2 advisor tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add orchestrator/nodes/advisor.py tests/test_nodes.py
git commit -m "Add advisor node (Opus) with root cause analysis"
```

---

## Task 7: Generator node

**Files:**
- Create: `orchestrator/nodes/generator.py`
- Modify: `tests/test_nodes.py`

- [ ] **Step 1: Append generator tests to `tests/test_nodes.py`**

```python
# append to tests/test_nodes.py

# ---------------------------------------------------------------------------
# Generator node
# ---------------------------------------------------------------------------

def test_generator_node_returns_implementation():
    from orchestrator.nodes.generator import generator_node
    from orchestrator.state import Implementation, SprintContract, Task

    valid_impl_json = json.dumps({
        "files_written": ["src/main.py", "tests/test_main.py"],
        "summary": "Implemented main module with tests",
    })
    mock_client = _make_client_mock(valid_impl_json)

    contract = SprintContract(
        goal="Build main module",
        tasks=[
            Task(
                id="T1",
                description="Write main.py",
                acceptance_criteria=["File exists", "Has main() function"],
            )
        ],
        constraints=["Use stdlib only"],
    )
    state: GraphState = {
        "idea": "Build something",
        "phase": "implementation",
        "revision_count": 0,
        "advisor_used": False,
        "plan": None,
        "sprint_contract": contract,
        "implementation": None,
        "evaluation": None,
        "advisor_memo": None,
    }

    with patch("orchestrator.nodes.generator.AgentClient", return_value=mock_client):
        result = generator_node(state)

    assert "implementation" in result
    assert isinstance(result["implementation"], Implementation)
    assert "src/main.py" in result["implementation"].files_written


def test_generator_node_uses_file_write_tools():
    from orchestrator.nodes.generator import generator_node, _GENERATOR_TOOLS
    from orchestrator.state import SprintContract, Task

    valid_impl_json = json.dumps({
        "files_written": ["src/app.py"],
        "summary": "Implemented app",
    })
    mock_client = _make_client_mock(valid_impl_json)

    contract = SprintContract(
        goal="Build app",
        tasks=[Task(id="T1", description="Write app.py", acceptance_criteria=["Done"])],
        constraints=[],
    )
    state: GraphState = {
        "idea": "Build app",
        "phase": "implementation",
        "revision_count": 0,
        "advisor_used": False,
        "plan": None,
        "sprint_contract": contract,
        "implementation": None,
        "evaluation": None,
        "advisor_memo": None,
    }

    captured_kwargs = {}

    def capture_init(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_client

    with patch("orchestrator.nodes.generator.AgentClient", side_effect=lambda **kw: capture_init(**kw)):
        generator_node(state)

    assert captured_kwargs.get("tools") == _GENERATOR_TOOLS
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/test_nodes.py -v -k "generator"
```

Expected: `ImportError`

- [ ] **Step 3: Create `orchestrator/nodes/generator.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from orchestrator.agents.client import AgentClient
from orchestrator.state import GraphState, Implementation

_SYSTEM_PROMPT = """\
You are a software implementation agent. Given an approved sprint contract,
implement all tasks by writing files to disk using the write tool.

For each task:
1. Write the implementation file(s) using the write tool.
2. Write a corresponding test file using the write tool.

After writing all files, respond with ONLY this JSON (no prose, no markdown fences):
{
  "files_written": ["<exact path 1>", "<exact path 2>"],
  "summary": "<one-paragraph summary of what was implemented>"
}
"""

_MODEL = "claude-sonnet-4-6"

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
    parts = [
        f"Sprint goal: {contract.goal}",
        "\nTasks to implement:",
    ]
    for task in contract.tasks:
        parts.append(f"\n  [{task.id}] {task.description}")
        parts.append("  Acceptance criteria:")
        for criterion in task.acceptance_criteria:
            parts.append(f"    - {criterion}")
    if contract.constraints:
        parts.append("\nConstraints:")
        for c in contract.constraints:
            parts.append(f"  - {c}")

    if state.get("evaluation") and state["evaluation"].verdict == "fail":
        eval_ = state["evaluation"]
        parts.append("\nPrevious implementation failed evaluation. Blockers:")
        for b in eval_.blockers:
            parts.append(f"  - {b}")
        parts.append("Required fixes:")
        for a in eval_.next_actions:
            parts.append(f"  - {a}")

    parts.append(
        "\nAfter writing all files, respond ONLY with the JSON summary as specified."
    )
    return "\n".join(parts)


def _parse_implementation(text: str) -> Implementation:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    # Find the last JSON object in case there's preamble text
    start = text.rfind("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    data = json.loads(text)
    return Implementation.model_validate(data)


def _persist_implementation(impl: Implementation) -> None:
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    path = artifacts / "implementation.md"
    content = f"# Implementation Summary\n\n{impl.summary}\n\n"
    content += "## Files Written\n\n" + "\n".join(
        f"- `{f}`" for f in impl.files_written
    )
    path.write_text(content)


def generator_node(state: GraphState) -> dict:
    client = AgentClient(
        name="Generator",
        model=_MODEL,
        system_prompt=_SYSTEM_PROMPT,
        tools=_GENERATOR_TOOLS,
    )
    message = _build_message(state)

    response = client.run(message)
    try:
        impl = _parse_implementation(response)
    except (json.JSONDecodeError, ValidationError) as exc:
        retry = response + "\n\nNow respond ONLY with the JSON summary. No other text."
        response = client.run(retry)
        try:
            impl = _parse_implementation(response)
        except (json.JSONDecodeError, ValidationError) as exc2:
            raise ValueError(
                f"Failed to parse generator response after retry: {exc2}"
            ) from exc2

    _persist_implementation(impl)
    return {
        "implementation": impl,
        "phase": "implementation",
        "revision_count": state.get("revision_count", 0) + 1,
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_nodes.py -v -k "generator"
```

Expected: all 2 generator tests `PASSED`

- [ ] **Step 5: Run all node tests together**

```bash
uv run pytest tests/test_nodes.py -v
```

Expected: all 12 node tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git add orchestrator/nodes/generator.py tests/test_nodes.py
git commit -m "Add generator node with file-write tools and implementation parsing"
```

---

## Task 8: Routing function and LangGraph graph

**Files:**
- Create: `orchestrator/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_graph.py
import pytest
from orchestrator.state import EvaluationResult, GraphState


def _state_with_eval(verdict: str, revision: int, advisor_used: bool, phase: str = "planning") -> GraphState:
    return {
        "idea": "test",
        "phase": phase,
        "revision_count": revision,
        "advisor_used": advisor_used,
        "plan": None,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": EvaluationResult(
            verdict=verdict,
            blockers=["issue"] if verdict == "fail" else [],
            next_actions=["fix it"] if verdict == "fail" else [],
        ),
        "advisor_memo": None,
    }


# ---------------------------------------------------------------------------
# route_after_evaluator — pure function tests
# ---------------------------------------------------------------------------

def test_route_planning_pass_goes_to_generator():
    from orchestrator.graph import route_after_evaluator

    state = _state_with_eval("pass", revision=1, advisor_used=False, phase="planning")
    assert route_after_evaluator(state) == "to_generator"


def test_route_implementation_pass_is_done():
    from orchestrator.graph import route_after_evaluator

    state = _state_with_eval("pass", revision=1, advisor_used=False, phase="implementation")
    assert route_after_evaluator(state) == "done"


def test_route_fail_low_count_revises():
    from orchestrator.graph import route_after_evaluator

    state = _state_with_eval("fail", revision=1, advisor_used=False)
    assert route_after_evaluator(state) == "revise"


def test_route_fail_count_2_revises():
    from orchestrator.graph import route_after_evaluator

    state = _state_with_eval("fail", revision=2, advisor_used=False)
    assert route_after_evaluator(state) == "revise"


def test_route_fail_count_3_no_advisor_routes_to_advisor():
    from orchestrator.graph import route_after_evaluator

    state = _state_with_eval("fail", revision=3, advisor_used=False)
    assert route_after_evaluator(state) == "advisor"


def test_route_fail_count_3_advisor_used_hard_stops():
    from orchestrator.graph import route_after_evaluator

    state = _state_with_eval("fail", revision=3, advisor_used=True)
    assert route_after_evaluator(state) == "hard_stop"


def test_route_fail_high_count_advisor_used_hard_stops():
    from orchestrator.graph import route_after_evaluator

    state = _state_with_eval("fail", revision=5, advisor_used=True)
    assert route_after_evaluator(state) == "hard_stop"


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def test_graph_compiles():
    from orchestrator.graph import build_graph

    graph = build_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    from orchestrator.graph import build_graph

    graph = build_graph()
    node_names = set(graph.nodes)
    assert "planner" in node_names
    assert "evaluator" in node_names
    assert "advisor" in node_names
    assert "generator" in node_names
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: `ImportError` on `from orchestrator.graph import route_after_evaluator`

- [ ] **Step 3: Create `orchestrator/graph.py`**

```python
from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from orchestrator.nodes.advisor import advisor_node
from orchestrator.nodes.evaluator import evaluator_node
from orchestrator.nodes.generator import generator_node
from orchestrator.nodes.planner import planner_node
from orchestrator.state import GraphState


def route_after_evaluator(
    state: GraphState,
) -> Literal["to_generator", "done", "revise", "advisor", "hard_stop"]:
    """Determine next step after evaluator runs."""
    eval_ = state["evaluation"]
    phase = state["phase"]
    revision_count = state.get("revision_count", 0)
    advisor_used = state.get("advisor_used", False)

    if eval_.verdict == "pass":
        if phase == "planning":
            return "to_generator"  # approved plan → hand to generator
        return "done"              # approved implementation → finished

    # Verdict is "fail"
    if revision_count >= 3 and advisor_used:
        return "hard_stop"         # advisor already tried, still failing
    if revision_count >= 3 and not advisor_used:
        return "advisor"           # escalate to advisor
    return "revise"                # still within retry budget


def _mark_done(state: GraphState) -> dict:
    return {"phase": "done"}


def _mark_failed(state: GraphState) -> dict:
    return {"phase": "failed"}


def _set_implementation_phase(state: GraphState) -> dict:
    return {"phase": "implementation", "revision_count": 0}


def build_graph():
    """Build and compile the agent orchestrator LangGraph."""
    builder = StateGraph(GraphState)

    # Register nodes
    builder.add_node("planner", planner_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("advisor", advisor_node)
    builder.add_node("generator", generator_node)
    builder.add_node("transition_to_impl", _set_implementation_phase)
    builder.add_node("mark_done", _mark_done)
    builder.add_node("mark_failed", _mark_failed)

    # Entry point
    builder.set_entry_point("planner")

    # Planner → Evaluator
    builder.add_edge("planner", "evaluator")

    # Evaluator → conditional routing
    builder.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {
            "to_generator": "transition_to_impl",
            "done": "mark_done",
            "revise": "planner",
            "advisor": "advisor",
            "hard_stop": "mark_failed",
        },
    )

    # Transitions
    builder.add_edge("transition_to_impl", "generator")
    builder.add_edge("advisor", "planner")
    builder.add_edge("generator", "evaluator")
    builder.add_edge("mark_done", END)
    builder.add_edge("mark_failed", END)

    return builder.compile()
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: all 9 tests `PASSED`

- [ ] **Step 5: Run the full test suite**

```bash
uv run pytest -v
```

Expected: all tests pass (state + client + nodes + graph)

- [ ] **Step 6: Commit**

```bash
git add orchestrator/graph.py tests/test_graph.py
git commit -m "Add LangGraph graph with routing logic and full test coverage"
```

---

## Task 9: CLI entry point

**Files:**
- Create: `main.py`

- [ ] **Step 1: Write the CLI — no test needed (thin wrapper, logic is in graph)**

```python
# main.py
"""
Agent Orchestrator CLI

Usage:
    python main.py "Build a REST API for todo items"
    python main.py "Build a REST API" --max-revisions 5
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(help="Run the agentic planner/evaluator/generator workflow.")


@app.command()
def run(
    idea: str = typer.Argument(..., help="The idea or feature request to implement"),
    max_revisions: int = typer.Option(3, help="Max evaluator revisions before advisor steps in"),
) -> None:
    """Turn an idea into an implemented sprint via LangGraph + Claude."""
    from orchestrator.graph import build_graph
    from orchestrator.state import GraphState

    typer.echo(f"\nStarting orchestrator for: {idea!r}\n")

    graph = build_graph()

    initial_state: GraphState = {
        "idea": idea,
        "phase": "planning",
        "revision_count": 0,
        "advisor_used": False,
        "plan": None,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": None,
        "advisor_memo": None,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        typer.echo(f"\nOrchestrator failed with exception: {exc}", err=True)
        raise typer.Exit(code=1)

    phase = final_state.get("phase")

    if phase == "done":
        impl = final_state.get("implementation")
        typer.echo("\nWorkflow complete.")
        if impl:
            typer.echo(f"Files written: {', '.join(impl.files_written)}")
            typer.echo(f"Summary: {impl.summary}")
        typer.echo("\nSee artifacts/ for all outputs.")

    elif phase == "failed":
        typer.echo("\nWorkflow could not converge.", err=True)
        eval_ = final_state.get("evaluation")
        if eval_:
            typer.echo("\nLast evaluation result:", err=True)
            typer.echo(eval_.model_dump_json(indent=2), err=True)
        advisor_memo = final_state.get("advisor_memo")
        if advisor_memo:
            typer.echo("\nAdvisor memo:", err=True)
            typer.echo(advisor_memo.model_dump_json(indent=2), err=True)
        typer.echo("\nAll artifacts preserved in artifacts/", err=True)
        raise typer.Exit(code=1)

    else:
        typer.echo(f"\nUnexpected final phase: {phase!r}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Verify the CLI parses correctly (no API calls)**

```bash
uv run python main.py --help
```

Expected output:
```
 Usage: main.py [OPTIONS] IDEA

 Turn an idea into an implemented sprint via LangGraph + Claude.

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    idea      TEXT  The idea or feature request to implement [required]      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "Add Typer CLI entry point for end-to-end workflow invocation"
```

---

## Task 10: End-to-end smoke test (dry-run with mocked graph)

**Files:**
- Modify: `tests/test_graph.py`

- [ ] **Step 1: Add smoke test with fully mocked nodes**

```python
# append to tests/test_graph.py
from unittest.mock import patch, MagicMock


def test_graph_planning_pass_reaches_done():
    """Smoke test: planner → evaluator (pass) → transition → generator → evaluator (pass) → done."""
    from orchestrator.graph import build_graph
    from orchestrator.state import (
        Plan,
        EvaluationResult,
        SprintContract,
        Task,
        Implementation,
    )

    approved_plan = Plan(
        summary="Build a simple module",
        steps=["Write module", "Write tests"],
        open_questions=[],
    )
    eval_pass = EvaluationResult(verdict="pass", blockers=[], next_actions=[])
    contract = SprintContract(
        goal="Build a simple module",
        tasks=[Task(id="T1", description="Write module", acceptance_criteria=["Done"])],
        constraints=[],
    )
    impl = Implementation(files_written=["src/module.py"], summary="Module written")

    def mock_planner(state):
        return {"plan": approved_plan, "revision_count": 1}

    def mock_evaluator(state):
        updates = {"evaluation": eval_pass}
        if state.get("phase") == "planning":
            updates["sprint_contract"] = contract
        return updates

    def mock_generator(state):
        return {"implementation": impl, "phase": "implementation", "revision_count": 1}

    graph = build_graph()

    with (
        patch("orchestrator.nodes.planner.planner_node", side_effect=mock_planner),
        patch("orchestrator.nodes.evaluator.evaluator_node", side_effect=mock_evaluator),
        patch("orchestrator.nodes.generator.generator_node", side_effect=mock_generator),
    ):
        # Re-build so patches are picked up
        from orchestrator import graph as graph_module
        with (
            patch.object(graph_module, "planner_node", mock_planner),
            patch.object(graph_module, "evaluator_node", mock_evaluator),
            patch.object(graph_module, "generator_node", mock_generator),
        ):
            g = graph_module.build_graph()
            initial = {
                "idea": "Build something",
                "phase": "planning",
                "revision_count": 0,
                "advisor_used": False,
                "plan": None,
                "sprint_contract": None,
                "implementation": None,
                "evaluation": None,
                "advisor_memo": None,
            }
            result = g.invoke(initial)

    assert result["phase"] == "done"


def test_graph_hard_stop_after_advisor():
    """Smoke test: evaluator fails 3+ times with advisor used → phase=failed."""
    from orchestrator.graph import build_graph
    from orchestrator.state import Plan, EvaluationResult, AdvisorMemo

    failing_plan = Plan(summary="Bad plan", steps=["Step 1"], open_questions=[])
    eval_fail = EvaluationResult(verdict="fail", blockers=["issue"], next_actions=["fix"])
    memo = AdvisorMemo(
        analysis="root cause",
        recommendations=["fix it"],
        suggested_approach=None,
    )

    call_count = {"n": 0}

    def mock_planner(state):
        call_count["n"] += 1
        return {"plan": failing_plan, "revision_count": state.get("revision_count", 0) + 1}

    def mock_evaluator(state):
        return {"evaluation": eval_fail}

    def mock_advisor(state):
        return {"advisor_memo": memo, "advisor_used": True, "revision_count": 0}

    from orchestrator import graph as graph_module

    with (
        patch.object(graph_module, "planner_node", mock_planner),
        patch.object(graph_module, "evaluator_node", mock_evaluator),
        patch.object(graph_module, "advisor_node", mock_advisor),
    ):
        g = graph_module.build_graph()
        initial = {
            "idea": "Bad idea",
            "phase": "planning",
            "revision_count": 0,
            "advisor_used": False,
            "plan": None,
            "sprint_contract": None,
            "implementation": None,
            "evaluation": None,
            "advisor_memo": None,
        }
        result = g.invoke(initial)

    assert result["phase"] == "failed"
```

- [ ] **Step 2: Run the smoke tests**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: all 11 graph tests `PASSED`

- [ ] **Step 3: Run the full test suite one final time**

```bash
uv run pytest -v
```

Expected: all tests `PASSED`, zero failures

- [ ] **Step 4: Commit**

```bash
git add tests/test_graph.py
git commit -m "Add end-to-end smoke tests for happy path and hard stop"
```

---

## Final State

After completing all 10 tasks, the repo contains:

```
agent-orchestrator/
├── pyproject.toml          # uv deps
├── uv.lock
├── .python-version         # 3.11
├── .env.example            # ANTHROPIC_API_KEY
├── main.py                 # python main.py "idea"
├── orchestrator/
│   ├── __init__.py
│   ├── state.py            # GraphState + all schemas
│   ├── graph.py            # LangGraph + route_after_evaluator
│   ├── agents/
│   │   ├── __init__.py
│   │   └── client.py       # AgentClient
│   └── nodes/
│       ├── __init__.py
│       ├── planner.py      # planner_node
│       ├── evaluator.py    # evaluator_node
│       ├── advisor.py      # advisor_node
│       └── generator.py    # generator_node
├── artifacts/
│   └── .gitkeep
└── tests/
    ├── __init__.py
    ├── test_state.py
    ├── test_client.py
    ├── test_nodes.py
    └── test_graph.py
```

To run against the real API:
```bash
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env
python main.py "Build a simple Python CLI that counts words in a file"
```
