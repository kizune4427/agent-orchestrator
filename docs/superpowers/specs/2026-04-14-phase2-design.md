# Phase 2 Design — Agent Orchestrator

**Date:** 2026-04-14  
**Status:** Approved  
**Implementation order:** 2.3 → 2.1 → 2.2 → 2.5 → 2.4

---

## Overview

Phase 2 adds five capabilities to the Phase 1 LangGraph orchestrator:

| Feature | Description |
|---------|-------------|
| 2.3 | Persistent run history with unique run IDs |
| 2.1 | Per-role model switching + OpenRouter backend |
| 2.2 | Streaming node-by-node output |
| 2.5 | Human-in-the-loop checkpoints |
| 2.4 | Planner-driven parallel branch exploration |

All features share a single architectural primitive: **`RunConfig`**.

---

## Core Infrastructure — `RunConfig`

`RunConfig` is a frozen Pydantic model built in `main.py` from CLI args and env vars. It is stored in `GraphState` and read by nodes and the client factory.

```python
class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str                          # YYYYMMDD-HHMMSS-{sha6}
    backend: Literal["anthropic", "openrouter"] = "anthropic"
    planner_model: str = "claude-sonnet-4-6"
    evaluator_model: str = "claude-sonnet-4-6"
    advisor_model: str = "claude-opus-4-6"
    generator_model: str = "claude-sonnet-4-6"
    auto_approve: bool = False
    parallel: bool = False
    from_node: Literal["planner", "evaluator", "generator"] | None = None
```

`GraphState` gains one field: `run_config: RunConfig`.

---

## 2.3 — Persistent Run History

**Run ID format:** `{YYYYMMDD-HHMMSS}-{sha6}` (sha6 = first 6 chars of `uuid4().hex`)

**Artifact layout:**
```
artifacts/
  runs.jsonl                   # append-only index
  {run_id}/
    plan_v{n}.md
    eval_{phase}_v{n}.json
    advisor_memo.md
    implementation.md
    branches.json              # parallel branches only
```

**`runs.jsonl` entry:**
```json
{"run_id": "20260414-143022-a3f9b1", "idea": "...", "phase": "done", "timestamp": "2026-04-14T14:30:22Z"}
```

**CLI:**
- `main.py` prints `Run ID: 20260414-143022-a3f9b1` on start
- `--run-id <id>` loads prior run's artifacts into state; defaults entry point to `planner`
- `--from-node planner|evaluator|generator` sets graph entry point; requires `--run-id`
- `--from-node` without `--run-id` is an error

---

## 2.1 — Model Switching

### Client abstraction

New `orchestrator/agents/base.py`:
```python
from abc import ABC, abstractmethod

class BaseAgentClient(ABC):
    @abstractmethod
    def run(self, user_message: str) -> str: ...
```

`AgentClient` (current) inherits from `BaseAgentClient`. Implementation unchanged.

### OpenRouter client

New `orchestrator/agents/openrouter.py`:
- Uses `httpx` to POST to OpenRouter's OpenAI-compatible endpoint
- API key read from `OPENROUTER_API_KEY` env var
- Constructor: `model: str`, `system_prompt: str`
- Same `.run(user_message) -> str` interface

### Client factory

New `orchestrator/agents/factory.py`:
```python
def make_client(
    role: Literal["planner", "evaluator", "advisor", "generator"],
    config: RunConfig,
    system_prompt: str,
) -> BaseAgentClient:
    model = getattr(config, f"{role}_model")
    if config.backend == "openrouter":
        return OpenRouterClient(model=model, system_prompt=system_prompt)
    return AgentClient(model=model, system_prompt=system_prompt)
```

Each node calls `make_client(role, state["run_config"], SYSTEM_PROMPT)` instead of constructing `AgentClient` directly.

### CLI flags
```
--backend anthropic|openrouter   (env: LLM_BACKEND)
--planner-model MODEL
--evaluator-model MODEL
--advisor-model MODEL
--generator-model MODEL
```

---

## 2.2 — Streaming Output

`main.py` switches from `graph.invoke()` to `graph.stream()`. A `print_node_summary(node_name, state_delta)` function formats output per node type.

**Format:**
```
━━━ planner ━━━━━━━━━━━━━━━━━━━━━━
Goal: Build a REST API for todo items
Steps:
  1. Define data models
  2. Implement CRUD endpoints
Open questions: none
────────────────────────────────────

━━━ evaluator ━━━━━━━━━━━━━━━━━━━━
Verdict: PASS
────────────────────────────────────

━━━ generator ━━━━━━━━━━━━━━━━━━━━
Files written:
  • generated/models.py
  • generated/routes.py
Summary: Implemented REST API with SQLite backend
────────────────────────────────────
```

Internal transition nodes (`transition_to_impl`, `mark_done`, `mark_failed`) print a single status line or nothing.

No new dependencies — plain `print()` with Unicode box chars.

---

## 2.5 — Human-in-the-Loop Checkpoints

Three checkpoints pause the streaming loop before the next node runs:

1. **After planner** — shows plan summary, before evaluator
2. **After plan approved** (evaluator PASS on planning phase) — shows sprint contract, before generator
3. **After generator** — shows files written, before final evaluator

### Checkpoint function

```python
def checkpoint(label: str, summary: str, auto_approve: bool) -> str | None:
    """
    Returns:
      None      → approved
      str       → feedback to append to idea
      "SKIP_ALL" → set auto_approve=True and continue
    Raises SystemExit(0) on quit.
    """
    if auto_approve:
        return None
    print(f"\n{summary}\n")
    print(f"[{label}] (a)pprove / (f)eedback / (s)kip all / (q)uit: ", end="")
    choice = input().strip().lower()
    if choice == "a":
        return None
    if choice == "f":
        return input("Feedback: ").strip()
    if choice == "s":
        return "SKIP_ALL"
    if choice == "q":
        raise SystemExit(0)
```

### Feedback injection

Feedback is appended to `state["idea"]` in the streaming loop in `main.py` before the next node runs. Nodes are not aware of the checkpoint mechanism.

`SKIP_ALL` causes `main.py` to replace `RunConfig` with `auto_approve=True` (new instance, since `RunConfig` is frozen) before resuming.

### CLI

`--auto-approve` sets `RunConfig.auto_approve = True` from the start, skipping all checkpoints (CI/batch use).

---

## 2.4 — Planner-Driven Parallel Branches

Enabled with `--parallel` flag. Planner decides the paths; no static persona file.

### New models

```python
class PathSpec(BaseModel):
    name: str    # e.g. "event-driven"
    focus: str   # one-sentence exploration angle

class BranchResult(BaseModel):
    name: str
    plan: Plan
    evaluation: EvaluationResult
```

### Two-stage flow

**Stage 1 — Path decider:**  
The planner makes a first API call returning a `PathSpec` list. Prompt asks: *"Given this idea, what are the distinct implementation paths worth exploring? Return 2–4 paths."*

**Stage 2 — Subagents:**  
`asyncio.gather` spawns one `AgentClient.run()` per `PathSpec`, each seeded with the standard planner prompt + `focus` injected. Returns N `Plan` objects. Each plan runs through its own evaluator call (also parallel). Produces N `BranchResult` objects.

**Selector:**  
Single evaluator call in selector mode — given all `(name, plan, evaluation)` pairs, picks the strongest plan. If the planner returns only 1 path, skip the selector entirely.

### Artifact layout

- **Winner:** full artifact set at `artifacts/{run_id}/` root
- **Losers:** `artifacts/{run_id}/branches.json` — array of `{name, plan, evaluation}`

### CLI

`--parallel` opts in. Without it, planner skips Stage 1 and runs a single plan as in Phase 1. `--parallel` without sufficient paths (planner returns 1) degrades gracefully to single-branch.

---

## File changes summary

| File | Change |
|------|--------|
| `orchestrator/state.py` | Add `RunConfig`, `PathSpec`, `BranchResult`; add `run_config` to `GraphState` |
| `orchestrator/agents/base.py` | New — `BaseAgentClient` ABC |
| `orchestrator/agents/factory.py` | New — `make_client()` factory |
| `orchestrator/agents/openrouter.py` | New — `OpenRouterClient` |
| `orchestrator/agents/client.py` | Inherit from `BaseAgentClient` |
| `orchestrator/nodes/*.py` | Use `make_client()` instead of direct `AgentClient` |
| `orchestrator/graph.py` | `build_graph(run_config)` — includes parallel nodes when `run_config.parallel` |
| `main.py` | `RunConfig` construction, `graph.stream()`, streaming loop, checkpoints, run history |
| `artifacts/` | Restructured to `artifacts/{run_id}/` |
