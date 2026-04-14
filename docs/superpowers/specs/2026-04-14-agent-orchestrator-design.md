# Agent Orchestrator — Design Spec

**Date:** 2026-04-14
**Status:** Approved

## Goal

Build a local-first agentic workflow that turns a user idea into a plan, then into implementation, with evaluator-gated revision loops. v1 proves the planner/evaluator loop end-to-end via a CLI script.

---

## Decisions

| Concern | Decision |
|---------|----------|
| Orchestrator | LangGraph (single flat graph) |
| Agent runtime | Anthropic Agents API (`/v1/agents`, `/v1/sessions`) |
| Models | Planner: Sonnet, Evaluator: Sonnet, Advisor: Opus, Generator: Sonnet |
| User interface | CLI script (`python main.py "idea"`); interactive REPL deferred to phase 2 |
| Inter-agent communication | Pydantic schemas in-memory + file artifacts for persistence/readability |
| Generator tools | File-write tools only; other agents are reasoning-only (no tools) |
| Loop termination | Advisor intervenes after 3 evaluator failures; hard stop if still failing after advisor |
| Dependency manager | uv |

---

## Project Structure

```
agent-orchestrator/
├── pyproject.toml
├── uv.lock
├── .python-version
├── main.py                     # CLI entry point
├── orchestrator/
│   ├── __init__.py
│   ├── graph.py                # LangGraph graph definition + compilation
│   ├── state.py                # GraphState TypedDict + all Pydantic schemas
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── planner.py
│   │   ├── evaluator.py        # shared node for planning and implementation phases
│   │   ├── advisor.py
│   │   └── generator.py
│   └── agents/
│       ├── __init__.py
│       └── client.py           # Anthropic Agents API wrapper
├── artifacts/                  # human-readable outputs written per run
│   └── .gitkeep
└── docs/
    └── superpowers/specs/
```

**Key dependencies:** `langgraph`, `anthropic`, `pydantic`, `typer`

---

## Graph State

```python
class GraphState(TypedDict):
    idea: str
    phase: Literal["planning", "implementation", "done", "failed"]
    plan: Plan | None
    sprint_contract: SprintContract | None
    implementation: Implementation | None
    evaluation: EvaluationResult | None
    advisor_memo: AdvisorMemo | None
    revision_count: int          # resets when transitioning planning → implementation
    advisor_used: bool           # tracks whether advisor has already intervened
```

---

## Pydantic Schemas

```python
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
    suggested_approach: str | None

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
```

Each node writes its output to state AND persists a human-readable artifact under `artifacts/` (e.g. `artifacts/plan_v1.md`, `artifacts/eval_planning_v1.json`, `artifacts/advisor_memo.md`).

---

## Graph & Routing Logic

```
[START] → planner → evaluator ──pass──→ generator → evaluator ──pass──→ [END: done]
               ↑        │                    ↑           │
               │      fail                  │          fail
               │    (< 3 revisions)         │        (< 3 revisions)
               │        │                  └──────────┘
               │        ↓
               │     advisor ──→ planner (advisor_used=True, revision_count reset to 0)
               │
               └── fail (≥ 3 AND advisor_used) → [END: failed]
```

**`route_after_evaluator` conditional edge:**

```python
def route_after_evaluator(state: GraphState) -> str:
    if state["evaluation"].verdict == "pass":
        if state["phase"] == "planning":
            return "to_generator"   # planning done → hand sprint contract to generator
        return "done"               # implementation done → END with phase="done"
    if state["revision_count"] >= 3 and state["advisor_used"]:
        return "hard_stop"          # → END with phase="failed"
    if state["revision_count"] >= 3 and not state["advisor_used"]:
        return "advisor"            # → advisor node
    return "revise"                 # → planner (planning) or generator (implementation)
```

The evaluator node reads `state["phase"]` to adjust its system prompt: it evaluates a plan during the planning phase and an implementation during the implementation phase.

---

## Agents API Client

`orchestrator/agents/client.py` — thin wrapper:

```python
class AgentClient:
    def __init__(self, model: str, system_prompt: str, tools: list = [])
    def run(self, messages: list[dict]) -> str
    # Creates an agent + session, invokes it, returns final text response
```

**Node → model → tools mapping:**

| Node | Model | Tools |
|------|-------|-------|
| planner | claude-sonnet-4-6 | none |
| evaluator | claude-sonnet-4-6 | none |
| advisor | claude-opus-4-6 | none |
| generator | claude-sonnet-4-6 | file-write |

---

## CLI Entry Point

```
python main.py "build a REST API for todo items"
```

**`main.py`:**

```python
@app.command()
def run(idea: str, max_revisions: int = 3):
    graph = build_graph()
    initial_state = GraphState(idea=idea, phase="planning", revision_count=0,
                               advisor_used=False, ...)
    final_state = graph.invoke(initial_state)

    if final_state["phase"] == "done":
        typer.echo("Done. See artifacts/ for outputs.")
    elif final_state["phase"] == "failed":
        typer.echo("Could not converge. Last evaluation:")
        typer.echo(final_state["evaluation"].model_dump_json(indent=2))
```

---

## Error Handling

- **Agent JSON parse failure:** Node catches `ValidationError`, retries once with an explicit "respond only with valid JSON" re-prompt. Raises on second failure — not silently swallowed.
- **Anthropic API errors:** Bubble up to CLI. No retry logic in v1.
- **Hard stop:** `phase="failed"` written to state; final evaluation and advisor memo printed to terminal; all artifacts preserved in `artifacts/` for inspection.
- No logging framework in v1 — `typer.echo` is sufficient.

---

## Out of Scope for v1

- Interactive REPL / streaming output (phase 2)
- Parallel branches
- Codex plugin integration
- Multi-provider model support
- Memory / persistent context across runs
