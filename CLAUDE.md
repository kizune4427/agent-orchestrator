# CLAUDE.md

## Goal

Local-first agentic workflow: idea → plan → implementation, with evaluator-gated revision loops.

## Stack

Python 3.11, uv, LangGraph, Anthropic Managed Agents API, Pydantic v2, Typer

## Phase 1 — Complete ✅

- LangGraph `StateGraph`: planner → evaluator → (advisor on 3× fail) → generator
- `AgentClient` wraps `/v1/agents`, `/v1/sessions`, `/v1/environments`
- Roles: Planner (Sonnet), Evaluator (Sonnet), Advisor (Opus), Generator (Sonnet)
- Generator returns file contents as inline JSON; written locally to `generated/`
- Artifacts: `artifacts/plan_v{n}.md`, `eval_{phase}_v{n}.json`, `advisor_memo.md`, `implementation.md`
- CLI: `uv run python main.py "idea"` — `ORCHESTRATOR_DEBUG=1` for event tracing

## Phase 2 — Planned

**2.1 Model switching**
- `LLM_BACKEND=anthropic` (default) or `openrouter`
- CLI flags: `--planner-model`, `--evaluator-model`, `--advisor-model`, `--generator-model`
- `AgentClient` dispatches to Anthropic Managed Agents or OpenRouter based on backend

**2.2 Streaming output**
- `graph.stream()` instead of `graph.invoke()`; print node results as they land
- Show plan summary, verdict+blockers, advisor excerpt, files written — progressively

**2.3 Persistent run history**
- Each run gets a unique ID (timestamp + short hash); artifacts go to `artifacts/{run_id}/`
- `artifacts/runs.jsonl` — append-only index: run ID, idea, final phase, timestamp
- `main.py` prints the run ID on start; `--run-id` to reference a prior run

**2.4 Parallel branches**
- N planners run in parallel (`asyncio`), each seeded with a different persona (simplicity / scalability / robustness) to ensure divergent directions; personas defined in `personas.yaml`
- After all branches settle, the **Evaluator** runs in selector mode: given all N `(plan, eval)` pairs, it picks the strongest; if only 1 branch ran, skip the selector
- `--branches N` (default 2); `--parallel` to opt in (sequential is default)

**2.5 Human-in-the-loop**
- Checkpoints after: plan drafted, plan approved (show sprint contract), files generated
- Each checkpoint: `[approve]` / `[feedback → injected into next node]` / `[skip all]` / `[abort]`
- `--auto-approve` skips all checkpoints from the start (CI/batch use)

## Avoid

- Giant prompts — pass structured artifacts between nodes
- Vague evaluator output — blockers and next_actions must be actionable
- Silently picking a branch by index — always use the evaluator selector
- Large multi-feature generator tasks — keep sprint contracts small
