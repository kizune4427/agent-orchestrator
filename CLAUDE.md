# CLAUDE.md

## Goal

Local-first agentic workflow: idea → plan → implementation, with evaluator-gated revision loops.

## Stack

Python 3.11, uv, LangGraph, Anthropic Managed Agents API, Pydantic v2, Typer

## Current state

Phases 1 and 2 complete. Run: `uv run python main.py "idea"` — `ORCHESTRATOR_DEBUG=1` for tracing.

**Graph:** planner → evaluator → (advisor on 3× fail) → generator. With `--parallel`: planner spawns N branches (personas from `personas.yaml`), selector picks the strongest before evaluation.

**Key files:**
- `orchestrator/config.py` — `RunConfig` (frozen Pydantic; all CLI flags live here)
- `orchestrator/agents/factory.py` — `make_client(role, config, prompt)` dispatches to `AgentClient` or `OpenRouterClient`
- `orchestrator/history.py` — run ID generation, `artifact_dir`, `runs.jsonl` append
- `orchestrator/nodes/` — planner, evaluator (+ selector), advisor, generator
- `orchestrator/graph.py` — `build_graph(run_config)` wires nodes; entry point shifts on `from_node`
- `orchestrator/streaming.py` — `print_node_summary()` for progressive output
- `orchestrator/checkpoint.py` — HITL `checkpoint()` with approve/feedback/skip/abort
- `personas.yaml` — parallel branch personas (simplicity, scalability, robustness, speed)
- `artifacts/{run_id}/` — per-run artifacts; `artifacts/runs.jsonl` — append-only run index

## Constraints

- Pass structured artifacts between nodes — no giant prompts
- Evaluator blockers and next_actions must be actionable
- Selector always uses LLM judgment — never pick branch by index silently
- Keep sprint contracts small — one focused feature per generator task
