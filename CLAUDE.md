# CLAUDE.md

## Goal

Local-first agentic workflow: idea → plan → implementation, with evaluator-gated revision loops.

## Stack

Python 3.11, uv, LangGraph, Anthropic Managed Agents API, Pydantic v2, Typer

## Run

```bash
uv run python main.py "idea"               # default: Anthropic backend
uv run python main.py "idea" --backend openrouter \
  --planner-model openai/gpt-4o \
  --generator-model anthropic/claude-sonnet-4.6
ORCHESTRATOR_DEBUG=1 uv run python main.py "idea"   # full event tracing
```

OpenRouter model IDs: `provider/name-version` with dots, e.g. `anthropic/claude-sonnet-4.6`. Unprefixed Anthropic SDK names (`claude-sonnet-4-6`) are rejected at startup. Unspecified roles fall back to backend-appropriate defaults automatically.

## Graph

`planner → evaluator → (advisor on 3× fail) → generator → evaluator → done`

With `--parallel`: planner spawns N branches (personas from `personas.yaml`), selector LLM-picks the strongest, then normal eval loop.

## Key files

| File | Purpose |
|------|---------|
| `orchestrator/config.py` | `RunConfig` — frozen Pydantic, all CLI flags; `from_env()` picks backend-correct default models |
| `orchestrator/agents/factory.py` | `make_client(role, config, prompt)` → `AgentClient` or `OpenRouterClient` |
| `orchestrator/agents/openrouter.py` | OpenRouter HTTP client; validates model prefix, surfaces 400 body |
| `orchestrator/nodes/evaluator.py` | Evaluator + selector; robust JSON parse (fence-strip + regex fallback) |
| `orchestrator/nodes/generator.py` | Generator; prints live status before LLM call; same robust JSON parse |
| `orchestrator/graph.py` | `build_graph(run_config)` — wires nodes, shifts entry on `from_node` |
| `orchestrator/streaming.py` | `print_node_summary()` — post-node formatted output |
| `orchestrator/checkpoint.py` | HITL `checkpoint()` — approve / feedback / skip-all / abort |
| `orchestrator/history.py` | Run ID generation, `artifact_dir`, `runs.jsonl` append |
| `personas.yaml` | Parallel branch personas (simplicity, scalability, robustness, speed) |

Artifacts: `artifacts/{run_id}/` per run; `artifacts/runs.jsonl` append-only index.

## Constraints

- Pass structured artifacts between nodes — no giant prompts
- Evaluator blockers and next_actions must be actionable
- Selector always uses LLM judgment — never pick branch by index silently
- Keep sprint contracts small — one focused feature per generator task
