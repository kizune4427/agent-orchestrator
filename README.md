# agent-orchestrator
A multi-agent orchestration framework built with LangGraph and the Claude Agent SDK.

Turns a user idea into an implemented sprint via an evaluator-gated revision loop:
**Planner → Evaluator → (Advisor if stuck) → Generator → Evaluator → done.**

## Quick Start

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) — `pip install uv`
- An [Anthropic API key](https://console.anthropic.com/)

### 1. Clone and install

```bash
git clone <repo-url>
cd agent-orchestrator
uv sync --extra dev
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=your_key_here
```

### 3. Run

```bash
uv run python main.py "Build a simple Python CLI that counts words in a file"
```

## CLI Reference

```
uv run python main.py "Your idea here" [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--max-revisions N` | `3` | Max evaluator revisions before advisor steps in |
| `--backend VALUE` | `anthropic` | LLM backend: `anthropic` or `openrouter` |
| `--planner-model MODEL` | `claude-sonnet-4-6` | Model for the planner role |
| `--evaluator-model MODEL` | `claude-sonnet-4-6` | Model for the evaluator role |
| `--advisor-model MODEL` | `claude-opus-4-6` | Model for the advisor role |
| `--generator-model MODEL` | `claude-sonnet-4-6` | Model for the generator role |
| `--auto-approve` | off | Skip all human-in-the-loop checkpoints |
| `--parallel` | off | Run N planner branches in parallel, then select the best |
| `--branches N` | `2` | Number of parallel branches (requires `--parallel`) |
| `--run-id ID` | auto-generated | Reuse a prior run ID (see run history) |
| `--from-node NODE` | `planner` | Entry node when resuming: `planner`, `evaluator`, or `generator` |

### Examples

```bash
# Basic run
uv run python main.py "Build a REST API for todo items"

# More revision budget
uv run python main.py "Build a REST API" --max-revisions 5

# Skip all checkpoints (CI/batch use)
uv run python main.py "Build a REST API" --auto-approve

# Use OpenRouter with a specific model
uv run python main.py "Build a REST API" \
  --backend openrouter \
  --planner-model openai/gpt-4o \
  --generator-model anthropic/claude-opus-4-6

# Run 3 parallel branches and let the evaluator pick the best plan
uv run python main.py "Build a REST API" --parallel --branches 3

# Resume from a prior run at the generator
uv run python main.py "Build a REST API" \
  --run-id 20260414-120000-abc123 \
  --from-node generator
```

### Environment variables

Model flags can also be set via env vars (CLI flag takes precedence):

| Variable | Default |
|----------|---------|
| `LLM_BACKEND` | `anthropic` |
| `PLANNER_MODEL` | `claude-sonnet-4-6` |
| `EVALUATOR_MODEL` | `claude-sonnet-4-6` |
| `ADVISOR_MODEL` | `claude-opus-4-6` |
| `GENERATOR_MODEL` | `claude-sonnet-4-6` |
| `OPENROUTER_API_KEY` | *(required when `--backend openrouter`)* |

Set `ORCHESTRATOR_DEBUG=1` for full LangGraph event tracing.

## Outputs

Each run gets a unique ID (`YYYYMMDD-HHMMSS-{rand6}`) printed at startup.
Artifacts land in `artifacts/{run_id}/`:

| File | Contents |
|------|----------|
| `plan_v{n}.md` | Plan produced by each planner revision |
| `plan_selected.md` | Winning plan when `--parallel` is used |
| `eval_planning_v{n}.json` | Evaluator verdict on each plan |
| `advisor_memo.md` | Advisor analysis (if invoked) |
| `implementation.md` | Generator summary |
| `eval_implementation_v{n}.json` | Evaluator verdict on the implementation |
| `branches.json` | Losing parallel branches (when `--parallel`) |

`artifacts/runs.jsonl` is an append-only index of all runs (run ID, idea, final phase, timestamp).

## How It Works

### Sequential (default)

```
idea
 └─► planner ──► evaluator ──┬─ pass ──► generator ──► evaluator ──┬─ pass ──► done
                              │                                      └─ fail ──► generator (retry)
                              ├─ fail (< 3 tries) ──────────────────────────────► planner (retry)
                              ├─ fail (3 tries, no advisor yet) ──► advisor ────► planner (retry)
                              └─ fail (advisor already used) ──────────────────► failed
```

### Parallel (`--parallel`)

```
idea
 └─► planner ──► [branch-1, branch-2, ...] ──► selector ──► evaluator ──► ...
                  (concurrent, each seeded        (evaluator picks
                   with a different persona)       the strongest plan)
```

Personas are defined in `personas.yaml` (simplicity-first, scalability-first, robustness-first, speed-of-delivery). The evaluator selector never picks by index — it always makes an LLM-based judgment.

### Human-in-the-loop checkpoints

When running interactively (without `--auto-approve`), the workflow pauses at three points:

1. **After plan drafted** — approve, give feedback (restarts planner with feedback injected), skip all, or abort
2. **After plan approved** — review the sprint contract before generation begins
3. **After files generated** — review the implementation

| Input | Effect |
|-------|--------|
| `a` | Approve and continue |
| `f` | Provide feedback (plan checkpoint: restarts from planner) |
| `s` | Skip all remaining checkpoints |
| `q` | Abort the run |

| Agent | Default Model | Role |
|-------|---------------|------|
| Planner | claude-sonnet-4-6 | Drafts the plan/design |
| Evaluator | claude-sonnet-4-6 | Pass/fail verdict with blockers |
| Advisor | claude-opus-4-6 | Root-cause analysis on repeated failures |
| Generator | claude-sonnet-4-6 | Implements the approved sprint contract |

## Development

```bash
# Run tests
uv run pytest -v

# Run a single test file
uv run pytest tests/test_graph.py -v

# Run the CLI
uv run python main.py "Your idea here"
```
