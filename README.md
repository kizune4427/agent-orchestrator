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

Optional flag:

```bash
uv run python main.py "Your idea here" --max-revisions 5
```

### 4. Check outputs

Artifacts are written to `artifacts/` after each stage:

| File | Contents |
|------|----------|
| `artifacts/plan_v{n}.md` | Plan produced by each planner revision |
| `artifacts/eval_planning_v{n}.json` | Evaluator verdict on each plan |
| `artifacts/advisor_memo.md` | Advisor analysis (if invoked) |
| `artifacts/implementation.md` | Generator summary |
| `artifacts/eval_implementation_v{n}.json` | Evaluator verdict on the implementation |

## How It Works

```
idea
 └─► planner ──► evaluator ──┬─ pass ──► generator ──► evaluator ──┬─ pass ──► done
                              │                                      └─ fail ──► generator (retry)
                              ├─ fail (< 3 tries) ──────────────────────────────► planner (retry)
                              ├─ fail (3 tries, no advisor yet) ──► advisor ────► planner (retry)
                              └─ fail (advisor already used) ──────────────────► failed
```

| Agent | Model | Role |
|-------|-------|------|
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
