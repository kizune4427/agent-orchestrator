# CLAUDE.md

## Goal

Build a local-first agentic workflow that turns a user idea into a plan, then into
implementation, with evaluator-gated revision loops.

---

## Phase 1 — Complete ✅

**What was built:**

- LangGraph flat `StateGraph` with five nodes: planner, evaluator, advisor, generator,
  routing transitions
- `AgentClient` — thin wrapper around the Anthropic Managed Agents API
  (`/v1/agents`, `/v1/sessions`, `/v1/environments`)
- Pydantic schemas: `Plan`, `EvaluationResult`, `AdvisorMemo`, `SprintContract`,
  `Implementation`, `GraphState`
- Four agent roles (all Claude-family):
  - **Planner** — `claude-sonnet-4-6`, drafts structured plans as JSON
  - **Evaluator** — `claude-sonnet-4-6`, pass/fail with blockers and next actions
  - **Advisor** — `claude-opus-4-6`, root-cause analysis after 3 consecutive failures
  - **Generator** — `claude-sonnet-4-6`, returns file contents inline; written locally
    under `generated/`
- Evaluator-gated revision loop: fail → planner retries; fail ×3 → advisor; fail after
  advisor → hard stop
- Artifact persistence: plans → `artifacts/plan_v{n}.md`, eval results →
  `artifacts/eval_{phase}_v{n}.json`, advisor memo → `artifacts/advisor_memo.md`,
  implementation summary → `artifacts/implementation.md`
- Typer CLI: `uv run python main.py "idea"` with `--max-revisions` flag
- `ORCHESTRATOR_DEBUG=1` env var for stream-event tracing

**Tech stack:** Python 3.11, uv, langgraph, anthropic ≥0.94, pydantic v2, typer,
python-dotenv

---

## Phase 2 — Planned

### 2.1 Model switching via OpenRouter

Allow each agent role to use any model reachable through OpenRouter, in addition to
the native Anthropic API.

- Add `LLM_BACKEND` env var: `anthropic` (default) or `openrouter`
- When `openrouter`, use the OpenRouter-compatible endpoint with the `openai` Python
  client (or `httpx` directly); model strings follow OpenRouter convention
  (e.g. `openai/gpt-4o`, `meta-llama/llama-3-70b-instruct`)
- `AgentClient` gains a `backend` param; internal dispatch picks Anthropic Managed
  Agents or OpenRouter accordingly
- Expose `--planner-model`, `--evaluator-model`, `--advisor-model`,
  `--generator-model` as CLI flags in `main.py` for per-run overrides
- Do **not** use OpenRouter for v1 (Phase 1 constraint — now lifted for Phase 2)

### 2.2 Streaming output

Surface live progress to the terminal as each node completes, rather than blocking
until the full workflow finishes.

- Switch `graph.invoke()` → `graph.stream()` in `main.py`; print state deltas after
  each node
- Print node name + key result on each step:
  - After planner: plan summary + step count
  - After evaluator: verdict + blockers (if any)
  - After advisor: analysis excerpt
  - After generator: files written list
- For `AgentClient`, forward token-level stream events to a callback so callers can
  print partial text (optional, Phase 2 stretch goal)

### 2.3 Persistent run history

Keep a durable record of every run so plans and evaluation history survive across
sessions.

- Each run gets a unique run ID (timestamp + short hash)
- Artifacts written to `artifacts/{run_id}/` instead of flat `artifacts/`
- Run index appended to `artifacts/runs.jsonl` — one line per run with: run ID,
  idea, final phase, timestamp, files generated
- CLI flag `--run-id` to resume or inspect a prior run
- `main.py` prints the run ID at start so the user can reference it

### 2.4 Parallel branches

Run N independent planner branches in parallel, each exploring a different direction,
then select the best result.

**Divergence strategy — persona seeding:**

Each branch is seeded with a different perspective injected into the planner's system
prompt. Example personas for N=3:

| Branch | Persona seed |
|--------|-------------|
| A | "Prioritise simplicity: prefer the fewest moving parts and minimal dependencies." |
| B | "Prioritise scalability: design for growth, extensibility, and clean abstractions." |
| C | "Prioritise robustness: emphasise error handling, observability, and defensive design." |

Personas are defined in config (e.g. `personas.yaml`) so users can customise them.

**Selection — evaluator as selector:**

- All branches run their full planner → evaluator loop independently (in parallel
  via `asyncio`)
- After all branches settle, the **Evaluator** is called once more in "selector mode"
  with all N `(plan, evaluation)` pairs; it returns the index of the strongest
  candidate based on verdict, fewest blockers, and fit to the original idea
- If only 1 branch produced a result, skip the selector call and use it directly
- If ≥1 branch passed evaluation, the selector picks the strongest passing plan
- If no branch passed, the selector picks the least-failing plan and feeds it back
  to the planner as advisor-style guidance for the next revision
- The winning plan's sprint contract is handed to the generator
- No new agent role is introduced — the evaluator handles selection via a different
  system prompt in selector mode

**Flags:**
- `--branches N` — number of parallel branches (default: 2; minimum: 1)
- `--parallel` — opt-in flag to enable parallel execution (sequential is default,
  so single-branch runs are unaffected)

### 2.5 Human-in-the-loop approvals

Pause at key checkpoints for user review, with a runtime "skip all" escape hatch.

**Checkpoints (default: all enabled):**

1. After planner produces a plan — show plan, ask approve / reject+feedback / skip all
2. After evaluator passes the plan — show sprint contract, ask approve / regenerate /
   skip all
3. After generator produces files — show file list + summary, ask approve / retry /
   skip all

**Behaviour:**

- At each checkpoint, print the artifact and prompt:
  ```
  [approve]  continue
  [feedback] send note back to the agent
  [skip]     skip all remaining approvals for this run
  [abort]    stop the run
  ```
- Selecting **skip** at any checkpoint disables all subsequent checkpoints for the
  current run (equivalent to `--auto-approve` mode)
- `--auto-approve` CLI flag skips all checkpoints from the start (CI/batch use)
- Feedback entered at a checkpoint is injected into the next node's message as
  additional human context

---

## Architecture decisions (carried forward)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Orchestrator | LangGraph `StateGraph` | Explicit state machine; easy to extend |
| Agent runtime | Anthropic Managed Agents API | Native session/streaming support |
| Schemas | Pydantic v2 | Validated in-memory; serialised as artifacts |
| Model hierarchy | Sonnet for routine roles; Opus for advisor/arbiter | Cost/quality balance |
| Generator output | Inline file contents as JSON | Local-first; no cloud filesystem dependency |
| Dependency manager | uv | Fast, reproducible |

## Avoid

- Overbuilding memory before the core loop is stable
- Adding many model providers simultaneously — OpenRouter first, then expand
- Giant prompts — use structured artifacts passed between nodes
- Vague evaluator output — blockers and next_actions must be actionable
- Large multi-feature generator tasks — keep sprint contracts small and focused
- Skipping the arbiter when branches disagree — never silently pick by index
