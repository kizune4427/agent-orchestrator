# CLAUDE.md

## Goal

Build a local-first agentic workflow that turns a user idea into a plan, then into implementation, with evaluator-gated revision loops.

## Current decision

Use:

- **LangGraph** as the top-level orchestrator
- **Claude Code / Claude Agent SDK** as the main runtime
- **Claude-family models** for v1 roles
- **Codex plugin in Claude Code** as optional delegated help later

Do **not** use OpenRouter for v1.

## Target workflow

1. User gives an idea.
2. **Planner** drafts a plan/design.
3. **Evaluator** reviews it.
4. If needed, **Advisor** helps with complex or blocked tasks.
5. Planner revises until the plan passes.
6. **Generator** implements the approved sprint contract.
7. Evaluator reviews implementation.
8. Generator revises until success criteria are met.
9. Parallel branches are a later extension, not required for v1.

## v1 model roles

- **Planner**: Sonnet
- **Advisor**: Opus
- **Evaluator**: Sonnet by default
- **Generator**: Sonnet with Codex Claude Code plugins

## Initial subagents

- **planner**: drafts compact plan/design artifacts
- **advisor**: stronger reasoning for ambiguity and deadlocks
- **evaluator**: returns pass/fail, blockers, next actions
- **generator**: implements only the approved sprint contract

## Next steps

1. Bootstrap repo structure and Python environment.
2. Create Claude Code subagents.
3. Define structured schemas.
4. Build minimal LangGraph flow.
5. Prove planner/evaluator loop first.
6. Hand off approved sprint contract to Claude Code manually.
7. Automate generator invocation later.
8. Add Codex plugin usage if needed.
9. Add parallel branches later.

## Avoid early

- overbuilding memory
- adding parallelism before the core loop works
- adding many model providers at once
- relying on giant prompts instead of artifacts
- vague evaluator output
- large multi-feature generator tasks

