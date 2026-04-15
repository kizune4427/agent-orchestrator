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
    from orchestrator.streaming import print_node_summary
    from orchestrator.checkpoint import checkpoint

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

    # Track auto_approve flag across checkpoints (set from run_config by Task 9)
    _auto_approve = False

    final_state: dict = {}

    try:
        for chunk in graph.stream(initial_state):
            for node_name, state_delta in chunk.items():
                print_node_summary(node_name, state_delta)
                final_state.update(state_delta)

                # Checkpoint 1: after planner drafts a plan
                if node_name == "planner" and "plan" in state_delta:
                    plan = state_delta["plan"]
                    summary = f"Plan: {plan.summary}\nSteps: {len(plan.steps)}"
                    feedback = checkpoint("plan-review", summary, _auto_approve)
                    if feedback == "SKIP_ALL":
                        _auto_approve = True
                    elif feedback:
                        final_state["idea"] = final_state.get("idea", "") + f"\n\nHuman feedback: {feedback}"

                # Checkpoint 2: after evaluator approves plan (sprint contract generated)
                elif node_name == "evaluator" and "sprint_contract" in state_delta:
                    contract = state_delta["sprint_contract"]
                    summary = f"Sprint goal: {contract.goal}\nTasks: {len(contract.tasks)}"
                    feedback = checkpoint("sprint-review", summary, _auto_approve)
                    if feedback == "SKIP_ALL":
                        _auto_approve = True
                    elif feedback:
                        final_state["idea"] = final_state.get("idea", "") + f"\n\nHuman feedback: {feedback}"

                # Checkpoint 3: after generator writes files
                elif node_name == "generator" and "implementation" in state_delta:
                    impl = state_delta["implementation"]
                    summary = f"Files written: {len(impl.files_written)}\n" + "\n".join(impl.files_written)
                    feedback = checkpoint("impl-review", summary, _auto_approve)
                    if feedback == "SKIP_ALL":
                        _auto_approve = True
                    elif feedback:
                        final_state["idea"] = final_state.get("idea", "") + f"\n\nHuman feedback: {feedback}"

    except SystemExit:
        typer.echo("\nAborted by user.", err=True)
        raise typer.Exit(code=0)
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
