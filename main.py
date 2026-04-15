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

    try:
        final_state = graph.invoke(initial_state)
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
