"""
Agent Orchestrator CLI

Usage:
    uv run python main.py "Build a REST API for todo items"
    uv run python main.py "Build a REST API" --max-revisions 5 --auto-approve
    uv run python main.py "idea" --backend openrouter --planner-model openai/gpt-4o
    uv run python main.py "idea" --run-id 20260414-120000-abc123 --from-node evaluator
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(help="Run the agentic planner/evaluator/generator workflow.")


@app.command()
def run(
    idea: str = typer.Argument(..., help="The idea or feature request to implement"),
    max_revisions: int = typer.Option(3, help="Max evaluator revisions before advisor steps in"),
    backend: Optional[str] = typer.Option(None, help="LLM backend: anthropic (default) or openrouter"),
    planner_model: Optional[str] = typer.Option(None, help="Model for planner role"),
    evaluator_model: Optional[str] = typer.Option(None, help="Model for evaluator role"),
    advisor_model: Optional[str] = typer.Option(None, help="Model for advisor role"),
    generator_model: Optional[str] = typer.Option(None, help="Model for generator role"),
    auto_approve: bool = typer.Option(False, "--auto-approve", help="Skip all HITL checkpoints"),
    parallel: bool = typer.Option(False, "--parallel", help="Enable planner-driven parallel branches"),
    branches: int = typer.Option(2, "--branches", help="Number of parallel planner branches (requires --parallel)"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Resume or reference a prior run"),
    from_node: Optional[str] = typer.Option(None, "--from-node", help="Entry node when using --run-id: planner|evaluator|generator"),
) -> None:
    """Turn an idea into an implemented sprint via LangGraph + Claude."""
    from orchestrator.config import RunConfig
    from orchestrator.graph import build_graph
    from orchestrator.history import append_run_index, artifact_dir, generate_run_id
    from orchestrator.state import GraphState
    from orchestrator.streaming import print_node_summary
    from orchestrator.checkpoint import checkpoint

    # Validate --from-node requires --run-id
    if from_node and not run_id:
        typer.echo("Error: --from-node requires --run-id", err=True)
        raise typer.Exit(code=1)

    if from_node and from_node not in ("planner", "evaluator", "generator"):
        typer.echo(f"Error: --from-node must be one of: planner, evaluator, generator", err=True)
        raise typer.Exit(code=1)

    # Generate or reuse run ID
    active_run_id = run_id or generate_run_id()
    typer.echo(f"\nRun ID: {active_run_id}")
    typer.echo(f"Starting orchestrator for: {idea!r}\n")

    run_config = RunConfig.from_env(
        run_id=active_run_id,
        backend=backend,
        planner_model=planner_model,
        evaluator_model=evaluator_model,
        advisor_model=advisor_model,
        generator_model=generator_model,
        auto_approve=auto_approve,
        parallel=parallel,
        branches=branches,
        from_node=from_node,
    )

    graph = build_graph(run_config)

    initial_phase = "planning"
    if from_node == "generator":
        initial_phase = "implementation"

    initial_state: GraphState = {
        "idea": idea,
        "phase": initial_phase,
        "revision_count": 0,
        "advisor_used": False,
        "plan": None,
        "sprint_contract": None,
        "implementation": None,
        "evaluation": None,
        "advisor_memo": None,
        "run_config": run_config,
    }

    _auto_approve = run_config.auto_approve

    while True:
        final_state: dict = {}
        restart_requested = False
        last_node_reached: Optional[str] = None

        try:
            for chunk in graph.stream(initial_state):
                for node_name, state_delta in chunk.items():
                    last_node_reached = node_name
                    print_node_summary(node_name, state_delta)
                    final_state.update(state_delta)

                    if node_name == "planner" and "plan" in state_delta:
                        plan = state_delta["plan"]
                        summary = f"Plan: {plan.summary}\nSteps: {len(plan.steps)}"
                        feedback = checkpoint("plan-review", summary, _auto_approve)
                        if feedback == "SKIP_ALL":
                            _auto_approve = True
                        elif feedback:
                            # Inject feedback into idea and restart from planner
                            initial_state = {
                                **initial_state,
                                "idea": initial_state["idea"] + f"\n\nHuman feedback: {feedback}",
                                "plan": None,
                                "sprint_contract": None,
                                "implementation": None,
                                "evaluation": None,
                                "advisor_memo": None,
                                "revision_count": 0,
                                "advisor_used": False,
                            }
                            restart_requested = True
                            break

                    elif node_name == "evaluator" and "sprint_contract" in state_delta:
                        contract = state_delta["sprint_contract"]
                        summary = f"Sprint goal: {contract.goal}\nTasks: {len(contract.tasks)}"
                        feedback = checkpoint("sprint-review", summary, _auto_approve)
                        if feedback == "SKIP_ALL":
                            _auto_approve = True
                        elif feedback:
                            typer.echo(
                                "[checkpoint] Sprint feedback noted. "
                                "Will apply if workflow restarts."
                            )

                    elif node_name == "generator" and "implementation" in state_delta:
                        impl = state_delta["implementation"]
                        summary = "Files written:\n" + "\n".join(
                            f"  • {f}" for f in impl.files_written
                        )
                        feedback = checkpoint("impl-review", summary, _auto_approve)
                        if feedback == "SKIP_ALL":
                            _auto_approve = True
                        elif feedback:
                            typer.echo("[checkpoint] Implementation feedback noted.")

                if restart_requested:
                    break

        except SystemExit:
            typer.echo("\nAborted by user.", err=True)
            append_run_index(run_id=active_run_id, idea=idea, phase="aborted")
            raise typer.Exit(code=0)
        except Exception as exc:
            err_type = type(exc).__name__
            err_msg = str(exc)
            typer.echo(f"\nOrchestrator failed with {err_type}: {err_msg}", err=True)

            # Dump full traceback to artifacts/{run_id}/error.log
            adir = artifact_dir(active_run_id)
            adir.mkdir(parents=True, exist_ok=True)
            tb = traceback.format_exc()
            log_body = (
                f"Run ID: {active_run_id}\n"
                f"Idea: {idea}\n"
                f"Last node reached: {last_node_reached}\n"
                f"Phase at crash: {final_state.get('phase') or initial_state.get('phase')}\n"
                f"Revision count: {final_state.get('revision_count', 0)}\n"
                f"Advisor used: {final_state.get('advisor_used', False)}\n"
                f"\n--- Traceback ---\n{tb}"
            )
            (adir / "error.log").write_text(log_body)
            typer.echo(f"Error details written to {adir / 'error.log'}", err=True)

            append_run_index(
                run_id=active_run_id,
                idea=idea,
                phase="error",
                error_type=err_type,
                error_message=err_msg[:500],
                last_phase_reached=last_node_reached,
            )
            raise typer.Exit(code=1)

        if restart_requested:
            typer.echo("\nRestarting from planner with your feedback...\n")
            continue  # Restart the while loop

        break  # Stream completed normally

    phase = final_state.get("phase")
    append_run_index(run_id=active_run_id, idea=idea, phase=phase or "unknown")

    if phase == "done":
        impl = final_state.get("implementation")
        typer.echo("\nWorkflow complete.")
        if impl:
            typer.echo(f"Files written: {', '.join(impl.files_written)}")
            typer.echo(f"Summary: {impl.summary}")
        typer.echo(f"\nArtifacts: artifacts/{active_run_id}/")

    elif phase == "failed":
        typer.echo("\nWorkflow could not converge.", err=True)
        eval_ = final_state.get("evaluation")
        if eval_:
            typer.echo("\nLast evaluation result:", err=True)
            typer.echo(eval_.model_dump_json(indent=2), err=True)
        typer.echo(f"\nAll artifacts preserved in artifacts/{active_run_id}/", err=True)
        raise typer.Exit(code=1)

    else:
        typer.echo(f"\nUnexpected final phase: {phase!r}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
