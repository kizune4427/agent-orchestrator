from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from orchestrator.config import RunConfig
from orchestrator.nodes.advisor import advisor_node
from orchestrator.nodes.evaluator import evaluator_node
from orchestrator.nodes.generator import generator_node
from orchestrator.nodes.planner import planner_node
from orchestrator.state import GraphState


def route_after_evaluator(
    state: GraphState,
) -> Literal["to_generator", "done", "revise", "advisor", "hard_stop"]:
    """Determine next step after evaluator runs."""
    eval_ = state["evaluation"]
    phase = state["phase"]
    revision_count = state.get("revision_count", 0)
    advisor_used = state.get("advisor_used", False)

    if eval_.verdict == "pass":
        if phase == "planning":
            return "to_generator"  # approved plan → hand to generator
        return "done"              # approved implementation → finished

    # Verdict is "fail"
    if revision_count >= 3 and advisor_used:
        return "hard_stop"         # advisor already tried, still failing
    if revision_count >= 3 and not advisor_used:
        return "advisor"           # escalate to advisor
    return "revise"                # still within retry budget


def _mark_done(state: GraphState) -> dict:
    return {"phase": "done"}


def _mark_failed(state: GraphState) -> dict:
    return {"phase": "failed"}


def _set_implementation_phase(state: GraphState) -> dict:
    return {"phase": "implementation", "revision_count": 0}


def build_graph(run_config: RunConfig | None = None):
    """Build and compile the agent orchestrator LangGraph.

    run_config is accepted for future use (parallel branch routing).
    Currently the graph topology is the same regardless of run_config.
    """
    builder = StateGraph(GraphState)

    # Register nodes
    builder.add_node("planner", planner_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("advisor", advisor_node)
    builder.add_node("generator", generator_node)
    builder.add_node("transition_to_impl", _set_implementation_phase)
    builder.add_node("mark_done", _mark_done)
    builder.add_node("mark_failed", _mark_failed)

    # Entry point
    builder.set_entry_point("planner")

    # Planner → Evaluator
    builder.add_edge("planner", "evaluator")

    # Evaluator → conditional routing
    builder.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {
            "to_generator": "transition_to_impl",
            "done": "mark_done",
            "revise": "planner",
            "advisor": "advisor",
            "hard_stop": "mark_failed",
        },
    )

    # Transitions
    builder.add_edge("transition_to_impl", "generator")
    builder.add_edge("advisor", "planner")
    builder.add_edge("generator", "evaluator")
    builder.add_edge("mark_done", END)
    builder.add_edge("mark_failed", END)

    return builder.compile()
