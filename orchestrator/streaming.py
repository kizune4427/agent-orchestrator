from __future__ import annotations

from orchestrator.state import AdvisorMemo, EvaluationResult, Implementation, Plan

_HEADER = "━" * 4
_DIVIDER = "─" * 36
_INTERNAL_NODES = {"transition_to_impl", "mark_done", "mark_failed"}


def print_node_summary(node_name: str, state_delta: dict) -> None:
    """Print a formatted summary block when a node completes."""
    if node_name in _INTERNAL_NODES:
        return

    print(f"\n{_HEADER} {node_name} {_HEADER}")

    if "plan" in state_delta and isinstance(state_delta["plan"], Plan):
        plan: Plan = state_delta["plan"]
        print(f"Goal: {plan.summary}")
        print("Steps:")
        for step in plan.steps:
            print(f"  • {step}")
        if plan.open_questions:
            print("Open questions:")
            for q in plan.open_questions:
                print(f"  ? {q}")
        else:
            print("Open questions: none")

    if "evaluation" in state_delta and isinstance(state_delta["evaluation"], EvaluationResult):
        result: EvaluationResult = state_delta["evaluation"]
        verdict_label = "PASS" if result.verdict == "pass" else "FAIL"
        print(f"Verdict: {verdict_label}")
        if result.blockers:
            print("Blockers:")
            for b in result.blockers:
                print(f"  ✗ {b}")
        if result.next_actions:
            print("Next actions:")
            for a in result.next_actions:
                print(f"  → {a}")

    if "implementation" in state_delta and isinstance(state_delta["implementation"], Implementation):
        impl: Implementation = state_delta["implementation"]
        print("Files written:")
        for f in impl.files_written:
            print(f"  • {f}")
        print(f"Summary: {impl.summary}")

    if "advisor_memo" in state_delta and isinstance(state_delta["advisor_memo"], AdvisorMemo):
        memo: AdvisorMemo = state_delta["advisor_memo"]
        print(f"Analysis: {memo.analysis}")
        if memo.recommendations:
            print("Recommendations:")
            for r in memo.recommendations:
                print(f"  • {r}")

    print(_DIVIDER)
