from __future__ import annotations


_MENU = "(a)pprove / (f)eedback / (s)kip all / (q)uit"


def checkpoint(label: str, summary: str, auto_approve: bool) -> str | None:
    """Pause and prompt the user at a workflow checkpoint.

    Returns:
        None      — approved, continue
        str       — feedback text to inject into next node
        "SKIP_ALL" — set auto_approve=True for remaining checkpoints
    Raises:
        SystemExit(0) — user chose to abort
    """
    if auto_approve:
        return None

    print(f"\n{summary}\n")
    print(f"[{label}] {_MENU}: ", end="", flush=True)
    choice = input().strip().lower()

    if choice == "a":
        return None
    if choice == "f":
        print("Feedback: ", end="", flush=True)
        return input().strip()
    if choice == "s":
        return "SKIP_ALL"
    if choice == "q":
        raise SystemExit(0)
    # Unrecognized input — treat as approve
    return None
