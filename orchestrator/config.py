from __future__ import annotations

import os
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    backend: Literal["anthropic", "openrouter"] = "anthropic"
    planner_model: str = "claude-sonnet-4-6"
    evaluator_model: str = "claude-sonnet-4-6"
    advisor_model: str = "claude-opus-4-6"
    generator_model: str = "claude-sonnet-4-6"
    auto_approve: bool = False
    parallel: bool = False
    branches: int = 2
    from_node: Optional[Literal["planner", "evaluator", "generator"]] = None

    @classmethod
    def from_env(
        cls,
        run_id: str,
        *,
        backend: Optional[str] = None,
        planner_model: Optional[str] = None,
        evaluator_model: Optional[str] = None,
        advisor_model: Optional[str] = None,
        generator_model: Optional[str] = None,
        auto_approve: bool = False,
        parallel: bool = False,
        branches: int = 2,
        from_node: Optional[Literal["planner", "evaluator", "generator"]] = None,
    ) -> "RunConfig":
        resolved_backend = backend or os.environ.get("LLM_BACKEND", "anthropic")

        # Default model IDs differ by backend — OpenRouter requires provider-prefixed
        # dot-version IDs (e.g. 'anthropic/claude-sonnet-4.6'), while the Anthropic
        # SDK uses dash-version IDs (e.g. 'claude-sonnet-4-6').
        if resolved_backend == "openrouter":
            _default_sonnet = "anthropic/claude-sonnet-4.6"
            _default_opus = "anthropic/claude-opus-4.6"
        else:
            _default_sonnet = "claude-sonnet-4-6"
            _default_opus = "claude-opus-4-6"

        return cls(
            run_id=run_id,
            backend=resolved_backend,
            planner_model=planner_model or os.environ.get("PLANNER_MODEL", _default_sonnet),
            evaluator_model=evaluator_model or os.environ.get("EVALUATOR_MODEL", _default_sonnet),
            advisor_model=advisor_model or os.environ.get("ADVISOR_MODEL", _default_opus),
            generator_model=generator_model or os.environ.get("GENERATOR_MODEL", _default_sonnet),
            auto_approve=auto_approve,
            parallel=parallel,
            branches=branches,
            from_node=from_node,
        )
