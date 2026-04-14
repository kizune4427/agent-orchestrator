from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel
from typing_extensions import TypedDict


class Plan(BaseModel):
    summary: str
    steps: list[str]
    open_questions: list[str]


class EvaluationResult(BaseModel):
    verdict: Literal["pass", "fail"]
    blockers: list[str]
    next_actions: list[str]


class AdvisorMemo(BaseModel):
    analysis: str
    recommendations: list[str]
    suggested_approach: Optional[str] = None


class Task(BaseModel):
    id: str
    description: str
    acceptance_criteria: list[str]


class SprintContract(BaseModel):
    goal: str
    tasks: list[Task]
    constraints: list[str]


class Implementation(BaseModel):
    files_written: list[str]
    summary: str


class GraphState(TypedDict, total=False):
    idea: str
    phase: Literal["planning", "implementation", "done", "failed"]
    plan: Optional[Plan]
    sprint_contract: Optional[SprintContract]
    implementation: Optional[Implementation]
    evaluation: Optional[EvaluationResult]
    advisor_memo: Optional[AdvisorMemo]
    revision_count: int
    advisor_used: bool
