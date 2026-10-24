"""Workflow state and the value types returned to callers."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from copilot.domain.models import IncidentBriefing
from copilot.domain.rbac import Role


class State(TypedDict, total=False):
    """Workflow state. Plain JSON-compatible values only."""

    run_id: str
    role: str
    snapshot: dict[str, Any]
    quality_issues: list[str]
    hazards: list[str]
    flood_risk: bool
    severity: str
    review_reasons: list[str]
    warnings: list[str]
    retrieved: list[dict[str, Any]]
    action_items: list[dict[str, Any]]
    alert: str | None
    route: str
    approval_status: str
    trace: Annotated[list[dict[str, Any]], operator.add]


@dataclass(frozen=True, slots=True)
class Decision:
    """A human decision on a case awaiting review."""

    approved: bool
    actor: str
    role: Role
    note: str = ""


@dataclass(frozen=True, slots=True)
class RunResult:
    """A briefing plus whether the workflow is paused for a human decision."""

    briefing: IncidentBriefing
    awaiting_decision: bool
