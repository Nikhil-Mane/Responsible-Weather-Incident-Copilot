"""Comms Drafter: drafts the local-team alert (always a DRAFT)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from copilot.agents.crew.runtime import make_agent
from copilot.agents.prompts import SYSTEM_ALERT, render_facts, sanitize
from copilot.agents.schemas import AlertDraft

if TYPE_CHECKING:
    from collections.abc import Sequence

    from copilot.domain.models import ActionItem
    from copilot.drafting.base import DraftContext


def build_comms_drafter(crewai: Any, llm: Any) -> Any:
    """The Comms Drafter agent."""
    return make_agent(
        crewai,
        llm,
        "Comms Drafter",
        "Draft an internal local-team alert that is clearly marked as a draft.",
        "You write precise internal notices and never claim anything has been sent.",
    )


def build_alert_task(
    crewai: Any, agent: Any, context: DraftContext, actions: Sequence[ActionItem]
) -> Any:
    """The drafter's task: the alert text, validated as an `AlertDraft`."""
    listed = "\n".join(f"- {sanitize(a.text, 200)}" for a in actions) or "- none"
    return crewai.Task(
        description=f"{SYSTEM_ALERT}\n\nVerified facts:\n{render_facts(context)}\n\n"
        f"Readiness actions to include:\n{listed}\n\nWrite the draft alert.",
        expected_output="A JSON object with the alert text.",
        agent=agent,
        output_pydantic=AlertDraft,
    )
