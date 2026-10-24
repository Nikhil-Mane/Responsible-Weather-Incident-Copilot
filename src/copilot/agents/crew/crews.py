"""Assembling the roles into sequential crews."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from copilot.agents.crew.comms_drafter import build_alert_task, build_comms_drafter
from copilot.agents.crew.hazard_analyst import build_analysis_task, build_hazard_analyst
from copilot.agents.crew.response_planner import build_planning_task, build_response_planner
from copilot.agents.crew.runtime import load_crewai
from copilot.agents.prompts import render_facts, render_playbooks

if TYPE_CHECKING:
    from collections.abc import Sequence

    from copilot.domain.models import ActionItem
    from copilot.drafting.base import DraftContext


def _crew(crewai: Any, agents: list[Any], tasks: list[Any]) -> Any:
    return crewai.Crew(
        agents=agents,
        tasks=tasks,
        process=crewai.Process.sequential,
        verbose=False,
        memory=False,
        cache=False,
        tracing=False,  # never collect or share execution traces (they contain prompts)
    )


def build_planning_crew(llm: Any, context: DraftContext) -> Any:
    """Hazard Analyst then Response Planner, producing a validated `ActionPlan`."""
    crewai = load_crewai()
    analyst = build_hazard_analyst(crewai, llm)
    planner = build_response_planner(crewai, llm)
    analysis = build_analysis_task(crewai, analyst, render_facts(context))
    plan = build_planning_task(crewai, planner, render_playbooks(context), analysis)
    return _crew(crewai, [analyst, planner], [analysis, plan])


def build_comms_crew(llm: Any, context: DraftContext, actions: Sequence[ActionItem]) -> Any:
    """Comms Drafter producing a validated `AlertDraft`."""
    crewai = load_crewai()
    drafter = build_comms_drafter(crewai, llm)
    task = build_alert_task(crewai, drafter, context, actions)
    return _crew(crewai, [drafter], [task])
