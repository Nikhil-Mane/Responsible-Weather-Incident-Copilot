"""Response Planner: proposes playbook-grounded readiness actions."""

from __future__ import annotations

from typing import Any

from copilot.agents.crew.runtime import make_agent
from copilot.agents.prompts import SYSTEM_ACTIONS
from copilot.agents.schemas import ActionPlan


def build_response_planner(crewai: Any, llm: Any) -> Any:
    """The Response Planner agent."""
    return make_agent(
        crewai,
        llm,
        "Response Planner",
        "Propose short readiness actions that are supported by the approved playbooks.",
        "You turn approved procedures into preparation steps. You never invent procedures.",
    )


def build_planning_task(crewai: Any, agent: Any, playbooks: str, analysis_task: Any) -> Any:
    """The planner's task: actions with citations, validated as an `ActionPlan`."""
    return crewai.Task(
        description=f"{SYSTEM_ACTIONS}\n\nApproved playbooks:\n{playbooks}\n\n"
        "Using the analysis, propose readiness actions. Cite playbook IDs for every action.",
        expected_output="A JSON object with a list of actions, each with text and citations.",
        agent=agent,
        context=[analysis_task],
        output_pydantic=ActionPlan,
    )
