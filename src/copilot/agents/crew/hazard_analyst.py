"""Hazard Analyst: explains the verified hazards and severity (no tools, no delegation)."""

from __future__ import annotations

from typing import Any

from copilot.agents.crew.runtime import make_agent
from copilot.agents.prompts import SYSTEM_ACTIONS


def build_hazard_analyst(crewai: Any, llm: Any) -> Any:
    """The Hazard Analyst agent."""
    return make_agent(
        crewai,
        llm,
        "Hazard Analyst",
        "Explain, from the verified facts only, which hazards are present and why the "
        "severity is what it is.",
        "You are a careful operations analyst. You never speculate beyond the data.",
    )


def build_analysis_task(crewai: Any, agent: Any, facts: str) -> Any:
    """The analyst's task: a two-sentence summary of hazards and severity."""
    return crewai.Task(
        description=f"{SYSTEM_ACTIONS}\n\nVerified facts:\n{facts}\n\n"
        "Summarise the hazards and severity in two sentences.",
        expected_output="Two sentences describing the hazards and severity.",
        agent=agent,
    )
