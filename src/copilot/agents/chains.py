"""LangChain (LCEL) chains: prompt followed by schema-constrained model output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.prompts import ChatPromptTemplate

from copilot.agents.prompts import SYSTEM_ACTIONS, SYSTEM_ALERT
from copilot.agents.schemas import ActionPlan, AlertDraft

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable


def build_planner_chain(model: Any) -> Runnable[dict[str, str], ActionPlan]:
    """LCEL chain: prompt -> model with schema-constrained output."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_ACTIONS),
            (
                "human",
                "Verified facts:\n{facts}\n\nApproved playbooks:\n{playbooks}\n\nPropose actions.",
            ),
        ]
    )
    return prompt | model.with_structured_output(ActionPlan)  # type: ignore[no-any-return]


def build_alert_chain(model: Any) -> Runnable[dict[str, str], AlertDraft]:
    """LCEL chain for the draft local-team alert."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_ALERT),
            (
                "human",
                "Verified facts:\n{facts}\n\nReadiness actions to include:\n{actions}\n\n"
                "Write the draft alert.",
            ),
        ]
    )
    return prompt | model.with_structured_output(AlertDraft)  # type: ignore[no-any-return]
