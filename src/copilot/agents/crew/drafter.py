"""`Drafter` that runs the CrewAI crews. Failures surface as `ProviderError` for fallback."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from copilot.agents.crew.crews import build_comms_crew, build_planning_crew
from copilot.agents.crew.runtime import build_crew_llm, flush_events
from copilot.agents.schemas import ActionPlan, AlertDraft
from copilot.core.exceptions import ProviderError
from copilot.domain.models import ActionItem
from copilot.llm.resilience import CircuitBreaker, call_with_resilience

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from copilot.core.config import Settings
    from copilot.drafting.base import DraftContext


def _final_output(result: Any, schema: type) -> Any:
    """Pull the validated pydantic output of the last task out of a crew result."""
    candidate = getattr(result, "pydantic", None)
    if candidate is None:
        tasks = getattr(result, "tasks_output", None) or []
        candidate = getattr(tasks[-1], "pydantic", None) if tasks else None
    if not isinstance(candidate, schema):
        msg = f"invalid structured output: crew did not return a {schema.__name__}"
        raise ProviderError(msg, retryable=True)
    return candidate


class CrewDrafter:
    """`Drafter` that runs CrewAI crews. Failures surface as `ProviderError` for fallback."""

    def __init__(
        self,
        llm: Any,
        *,
        retries: int = 2,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._llm = llm
        self._retries = retries
        self._breaker = breaker
        self._sleep = sleep

    def draft_actions(self, context: DraftContext) -> list[ActionItem]:
        """Run the Hazard Analyst and Response Planner crew."""

        def operation() -> list[ActionItem]:
            result = build_planning_crew(self._llm, context).kickoff()
            flush_events()
            plan: ActionPlan = _final_output(result, ActionPlan)
            return [
                ActionItem(text=a.text.strip(), citations=tuple(a.citations)) for a in plan.actions
            ]

        return call_with_resilience(
            operation, retries=self._retries, breaker=self._breaker, sleep=self._sleep
        )

    def draft_alert(self, context: DraftContext, actions: Sequence[ActionItem]) -> str:
        """Run the Comms Drafter crew."""

        def operation() -> str:
            result = build_comms_crew(self._llm, context, actions).kickoff()
            flush_events()
            draft: AlertDraft = _final_output(result, AlertDraft)
            return draft.text.strip()

        return call_with_resilience(
            operation, retries=self._retries, breaker=self._breaker, sleep=self._sleep
        )


def build_crew_drafter(
    settings: Settings, breaker: CircuitBreaker | None = None
) -> CrewDrafter | None:
    """A `CrewDrafter` when Azure is configured and CrewAI is installed, otherwise `None`."""
    llm = build_crew_llm(settings)
    if llm is None:
        return None
    return CrewDrafter(llm, retries=settings.llm_max_retries, breaker=breaker or CircuitBreaker())
