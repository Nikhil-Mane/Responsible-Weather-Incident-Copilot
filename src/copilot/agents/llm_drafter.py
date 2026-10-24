"""`Drafter` that uses LCEL chains."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from copilot.agents.prompts import render_facts, render_playbooks, sanitize
from copilot.agents.schemas import ActionPlan, AlertDraft
from copilot.domain.models import ActionItem
from copilot.drafting.base import DraftContext
from copilot.llm.resilience import CircuitBreaker, call_with_resilience
from copilot.llm.usage import UsageTracker

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.runnables import Runnable, RunnableConfig


class LLMDrafter:
    """`Drafter` that uses LCEL chains. Failures surface as `ProviderError` for fallback."""

    def __init__(  # noqa: PLR0913 - dependency-injection constructor; tuning args are optional
        self,
        planner: Runnable[dict[str, str], ActionPlan],
        alert: Runnable[dict[str, str], AlertDraft],
        *,
        retries: int = 3,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
        usage: UsageTracker | None = None,
        extra_callbacks: Callable[[], list[Any]] | None = None,
    ) -> None:
        self._planner = planner
        self._alert = alert
        self._retries = retries
        self._breaker = breaker
        self._sleep = sleep
        self.usage = usage
        self._extra_callbacks = extra_callbacks

    @property
    def _config(self) -> RunnableConfig:
        """Callbacks for one model call: usage counting plus any observer handlers."""
        callbacks: list[Any] = [self.usage] if self.usage else []
        if self._extra_callbacks:
            callbacks.extend(self._extra_callbacks())
        return {"callbacks": callbacks} if callbacks else {}

    def _named(self, name: str) -> RunnableConfig:
        """The call configuration with a readable name for traces."""
        return {**self._config, "run_name": name}

    def draft_actions(self, context: DraftContext) -> list[ActionItem]:
        """Ask the model for grounded actions."""
        payload = {"facts": render_facts(context), "playbooks": render_playbooks(context)}

        def operation() -> list[ActionItem]:
            plan = self._planner.invoke(payload, config=self._named("lcel:action_planner"))
            if not isinstance(plan, ActionPlan):
                plan = ActionPlan.model_validate(plan)
            return [
                ActionItem(text=a.text.strip(), citations=tuple(a.citations)) for a in plan.actions
            ]

        return call_with_resilience(
            operation, retries=self._retries, breaker=self._breaker, sleep=self._sleep
        )

    def draft_alert(self, context: DraftContext, actions: Sequence[ActionItem]) -> str:
        """Ask the model for the draft alert text."""
        payload = {
            "facts": render_facts(context),
            "actions": "\n".join(f"- {sanitize(a.text, 200)}" for a in actions) or "- none",
        }

        def operation() -> str:
            draft = self._alert.invoke(payload, config=self._named("lcel:alert_drafter"))
            if not isinstance(draft, AlertDraft):
                draft = AlertDraft.model_validate(draft)
            return draft.text.strip()

        return call_with_resilience(
            operation, retries=self._retries, breaker=self._breaker, sleep=self._sleep
        )
