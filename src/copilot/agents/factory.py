"""Choosing the LLM drafter or the deterministic one."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from copilot.agents.chains import build_alert_chain, build_planner_chain
from copilot.agents.llm_drafter import LLMDrafter
from copilot.drafting.base import Drafter
from copilot.drafting.rule_based import RuleBasedDrafter
from copilot.llm.azure import build_chat_model
from copilot.llm.resilience import CircuitBreaker
from copilot.llm.usage import UsageTracker

if TYPE_CHECKING:
    from collections.abc import Callable

    from copilot.core.config import Settings


def build_drafter(
    settings: Settings,
    breaker: CircuitBreaker | None = None,
    extra_callbacks: Callable[[], list[Any]] | None = None,
) -> Drafter:
    """The LLM drafter when Azure is configured, otherwise the deterministic drafter."""
    model = build_chat_model(settings)
    if model is None:
        return RuleBasedDrafter()
    shared = breaker or CircuitBreaker()
    return LLMDrafter(
        build_planner_chain(model),
        build_alert_chain(model),
        retries=settings.llm_max_retries,
        breaker=shared,
        usage=UsageTracker(),
        extra_callbacks=extra_callbacks,
    )
