"""The `Observer` contract and the no-op default."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    from copilot.domain.models import IncidentBriefing
    from copilot.evaluation.report import Report


class Observer(Protocol):
    """Receives what happens during a run. Must never raise into the caller."""

    def record_run(self, briefing: IncidentBriefing, actor_role: str, *, since: int = 0) -> None:
        """Record a run's steps from index `since` on (after an approval, only the new ones)."""
        ...

    def record_send(self, run_id: str, receipt: Mapping[str, Any]) -> None:
        """Record the (simulated) send of an approved alert."""
        ...

    def record_eval(self, report: Report) -> None:
        """Record evaluation metrics and per-case results."""
        ...

    def llm_callbacks(self) -> list[Any]:
        """LangChain callbacks that should observe model calls (may be empty)."""
        ...

    def instrument_agents(self) -> None:
        """Start observing CrewAI agents, tasks and model calls (no-op without CrewAI)."""
        ...

    def flush(self) -> None:
        """Send anything buffered. Call before a short-lived process exits."""
        ...


class NoopObserver:
    """Default observer: does nothing, so the system has no hard dependency on Langfuse."""

    def record_run(self, briefing: IncidentBriefing, actor_role: str, *, since: int = 0) -> None:
        """No-op."""

    def record_send(self, run_id: str, receipt: Mapping[str, Any]) -> None:
        """No-op."""

    def record_eval(self, report: Report) -> None:
        """No-op."""

    def llm_callbacks(self) -> list[Any]:
        """No callbacks."""
        return []

    def instrument_agents(self) -> None:
        """No-op."""

    def flush(self) -> None:
        """No-op."""
