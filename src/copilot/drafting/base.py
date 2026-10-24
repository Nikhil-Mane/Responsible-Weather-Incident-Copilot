"""The `Drafter` contract and the context a drafter receives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from copilot.domain.models import ActionItem

if TYPE_CHECKING:
    from collections.abc import Sequence

    from copilot.domain.models import WeatherSnapshot
    from copilot.domain.rules import Assessment
    from copilot.retrieval.hybrid import RetrievedPlaybook


# Playbooks that list operational actions. Standards (communications, data quality) do not.
ACTION_TOPICS = frozenset({"flood", "winter", "heat"})


@dataclass(frozen=True, slots=True)
class DraftContext:
    """Everything a drafter may use: verified facts, assessment and retrieved guidance."""

    snapshot: WeatherSnapshot
    assessment: Assessment
    playbooks: tuple[RetrievedPlaybook, ...]


class Drafter(Protocol):
    """Produces recommendations and the draft alert from grounded context."""

    def draft_actions(self, context: DraftContext) -> list[ActionItem]:
        """Return readiness actions, each citing the playbooks it came from."""
        ...

    def draft_alert(self, context: DraftContext, actions: Sequence[ActionItem]) -> str:
        """Return the local-team alert text. It must be a draft."""
        ...
