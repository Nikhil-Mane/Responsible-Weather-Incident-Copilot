"""Output guardrails: citation enforcement and draft-alert checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from copilot.retrieval.tokenizer import tokenize

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from copilot.domain.models import ActionItem, WeatherSnapshot

DRAFT_PREFIX = "DRAFT:"
APPROVAL_PHRASE = "incident-command"
MIN_GROUNDING_OVERLAP = 0.6


@dataclass(frozen=True, slots=True)
class CitationCheck:
    """Actions kept and dropped by citation enforcement."""

    kept: tuple[ActionItem, ...]
    dropped: tuple[str, ...]


def enforce_action_citations(
    actions: Sequence[ActionItem], retrieved_ids: Iterable[str]
) -> CitationCheck:
    """Keep only actions whose every citation is a playbook that was actually retrieved."""
    allowed = frozenset(retrieved_ids)
    kept: list[ActionItem] = []
    dropped: list[str] = []
    for action in actions:
        if action.citations and all(c in allowed for c in action.citations):
            kept.append(action)
        else:
            dropped.append(action.text)
    return CitationCheck(kept=tuple(kept), dropped=tuple(dropped))


def known_facts_text(snapshot: WeatherSnapshot) -> str:
    """Verified place names from the incident (city, areas, closures, assets).

    An action may legitimately mention them ("inspect drainage hotspots in Riverside"), so they
    are not counted for or against its grounding in the playbook.
    """
    return " ".join(
        [
            snapshot.city,
            *snapshot.affected_areas,
            *snapshot.road_closures,
            *snapshot.critical_assets,
        ]
    )


def grounding_overlap(action: str, source: str, *, known_facts: str = "") -> float:
    """Share of the action's instructional words that appear in its cited source text.

    Words that are verified incident facts (`known_facts`) are set aside first. An action made
    only of such words has no instructional content and scores 0.
    """
    words = set(tokenize(action)) - set(tokenize(known_facts))
    if not words:
        return 0.0
    return len(words & set(tokenize(source))) / len(words)


def enforce_grounding(
    actions: Sequence[ActionItem],
    source_text: Mapping[str, str],
    *,
    min_overlap: float = MIN_GROUNDING_OVERLAP,
    known_facts: str = "",
) -> CitationCheck:
    """Drop actions whose wording is not supported by the playbooks they cite.

    A model can cite a real playbook and still invent an instruction. Requiring most of an
    action's instructional words to appear in the cited text catches that, while still allowing
    it to localise a real action with verified place names (`known_facts`).
    """
    kept: list[ActionItem] = []
    dropped: list[str] = []
    for action in actions:
        cited = " ".join(source_text.get(c, "") for c in action.citations)
        if grounding_overlap(action.text, cited, known_facts=known_facts) >= min_overlap:
            kept.append(action)
        else:
            dropped.append(action.text)
    return CitationCheck(kept=tuple(kept), dropped=tuple(dropped))


def alert_problems(alert: str, *, event: str, areas: Sequence[str]) -> list[str]:
    """Return what is missing from a draft alert (PLAYBOOK-COMMS-004). Empty means valid."""
    problems: list[str] = []
    lowered = alert.lower()
    if not alert.startswith(DRAFT_PREFIX):
        problems.append(f"must start with '{DRAFT_PREFIX}'")
    if event.lower() not in lowered:
        problems.append("must name the event")
    if areas and not any(area.lower() in lowered for area in areas):
        problems.append("must name the affected area")
    if APPROVAL_PHRASE not in lowered or "approv" not in lowered:
        problems.append("must state the approval status and await incident-command direction")
    if "readiness" not in lowered and "prepare" not in lowered and "inspect" not in lowered:
        problems.append("must name a readiness action")
    return problems
