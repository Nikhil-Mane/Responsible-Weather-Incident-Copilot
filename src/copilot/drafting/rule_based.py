"""The deterministic, offline drafter: every action is lifted from a retrieved playbook."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from copilot.domain.models import ActionItem
from copilot.drafting.base import ACTION_TOPICS, DraftContext

if TYPE_CHECKING:
    from collections.abc import Sequence


_IMPERATIVES = frozenset(
    {
        "inspect",
        "verify",
        "coordinate",
        "prepare",
        "stage",
        "assess",
        "communicate",
        "review",
        "confirm",
        "monitor",
        "check",
    }
)


_MAX_ALERT_ACTIONS = 3


def extract_actions(content: str) -> list[str]:
    """Pull the imperative action phrases out of a playbook's first sentence."""
    sentence = re.split(r"(?<=[.!?])\s", content.strip(), maxsplit=1)[0].rstrip(".")
    parts = [re.sub(r"^(and|or)\s+", "", p.strip()) for p in sentence.split(",")]
    for index, part in enumerate(parts):
        words = part.split()
        if words and words[0].lower() in _IMPERATIVES:
            return [p for p in parts[index:] if p]
    return []


class RuleBasedDrafter:
    """Deterministic drafter that only restates approved playbook actions."""

    def draft_actions(self, context: DraftContext) -> list[ActionItem]:
        """One action per playbook phrase, in retrieval order, without duplicates."""
        seen: set[str] = set()
        items: list[ActionItem] = []
        for retrieved in context.playbooks:
            if retrieved.playbook.topic not in ACTION_TOPICS:
                continue
            for phrase in extract_actions(retrieved.playbook.content):
                text = phrase[0].upper() + phrase[1:]
                if text.lower() in seen:
                    continue
                seen.add(text.lower())
                items.append(ActionItem(text=text, citations=(retrieved.citation,)))
        return items

    def draft_alert(self, context: DraftContext, actions: Sequence[ActionItem]) -> str:
        """Compose a draft that names the event, area, readiness action and approval status."""
        snapshot = context.snapshot
        areas = ", ".join(snapshot.affected_areas) or "to be confirmed"
        readiness = "; ".join(a.text.lower() for a in actions[:_MAX_ALERT_ACTIONS])
        readiness = readiness or "confirm readiness with the local response team"
        return (
            f"DRAFT: {snapshot.event} in {snapshot.city}. Affected area: {areas}. "
            f"Readiness action: {readiness}. "
            "Approval status: pending incident-commander approval; "
            "await incident-command direction. Not sent."
        )
