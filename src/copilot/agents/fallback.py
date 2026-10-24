"""Falls back to the deterministic drafter when a model fails."""

from __future__ import annotations

from typing import TYPE_CHECKING

from copilot.domain.models import ActionItem
from copilot.drafting.base import DraftContext, Drafter
from copilot.drafting.rule_based import RuleBasedDrafter
from copilot.llm.errors import classify_error

if TYPE_CHECKING:
    from collections.abc import Sequence


class FallbackDrafter:
    """Tries the primary drafter and falls back to the deterministic one on any failure."""

    def __init__(self, primary: Drafter, fallback: Drafter | None = None) -> None:
        self._primary = primary
        self._fallback = fallback or RuleBasedDrafter()
        self.last_fallback_reason = ""

    def draft_actions(self, context: DraftContext) -> list[ActionItem]:
        """Actions from the primary drafter, or from the fallback if it fails."""
        try:
            return self._primary.draft_actions(context)
        except Exception as exc:
            self.last_fallback_reason = str(classify_error(exc))
            return self._fallback.draft_actions(context)

    def draft_alert(self, context: DraftContext, actions: Sequence[ActionItem]) -> str:
        """Alert from the primary drafter, or from the fallback if it fails."""
        try:
            return self._primary.draft_alert(context, actions)
        except Exception as exc:
            self.last_fallback_reason = str(classify_error(exc))
            return self._fallback.draft_alert(context, actions)
