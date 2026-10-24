"""Token usage accounting for cost reporting."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import BaseCallbackHandler

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult


class UsageTracker(BaseCallbackHandler):
    """LangChain callback that totals token usage across model calls (for cost reporting)."""

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Add the usage reported by the provider for one completion."""
        with self._lock:
            self.calls += 1
            for generations in response.generations:
                for generation in generations:
                    usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                    if usage:
                        self.input_tokens += int(usage.get("input_tokens", 0))
                        self.output_tokens += int(usage.get("output_tokens", 0))

    def snapshot(self) -> tuple[int, int, int]:
        """(calls, input tokens, output tokens) so far."""
        with self._lock:
            return self.calls, self.input_tokens, self.output_tokens
