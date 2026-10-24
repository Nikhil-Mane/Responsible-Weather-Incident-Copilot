"""Mapping provider and parsing failures to `ProviderError`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import openai
from pydantic import ValidationError

from copilot.core.exceptions import ProviderError
from copilot.core.logging_config import redact

if TYPE_CHECKING:
    pass


_TRANSIENT = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


_CONTENT_FILTER_MARKERS = ("content_filter", "ResponsibleAIPolicyViolation")


def classify_error(exc: BaseException) -> ProviderError:
    """Map any provider or parsing failure to a `ProviderError`, without leaking secrets."""
    if isinstance(exc, ProviderError):
        return exc
    detail = redact(str(exc))[:300]
    if isinstance(exc, openai.BadRequestError):
        blocked = any(marker in str(exc) for marker in _CONTENT_FILTER_MARKERS)
        message = (
            "the provider's content filter blocked the request"
            if blocked
            else f"the provider rejected the request: {detail}"
        )
        return ProviderError(message)
    message, retryable = _describe(exc, detail)
    return ProviderError(message, retryable=retryable)


def _describe(exc: BaseException, detail: str) -> tuple[str, bool]:
    """Message and whether another attempt is worthwhile."""
    if isinstance(exc, _TRANSIENT):
        return f"{type(exc).__name__}: {detail}", True
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return "the provider refused the credentials", False
    if isinstance(exc, ValidationError | ValueError):
        return f"invalid structured output: {detail}", True
    if isinstance(exc, TimeoutError):
        return "the provider call timed out", True
    return f"{type(exc).__name__}: {detail}", False
