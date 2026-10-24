"""Exception hierarchy. Every expected failure maps to one of these types."""

from __future__ import annotations


class CopilotError(Exception):
    """Base class for all expected errors raised by this package."""


class ConfigurationError(CopilotError):
    """Settings are missing or invalid."""


class InputValidationError(CopilotError):
    """Incoming incident data is malformed, oversized or out of range."""


class DataQualityError(CopilotError):
    """Input is well-formed but not trustworthy enough for a recommendation."""


class RetrievalError(CopilotError):
    """Playbook retrieval failed."""


class ProviderError(CopilotError):
    """An external model provider failed (timeout, throttling, content filter, bad output).

    `retryable` marks transient failures that are worth another attempt.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ApprovalError(CopilotError):
    """An approval token is missing, expired, reused or does not match the alert."""


class LockTimeoutError(CopilotError):
    """A file lock could not be acquired in time."""


class AuditIntegrityError(CopilotError):
    """The audit log failed its hash-chain verification."""


class AuthorizationError(CopilotError):
    """The caller's role does not permit the requested action."""
