"""Bounded retries with jittered backoff and a circuit breaker."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, TypeVar

from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from copilot.core.exceptions import ProviderError
from copilot.core.logging_config import get_logger
from copilot.llm.errors import classify_error

if TYPE_CHECKING:
    from collections.abc import Callable


_LOG = get_logger(__name__)


T = TypeVar("T")


class CircuitBreaker:
    """Opens after repeated failures so a struggling provider is not hammered."""

    def __init__(
        self,
        failure_threshold: int = 3,
        reset_after: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = failure_threshold
        self._reset_after = reset_after
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        """True while calls are being refused."""
        with self._lock:
            return self._opened_at is not None and (
                self._clock() - self._opened_at < self._reset_after
            )

    def record_success(self) -> None:
        """A call worked: close the breaker."""
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        """A call failed: open the breaker once the threshold is reached."""
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                self._opened_at = self._clock()


def call_with_resilience(
    operation: Callable[[], T],
    *,
    retries: int,
    breaker: CircuitBreaker | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run a provider call with bounded retries, jittered backoff and a circuit breaker."""

    def attempt() -> T:
        if breaker is not None and breaker.is_open:
            msg = "the circuit breaker is open after repeated provider failures"
            raise ProviderError(msg)
        try:
            result = operation()
        except Exception as exc:
            error = classify_error(exc)
            if breaker is not None:
                breaker.record_failure()
            raise error from exc
        if breaker is not None:
            breaker.record_success()
        return result

    retrying = Retrying(
        stop=stop_after_attempt(retries + 1),
        wait=wait_exponential(multiplier=0.5, max=8.0) + wait_random(0, 0.5),
        retry=retry_if_exception(
            lambda exc: (
                isinstance(exc, ProviderError)
                and exc.retryable
                and not (breaker is not None and breaker.is_open)
            )
        ),
        sleep=sleep,
        reraise=False,
    )
    try:
        return retrying(attempt)
    except RetryError as exc:
        last = exc.last_attempt.exception()
        if isinstance(last, ProviderError):
            _LOG.warning("provider call failed after %d attempts: %s", retries + 1, last)
            raise last from exc
        raise ProviderError(str(last)) from exc
