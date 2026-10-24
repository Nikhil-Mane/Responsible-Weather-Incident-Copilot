"""Structured logging with run-ID correlation and secret redaction."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="-")

_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|secret|token|authorization)(\"?\s*[:=]\s*\"?)([^\s\",}]+)"),
)
_REDACTED = "***REDACTED***"
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


def set_run_id(run_id: str) -> contextvars.Token[str]:
    """Bind a run ID to the current context so every log line carries it."""
    return _run_id.set(run_id)


def reset_run_id(token: contextvars.Token[str]) -> None:
    """Restore the previous run ID."""
    _run_id.reset(token)


def get_run_id() -> str:
    """Return the run ID bound to the current context."""
    return _run_id.get()


def redact(text: str) -> str:
    """Mask values that look like credentials."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}{_REDACTED}", text)
    return text


class _RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", "-"),
            "message": redact(record.getMessage()),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key != "run_id":
                payload[key] = value
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


class _TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s [%(run_id)s] %(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure the root logger once. Safe to call repeatedly."""
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(_RunIdFilter())
    handler.setFormatter(_JsonFormatter() if json_output else _TextFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)
