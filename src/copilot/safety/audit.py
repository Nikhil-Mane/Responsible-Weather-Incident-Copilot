"""Append-only audit log with a hash chain, so tampering is detectable."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from copilot.core.exceptions import AuditIntegrityError
from copilot.core.locking import FileLock
from copilot.core.logging_config import redact

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

GENESIS = "0" * 64


def _digest(previous: str, body: Mapping[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((previous + canonical).encode("utf-8")).hexdigest()


class AuditLog:
    """JSON Lines file. Each entry commits to the previous entry's hash."""

    def __init__(self, path: Path, clock: Callable[[], datetime] | None = None) -> None:
        self._path = path
        self._lock_path = path.with_suffix(path.suffix + ".lock")
        self._clock = clock or (lambda: datetime.now(UTC))

    def append(
        self,
        event: str,
        *,
        run_id: str,
        actor: str = "system",
        detail: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add one entry and return it."""
        with FileLock(self._lock_path):
            entries = self._read()
            previous = entries[-1]["hash"] if entries else GENESIS
            body = {
                "seq": len(entries) + 1,
                "at": self._clock().isoformat(),
                "event": event,
                "run_id": run_id,
                "actor": actor,
                "detail": json.loads(redact(json.dumps(dict(detail or {}), default=str))),
                "prev_hash": previous,
            }
            entry = {**body, "hash": _digest(previous, body)}
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
            return entry

    def entries(self) -> list[dict[str, Any]]:
        """All entries, oldest first."""
        return self._read()

    def verify(self) -> int:
        """Check the whole chain. Returns the entry count or raises `AuditIntegrityError`."""
        previous = GENESIS
        entries = self._read()
        for index, entry in enumerate(entries, start=1):
            body = {k: v for k, v in entry.items() if k != "hash"}
            if entry.get("seq") != index:
                msg = f"entry {index}: sequence number is {entry.get('seq')}"
                raise AuditIntegrityError(msg)
            if entry.get("prev_hash") != previous:
                msg = f"entry {index}: previous-hash link is broken"
                raise AuditIntegrityError(msg)
            if entry.get("hash") != _digest(previous, body):
                msg = f"entry {index}: content does not match its hash"
                raise AuditIntegrityError(msg)
            previous = entry["hash"]
        return len(entries)

    def _read(self) -> list[dict[str, Any]]:
        if not self._path.is_file():
            return []
        entries: list[dict[str, Any]] = []
        for number, line in enumerate(self._path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                msg = f"line {number}: not valid JSON"
                raise AuditIntegrityError(msg) from exc
        return entries
