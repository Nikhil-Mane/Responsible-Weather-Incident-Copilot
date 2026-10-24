"""A small cross-platform advisory file lock (atomic create), used to serialise approvals."""

from __future__ import annotations

import contextlib
import os
import time
from typing import TYPE_CHECKING

from copilot.core.exceptions import LockTimeoutError

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


class FileLock:
    """Exclusive lock held by creating `path` atomically. Stale locks are broken."""

    def __init__(
        self,
        path: Path,
        *,
        timeout: float = 5.0,
        poll_interval: float = 0.02,
        stale_after: float = 30.0,
    ) -> None:
        self._path = path
        self._timeout = timeout
        self._poll = poll_interval
        self._stale_after = stale_after
        self._fd: int | None = None

    def __enter__(self) -> FileLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                self._fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except (FileExistsError, PermissionError):
                # Windows reports PermissionError, not FileExistsError, when the lock file
                # has just been deleted by its previous holder and is still pending removal.
                self._break_if_stale()
                if time.monotonic() >= deadline:
                    msg = f"could not acquire lock {self._path.name} within {self._timeout}s"
                    raise LockTimeoutError(msg) from None
                time.sleep(self._poll)

    def _break_if_stale(self) -> None:
        try:
            age = time.time() - self._path.stat().st_mtime
        except (FileNotFoundError, PermissionError):
            return
        if age > self._stale_after:
            with contextlib.suppress(PermissionError):
                self._path.unlink(missing_ok=True)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._path.unlink(missing_ok=True)
