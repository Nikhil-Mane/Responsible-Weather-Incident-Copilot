"""Persistence of finished briefings so a run can be reviewed and approved later."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import ValidationError

from copilot.core.exceptions import InputValidationError
from copilot.domain.models import IncidentBriefing

if TYPE_CHECKING:
    from pathlib import Path

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_run_id(run_id: str) -> str:
    """Reject IDs that could escape the store directory."""
    if not _RUN_ID.fullmatch(run_id):
        msg = "run ID must be 1-64 letters, digits, '-' or '_'"
        raise InputValidationError(msg)
    return run_id


class RunStore:
    """Stores one JSON file per run, written atomically."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def _path(self, run_id: str) -> Path:
        return self._dir / f"{validate_run_id(run_id)}.json"

    def save(self, briefing: IncidentBriefing) -> Path:
        """Write the briefing, replacing any earlier version of the same run."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(briefing.run_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(briefing.model_dump_json(indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        return path

    def load(self, run_id: str) -> IncidentBriefing:
        """Read a stored briefing."""
        path = self._path(run_id)
        try:
            return IncidentBriefing.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            msg = f"no stored run '{run_id}'"
            raise InputValidationError(msg) from exc
        except ValidationError as exc:
            msg = f"stored run '{run_id}' is corrupt"
            raise InputValidationError(msg) from exc
