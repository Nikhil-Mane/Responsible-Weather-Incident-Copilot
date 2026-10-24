"""Approved playbooks: the record schema, the in-memory type and loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from copilot.core.exceptions import RetrievalError

if TYPE_CHECKING:
    pass


_TOPIC_BY_CODE = {
    "FLOOD": "flood",
    "WINTER": "winter",
    "HEAT": "heat",
    "COMMS": "communications",
    "DATA": "data_quality",
}


class PlaybookRecord(BaseModel):
    """Schema of one extracted playbook."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    title: str
    keywords: tuple[str, ...]
    keywords_clipped: bool = False
    content: str


@dataclass(frozen=True, slots=True)
class Playbook:
    """An approved operational procedure."""

    document_id: str
    title: str
    keywords: tuple[str, ...]
    content: str
    topic: str

    @property
    def text(self) -> str:
        """Text used for indexing: title, keywords and body."""
        return f"{self.title}. {' '.join(self.keywords)}. {self.content}"


def topic_for(document_id: str) -> str:
    """Derive the metadata topic from the ID, for example PLAYBOOK-FLOOD-001 -> flood."""
    parts = document_id.split("-")
    code = parts[1] if len(parts) >= 3 else document_id
    return _TOPIC_BY_CODE.get(code.upper(), code.lower())


def load_playbooks(path: Path) -> tuple[Playbook, ...]:
    """Load playbooks from the generated JSON file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = [PlaybookRecord.model_validate(item) for item in raw]
    except (OSError, ValueError) as exc:
        msg = f"cannot load playbooks from {path}: {exc}"
        raise RetrievalError(msg) from exc
    if not records:
        msg = f"no playbooks found in {path}"
        raise RetrievalError(msg)
    return tuple(
        Playbook(r.document_id, r.title, r.keywords, r.content, topic_for(r.document_id))
        for r in records
    )
