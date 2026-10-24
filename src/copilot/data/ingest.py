"""Safe loading of incident JSON: size and depth limits, no NaN, strict schema."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from copilot.core.exceptions import InputValidationError
from copilot.domain.models import WeatherSnapshot

if TYPE_CHECKING:
    from pathlib import Path


def _reject_constant(name: str) -> Any:
    msg = f"non-finite number '{name}' is not allowed"
    raise InputValidationError(msg)


def _depth(value: object, limit: int) -> None:
    """Iteratively check nesting depth so hostile input cannot exhaust the stack."""
    stack: list[tuple[object, int]] = [(value, 1)]
    while stack:
        item, level = stack.pop()
        if level > limit:
            msg = f"JSON nesting exceeds the limit of {limit}"
            raise InputValidationError(msg)
        if isinstance(item, dict):
            stack.extend((child, level + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, level + 1) for child in item)


def parse_incident(
    raw: bytes | str, *, max_bytes: int = 262_144, max_depth: int = 12
) -> WeatherSnapshot:
    """Parse and validate one incident. Raises `InputValidationError` for any problem."""
    size = len(raw.encode("utf-8")) if isinstance(raw, str) else len(raw)
    if size > max_bytes:
        msg = f"input is {size} bytes; the limit is {max_bytes}"
        raise InputValidationError(msg)
    try:
        payload = json.loads(raw, parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        msg = f"input is not valid JSON: {exc}"
        raise InputValidationError(msg) from exc
    if not isinstance(payload, dict):
        msg = "input must be a JSON object"
        raise InputValidationError(msg)
    _depth(payload, max_depth)
    try:
        return WeatherSnapshot.model_validate(payload)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'input'}: {err['msg']}"
            for err in exc.errors()
        )
        msg = f"incident failed validation: {problems}"
        raise InputValidationError(msg) from exc


def load_incident(path: Path, *, max_bytes: int = 262_144, max_depth: int = 12) -> WeatherSnapshot:
    """Read an incident file, checking its size before reading it into memory."""
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            msg = f"{path} is not a file"
            raise InputValidationError(msg)
        if resolved.stat().st_size > max_bytes:
            msg = f"{path.name} is larger than the {max_bytes} byte limit"
            raise InputValidationError(msg)
        raw = resolved.read_bytes()
    except OSError as exc:
        msg = f"cannot read {path}: {exc}"
        raise InputValidationError(msg) from exc
    return parse_incident(raw, max_bytes=max_bytes, max_depth=max_depth)
