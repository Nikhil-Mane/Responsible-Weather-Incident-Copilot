"""Extract structured data from the supplied PDF spreadsheet printouts.

The PDFs are the only data source. They are printed spreadsheets, so plain text order is
unreliable: columns are recovered from word coordinates (scenario dataset), from column
order (single-row scenario files) or from character positions (playbooks, whose keyword
cells are clipped by the next column).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from copilot.core.exceptions import InputValidationError
from copilot.domain.models import Severity

# Canonical column order of the flood example. Heatwave and winter files are prefixes of it.
SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "city",
    "event",
    "severity",
    "snowfall_rate_cm_per_hour",
    "wind_gust_kmh",
    "temperature_c",
    "road_surface_temperature_c",
    "visibility_m",
    "rainfall_mm_per_hour",
    "river_level_m",
    "flood_warning",
    "forecast_rainfall_mm",
    "river_rise_cm_per_hour",
    "affected_areas",
    "road_closures",
    "observation_timestamp",
    "forecast_horizon_hours",
    "source_id",
    "source_confidence",
    "latitude",
    "longitude",
    "geo_zone",
    "population_at_risk",
    "critical_assets",
    "drainage_capacity_percent",
    "incident_status",
)

_LIST_COLUMNS = frozenset({"affected_areas", "road_closures", "critical_assets"})
_BOOL_COLUMNS = frozenset({"flood_warning"})
_TEXT_COLUMNS = frozenset(
    {"city", "severity", "observation_timestamp", "source_id", "geo_zone", "incident_status"}
)
_SEVERITIES = frozenset(s.value for s in Severity)

# Dataset header -> snapshot field. Header names are matched exactly as printed.
DATASET_FIELD_MAP: dict[str, str] = {
    "snowfall_rate": "snowfall_rate_cm_per_hour",
    "wind_gusts": "wind_gust_kmh",
    "visibility": "visibility_m",
    "rainfall_rate": "rainfall_mm_per_hour",
    "affected_areas": "affected_areas",
    "river_level": "river_level_m",
    "flood_warning": "flood_warning",
    "river_rise": "river_rise_cm_per_hour",
    "forecast_rainfall": "forecast_rainfall_mm",
    "road_closures": "road_closures",
    "temperature": "temperature_c",
    "road_temperature": "road_surface_temperature_c",
}
_DATASET_LABEL_COLUMNS = ("id", "name", "category", "severity")
_DATASET_FLAG_COLUMNS = ("expects_approval", "expects_local_alert")
_ROW_Y_TOLERANCE = 2.0


@dataclass(frozen=True, slots=True)
class Word:
    """A positioned word from a PDF page."""

    x0: float
    x1: float
    y0: float
    text: str


def _page_words(path: Path) -> list[Word]:
    with pymupdf.open(path) as doc:
        if doc.page_count != 1:
            msg = f"{path.name}: expected a single page, found {doc.page_count}"
            raise InputValidationError(msg)
        raw = doc[0].get_text("words")
    return [Word(w[0], w[2], w[1], w[4]) for w in raw]


def _group_rows(words: list[Word]) -> list[list[Word]]:
    """Group words into visual rows by y position, each row ordered left to right."""
    rows: list[list[Word]] = []
    for word in sorted(words, key=lambda w: (w.y0, w.x0)):
        if rows and abs(rows[-1][0].y0 - word.y0) <= _ROW_Y_TOLERANCE:
            rows[-1].append(word)
        else:
            rows.append([word])
    return [sorted(r, key=lambda w: w.x0) for r in rows]


def _parse_scalar(text: str) -> Any:
    upper = text.upper()
    if upper in {"TRUE", "FALSE"}:
        return upper == "TRUE"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_list(tokens: list[str]) -> list[str]:
    try:
        value = json.loads(" ".join(tokens))
    except json.JSONDecodeError as exc:
        msg = f"cannot parse list value: {' '.join(tokens)!r}"
        raise InputValidationError(msg) from exc
    if not isinstance(value, list):
        msg = f"expected a list, got {value!r}"
        raise InputValidationError(msg)
    return [str(item) for item in value]


def _consume_list(tokens: list[str], start: int) -> tuple[list[str], int]:
    end = start
    while end < len(tokens) and not tokens[end].rstrip(",").endswith("]"):
        end += 1
    if end >= len(tokens):
        msg = "unterminated list value"
        raise InputValidationError(msg)
    return _parse_list(tokens[start : end + 1]), end + 1


def _consume_event(tokens: list[str], start: int) -> tuple[str, int]:
    """The event name is multi-word; it ends where the severity word begins."""
    for end in range(start + 1, len(tokens)):
        if tokens[end].lower() in _SEVERITIES:
            return " ".join(tokens[start:end]), end
    msg = "could not locate the severity value after the event name"
    raise InputValidationError(msg)


def _header_columns(header: list[Word]) -> tuple[str, ...]:
    """Match the printed header (spaces or underscores) to the canonical column order."""
    joined = "_".join(w.text.lower() for w in header)
    for count in range(len(SNAPSHOT_COLUMNS), 0, -1):
        if joined == "_".join(SNAPSHOT_COLUMNS[:count]):
            return SNAPSHOT_COLUMNS[:count]
    msg = f"unrecognised header: {joined!r}"
    raise InputValidationError(msg)


def extract_snapshot(path: Path) -> dict[str, Any]:
    """Extract the single data row of a scenario PDF (flood, heatwave or winter)."""
    rows = _group_rows(_page_words(path))
    if len(rows) != 2:
        msg = f"{path.name}: expected a header row and one data row, found {len(rows)} rows"
        raise InputValidationError(msg)
    columns = _header_columns(rows[0])
    tokens = [w.text for w in rows[1]]

    record: dict[str, Any] = {}
    cursor = 0
    for column in columns:
        if cursor >= len(tokens):
            msg = f"{path.name}: ran out of values at column '{column}'"
            raise InputValidationError(msg)
        if column == "event":
            record[column], cursor = _consume_event(tokens, cursor)
        elif column in _LIST_COLUMNS:
            record[column], cursor = _consume_list(tokens, cursor)
        elif column in _TEXT_COLUMNS:
            record[column] = tokens[cursor]
            cursor += 1
        else:
            record[column] = _parse_scalar(tokens[cursor])
            cursor += 1
    if cursor != len(tokens):
        msg = f"{path.name}: {len(tokens) - cursor} unused values after the last column"
        raise InputValidationError(msg)
    for column in _BOOL_COLUMNS & record.keys():
        if not isinstance(record[column], bool):
            msg = f"{path.name}: '{column}' is not TRUE/FALSE"
            raise InputValidationError(msg)
    return record


def extract_scenarios(path: Path) -> list[dict[str, Any]]:
    """Extract the labeled scenarios, assigning each value to a column by x position."""
    rows = _group_rows(_page_words(path))
    header, body = rows[0], rows[1:]
    starts = [(w.x0, w.text) for w in header]

    def column_of(word: Word) -> str:
        name = starts[0][1]
        for start, candidate in starts:
            if word.x0 + 1.0 >= start:
                name = candidate
            else:
                break
        return name

    scenarios: list[dict[str, Any]] = []
    for row in body:
        cells: dict[str, list[str]] = {}
        for word in row:
            cells.setdefault(column_of(word), []).append(word.text)
        scenarios.append(_scenario_from_cells(cells))
    return scenarios


def _scenario_from_cells(cells: dict[str, list[str]]) -> dict[str, Any]:
    scenario: dict[str, Any] = {}
    for column in _DATASET_LABEL_COLUMNS:
        scenario[column] = " ".join(cells[column])
    for column in _DATASET_FLAG_COLUMNS:
        value = _parse_scalar(cells[column][0])
        if not isinstance(value, bool):
            msg = f"{scenario['id']}: '{column}' is not TRUE/FALSE"
            raise InputValidationError(msg)
        scenario[column] = value

    fields: dict[str, Any] = {}
    for header, field in DATASET_FIELD_MAP.items():
        tokens = cells.get(header)
        if not tokens:
            continue
        fields[field] = _parse_list(tokens) if field in _LIST_COLUMNS else _parse_scalar(tokens[0])
    scenario["fields"] = fields
    return scenario


def _page_chars(path: Path) -> list[tuple[float, float, str]]:
    with pymupdf.open(path) as doc:
        raw = doc[0].get_text("rawdict")
    chars: list[tuple[float, float, str]] = []
    for block in raw["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                chars.extend((c["bbox"][0], c["bbox"][1], c["c"]) for c in span["chars"])
    return chars


def _text_from_chars(chars: list[tuple[float, float, str]]) -> str:
    ordered = sorted(chars, key=lambda c: c[0])
    return "".join(c[2] for c in ordered).strip()


def _complete_keywords(clipped: str) -> tuple[list[str], bool]:
    """Keep only fully printed quoted keywords; report whether the cell was clipped."""
    parts = clipped.split('"')
    # parts at odd indexes are inside quotes; the last quote may be unterminated
    closed = clipped.rstrip().endswith("]")
    quoted = parts[1::2]
    if not closed and len(parts) % 2 == 0:
        quoted = quoted[:-1]
    return [kw for kw in quoted if kw.strip()], not closed


def extract_playbooks(path: Path) -> list[dict[str, Any]]:
    """Extract playbooks using character positions, because keyword cells are clipped."""
    words = _page_words(path)
    rows = _group_rows(words)
    header, body = rows[0], rows[1:]
    starts = {w.text: w.x0 for w in header}
    needed = ("document_id", "title", "keywords", "content")
    if any(name not in starts for name in needed):
        msg = f"{path.name}: unexpected header {[w.text for w in header]}"
        raise InputValidationError(msg)
    bounds = [starts[name] for name in needed]

    chars = _page_chars(path)
    playbooks: list[dict[str, Any]] = []
    for row in body:
        row_y = row[0].y0
        row_chars = [c for c in chars if abs(c[1] - row_y) <= _ROW_Y_TOLERANCE + 2]
        cells: dict[str, str] = {}
        for index, name in enumerate(needed):
            low = bounds[index] - 0.5
            high = bounds[index + 1] - 0.5 if index + 1 < len(bounds) else float("inf")
            cells[name] = _text_from_chars([c for c in row_chars if low <= c[0] < high])
        keywords, clipped = _complete_keywords(cells["keywords"])
        playbooks.append(
            {
                "document_id": cells["document_id"],
                "title": cells["title"],
                "keywords": keywords,
                "keywords_clipped": clipped,
                "content": cells["content"],
            }
        )
    return playbooks


SOURCE_FILES: dict[str, str] = {
    "flood_warning": "flood_warning.pdf",
    "heatwave_advisory": "heatwave_advisory.pdf",
    "winter_storm": "winter_storm.pdf",
    "operational_playbooks": "operational_playbooks.pdf",
    "scenario_dataset": "scenario_dataset.pdf",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    """Write atomically so a crash never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def extract_all(data_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Extract every supplied PDF into `out_dir` as JSON and write a manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: data_dir / filename for name, filename in SOURCE_FILES.items()}
    missing = [str(p) for p in sources.values() if not p.is_file()]
    if missing:
        msg = f"missing source files: {', '.join(missing)}"
        raise InputValidationError(msg)

    outputs: dict[str, object] = {
        "flood_warning": extract_snapshot(sources["flood_warning"]),
        "heatwave_advisory": extract_snapshot(sources["heatwave_advisory"]),
        "winter_storm": extract_snapshot(sources["winter_storm"]),
        "operational_playbooks": extract_playbooks(sources["operational_playbooks"]),
        "scenario_dataset": extract_scenarios(sources["scenario_dataset"]),
    }
    manifest: dict[str, Any] = {"files": {}}
    for name, payload in outputs.items():
        _write_json(out_dir / f"{name}.json", payload)
        manifest["files"][name] = {
            "source": sources[name].name,
            "sha256": _sha256(sources[name]),
            "records": len(payload) if isinstance(payload, list) else 1,
        }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest
