"""Data-integrity tests for PDF extraction, plus the golden check of the rules."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from copilot.core.exceptions import InputValidationError
from copilot.data.extract import (
    SOURCE_FILES,
    _complete_keywords,
    _parse_list,
    extract_all,
    extract_playbooks,
    extract_scenarios,
    extract_snapshot,
)
from copilot.domain.models import WeatherSnapshot
from copilot.domain.rules import assess
from copilot.domain.validation import check_quality

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

pytestmark = pytest.mark.skipif(
    not all((DATA_DIR / name).is_file() for name in SOURCE_FILES.values()),
    reason="supplied PDFs are not present in data/",
)


@pytest.fixture(scope="module")
def scenarios() -> list[dict[str, Any]]:
    return extract_scenarios(DATA_DIR / SOURCE_FILES["scenario_dataset"])


def _snapshot_for(scenario: dict[str, Any]) -> WeatherSnapshot:
    return WeatherSnapshot.model_validate(
        {
            "city": "Northbridge",
            "event": scenario["name"],
            "severity": scenario["severity"],
            **scenario["fields"],
        }
    )


def test_there_are_24_labeled_scenarios(scenarios) -> None:
    assert [s["id"] for s in scenarios] == [f"WX-{n:03d}" for n in range(1, 25)]
    for scenario in scenarios:
        assert isinstance(scenario["expects_approval"], bool)
        assert isinstance(scenario["expects_local_alert"], bool)


@pytest.mark.parametrize(
    ("scenario_id", "field", "value"),
    [
        ("WX-003", "rainfall_mm_per_hour", 24),
        ("WX-004", "rainfall_mm_per_hour", 25),
        ("WX-005", "river_level_m", 4.5),
        ("WX-007", "river_rise_cm_per_hour", 10),
        ("WX-011", "snowfall_rate_cm_per_hour", 3),
        ("WX-012", "visibility_m", 300),
        ("WX-017", "temperature_c", 34.9),
        ("WX-019", "road_surface_temperature_c", 45),
        ("WX-023", "river_level_m", 3.8),
    ],
)
def test_threshold_cells_land_in_the_right_column(scenarios, scenario_id, field, value) -> None:
    scenario = next(s for s in scenarios if s["id"] == scenario_id)
    assert scenario["fields"][field] == value


def test_multiword_lists_are_reassembled(scenarios) -> None:
    wx009 = next(s for s in scenarios if s["id"] == "WX-009")
    assert wx009["fields"]["affected_areas"] == ["Riverside", "Old Town", "Industrial Park"]
    assert wx009["fields"]["road_closures"] == ["Harbor Road", "River Lane"]


def test_every_scenario_is_a_valid_snapshot(scenarios) -> None:
    for scenario in scenarios:
        assert _snapshot_for(scenario).city == "Northbridge"


def test_rules_reproduce_every_golden_label(scenarios) -> None:
    """Route and alert expectations for all 24 labeled scenarios, with the documented rules."""
    mismatches = []
    for scenario in scenarios:
        result = assess(_snapshot_for(scenario))
        got = (result.requires_incident_command_review, result.flood_risk)
        want = (scenario["expects_approval"], scenario["expects_local_alert"])
        if got != want:
            mismatches.append((scenario["id"], scenario["name"], got, want))
    assert mismatches == []


def test_flood_example_is_complete_and_trusted() -> None:
    record = extract_snapshot(DATA_DIR / SOURCE_FILES["flood_warning"])
    snapshot = WeatherSnapshot.model_validate(record)
    assert snapshot.rainfall_mm_per_hour == 42
    assert snapshot.river_level_m == 4.9
    assert snapshot.river_rise_cm_per_hour == 14
    assert snapshot.affected_areas == ("Riverside", "Old Town")
    assert snapshot.road_closures == ("Harbor Road",)
    assert snapshot.critical_assets == ("Riverside Care Home", "Northbridge Substation")
    assert check_quality(snapshot, snapshot.observation_timestamp).passed  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["heatwave_advisory", "winter_storm"])
def test_files_without_provenance_are_held(name: str) -> None:
    record = extract_snapshot(DATA_DIR / SOURCE_FILES[name])
    snapshot = WeatherSnapshot.model_validate(record)
    report = check_quality(snapshot, datetime(2026, 9, 1, 9, 5, tzinfo=UTC))
    assert "missing_source" in report.codes
    assert "missing_timestamp" in report.codes


def test_five_playbooks_with_expected_ids() -> None:
    playbooks = extract_playbooks(DATA_DIR / SOURCE_FILES["operational_playbooks"])
    assert [p["document_id"] for p in playbooks] == [
        "PLAYBOOK-FLOOD-001",
        "PLAYBOOK-WINTER-002",
        "PLAYBOOK-HEAT-003",
        "PLAYBOOK-COMMS-004",
        "PLAYBOOK-DATA-005",
    ]
    flood = playbooks[0]
    assert flood["content"].startswith("For verified flood risk")
    assert flood["keywords_clipped"] is True
    assert "flood" in flood["keywords"]
    assert all(p["content"] and p["title"] for p in playbooks)


def test_extract_all_writes_json_and_manifest(tmp_path: Path) -> None:
    manifest = extract_all(DATA_DIR, tmp_path)
    assert manifest["files"]["scenario_dataset"]["records"] == 24
    assert manifest["files"]["operational_playbooks"]["records"] == 5
    assert len(manifest["files"]["flood_warning"]["sha256"]) == 64
    assert (tmp_path / "manifest.json").is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_source_is_reported(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="missing source files"):
        extract_all(tmp_path, tmp_path / "out")


@pytest.mark.parametrize(
    ("clipped", "expected", "was_clipped"),
    [
        ('["a", "b", "c"]', ["a", "b", "c"], False),
        ('["flood", "road clo', ["flood"], True),
        ('["alert", "local', ["alert"], True),
    ],
)
def test_complete_keywords_drops_clipped_tail(clipped, expected, was_clipped) -> None:
    keywords, is_clipped = _complete_keywords(clipped)
    assert keywords == expected
    assert is_clipped is was_clipped


def test_parse_list_rejects_non_lists() -> None:
    assert _parse_list(['["Old', 'Town"]']) == ["Old Town"]
    with pytest.raises(InputValidationError):
        _parse_list(['{"a": 1}'])
    with pytest.raises(InputValidationError):
        _parse_list(["[broken"])
