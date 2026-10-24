"""Data-quality gate and typed-contract tests, including failure paths."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from copilot.domain.models import WeatherSnapshot
from copilot.domain.validation import IssueCode, QualityPolicy, check_quality


def test_trusted_flood_snapshot_passes(make_snapshot, now) -> None:
    report = check_quality(make_snapshot(), now)
    assert report.passed
    assert report.codes == ()


def test_stale_observation_is_held(make_snapshot, now) -> None:
    stale = make_snapshot(observation_timestamp=(now - timedelta(minutes=61)).isoformat())
    assert IssueCode.STALE.value in check_quality(stale, now).codes


def test_observation_at_the_age_limit_is_accepted(make_snapshot, now) -> None:
    edge = make_snapshot(observation_timestamp=(now - timedelta(minutes=60)).isoformat())
    assert check_quality(edge, now).passed


def test_future_timestamp_is_held(make_snapshot, now) -> None:
    future = make_snapshot(observation_timestamp=(now + timedelta(minutes=30)).isoformat())
    assert IssueCode.FUTURE_TIMESTAMP.value in check_quality(future, now).codes


@pytest.mark.parametrize(("confidence", "ok"), [(0.69, False), (0.7, True), (0.96, True)])
def test_confidence_boundary(make_snapshot, now, confidence: float, ok: bool) -> None:
    assert check_quality(make_snapshot(source_confidence=confidence), now).passed is ok


def test_missing_provenance_is_held(make_weather, now) -> None:
    """Heatwave and winter inputs as supplied carry no provenance fields."""
    report = check_quality(make_weather(temperature_c=39), now)
    assert {
        IssueCode.MISSING_SOURCE.value,
        IssueCode.MISSING_TIMESTAMP.value,
        IssueCode.MISSING_CONFIDENCE.value,
        IssueCode.MISSING_GEO.value,
        IssueCode.MISSING_STATUS.value,
    } <= set(report.codes)


def test_untrusted_source_is_held(make_snapshot, now) -> None:
    report = check_quality(make_snapshot(source_id="social-media-post-1"), now)
    assert IssueCode.UNTRUSTED_SOURCE.value in report.codes


def test_coordinates_can_replace_geo_zone(make_snapshot, now) -> None:
    assert check_quality(make_snapshot(geo_zone=None), now).passed


def test_no_geo_context_is_held(make_snapshot, now) -> None:
    report = check_quality(make_snapshot(geo_zone=None, latitude=None, longitude=None), now)
    assert IssueCode.MISSING_GEO.value in report.codes


def test_forecast_horizon_limit(make_snapshot, now) -> None:
    assert check_quality(make_snapshot(forecast_horizon_hours=48), now).passed
    far = check_quality(make_snapshot(forecast_horizon_hours=49), now)
    assert IssueCode.FORECAST_TOO_FAR.value in far.codes


@pytest.mark.parametrize("status", ["resolved", "closed", "Cancelled"])
def test_inactive_incident_is_held(make_snapshot, now, status: str) -> None:
    assert (
        IssueCode.INACTIVE_INCIDENT.value
        in check_quality(make_snapshot(incident_status=status), now).codes
    )


def test_policy_is_configurable(make_snapshot, now) -> None:
    strict = QualityPolicy(min_source_confidence=0.99)
    assert IssueCode.LOW_CONFIDENCE.value in check_quality(make_snapshot(), now, strict).codes


def test_naive_clock_is_rejected(make_snapshot) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        check_quality(make_snapshot(), datetime(2026, 9, 1, 9, 5))


@pytest.mark.parametrize(
    "bad",
    [
        {"rainfall_mm_per_hour": -1},
        {"rainfall_mm_per_hour": float("nan")},
        {"river_level_m": float("inf")},
        {"latitude": 91},
        {"source_confidence": 1.2},
        {"observation_timestamp": "2026-09-01T09:00:00"},
        {"severity": "catastrophic"},
        {"unexpected_field": 1},
    ],
)
def test_invalid_input_is_rejected_at_the_boundary(bad: dict[str, object]) -> None:
    payload = {"city": "Northbridge", "event": "x", "severity": "low", **bad}
    with pytest.raises(ValidationError):
        WeatherSnapshot.model_validate(payload)


def test_snapshot_is_immutable(make_snapshot) -> None:
    snapshot = make_snapshot()
    with pytest.raises(ValidationError):
        snapshot.severity = "low"  # type: ignore[assignment]
