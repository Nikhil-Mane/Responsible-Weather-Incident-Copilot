"""Boundary and invariant tests for the deterministic rules.

Scenario IDs in test names refer to the labeled dataset (WX-xxx).
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from copilot.domain.models import Hazard, Severity, WeatherSnapshot
from copilot.domain.rules import assess, detect_hazards


def _weather(**overrides: object) -> WeatherSnapshot:
    """Plain builder for property tests (fixtures are not reset between generated inputs)."""
    base = {"city": "Northbridge", "event": "Test", "severity": "low"}
    return WeatherSnapshot.model_validate({**base, **overrides})


@pytest.mark.parametrize(
    ("rain", "expected"),
    [(24.9, False), (25.0, True), (25.1, True)],
    ids=["WX-003-below", "WX-004-at", "above"],
)
def test_rainfall_threshold_is_inclusive(make_weather, rain: float, expected: bool) -> None:
    result = assess(make_weather(rainfall_mm_per_hour=rain))
    assert result.flood_risk is expected


@pytest.mark.parametrize(("level", "expected"), [(4.4, False), (4.5, True), (4.6, True)])
def test_river_level_threshold_is_inclusive(make_weather, level: float, expected: bool) -> None:
    assert assess(make_weather(river_level_m=level)).flood_risk is expected


def test_active_flood_warning_alone_is_flood_risk(make_weather) -> None:
    result = assess(make_weather(flood_warning=True))
    assert result.flood_risk is True
    assert result.hazards == (Hazard.ACTIVE_FLOOD_WARNING,)
    assert result.requires_incident_command_review is False


@pytest.mark.parametrize(("rise", "review"), [(9.9, False), (10.0, True), (14.0, True)])
def test_rapid_rise_requires_review(make_weather, rise: float, review: bool) -> None:
    result = assess(make_weather(river_level_m=4.6, river_rise_cm_per_hour=rise))
    assert result.requires_incident_command_review is review


@pytest.mark.parametrize(
    ("overrides", "hazard", "present"),
    [
        ({"snowfall_rate_cm_per_hour": 2.9}, Hazard.HEAVY_SNOWFALL, False),
        ({"snowfall_rate_cm_per_hour": 3.0}, Hazard.HEAVY_SNOWFALL, True),
        ({"visibility_m": 301}, Hazard.LOW_VISIBILITY, False),
        ({"visibility_m": 300}, Hazard.LOW_VISIBILITY, True),
        ({"wind_gust_kmh": 59.9}, Hazard.HIGH_WIND, False),
        ({"wind_gust_kmh": 60}, Hazard.HIGH_WIND, True),
        ({"temperature_c": 34.9}, Hazard.EXTREME_HEAT, False),
        ({"temperature_c": 35}, Hazard.EXTREME_HEAT, True),
        ({"road_surface_temperature_c": 44.9}, Hazard.HOT_ROAD_SURFACE, False),
        ({"road_surface_temperature_c": 45}, Hazard.HOT_ROAD_SURFACE, True),
    ],
)
def test_weather_thresholds(make_weather, overrides, hazard: Hazard, present: bool) -> None:
    assert (hazard in detect_hazards(make_weather(**overrides))) is present


def test_missing_values_are_not_hazards(make_weather) -> None:
    assert detect_hazards(make_weather()) == ()


def test_clear_baseline_is_low_and_needs_no_review(make_weather) -> None:
    result = assess(make_weather(wind_gust_kmh=12, visibility_m=10_000))
    assert result.severity is Severity.LOW
    assert result.requires_incident_command_review is False


@pytest.mark.parametrize("severity", ["severe", "extreme"])
def test_severe_or_extreme_requires_review(make_weather, severity: str) -> None:
    result = assess(make_weather(severity=severity))
    assert result.requires_incident_command_review is True


def test_two_hazards_do_not_require_review_on_their_own(make_weather) -> None:
    """WX-020: air 39 and road 51 are two heat hazards and expect no approval."""
    result = assess(make_weather(severity="high", temperature_c=39, road_surface_temperature_c=51))
    assert len(result.hazards) == 2
    assert result.requires_incident_command_review is False


def test_three_hazards_require_review_even_if_source_severity_is_low(make_weather) -> None:
    result = assess(
        make_weather(
            severity="low",
            snowfall_rate_cm_per_hour=5,
            wind_gust_kmh=72,
            visibility_m=200,
        )
    )
    assert len(result.hazards) == 3
    assert result.severity is Severity.SEVERE
    assert result.requires_incident_command_review is True


def test_source_severity_is_never_downgraded(make_weather) -> None:
    assert assess(make_weather(severity="extreme")).severity is Severity.EXTREME


def test_trusted_flood_example_routes_to_review(make_snapshot) -> None:
    result = assess(make_snapshot())
    assert result.flood_risk is True
    assert Hazard.INTENSE_RAINFALL in result.hazards
    assert Hazard.RISING_RIVER_LEVEL in result.hazards
    assert result.requires_incident_command_review is True


@given(
    rain=st.floats(min_value=0, max_value=1_000, allow_nan=False),
    level=st.floats(min_value=0, max_value=100, allow_nan=False),
    warning=st.booleans(),
)
def test_flood_risk_matches_the_documented_rule(rain, level, warning) -> None:
    result = assess(_weather(rainfall_mm_per_hour=rain, river_level_m=level, flood_warning=warning))
    assert result.flood_risk is (rain >= 25.0 or level >= 4.5 or warning)


@given(
    snow=st.floats(min_value=0, max_value=100, allow_nan=False),
    gust=st.floats(min_value=0, max_value=300, allow_nan=False),
    vis=st.floats(min_value=0, max_value=20_000, allow_nan=False),
)
def test_severity_never_below_source_and_review_follows_hazard_count(snow, gust, vis) -> None:
    result = assess(
        _weather(
            severity="moderate",
            snowfall_rate_cm_per_hour=snow,
            wind_gust_kmh=gust,
            visibility_m=vis,
        )
    )
    assert result.severity.rank >= Severity.MODERATE.rank
    if len(result.hazards) >= 3:
        assert result.requires_incident_command_review is True
