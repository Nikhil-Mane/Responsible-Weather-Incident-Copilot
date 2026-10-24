"""Deterministic hazard thresholds and routing rules. Pure functions, no I/O.

Thresholds come from the case study and are cross-checked against the labeled
scenarios. Every boundary is inclusive (>=) except visibility (<=).
"""

from __future__ import annotations

from dataclasses import dataclass

from copilot.domain.models import Hazard, Severity, WeatherSnapshot

RULES_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class RuleConfig:
    """Versioned thresholds. Change values here and add a boundary test."""

    version: str = RULES_VERSION
    flood_rainfall_mm_per_hour: float = 25.0
    flood_river_level_m: float = 4.5
    rapid_rise_cm_per_hour: float = 10.0
    heavy_snowfall_cm_per_hour: float = 3.0
    low_visibility_m: float = 300.0
    high_wind_gust_kmh: float = 60.0
    extreme_heat_air_c: float = 35.0
    hot_road_surface_c: float = 45.0
    review_hazard_count: int = 3
    review_severities: frozenset[Severity] = frozenset({Severity.SEVERE, Severity.EXTREME})


DEFAULT_RULES = RuleConfig()


@dataclass(frozen=True, slots=True)
class Assessment:
    """Outcome of the deterministic rules for one snapshot."""

    hazards: tuple[Hazard, ...]
    flood_risk: bool
    rapid_rise: bool
    severity: Severity
    review_reasons: tuple[str, ...]

    @property
    def requires_incident_command_review(self) -> bool:
        """True when a human incident commander must review before any notification."""
        return bool(self.review_reasons)


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _at_most(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def detect_hazards(
    snapshot: WeatherSnapshot, rules: RuleConfig = DEFAULT_RULES
) -> tuple[Hazard, ...]:
    """Return the hazards present, in a stable order."""
    checks: tuple[tuple[Hazard, bool], ...] = (
        (
            Hazard.INTENSE_RAINFALL,
            _at_least(snapshot.rainfall_mm_per_hour, rules.flood_rainfall_mm_per_hour),
        ),
        (
            Hazard.RISING_RIVER_LEVEL,
            _at_least(snapshot.river_level_m, rules.flood_river_level_m),
        ),
        (Hazard.ACTIVE_FLOOD_WARNING, snapshot.flood_warning is True),
        (
            Hazard.RAPID_RIVER_RISE,
            _at_least(snapshot.river_rise_cm_per_hour, rules.rapid_rise_cm_per_hour),
        ),
        (
            Hazard.HEAVY_SNOWFALL,
            _at_least(snapshot.snowfall_rate_cm_per_hour, rules.heavy_snowfall_cm_per_hour),
        ),
        (Hazard.LOW_VISIBILITY, _at_most(snapshot.visibility_m, rules.low_visibility_m)),
        (Hazard.HIGH_WIND, _at_least(snapshot.wind_gust_kmh, rules.high_wind_gust_kmh)),
        (Hazard.EXTREME_HEAT, _at_least(snapshot.temperature_c, rules.extreme_heat_air_c)),
        (
            Hazard.HOT_ROAD_SURFACE,
            _at_least(snapshot.road_surface_temperature_c, rules.hot_road_surface_c),
        ),
    )
    return tuple(hazard for hazard, present in checks if present)


_FLOOD_HAZARDS = frozenset(
    {Hazard.INTENSE_RAINFALL, Hazard.RISING_RIVER_LEVEL, Hazard.ACTIVE_FLOOD_WARNING}
)


def assess(snapshot: WeatherSnapshot, rules: RuleConfig = DEFAULT_RULES) -> Assessment:
    """Apply the decision rules to a validated snapshot."""
    hazards = detect_hazards(snapshot, rules)
    flood_risk = any(h in _FLOOD_HAZARDS for h in hazards)
    rapid_rise = Hazard.RAPID_RIVER_RISE in hazards

    derived = Severity.LOW
    if hazards:
        derived = Severity.MODERATE
    if rapid_rise:
        derived = Severity.HIGH
    if len(hazards) >= rules.review_hazard_count:
        derived = Severity.SEVERE
    severity = Severity.highest(snapshot.severity, derived)

    reasons: list[str] = []
    if severity in rules.review_severities:
        reasons.append(f"severity is {severity.value}")
    if len(hazards) >= rules.review_hazard_count:
        reasons.append(f"{len(hazards)} concurrent hazards")
    if rapid_rise:
        reasons.append("rapid river rise")

    return Assessment(
        hazards=hazards,
        flood_risk=flood_risk,
        rapid_rise=rapid_rise,
        severity=severity,
        review_reasons=tuple(reasons),
    )
