"""Typed contracts: the verified data boundary and the incident briefing."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

_Finite = {"allow_inf_nan": False}
Rate = Annotated[float, Field(ge=0, le=1_000, **_Finite)]
Celsius = Annotated[float, Field(ge=-90, le=70, **_Finite)]
Percent = Annotated[float, Field(ge=0, le=100, **_Finite)]
Probability = Annotated[float, Field(ge=0, le=1, **_Finite)]


class Severity(StrEnum):
    """Ordered incident severity."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"
    EXTREME = "extreme"

    @property
    def rank(self) -> int:
        """Numeric order, so severities can be compared and maximised."""
        return tuple(Severity).index(self)

    @classmethod
    def highest(cls, *values: Severity) -> Severity:
        """Return the most severe of the given values."""
        return max(values, key=lambda s: s.rank)


class Hazard(StrEnum):
    """Hazards the deterministic rules can raise. Values are operator-facing labels."""

    INTENSE_RAINFALL = "intense rainfall"
    RISING_RIVER_LEVEL = "rising river level"
    ACTIVE_FLOOD_WARNING = "active flood warning"
    RAPID_RIVER_RISE = "rapid river rise"
    HEAVY_SNOWFALL = "heavy snowfall"
    LOW_VISIBILITY = "low visibility"
    HIGH_WIND = "high wind"
    EXTREME_HEAT = "extreme heat"
    HOT_ROAD_SURFACE = "hot road surface"


class Route(StrEnum):
    """Where a case goes after assessment."""

    HUMAN_REVIEW = "human_review"
    OPERATIONAL_BRIEFING = "operational_briefing"
    DATA_REVIEW = "data_review"


class ApprovalStatus(StrEnum):
    """Approval state of any alert or dispatch draft."""

    NOT_REQUIRED = "not_required"
    PENDING = "pending_commander_approval"
    APPROVED = "approved"
    HELD = "held"
    HOLD_FOR_DATA_REVIEW = "hold_for_data_review"


class WeatherSnapshot(BaseModel):
    """The only input the rest of the system trusts.

    Fields are optional where the supplied data omits them, so that "missing" stays
    distinguishable from zero. Quality and provenance are judged by `validation`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    city: str = Field(min_length=1, max_length=100)
    event: str = Field(min_length=1, max_length=200)
    severity: Severity

    snowfall_rate_cm_per_hour: Rate | None = None
    wind_gust_kmh: Annotated[float, Field(ge=0, le=500, **_Finite)] | None = None
    temperature_c: Celsius | None = None
    road_surface_temperature_c: Celsius | None = None
    visibility_m: Annotated[float, Field(ge=0, le=100_000, **_Finite)] | None = None

    rainfall_mm_per_hour: Rate | None = None
    forecast_rainfall_mm: Annotated[float, Field(ge=0, le=5_000, **_Finite)] | None = None
    river_level_m: Annotated[float, Field(ge=0, le=100, **_Finite)] | None = None
    river_rise_cm_per_hour: Annotated[float, Field(ge=0, le=1_000, **_Finite)] | None = None
    flood_warning: bool | None = None

    affected_areas: tuple[str, ...] = ()
    road_closures: tuple[str, ...] = ()
    critical_assets: tuple[str, ...] = ()
    population_at_risk: Annotated[int, Field(ge=0)] | None = None
    drainage_capacity_percent: Percent | None = None

    observation_timestamp: AwareDatetime | None = None
    forecast_horizon_hours: Annotated[int, Field(ge=0, le=720)] | None = None
    source_id: str | None = Field(default=None, max_length=200)
    source_confidence: Probability | None = None
    latitude: Annotated[float, Field(ge=-90, le=90, **_Finite)] | None = None
    longitude: Annotated[float, Field(ge=-180, le=180, **_Finite)] | None = None
    geo_zone: str | None = Field(default=None, max_length=100)
    incident_status: str | None = Field(default=None, max_length=50)


class TraceStep(BaseModel):
    """One audited workflow step."""

    model_config = ConfigDict(frozen=True)

    step: str
    status: str = "ok"
    at: str = ""
    detail: dict[str, object] = Field(default_factory=dict)


class ActionItem(BaseModel):
    """A recommended readiness action. It must cite at least one approved playbook."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    citations: tuple[str, ...] = Field(min_length=1)


class IncidentBriefing(BaseModel):
    """Structured response: facts, evidence, recommendation and approval state."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    rules_version: str
    run_id: str
    severity: Severity
    hazards: tuple[str, ...]
    actions: tuple[str, ...]
    action_items: tuple[ActionItem, ...] = ()
    citations: tuple[str, ...]
    local_team_alert: str | None
    alert_recipients: tuple[str, ...] = ()
    route: Route
    approval_status: ApprovalStatus
    review_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    generated_at: str = ""
    trace: tuple[str, ...]
    trace_detail: tuple[TraceStep, ...] = ()
