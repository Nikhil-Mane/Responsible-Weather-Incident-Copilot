"""Data-quality gate. Pure functions: the clock and policy are injected."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING

from copilot.domain.models import WeatherSnapshot

if TYPE_CHECKING:
    from copilot.core.config import Settings

_FUTURE_SKEW = timedelta(minutes=5)


class IssueCode(StrEnum):
    """Machine-readable reasons a snapshot is held for data-quality review."""

    MISSING_SOURCE = "missing_source"
    UNTRUSTED_SOURCE = "untrusted_source"
    MISSING_TIMESTAMP = "missing_timestamp"
    STALE = "stale_observation"
    FUTURE_TIMESTAMP = "future_timestamp"
    MISSING_CONFIDENCE = "missing_confidence"
    LOW_CONFIDENCE = "low_confidence"
    MISSING_GEO = "missing_geo_context"
    FORECAST_TOO_FAR = "forecast_horizon_too_far"
    MISSING_STATUS = "missing_incident_status"
    INACTIVE_INCIDENT = "inactive_incident"


@dataclass(frozen=True, slots=True)
class QualityIssue:
    """One data-quality finding."""

    code: IssueCode
    message: str


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    """Thresholds for the quality gate (PLAYBOOK-DATA-005)."""

    max_observation_age: timedelta = timedelta(minutes=60)
    min_source_confidence: float = 0.7
    max_forecast_horizon_hours: int = 48
    trusted_source_prefixes: tuple[str, ...] = ("official-",)
    inactive_statuses: frozenset[str] = frozenset({"resolved", "closed", "cancelled", "drill"})

    @classmethod
    def from_settings(cls, settings: Settings) -> QualityPolicy:
        """Build a policy from application settings."""
        return cls(
            max_observation_age=timedelta(minutes=settings.max_observation_age_minutes),
            min_source_confidence=settings.min_source_confidence,
            max_forecast_horizon_hours=settings.max_forecast_horizon_hours,
        )


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Result of the quality gate. Any issue holds the case for data review."""

    issues: tuple[QualityIssue, ...]

    @property
    def passed(self) -> bool:
        """True when no issue was found."""
        return not self.issues

    @property
    def codes(self) -> tuple[str, ...]:
        """Issue codes, for traces and tests."""
        return tuple(issue.code.value for issue in self.issues)


def _source_issues(snapshot: WeatherSnapshot, policy: QualityPolicy) -> list[QualityIssue]:
    if not snapshot.source_id:
        return [QualityIssue(IssueCode.MISSING_SOURCE, "no source ID; data is not traceable")]
    if not snapshot.source_id.startswith(policy.trusted_source_prefixes):
        return [
            QualityIssue(
                IssueCode.UNTRUSTED_SOURCE,
                f"source '{snapshot.source_id}' is not an approved official source",
            )
        ]
    return []


def _timestamp_issues(
    snapshot: WeatherSnapshot, policy: QualityPolicy, now: datetime
) -> list[QualityIssue]:
    observed = snapshot.observation_timestamp
    if observed is None:
        return [QualityIssue(IssueCode.MISSING_TIMESTAMP, "no observation timestamp")]
    if observed > now + _FUTURE_SKEW:
        return [QualityIssue(IssueCode.FUTURE_TIMESTAMP, "observation timestamp is in the future")]
    age = now - observed
    if age > policy.max_observation_age:
        minutes = int(age.total_seconds() // 60)
        limit = int(policy.max_observation_age.total_seconds() // 60)
        return [QualityIssue(IssueCode.STALE, f"observation is {minutes} min old (limit {limit})")]
    return []


def _confidence_issues(snapshot: WeatherSnapshot, policy: QualityPolicy) -> list[QualityIssue]:
    confidence = snapshot.source_confidence
    if confidence is None:
        return [QualityIssue(IssueCode.MISSING_CONFIDENCE, "no source confidence")]
    if confidence < policy.min_source_confidence:
        return [
            QualityIssue(
                IssueCode.LOW_CONFIDENCE,
                f"confidence {confidence:.2f} is below {policy.min_source_confidence:.2f}",
            )
        ]
    return []


def _context_issues(snapshot: WeatherSnapshot, policy: QualityPolicy) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    has_coordinates = snapshot.latitude is not None and snapshot.longitude is not None
    if not snapshot.geo_zone and not has_coordinates:
        issues.append(QualityIssue(IssueCode.MISSING_GEO, "no geo zone or coordinates"))
    horizon = snapshot.forecast_horizon_hours
    if horizon is not None and horizon > policy.max_forecast_horizon_hours:
        issues.append(
            QualityIssue(
                IssueCode.FORECAST_TOO_FAR,
                f"forecast horizon {horizon} h exceeds {policy.max_forecast_horizon_hours} h",
            )
        )
    status = (snapshot.incident_status or "").lower()
    if not status:
        issues.append(QualityIssue(IssueCode.MISSING_STATUS, "no incident status"))
    elif status in policy.inactive_statuses:
        issues.append(
            QualityIssue(IssueCode.INACTIVE_INCIDENT, f"incident status is '{status}', not active")
        )
    return issues


def check_quality(
    snapshot: WeatherSnapshot,
    now: datetime,
    policy: QualityPolicy | None = None,
) -> QualityReport:
    """Judge whether a snapshot is trustworthy enough to drive a recommendation.

    `now` must be timezone-aware so freshness is unambiguous.
    """
    if now.tzinfo is None:
        msg = "`now` must be timezone-aware"
        raise ValueError(msg)
    active = policy or QualityPolicy()
    current = now.astimezone(UTC)
    issues = [
        *_source_issues(snapshot, active),
        *_timestamp_issues(snapshot, active, current),
        *_confidence_issues(snapshot, active),
        *_context_issues(snapshot, active),
    ]
    return QualityReport(issues=tuple(issues))
