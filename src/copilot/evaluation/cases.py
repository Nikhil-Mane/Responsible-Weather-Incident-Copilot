"""Evaluation cases: labeled scenarios, the supplied example and failure paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from copilot.core.exceptions import InputValidationError
from copilot.domain.models import ApprovalStatus, Route, WeatherSnapshot

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


EVAL_NOW = datetime(2026, 9, 1, 9, 5, tzinfo=UTC)


FLOOD, WINTER, HEAT = "PLAYBOOK-FLOOD-001", "PLAYBOOK-WINTER-002", "PLAYBOOK-HEAT-003"


COMMS, DATA = "PLAYBOOK-COMMS-004", "PLAYBOOK-DATA-005"


# Scenarios that sit exactly on, or one step below, a documented threshold.
BOUNDARY_IDS = frozenset(
    {
        "WX-003",  # rainfall 24 (just below 25)
        "WX-004",  # rainfall 25
        "WX-005",  # river level 4.5
        "WX-007",  # river rise 10
        "WX-010",  # snowfall 2.9 (just below 3)
        "WX-011",  # snowfall 3
        "WX-012",  # visibility 300
        "WX-015",  # gusts 60
        "WX-017",  # 34.9 / 44.9 (just below)
        "WX-018",  # air 35
        "WX-019",  # road surface 45
    }
)


_PROVENANCE: dict[str, Any] = {
    "observation_timestamp": (EVAL_NOW - timedelta(minutes=5)).isoformat(),
    "forecast_horizon_hours": 6,
    "source_id": "official-eval-harness",
    "source_confidence": 0.95,
    "latitude": 51.5074,
    "longitude": -0.1278,
    "geo_zone": "Eval-Zone-01",
    "incident_status": "active",
}


@dataclass(frozen=True, slots=True)
class Expected:
    """What a case should produce."""

    route: Route
    alert: bool
    approval: ApprovalStatus
    required_playbooks: frozenset[str]


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One evaluation case."""

    case_id: str
    name: str
    kind: str  # scenario | example | failure
    snapshot: WeatherSnapshot
    expected: Expected
    boundary: bool = False
    derived: bool = False


def _snapshot(base: Mapping[str, Any], **overrides: Any) -> WeatherSnapshot:
    payload = {**base, **overrides}
    return WeatherSnapshot.model_validate({k: v for k, v in payload.items() if v is not None})


_NONE: frozenset[str] = frozenset()


_FLOOD_ALERT = frozenset({FLOOD, COMMS})


# Golden citation labels. The supplied dataset labels only approval and alert, so the
# playbooks each briefing must cite are labeled here by hand (derived labels). Scenarios
# below every threshold cite nothing; wind has no approved playbook, so a severe wind event
# cites only the communications standard.
EXPECTED_PLAYBOOKS: dict[str, frozenset[str]] = {
    "WX-001": _NONE,  # clear baseline
    "WX-002": _NONE,  # moderate rain
    "WX-003": _NONE,  # rainfall 24, below the flood threshold
    "WX-004": _FLOOD_ALERT,  # rainfall 25
    "WX-005": _FLOOD_ALERT,  # river 4.5
    "WX-006": _FLOOD_ALERT,  # active warning
    "WX-007": _FLOOD_ALERT,  # rapid rise
    "WX-008": _FLOOD_ALERT,  # severe flood
    "WX-009": _FLOOD_ALERT,  # extreme flood
    "WX-010": _NONE,  # snowfall 2.9, below threshold
    "WX-011": frozenset({WINTER}),  # snowfall 3
    "WX-012": frozenset({WINTER}),  # visibility 300
    "WX-013": frozenset({WINTER, COMMS}),  # severe winter storm
    "WX-014": frozenset({WINTER, COMMS}),  # extreme blizzard
    "WX-015": _NONE,  # gusts 60: high, but no wind playbook exists
    "WX-016": frozenset({COMMS}),  # severe wind: review, no wind playbook
    "WX-017": _NONE,  # 34.9 / 44.9, below thresholds
    "WX-018": frozenset({HEAT}),  # air 35
    "WX-019": frozenset({HEAT}),  # road surface 45
    "WX-020": frozenset({HEAT}),  # heatwave advisory
    "WX-021": frozenset({HEAT, COMMS}),  # severe heat emergency
    "WX-022": frozenset({FLOOD, WINTER, COMMS}),  # snow and flood
    "WX-023": _FLOOD_ALERT,  # localized road flooding
    "WX-024": _NONE,  # cold clear night
}


def _required_playbooks(scenario_id: str) -> frozenset[str]:
    try:
        return EXPECTED_PLAYBOOKS[scenario_id]
    except KeyError:
        msg = f"no expected-citation label for scenario {scenario_id}"
        raise InputValidationError(msg) from None


def scenario_cases(scenarios: Sequence[Mapping[str, Any]]) -> list[EvalCase]:
    """Labeled scenarios, wrapped with derived trusted provenance."""
    cases: list[EvalCase] = []
    for scenario in scenarios:
        approval, alert = bool(scenario["expects_approval"]), bool(scenario["expects_local_alert"])
        base = {
            "city": "Northbridge",
            "event": scenario["name"],
            "severity": scenario["severity"],
            **scenario["fields"],
            **_PROVENANCE,
        }
        cases.append(
            EvalCase(
                case_id=scenario["id"],
                name=scenario["name"],
                kind="scenario",
                snapshot=_snapshot(base),
                expected=Expected(
                    route=Route.HUMAN_REVIEW if approval else Route.OPERATIONAL_BRIEFING,
                    alert=alert,
                    approval=(
                        ApprovalStatus.PENDING if approval or alert else ApprovalStatus.NOT_REQUIRED
                    ),
                    required_playbooks=_required_playbooks(scenario["id"]),
                ),
                boundary=scenario["id"] in BOUNDARY_IDS,
                derived=True,
            )
        )
    return cases


def example_case(flood: Mapping[str, Any]) -> EvalCase:
    """The supplied production-style flood example, unmodified."""
    return EvalCase(
        case_id="EX-FLOOD",
        name="supplied flood warning",
        kind="example",
        snapshot=_snapshot(flood),
        expected=Expected(
            Route.HUMAN_REVIEW, True, ApprovalStatus.PENDING, frozenset({FLOOD, COMMS})
        ),
    )


_HELD = Expected(Route.DATA_REVIEW, False, ApprovalStatus.HOLD_FOR_DATA_REVIEW, frozenset({DATA}))


def failure_cases(
    flood: Mapping[str, Any],
    supplied: Mapping[str, Mapping[str, Any]],
) -> list[EvalCase]:
    """Data-quality failure paths: corrupt the trusted flood example, plus supplied files."""
    stale = (EVAL_NOW - timedelta(hours=3)).isoformat()
    future = (EVAL_NOW + timedelta(hours=1)).isoformat()
    mutations: dict[str, dict[str, Any]] = {
        "stale_observation": {"observation_timestamp": stale},
        "future_timestamp": {"observation_timestamp": future},
        "low_confidence": {"source_confidence": 0.4},
        "missing_source": {"source_id": None},
        "untrusted_source": {"source_id": "social-media-post-81"},
        "missing_timestamp": {"observation_timestamp": None},
        "missing_confidence": {"source_confidence": None},
        "missing_geo": {"geo_zone": None, "latitude": None, "longitude": None},
        "forecast_too_far": {"forecast_horizon_hours": 96},
        "inactive_incident": {"incident_status": "resolved"},
        "missing_status": {"incident_status": None},
    }
    cases = [
        EvalCase(
            f"FP-{index:02d}",
            name,
            "failure",
            _snapshot({**flood, **change}),
            _HELD,
            derived=True,
        )
        for index, (name, change) in enumerate(mutations.items(), start=1)
    ]
    for offset, (name, record) in enumerate(supplied.items(), start=len(cases) + 1):
        cases.append(
            EvalCase(
                f"FP-{offset:02d}",
                f"supplied_{name}_without_provenance",
                "failure",
                _snapshot(record),
                _HELD,
            )
        )
    return cases


def load_cases(generated_dir: Path) -> list[EvalCase]:
    """Build every case from the extracted JSON files."""

    def read(name: str) -> Any:
        return json.loads((generated_dir / f"{name}.json").read_text(encoding="utf-8"))

    flood = read("flood_warning")
    supplied = {"heatwave": read("heatwave_advisory"), "winter_storm": read("winter_storm")}
    return [
        *scenario_cases(read("scenario_dataset")),
        example_case(flood),
        *failure_cases(flood, supplied),
    ]
