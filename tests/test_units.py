"""Unit tests for ingest, rbac, store, guardrails and drafting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from copilot.core.exceptions import AuthorizationError, InputValidationError
from copilot.data.ingest import load_incident, parse_incident
from copilot.data.store import RunStore, validate_run_id
from copilot.domain.models import ActionItem, ApprovalStatus, IncidentBriefing, Route
from copilot.domain.rbac import (
    Permission,
    Role,
    allowed_playbook_ids,
    is_allowed,
    parse_role,
    require,
)
from copilot.domain.rules import assess
from copilot.drafting.base import DraftContext
from copilot.drafting.rule_based import RuleBasedDrafter, extract_actions
from copilot.retrieval.playbooks import Playbook
from copilot.safety.guardrails import alert_problems, enforce_action_citations

VALID = json.dumps({"city": "Northbridge", "event": "Test", "severity": "low"})


# ----- ingest -----------------------------------------------------------------------------


def test_parse_valid_incident() -> None:
    assert parse_incident(VALID).city == "Northbridge"
    assert parse_incident(VALID.encode()).event == "Test"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("not json", "not valid JSON"),
        ("[1, 2]", "must be a JSON object"),
        (
            '{"city": "x", "event": "y", "severity": "low", "rainfall_mm_per_hour": NaN}',
            "non-finite",
        ),
        ('{"city": "x", "event": "y", "severity": "low", "river_level_m": Infinity}', "non-finite"),
        ('{"city": "x"}', "failed validation"),
        ('{"city": "x", "event": "y", "severity": "low", "extra": 1}', "failed validation"),
    ],
)
def test_parse_rejects_bad_input(raw: str, message: str) -> None:
    with pytest.raises(InputValidationError, match=message):
        parse_incident(raw)


def test_parse_enforces_size_and_depth_limits() -> None:
    with pytest.raises(InputValidationError, match="limit"):
        parse_incident(VALID, max_bytes=10)
    deep = '{"a": ' * 20 + "1" + "}" * 20
    with pytest.raises(InputValidationError, match="nesting"):
        parse_incident(deep, max_depth=12)


def test_parse_survives_hostile_recursion() -> None:
    with pytest.raises(InputValidationError):
        parse_incident("[" * 100_000, max_bytes=1_000_000)


def test_load_incident_checks_the_file(tmp_path: Path) -> None:
    good = tmp_path / "ok.json"
    good.write_text(VALID, encoding="utf-8")
    assert load_incident(good).city == "Northbridge"
    with pytest.raises(InputValidationError, match="cannot read"):
        load_incident(tmp_path / "missing.json")
    with pytest.raises(InputValidationError, match="not a file"):
        load_incident(tmp_path)
    with pytest.raises(InputValidationError, match="larger than"):
        load_incident(good, max_bytes=5)


# ----- rbac -------------------------------------------------------------------------------


def test_only_the_commander_can_approve() -> None:
    assert [r for r in Role if is_allowed(r, Permission.APPROVE_ALERT)] == [Role.INCIDENT_COMMANDER]
    with pytest.raises(AuthorizationError, match="approve alert"):
        require(Role.OPERATOR, Permission.APPROVE_ALERT)
    require(Role.INCIDENT_COMMANDER, Permission.APPROVE_ALERT)


def test_reviewer_can_audit_but_not_run() -> None:
    assert is_allowed(Role.ASSURANCE_REVIEWER, Permission.VIEW_AUDIT)
    assert not is_allowed(Role.ASSURANCE_REVIEWER, Permission.RUN_ASSESSMENT)


def test_parse_role() -> None:
    assert parse_role(" Incident_Commander ") is Role.INCIDENT_COMMANDER
    with pytest.raises(AuthorizationError, match="unknown role"):
        parse_role("admin")


def test_response_team_cannot_read_the_data_quality_standard() -> None:
    books = [
        Playbook("PLAYBOOK-FLOOD-001", "F", (), "c", "flood"),
        Playbook("PLAYBOOK-DATA-005", "D", (), "c", "data_quality"),
    ]
    assert allowed_playbook_ids(Role.RESPONSE_TEAM, books) == {"PLAYBOOK-FLOOD-001"}
    assert allowed_playbook_ids(Role.INCIDENT_COMMANDER, books) == {
        "PLAYBOOK-FLOOD-001",
        "PLAYBOOK-DATA-005",
    }


# ----- store ------------------------------------------------------------------------------


def _briefing(run_id: str = "run-1") -> IncidentBriefing:
    return IncidentBriefing(
        rules_version="1.0.0",
        run_id=run_id,
        severity="low",
        hazards=(),
        actions=(),
        citations=("weather_snapshot",),
        local_team_alert=None,
        route=Route.OPERATIONAL_BRIEFING,
        approval_status=ApprovalStatus.NOT_REQUIRED,
        trace=("validated_input",),
    )


def test_store_round_trip_and_atomic_write(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    path = store.save(_briefing())
    assert store.load("run-1") == _briefing()
    assert path.name == "run-1.json"
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", "x" * 65, ".hidden", "a b"])
def test_store_rejects_unsafe_run_ids(tmp_path: Path, bad: str) -> None:
    with pytest.raises(InputValidationError):
        validate_run_id(bad)
    with pytest.raises(InputValidationError):
        RunStore(tmp_path).load(bad)


def test_store_reports_missing_and_corrupt_runs(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    with pytest.raises(InputValidationError, match="no stored run"):
        store.load("nope")
    (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
    with pytest.raises(InputValidationError, match="corrupt"):
        store.load("bad")


# ----- guardrails -------------------------------------------------------------------------


def test_action_citations_must_be_retrieved() -> None:
    good = ActionItem(text="Inspect drainage", citations=("A",))
    bad = ActionItem(text="Invented", citations=("A", "Z"))
    check = enforce_action_citations([good, bad], ["A", "B"])
    assert check.kept == (good,)
    assert check.dropped == ("Invented",)


def test_action_item_requires_a_citation() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ActionItem(text="x", citations=())


GOOD_ALERT = (
    "DRAFT: River flood warning in Northbridge. Affected area: Riverside. "
    "Readiness action: inspect drainage. Approval status: pending incident-commander "
    "approval; await incident-command direction."
)


def test_valid_alert_has_no_problems() -> None:
    assert alert_problems(GOOD_ALERT, event="River flood warning", areas=["Riverside"]) == []


@pytest.mark.parametrize(
    ("alert", "fragment"),
    [
        (GOOD_ALERT.replace("DRAFT:", "SENT:"), "must start with"),
        (GOOD_ALERT.replace("River flood warning", "Storm"), "name the event"),
        (GOOD_ALERT.replace("Riverside", "somewhere"), "affected area"),
        ("DRAFT: River flood warning in Northbridge. Riverside. Inspect.", "approval status"),
        (
            GOOD_ALERT.replace("Readiness action: inspect drainage", "Next: wait"),
            "readiness action",
        ),
    ],
)
def test_invalid_alerts_are_flagged(alert: str, fragment: str) -> None:
    problems = alert_problems(alert, event="River flood warning", areas=["Riverside"])
    assert any(fragment in p for p in problems)


# ----- drafting ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            "For verified flood risk, inspect drainage hotspots, verify flood equipment "
            "readiness, coordinate approved diversions, and prepare a local-team readiness "
            "alert. Public notifications require approval.",
            [
                "inspect drainage hotspots",
                "verify flood equipment readiness",
                "coordinate approved diversions",
                "prepare a local-team readiness alert",
            ],
        ),
        (
            "For heavy snow, low visibility, or icy roads, stage snow-clearing capability on "
            "priority routes, assess transport restrictions, and communicate travel risk "
            "through approved channels.",
            [
                "stage snow-clearing capability on priority routes",
                "assess transport restrictions",
                "communicate travel risk through approved channels",
            ],
        ),
        (
            "For extreme heat, prepare cooling-center readiness, coordinate welfare checks "
            "for vulnerable residents, and review hydration and work-rest controls for "
            "outdoor teams.",
            [
                "prepare cooling-center readiness",
                "coordinate welfare checks for vulnerable residents",
                "review hydration and work-rest controls for outdoor teams",
            ],
        ),
        ("Draft alerts must identify the event.", []),
    ],
)
def test_extract_actions(content: str, expected: list[str]) -> None:
    assert extract_actions(content) == expected


def test_rule_based_alert_without_actions_still_names_a_readiness_step(make_snapshot, now) -> None:
    snapshot = make_snapshot(affected_areas=[])
    alert = RuleBasedDrafter().draft_alert(DraftContext(snapshot, assess(snapshot), ()), [])
    assert "to be confirmed" in alert
    assert alert_problems(alert, event=snapshot.event, areas=[]) == []
