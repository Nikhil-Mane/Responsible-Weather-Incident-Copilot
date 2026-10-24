"""Workflow tests: routing, guardrails, human approval, checkpoint resume, RBAC."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from conftest import NOW
from copilot.core.exceptions import ApprovalError, AuthorizationError, ProviderError
from copilot.domain.models import ActionItem, ApprovalStatus, Route, WeatherSnapshot
from copilot.domain.rbac import Role
from copilot.drafting.base import DraftContext
from copilot.drafting.rule_based import RuleBasedDrafter
from copilot.workflow.graph import IncidentWorkflow, make_checkpointer
from copilot.workflow.state import Decision


@pytest.fixture
def workflow(real_retriever) -> IncidentWorkflow:
    return IncidentWorkflow(real_retriever, clock=lambda: NOW)


def _commander(approved: bool = True) -> Decision:
    return Decision(approved=approved, actor="J. Rivera", role=Role.INCIDENT_COMMANDER, note="ok")


# ----- flood: the full route -----------------------------------------------------------


def test_trusted_flood_pauses_for_the_commander(workflow, make_snapshot) -> None:
    result = workflow.run(make_snapshot())
    briefing = result.briefing

    assert result.awaiting_decision is True
    assert briefing.route is Route.HUMAN_REVIEW
    assert briefing.approval_status is ApprovalStatus.PENDING
    assert briefing.severity.value == "severe"
    assert "intense rainfall" in briefing.hazards
    assert "rising river level" in briefing.hazards
    assert briefing.local_team_alert is not None
    assert briefing.local_team_alert.startswith("DRAFT:")
    assert "incident-command direction" in briefing.local_team_alert
    assert "Riverside" in briefing.local_team_alert
    assert briefing.alert_recipients == ("response_team",)
    assert {"PLAYBOOK-FLOOD-001", "PLAYBOOK-COMMS-004"} <= set(briefing.citations)
    assert "official-river-gauge-northbridge-07" in briefing.citations
    assert "weather_snapshot" in briefing.citations


def test_flood_trace_records_every_step_in_order(workflow, make_snapshot) -> None:
    trace = workflow.run(make_snapshot()).briefing.trace
    assert trace == (
        "validated_input",
        "hazards_assessed",
        "retrieved_playbooks",
        "actions_drafted",
        "alert_drafted",
        "review_required",
    )


def test_every_action_cites_a_retrieved_playbook(workflow, make_snapshot) -> None:
    briefing = workflow.run(make_snapshot()).briefing
    assert briefing.actions
    assert "Inspect drainage hotspots" in briefing.actions
    for item in briefing.action_items:
        assert set(item.citations) <= set(briefing.citations)


def test_trace_uses_the_injected_clock(workflow, make_snapshot) -> None:
    detail = workflow.run(make_snapshot()).briefing.trace_detail
    assert {step.at for step in detail} == {NOW.isoformat()}


# ----- human approval ------------------------------------------------------------------


def test_commander_approval_resumes_the_run(workflow, make_snapshot) -> None:
    run_id = workflow.run(make_snapshot()).briefing.run_id
    result = workflow.decide(run_id, _commander(True))
    assert result.awaiting_decision is False
    assert result.briefing.approval_status is ApprovalStatus.APPROVED
    assert result.briefing.trace[-1] == "commander_approved"


def test_commander_can_hold(workflow, make_snapshot) -> None:
    run_id = workflow.run(make_snapshot()).briefing.run_id
    result = workflow.decide(run_id, _commander(False))
    assert result.briefing.approval_status is ApprovalStatus.HELD
    assert result.briefing.trace[-1] == "commander_held"


@pytest.mark.parametrize("role", [Role.OPERATOR, Role.RESPONSE_TEAM, Role.ASSURANCE_REVIEWER])
def test_only_the_commander_may_decide(workflow, make_snapshot, role: Role) -> None:
    run_id = workflow.run(make_snapshot()).briefing.run_id
    with pytest.raises(AuthorizationError):
        workflow.decide(run_id, Decision(approved=True, actor="someone", role=role))
    still = workflow.status(run_id)
    assert still.awaiting_decision is True
    assert still.briefing.approval_status is ApprovalStatus.PENDING


def test_a_decision_needs_a_named_approver(workflow, make_snapshot) -> None:
    run_id = workflow.run(make_snapshot()).briefing.run_id
    with pytest.raises(ApprovalError, match="named approver"):
        workflow.decide(run_id, Decision(True, "  ", Role.INCIDENT_COMMANDER))


def test_a_decision_cannot_be_applied_twice(workflow, make_snapshot) -> None:
    run_id = workflow.run(make_snapshot()).briefing.run_id
    workflow.decide(run_id, _commander())
    with pytest.raises(ApprovalError, match="not awaiting"):
        workflow.decide(run_id, _commander(False))


def test_unknown_run_is_reported(workflow) -> None:
    with pytest.raises(ApprovalError, match="unknown run"):
        workflow.status("does-not-exist")


def test_paused_run_survives_a_restart(real_retriever, make_snapshot, tmp_path: Path) -> None:
    db = tmp_path / "checkpoints.sqlite"
    first = IncidentWorkflow(real_retriever, checkpointer=make_checkpointer(db), clock=lambda: NOW)
    run_id = first.run(make_snapshot()).briefing.run_id
    del first

    second = IncidentWorkflow(real_retriever, checkpointer=make_checkpointer(db), clock=lambda: NOW)
    assert second.status(run_id).awaiting_decision is True
    resumed = second.decide(run_id, _commander())
    assert resumed.briefing.approval_status is ApprovalStatus.APPROVED


def test_the_workflow_has_no_way_to_send(workflow) -> None:
    assert not any(hasattr(workflow, name) for name in ("send", "notify", "dispatch"))


# ----- data quality --------------------------------------------------------------------


def test_stale_input_is_held_without_recommendations(workflow, make_snapshot) -> None:
    stale = make_snapshot(observation_timestamp=(NOW - timedelta(hours=3)).isoformat())
    result = workflow.run(stale)
    briefing = result.briefing

    assert result.awaiting_decision is False
    assert briefing.route is Route.DATA_REVIEW
    assert briefing.approval_status is ApprovalStatus.HOLD_FOR_DATA_REVIEW
    assert briefing.actions == ()
    assert briefing.local_team_alert is None
    assert briefing.hazards == ()
    assert "PLAYBOOK-DATA-005" in briefing.citations
    assert "PLAYBOOK-FLOOD-001" not in briefing.citations
    assert any("stale_observation" in reason for reason in briefing.review_reasons)
    assert briefing.trace == ("validated_input", "retrieved_playbooks", "data_review_required")


def test_low_confidence_flood_is_held(workflow, make_snapshot) -> None:
    result = workflow.run(make_snapshot(source_confidence=0.4))
    assert result.briefing.route is Route.DATA_REVIEW
    assert result.briefing.local_team_alert is None


def test_input_without_provenance_is_held(workflow, make_weather) -> None:
    """The supplied heatwave file carries no source, timestamp or confidence."""
    result = workflow.run(
        make_weather(severity="high", temperature_c=39, road_surface_temperature_c=51)
    )
    assert result.briefing.route is Route.DATA_REVIEW
    assert result.briefing.actions == ()


def test_supplied_heatwave_pdf_data_is_held(real_retriever, generated_dir) -> None:
    record = json.loads((generated_dir / "heatwave_advisory.json").read_text(encoding="utf-8"))
    workflow = IncidentWorkflow(real_retriever, clock=lambda: NOW)
    result = workflow.run(WeatherSnapshot.model_validate(record))
    assert result.briefing.route is Route.DATA_REVIEW


# ----- conditional routing -------------------------------------------------------------


def test_trusted_heat_is_an_operational_briefing(workflow, make_snapshot) -> None:
    heat = make_snapshot(
        event="Heatwave advisory",
        severity="high",
        temperature_c=39,
        road_surface_temperature_c=51,
        rainfall_mm_per_hour=0,
        river_level_m=2.1,
        flood_warning=False,
        river_rise_cm_per_hour=0,
        forecast_rainfall_mm=0,
        wind_gust_kmh=18,
        visibility_m=8000,
        affected_areas=["Central District", "Westside"],
        road_closures=[],
    )
    result = workflow.run(heat)
    briefing = result.briefing

    assert result.awaiting_decision is False
    assert briefing.route is Route.OPERATIONAL_BRIEFING
    assert briefing.approval_status is ApprovalStatus.NOT_REQUIRED
    assert briefing.local_team_alert is None
    assert "PLAYBOOK-HEAT-003" in briefing.citations
    assert "PLAYBOOK-COMMS-004" not in briefing.citations
    assert any("cooling-center" in a for a in briefing.actions)


def test_clear_weather_needs_no_playbooks(workflow, make_snapshot) -> None:
    clear = make_snapshot(
        event="Clear baseline",
        severity="low",
        rainfall_mm_per_hour=0,
        river_level_m=1.0,
        flood_warning=False,
        river_rise_cm_per_hour=0,
        wind_gust_kmh=12,
        visibility_m=10_000,
    )
    briefing = workflow.run(clear).briefing
    assert briefing.route is Route.OPERATIONAL_BRIEFING
    assert briefing.hazards == ()
    assert briefing.actions == ()
    assert briefing.citations == ("weather_snapshot", "official-river-gauge-northbridge-07")


def test_moderate_flood_drafts_an_alert_but_needs_no_review(workflow, make_snapshot) -> None:
    """WX-004: rainfall exactly 25 mm/h raises an alert but not an incident-command review."""
    briefing = workflow.run(
        make_snapshot(
            severity="high",
            rainfall_mm_per_hour=25,
            river_level_m=2.0,
            flood_warning=False,
            river_rise_cm_per_hour=0,
            forecast_rainfall_mm=0,
            affected_areas=["Riverside"],
        )
    ).briefing
    assert briefing.route is Route.OPERATIONAL_BRIEFING
    assert briefing.local_team_alert is not None
    assert briefing.approval_status is ApprovalStatus.PENDING


def test_severe_winter_reviews_without_a_flood_alert(workflow, make_snapshot) -> None:
    result = workflow.run(
        make_snapshot(
            event="Winter storm",
            severity="severe",
            snowfall_rate_cm_per_hour=5,
            wind_gust_kmh=72,
            visibility_m=200,
            rainfall_mm_per_hour=0,
            river_level_m=1.0,
            flood_warning=False,
            river_rise_cm_per_hour=0,
        )
    )
    assert result.briefing.route is Route.HUMAN_REVIEW
    assert result.briefing.local_team_alert is None
    assert "PLAYBOOK-WINTER-002" in result.briefing.citations


# ----- guardrails ----------------------------------------------------------------------


class _UncitedDrafter(RuleBasedDrafter):
    def draft_actions(self, context: DraftContext) -> list[ActionItem]:
        genuine = super().draft_actions(context)
        return [
            *genuine,
            ActionItem(text="Evacuate the district", citations=("PLAYBOOK-FAKE-999",)),
        ]


def test_actions_citing_unretrieved_sources_are_dropped(real_retriever, make_snapshot) -> None:
    workflow = IncidentWorkflow(real_retriever, drafter=_UncitedDrafter(), clock=lambda: NOW)
    briefing = workflow.run(make_snapshot()).briefing
    assert "Evacuate the district" not in briefing.actions
    step = next(s for s in briefing.trace_detail if s.step == "actions_drafted")
    assert step.detail["dropped_uncited"] == ["Evacuate the district"]


class _FailingDrafter(RuleBasedDrafter):
    def draft_actions(self, context: DraftContext) -> list[ActionItem]:
        msg = "service unavailable"
        raise ProviderError(msg)

    def draft_alert(self, context: DraftContext, actions) -> str:  # type: ignore[no-untyped-def]
        msg = "content filter"
        raise ProviderError(msg)


def test_provider_failure_falls_back_to_rules(real_retriever, make_snapshot) -> None:
    workflow = IncidentWorkflow(real_retriever, drafter=_FailingDrafter(), clock=lambda: NOW)
    briefing = workflow.run(make_snapshot()).briefing
    assert briefing.actions
    assert briefing.local_team_alert is not None
    statuses = {s.step: s.status for s in briefing.trace_detail}
    assert statuses["actions_drafted"] == "fallback"
    assert statuses["alert_drafted"] == "fallback"


class _SendingDrafter(RuleBasedDrafter):
    def draft_alert(self, context: DraftContext, actions) -> str:  # type: ignore[no-untyped-def]
        return "URGENT: Evacuate Riverside now. Sent to all residents."


def test_an_alert_that_is_not_a_draft_is_replaced(real_retriever, make_snapshot) -> None:
    workflow = IncidentWorkflow(real_retriever, drafter=_SendingDrafter(), clock=lambda: NOW)
    briefing = workflow.run(make_snapshot()).briefing
    assert briefing.local_team_alert is not None
    assert briefing.local_team_alert.startswith("DRAFT:")
    step = next(s for s in briefing.trace_detail if s.step == "alert_drafted")
    assert step.status == "fallback"
    assert step.detail["guardrail_problems"]


def test_missing_playbooks_force_human_review(real_retriever, make_snapshot) -> None:
    class Empty:
        playbooks = real_retriever.playbooks

        def search(self, query):  # type: ignore[no-untyped-def]
            return []

    workflow = IncidentWorkflow(Empty(), clock=lambda: NOW)  # type: ignore[arg-type]
    briefing = workflow.run(
        make_snapshot(
            severity="moderate",
            rainfall_mm_per_hour=25,
            river_level_m=2.0,
            flood_warning=False,
            river_rise_cm_per_hour=0,
        )
    ).briefing
    assert briefing.route is Route.HUMAN_REVIEW
    assert "no approved playbook matched the hazards" in briefing.review_reasons
    assert briefing.actions == ()


# ----- RBAC ----------------------------------------------------------------------------


def test_roles_without_run_permission_cannot_start_a_run(workflow, make_snapshot) -> None:
    with pytest.raises(AuthorizationError):
        workflow.run(make_snapshot(), role=Role.RESPONSE_TEAM)


def test_commander_can_run_and_sees_the_data_quality_standard(workflow, make_snapshot) -> None:
    stale = make_snapshot(source_confidence=0.1)
    briefing = workflow.run(stale, role=Role.INCIDENT_COMMANDER).briefing
    assert "PLAYBOOK-DATA-005" in briefing.citations


# ----- no matching playbook: warn, do not invent -------------------------------------------


def test_wind_only_has_no_playbook_so_it_warns_instead_of_inventing(
    workflow, make_snapshot
) -> None:
    """WX-015: gusts of 60 km/h are a hazard, but no approved wind playbook exists."""
    gusts = make_snapshot(
        event="High wind advisory",
        severity="high",
        wind_gust_kmh=60,
        rainfall_mm_per_hour=0,
        river_level_m=1.0,
        flood_warning=False,
        river_rise_cm_per_hour=0,
        visibility_m=1200,
    )
    briefing = workflow.run(gusts).briefing

    assert briefing.route is Route.OPERATIONAL_BRIEFING
    assert briefing.hazards == ("high wind",)
    assert briefing.actions == ()
    assert briefing.local_team_alert is None
    assert briefing.approval_status is ApprovalStatus.NOT_REQUIRED
    assert any(
        "no approved operational playbook matched: high wind" in w for w in briefing.warnings
    )
    assert not any(c.startswith("PLAYBOOK-") for c in briefing.citations)


def test_a_hot_road_surface_reaches_the_heat_playbook(workflow, make_snapshot) -> None:
    """WX-019: the hazard is worded differently from the playbook, but the topic matches."""
    surface = make_snapshot(
        event="heatwave_surface_threshold",
        severity="high",
        temperature_c=32,
        road_surface_temperature_c=45,
        rainfall_mm_per_hour=0,
        river_level_m=1.0,
        flood_warning=False,
        river_rise_cm_per_hour=0,
    )
    briefing = workflow.run(surface).briefing
    assert briefing.route is Route.OPERATIONAL_BRIEFING
    assert "PLAYBOOK-HEAT-003" in briefing.citations
    assert briefing.actions
    assert briefing.warnings == ()


def test_a_flood_alert_with_no_guidance_still_goes_to_a_person(
    real_retriever, make_snapshot
) -> None:
    class Empty:
        playbooks = real_retriever.playbooks

        def search(self, query):  # type: ignore[no-untyped-def]
            return []

    briefing = (
        IncidentWorkflow(Empty(), clock=lambda: NOW)
        .run(  # type: ignore[arg-type]
            make_snapshot(
                severity="moderate",
                rainfall_mm_per_hour=25,
                river_level_m=2.0,
                flood_warning=False,
                river_rise_cm_per_hour=0,
            )
        )
        .briefing
    )
    assert briefing.route is Route.HUMAN_REVIEW
    assert briefing.warnings
