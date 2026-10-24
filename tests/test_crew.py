"""CrewAI role orchestration: structure and safety, with no model calls."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

crewai = pytest.importorskip("crewai")

from conftest import NOW
from copilot.agents.crew.crews import build_comms_crew, build_planning_crew
from copilot.agents.crew.drafter import CrewDrafter, build_crew_drafter
from copilot.agents.crew.runtime import build_crew_llm
from copilot.agents.schemas import ActionPlan, AlertDraft, PlannedAction
from copilot.core.config import Settings
from copilot.core.exceptions import ProviderError
from copilot.domain.models import ActionItem
from copilot.domain.rules import assess
from copilot.drafting.base import DraftContext
from copilot.retrieval.query import build_query
from copilot.workflow.graph import IncidentWorkflow

AZURE = {
    "AZURE_OPENAI_API_KEY": "k-123456",
    "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
    "AZURE_OPENAI_API_VERSION": "2024-10-21",
    "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
}
GOOD_ALERT = (
    "DRAFT: River flood warning in Northbridge. Affected area: Riverside, Old Town. "
    "Readiness action: inspect drainage hotspots. Approval status: pending incident-commander "
    "approval; await incident-command direction."
)


def _no_sleep(_seconds: float) -> None:
    return None


@pytest.fixture
def llm() -> object:
    return build_crew_llm(Settings(_env_file=None, **AZURE))  # type: ignore[call-arg]


@pytest.fixture
def context(real_retriever, make_snapshot) -> DraftContext:
    snapshot = make_snapshot()
    result = assess(snapshot)
    return DraftContext(
        snapshot, result, tuple(real_retriever.search(build_query(result, snapshot.event)))
    )


def test_telemetry_is_disabled_before_crewai_runs(llm) -> None:
    assert os.environ["CREWAI_DISABLE_TELEMETRY"].lower() == "true"
    assert "OTEL_SDK_DISABLED" not in os.environ  # it would also disable Langfuse's OTel tracing
    assert os.environ["CREWAI_TRACING_ENABLED"].lower() == "false"


def test_crews_never_collect_execution_traces(llm, context) -> None:
    """CrewAI would otherwise offer to share traces (which contain prompts) on first run."""
    assert build_planning_crew(llm, context).tracing is False
    assert build_comms_crew(llm, context, []).tracing is False


def test_no_llm_without_azure_settings() -> None:
    assert build_crew_llm(Settings(_env_file=None)) is None  # type: ignore[call-arg]
    assert build_crew_drafter(Settings(_env_file=None)) is None  # type: ignore[call-arg]


def test_llm_targets_the_azure_deployment(llm) -> None:
    assert llm.model == "gpt-4o"


def test_planning_crew_has_scoped_roles_and_no_tools(llm, context) -> None:
    crew = build_planning_crew(llm, context)
    assert [a.role for a in crew.agents] == ["Hazard Analyst", "Response Planner"]
    assert crew.process == crewai.Process.sequential
    for agent in crew.agents:
        assert not agent.tools
        assert agent.allow_delegation is False
    assert crew.tasks[1].output_pydantic is ActionPlan
    assert crew.tasks[1].context == [crew.tasks[0]]
    assert "[PLAYBOOK-FLOOD-001]" in crew.tasks[1].description
    assert "untrusted data" in crew.tasks[0].description


def test_comms_crew_is_a_single_scoped_role(llm, context) -> None:
    crew = build_comms_crew(llm, context, [ActionItem(text="Inspect drainage", citations=("A",))])
    assert [a.role for a in crew.agents] == ["Comms Drafter"]
    assert not crew.agents[0].tools
    assert crew.tasks[0].output_pydantic is AlertDraft
    assert "Inspect drainage" in crew.tasks[0].description
    assert "DRAFT:" in crew.tasks[0].description


def test_crew_drafter_returns_validated_items(llm, context, monkeypatch) -> None:
    plan = ActionPlan(actions=[PlannedAction(text="Check drainage hotspots", citations=["P1"])])
    monkeypatch.setattr(
        crewai.Crew,
        "kickoff",
        lambda self, *a, **k: SimpleNamespace(
            pydantic=plan if len(self.tasks) == 2 else AlertDraft(text=GOOD_ALERT)
        ),
    )
    drafter = CrewDrafter(llm, retries=0, sleep=_no_sleep)
    items = drafter.draft_actions(context)
    assert items == [ActionItem(text="Check drainage hotspots", citations=("P1",))]
    assert drafter.draft_alert(context, items) == GOOD_ALERT


def test_result_can_come_from_the_last_task_output(llm, context, monkeypatch) -> None:
    plan = ActionPlan(actions=[PlannedAction(text="Check drainage hotspots", citations=["P1"])])
    monkeypatch.setattr(
        crewai.Crew,
        "kickoff",
        lambda self, *a, **k: SimpleNamespace(
            pydantic=None,
            tasks_output=[SimpleNamespace(pydantic=None), SimpleNamespace(pydantic=plan)],
        ),
    )
    assert CrewDrafter(llm, retries=0).draft_actions(context)[0].text == "Check drainage hotspots"


def test_unstructured_crew_output_is_a_provider_error(llm, context, monkeypatch) -> None:
    monkeypatch.setattr(
        crewai.Crew, "kickoff", lambda self, *a, **k: SimpleNamespace(pydantic=None)
    )
    with pytest.raises(ProviderError, match="did not return"):
        CrewDrafter(llm, retries=1, sleep=_no_sleep).draft_actions(context)


def test_crew_failures_are_mapped_and_redacted(llm, context, monkeypatch) -> None:
    def boom(self, *a, **k):  # type: ignore[no-untyped-def]
        msg = "upstream error api_key=sk-leak"
        raise RuntimeError(msg)

    monkeypatch.setattr(crewai.Crew, "kickoff", boom)
    with pytest.raises(ProviderError) as caught:
        CrewDrafter(llm, retries=0).draft_alert(context, [])
    assert "sk-leak" not in str(caught.value)


def test_workflow_applies_guardrails_to_crew_output(
    real_retriever, make_snapshot, llm, monkeypatch
) -> None:
    plan = ActionPlan(
        actions=[
            PlannedAction(text="Check drainage hotspots", citations=["PLAYBOOK-FLOOD-001"]),
            PlannedAction(text="Evacuate Old Town residents now", citations=["PLAYBOOK-FLOOD-001"]),
        ]
    )
    monkeypatch.setattr(
        crewai.Crew,
        "kickoff",
        lambda self, *a, **k: SimpleNamespace(
            pydantic=plan if len(self.tasks) == 2 else AlertDraft(text=GOOD_ALERT)
        ),
    )
    workflow = IncidentWorkflow(
        real_retriever, drafter=CrewDrafter(llm, retries=0), clock=lambda: NOW
    )
    briefing = workflow.run(make_snapshot()).briefing
    assert briefing.actions == ("Check drainage hotspots",)
    assert briefing.local_team_alert == GOOD_ALERT
