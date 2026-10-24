"""Observability: no-op default, Langfuse mapping, failure isolation and wiring."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import pytest

from conftest import NOW
from copilot.core.config import Settings
from copilot.domain.rbac import Role
from copilot.evaluation.cases import EVAL_NOW, load_cases
from copilot.evaluation.report import Usage, summarize
from copilot.evaluation.runner import run_eval
from copilot.observability.langfuse import LangfuseObserver, _clean, build_observer
from copilot.observability.observer import NoopObserver
from copilot.workflow.graph import IncidentWorkflow
from copilot.workflow.state import Decision


class FakeObservation:
    """Stands in for a Langfuse observation: records children, updates and the end call."""

    def __init__(self, fields: dict[str, Any], sink: list[FakeObservation]) -> None:
        self.fields = fields
        self.sink = sink
        self.children: list[FakeObservation] = []
        self.updates: list[dict[str, Any]] = []
        self.ended = False
        sink.append(self)

    def start_observation(self, **fields: Any) -> FakeObservation:
        child = FakeObservation(fields, self.sink)
        self.children.append(child)
        return child

    def update(self, **fields: Any) -> None:
        self.updates.append(fields)

    def end(self) -> None:
        self.ended = True


class FakeClient:
    """Stands in for the Langfuse SDK client."""

    def __init__(self) -> None:
        self.observations: list[FakeObservation] = []
        self.scores: list[dict[str, Any]] = []
        self.flushed = 0

    @staticmethod
    def create_trace_id(*, seed: str) -> str:
        return f"trace-{seed}"

    def start_observation(self, **fields: Any) -> FakeObservation:
        return FakeObservation(fields, self.observations)

    def create_score(self, **fields: Any) -> None:
        self.scores.append(fields)

    def flush(self) -> None:
        self.flushed += 1

    def roots(self, name: str | None = None) -> list[FakeObservation]:
        return [
            o
            for o in self.observations
            if "trace_context" in o.fields and (name is None or o.fields["name"] == name)
        ]


def _observer(client: Any, factory: Any = None) -> LangfuseObserver:
    return LangfuseObserver(client, factory, propagate=lambda **_: nullcontext())


def _commander() -> Decision:
    return Decision(True, "J. Rivera", Role.INCIDENT_COMMANDER)


def test_the_default_observer_does_nothing(real_retriever, make_snapshot) -> None:
    observer = NoopObserver()
    assert observer.llm_callbacks() == []
    observer.flush()
    workflow = IncidentWorkflow(real_retriever, clock=lambda: NOW)
    assert isinstance(workflow.observer, NoopObserver)
    assert workflow.run(make_snapshot()).briefing.trace


def test_observer_is_noop_without_langfuse_settings() -> None:
    assert isinstance(build_observer(Settings(_env_file=None)), NoopObserver)  # type: ignore[call-arg]


def test_a_configured_but_unreachable_langfuse_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    langfuse = pytest.importorskip("langfuse")

    def boom(*args: Any, **kwargs: Any) -> None:
        msg = "bad host"
        raise RuntimeError(msg)

    monkeypatch.setattr(langfuse, "Langfuse", boom)
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, LANGFUSE_PUBLIC_KEY="pk-lf-1", LANGFUSE_SECRET_KEY="sk-lf-1"
    )
    assert isinstance(build_observer(settings), NoopObserver)


def test_a_run_becomes_one_trace_with_an_observation_per_step(
    real_retriever, make_snapshot
) -> None:
    client = FakeClient()
    workflow = IncidentWorkflow(real_retriever, clock=lambda: NOW, observer=_observer(client))
    briefing = workflow.run(make_snapshot()).briefing

    (root,) = client.roots("incident_run")
    assert root.fields["trace_context"] == {"trace_id": f"trace-{briefing.run_id}"}
    assert [c.fields["name"] for c in root.children] == list(briefing.trace)
    assert all(c.fields["level"] == "DEFAULT" for c in root.children)
    assert all(c.ended for c in root.children) and root.ended
    types = {c.fields["name"]: c.fields["as_type"] for c in root.children}
    assert types["retrieved_playbooks"] == "retriever"
    assert types["validated_input"] == "guardrail"
    output = root.updates[-1]["output"]
    assert output["route"] == "human_review"
    assert "PLAYBOOK-FLOOD-001" in output["citations"]


def test_approval_sends_only_the_new_steps(real_retriever, make_snapshot) -> None:
    client = FakeClient()
    workflow = IncidentWorkflow(real_retriever, clock=lambda: NOW, observer=_observer(client))
    run_id = workflow.run(make_snapshot()).briefing.run_id
    workflow.decide(run_id, _commander())

    names = [c.fields["name"] for o in client.roots() for c in o.children]
    assert names.count("validated_input") == 1
    assert names[-1] == "commander_approved"
    assert [o.fields["name"] for o in client.roots()] == ["incident_run", "incident_decision"]
    assert len({o.fields["trace_context"]["trace_id"] for o in client.roots()}) == 1


def test_failures_and_fallbacks_are_flagged_by_level(real_retriever, make_snapshot) -> None:
    client = FakeClient()
    workflow = IncidentWorkflow(real_retriever, clock=lambda: NOW, observer=_observer(client))
    workflow.run(make_snapshot(source_confidence=0.2))
    levels = {c.fields["name"]: c.fields["level"] for c in client.roots()[0].children}
    assert levels["validated_input"] == "ERROR"
    assert levels["data_review_required"] == "ERROR"
    assert levels["retrieved_playbooks"] == "DEFAULT"


def test_the_simulated_send_is_recorded_and_a_duplicate_is_a_warning() -> None:
    client = FakeClient()
    observer = _observer(client)
    observer.record_send("r1", {"duplicate": False, "approver": "J. Rivera"})
    observer.record_send("r1", {"duplicate": True})

    first, second = client.roots("simulated_send")
    assert first.fields["as_type"] == "tool"
    assert first.fields["trace_context"] == {"trace_id": "trace-r1"}
    assert first.fields["level"] == "DEFAULT"
    assert second.fields["level"] == "WARNING"
    assert first.ended and second.ended


def test_secrets_in_step_details_are_redacted() -> None:
    cleaned = _clean({"note": "api_key=sk-secret-123", "token": "abc"})
    assert "sk-secret-123" not in str(cleaned)
    assert "abc" not in str(cleaned)


def test_a_failing_langfuse_never_breaks_a_run(real_retriever, make_snapshot) -> None:
    class Broken:
        def create_trace_id(self, **kwargs: Any) -> None:
            msg = "network down"
            raise ConnectionError(msg)

        start_observation = create_score = flush = create_trace_id

    observer = _observer(Broken())
    workflow = IncidentWorkflow(real_retriever, clock=lambda: NOW, observer=observer)
    briefing = workflow.run(make_snapshot()).briefing
    assert briefing.local_team_alert is not None
    workflow.flush()  # also swallowed
    observer.record_send("r1", {})
    observer.record_eval(summarize([], Usage()))
    observer.instrument_agents()


def test_flush_reaches_the_client(real_retriever) -> None:
    client = FakeClient()
    workflow = IncidentWorkflow(real_retriever, clock=lambda: NOW, observer=_observer(client))
    workflow.flush()
    assert client.flushed == 1


def test_evaluation_results_become_scores(real_retriever, generated_dir) -> None:
    client = FakeClient()
    observer = _observer(client)
    workflow = IncidentWorkflow(real_retriever, clock=lambda: EVAL_NOW, observer=observer)
    cases = load_cases(generated_dir)[:3]
    report = run_eval(workflow, real_retriever, cases)
    observer.record_eval(report)

    names = {s["name"] for s in client.scores}
    assert {"route_accuracy", "citation_accuracy", "latency_ms_mean", "case_passed"} <= names
    per_case = [s for s in client.scores if s["name"] == "case_passed"]
    assert len(per_case) == 3
    assert all(s["value"] == 1.0 for s in per_case)
    assert client.roots("evaluation")


def test_llm_callbacks_use_the_current_run_id() -> None:
    seen: list[str] = []

    def factory(trace_id: str) -> object:
        seen.append(trace_id)
        return object()

    assert len(_observer(FakeClient(), factory).llm_callbacks()) == 1
    assert _observer(FakeClient()).llm_callbacks() == []

    def broken(trace_id: str) -> object:
        msg = "no handler"
        raise RuntimeError(msg)

    assert _observer(FakeClient(), broken).llm_callbacks() == []
    assert seen == ["trace--"]


def test_instrument_agents_attaches_the_bridge_once() -> None:
    observer = _observer(FakeClient())
    observer.instrument_agents()
    observer.instrument_agents()  # idempotent, and a missing CrewAI is only logged
