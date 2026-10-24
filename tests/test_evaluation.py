"""The evaluation suite itself: case construction, scoring, gates and reporting."""

from __future__ import annotations

import json

import pytest

from copilot.core.exceptions import InputValidationError
from copilot.evaluation.cases import (
    BOUNDARY_IDS,
    EVAL_NOW,
    EXPECTED_PLAYBOOKS,
    load_cases,
    scenario_cases,
)
from copilot.evaluation.report import Floors, Usage, format_report, report_to_dict, summarize
from copilot.evaluation.runner import EvalOptions, run_eval
from copilot.workflow.graph import IncidentWorkflow


@pytest.fixture(scope="module")
def cases(generated_dir):
    return load_cases(generated_dir)


@pytest.fixture
def workflow(real_retriever) -> IncidentWorkflow:
    return IncidentWorkflow(real_retriever, clock=lambda: EVAL_NOW)


# ----- cases ------------------------------------------------------------------------------


def test_case_inventory(cases) -> None:
    kinds = [c.kind for c in cases]
    assert kinds.count("scenario") == 24
    assert kinds.count("example") == 1
    assert kinds.count("failure") == 13
    assert len({c.case_id for c in cases}) == len(cases)


def test_boundary_cases_are_marked(cases) -> None:
    marked = {c.case_id for c in cases if c.boundary}
    assert marked == set(BOUNDARY_IDS)
    assert len(marked) == 11


def test_every_scenario_has_an_expected_citation_label() -> None:
    assert sorted(EXPECTED_PLAYBOOKS) == [f"WX-{n:03d}" for n in range(1, 25)]


def test_a_scenario_without_a_label_is_rejected() -> None:
    scenario = {
        "id": "WX-999",
        "name": "x",
        "category": "flood",
        "severity": "low",
        "expects_approval": False,
        "expects_local_alert": False,
        "fields": {},
    }
    with pytest.raises(InputValidationError, match="WX-999"):
        scenario_cases([scenario])


def test_the_supplied_files_without_provenance_are_failure_cases(cases) -> None:
    names = {c.name for c in cases if c.kind == "failure"}
    assert "supplied_heatwave_without_provenance" in names
    assert "supplied_winter_storm_without_provenance" in names
    assert {"stale_observation", "low_confidence", "untrusted_source"} <= names


# ----- running ----------------------------------------------------------------------------


def test_the_deterministic_pipeline_passes_every_gate(workflow, real_retriever, cases) -> None:
    report = run_eval(workflow, real_retriever, cases)
    failed = [(r.case_id, r.problems) for r in report.results if not r.passed]

    assert failed == []
    assert report.passed
    metrics = report.metrics
    for name in (
        "route_accuracy",
        "alert_accuracy",
        "approval_accuracy",
        "citation_accuracy",
        "groundedness",
        "boundary_accuracy",
        "failure_path_accuracy",
        "retrieval_recall_at_3",
        "retrieval_mrr",
    ):
        assert metrics[name] == 1.0, name
    assert metrics["retrieval_precision_at_3"] >= 0.95
    assert metrics["action_relevance"] >= 0.95
    assert metrics["cases"] == 38


def test_a_retrieval_regression_fails_the_gates(real_retriever, cases) -> None:
    class Empty:
        playbooks = real_retriever.playbooks

        def search(self, query):  # type: ignore[no-untyped-def]
            return []

    broken = IncidentWorkflow(Empty(), clock=lambda: EVAL_NOW)  # type: ignore[arg-type]
    report = run_eval(broken, real_retriever, cases)
    assert not report.passed
    assert report.gates["citation_accuracy"] is False
    assert report.metrics["citation_accuracy"] < 1.0
    assert "FAIL" in format_report(report)
    assert any("citations" in problem for r in report.results for problem in r.problems)


def test_floors_can_be_relaxed_for_model_runs(real_retriever, cases) -> None:
    class Empty:
        playbooks = real_retriever.playbooks

        def search(self, query):  # type: ignore[no-untyped-def]
            return []

    broken = IncidentWorkflow(Empty(), clock=lambda: EVAL_NOW)  # type: ignore[arg-type]
    relaxed = EvalOptions(floors=Floors.uniform(0.0))
    assert run_eval(broken, real_retriever, cases, relaxed).passed


def test_model_usage_and_cost_are_reported(workflow, real_retriever, cases) -> None:
    class Tracker:
        def __init__(self) -> None:
            self.calls = 0

        def snapshot(self) -> tuple[int, int, int]:
            self.calls += 1
            return (0, 0, 0) if self.calls == 1 else (4, 2_000, 500)

    options = EvalOptions(usage_tracker=Tracker(), price_per_1k=(0.01, 0.03))  # type: ignore[arg-type]
    report = run_eval(workflow, real_retriever, cases[:3], options)
    assert report.usage == Usage(4, 2_000, 500, 0.035)


# ----- summary and reports ----------------------------------------------------------------


def test_an_empty_run_summarises_safely() -> None:
    report = summarize([], Usage())
    assert report.passed
    assert report.metrics["cases"] == 0
    assert report.metrics["latency_ms_max"] == 0.0


def test_floors_uniform_sets_every_gate() -> None:
    floors = Floors.uniform(0.9)
    assert (floors.route, floors.alert, floors.approval) == (0.9, 0.9, 0.9)
    assert (floors.citation, floors.grounded, floors.failure_path) == (0.9, 0.9, 0.9)


def test_the_report_is_json_serialisable(workflow, real_retriever, cases) -> None:
    report = run_eval(workflow, real_retriever, cases[:5])
    payload = json.loads(json.dumps(report_to_dict(report)))
    assert payload["passed"] is True
    assert len(payload["cases"]) == 5
    assert payload["metrics"]["route_accuracy"] == 1.0
    text = format_report(report)
    assert "route accuracy" in text
    assert text.rstrip().endswith("PASS")
