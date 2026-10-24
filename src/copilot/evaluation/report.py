"""Evaluation results: per-case outcome, metrics, gates and reporting."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Outcome of one case."""

    case_id: str
    name: str
    kind: str
    boundary: bool
    derived: bool
    expected_route: str
    actual_route: str
    route_ok: bool
    alert_ok: bool
    approval_ok: bool
    citations_ok: bool
    grounded_ok: bool
    precision_at_3: float
    recall_at_3: float
    reciprocal_rank: float
    action_relevance: float
    latency_ms: float
    problems: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """True when every check passed."""
        return not self.problems


@dataclass(frozen=True, slots=True)
class Floors:
    """Minimum acceptable rate for each gated metric (1.0 = every case)."""

    route: float = 1.0
    alert: float = 1.0
    approval: float = 1.0
    citation: float = 1.0
    grounded: float = 1.0
    failure_path: float = 1.0

    @classmethod
    def uniform(cls, value: float) -> Floors:
        """The same floor for every metric."""
        return cls(value, value, value, value, value, value)


@dataclass(frozen=True, slots=True)
class Usage:
    """Model usage during the run."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0


@dataclass(frozen=True, slots=True)
class Report:
    """Aggregated evaluation results."""

    results: tuple[CaseResult, ...]
    metrics: dict[str, float]
    usage: Usage
    gates: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when every gate is met."""
        return all(self.gates.values())


def _rate(results: Sequence[CaseResult], check: str) -> float:
    return sum(1 for r in results if getattr(r, check)) / len(results) if results else 1.0


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def summarize(results: Sequence[CaseResult], usage: Usage, floors: Floors | None = None) -> Report:
    """Aggregate per-case results into metrics and pass/fail gates."""
    floors = floors or Floors()
    labeled = [r for r in results if r.kind in {"scenario", "example"}]
    failures = [r for r in results if r.kind == "failure"]
    boundary = [r for r in results if r.boundary]
    retrieval = [r for r in labeled if r.kind != "failure"]
    latencies = [r.latency_ms for r in results]

    def mean(values: Sequence[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    metrics = {
        "cases": float(len(results)),
        "route_accuracy": _rate(labeled, "route_ok"),
        "alert_accuracy": _rate(labeled, "alert_ok"),
        "approval_accuracy": _rate(labeled, "approval_ok"),
        "citation_accuracy": _rate(labeled, "citations_ok"),
        "groundedness": _rate(results, "grounded_ok"),
        "boundary_accuracy": (
            sum(1 for r in boundary if r.passed) / len(boundary) if boundary else 1.0
        ),
        "failure_path_accuracy": (
            sum(1 for r in failures if r.passed) / len(failures) if failures else 1.0
        ),
        "action_relevance": mean([r.action_relevance for r in labeled]),
        "retrieval_precision_at_3": mean([r.precision_at_3 for r in retrieval]),
        "retrieval_recall_at_3": mean([r.recall_at_3 for r in retrieval]),
        "retrieval_mrr": mean([r.reciprocal_rank for r in retrieval]),
        "latency_ms_mean": mean(latencies),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "latency_ms_max": max(latencies, default=0.0),
    }
    gates = {
        "route_accuracy": metrics["route_accuracy"] >= floors.route,
        "alert_accuracy": metrics["alert_accuracy"] >= floors.alert,
        "approval_accuracy": metrics["approval_accuracy"] >= floors.approval,
        "citation_accuracy": metrics["citation_accuracy"] >= floors.citation,
        "groundedness": metrics["groundedness"] >= floors.grounded,
        "failure_path_accuracy": metrics["failure_path_accuracy"] >= floors.failure_path,
    }
    return Report(results=tuple(results), metrics=metrics, usage=usage, gates=gates)


def report_to_dict(report: Report) -> dict[str, Any]:
    """JSON-friendly report."""
    return {
        "passed": report.passed,
        "gates": report.gates,
        "metrics": {k: round(v, 4) for k, v in report.metrics.items()},
        "usage": asdict(report.usage),
        "failures": [
            {"case": r.case_id, "name": r.name, "problems": list(r.problems)}
            for r in report.results
            if not r.passed
        ],
        "cases": [asdict(r) | {"passed": r.passed} for r in report.results],
    }


def format_report(report: Report) -> str:
    """Human-readable summary."""
    m = report.metrics
    lines = [
        "Weather Incident Copilot: evaluation",
        f"  cases                  {int(m['cases'])}"
        f"  (scenarios + example: {sum(r.kind != 'failure' for r in report.results)},"
        f" failure paths: {sum(r.kind == 'failure' for r in report.results)})",
        f"  route accuracy         {m['route_accuracy']:.1%}",
        f"  alert accuracy         {m['alert_accuracy']:.1%}",
        f"  approval accuracy      {m['approval_accuracy']:.1%}",
        f"  citation accuracy      {m['citation_accuracy']:.1%}",
        f"  groundedness           {m['groundedness']:.1%}",
        f"  boundary accuracy      {m['boundary_accuracy']:.1%}",
        f"  failure-path accuracy  {m['failure_path_accuracy']:.1%}",
        f"  action relevance       {m['action_relevance']:.1%}",
        f"  retrieval P@3 / R@3 / MRR  {m['retrieval_precision_at_3']:.2f} /"
        f" {m['retrieval_recall_at_3']:.2f} / {m['retrieval_mrr']:.2f}",
        f"  latency ms (mean / p95 / max)  {m['latency_ms_mean']:.1f} /"
        f" {m['latency_ms_p95']:.1f} / {m['latency_ms_max']:.1f}",
        f"  model calls {report.usage.calls}, tokens in/out"
        f" {report.usage.input_tokens}/{report.usage.output_tokens},"
        f" estimated cost {report.usage.estimated_cost:.4f}",
    ]
    failed = [r for r in report.results if not r.passed]
    if failed:
        lines.append("  FAILURES:")
        lines.extend(f"    {r.case_id} {r.name}: {'; '.join(r.problems)}" for r in failed)
    lines.append("  result: " + ("PASS" if report.passed else "FAIL"))
    return "\n".join(lines)
