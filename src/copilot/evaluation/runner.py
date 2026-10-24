"""Running cases through the workflow and scoring them."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from copilot.evaluation.cases import EvalCase
from copilot.evaluation.report import CaseResult, Floors, Report, Usage, summarize
from copilot.retrieval.metrics import precision_at_k, recall_at_k, reciprocal_rank
from copilot.safety.guardrails import enforce_grounding, known_facts_text

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from copilot.domain.models import IncidentBriefing
    from copilot.llm.usage import UsageTracker
    from copilot.retrieval.hybrid import HybridRetriever
    from copilot.workflow.graph import IncidentWorkflow

_TOP_K = 3


def _evaluate(
    case: EvalCase, briefing: IncidentBriefing, sources: Mapping[str, str], latency_ms: float
) -> CaseResult:
    expected = case.expected
    cited = [c for c in briefing.citations if c.startswith("PLAYBOOK-")]
    required = expected.required_playbooks

    route_ok = briefing.route is expected.route
    alert_ok = (briefing.local_team_alert is not None) is expected.alert and (
        briefing.local_team_alert is None or briefing.local_team_alert.startswith("DRAFT:")
    )
    approval_ok = briefing.approval_status is expected.approval
    citations_ok = required <= set(cited) if required else not cited
    grounded = enforce_grounding(
        briefing.action_items, sources, known_facts=known_facts_text(case.snapshot)
    )
    grounded_ok = not grounded.dropped and all(
        set(item.citations) <= set(cited) for item in briefing.action_items
    )
    if case.kind == "failure":
        grounded_ok = grounded_ok and not briefing.actions

    relevant = sum(1 for item in briefing.action_items if set(item.citations) & required)
    relevance = relevant / len(briefing.action_items) if briefing.action_items else 1.0

    problems = tuple(
        message
        for ok, message in (
            (route_ok, f"route {briefing.route.value}, expected {expected.route.value}"),
            (
                alert_ok,
                f"alert present={briefing.local_team_alert is not None}, expected {expected.alert}",
            ),
            (
                approval_ok,
                f"approval {briefing.approval_status.value}, expected {expected.approval.value}",
            ),
            (citations_ok, f"citations {sorted(cited)} do not match required {sorted(required)}"),
            (grounded_ok, "an action is not grounded in its cited playbook"),
        )
        if not ok
    )
    return CaseResult(
        case_id=case.case_id,
        name=case.name,
        kind=case.kind,
        boundary=case.boundary,
        derived=case.derived,
        expected_route=expected.route.value,
        actual_route=briefing.route.value,
        route_ok=route_ok,
        alert_ok=alert_ok,
        approval_ok=approval_ok,
        citations_ok=citations_ok,
        grounded_ok=grounded_ok,
        precision_at_3=precision_at_k(cited, required, _TOP_K) if required else 1.0,
        recall_at_3=recall_at_k(cited, required, _TOP_K),
        reciprocal_rank=reciprocal_rank(cited, required),
        action_relevance=relevance,
        latency_ms=latency_ms,
        problems=problems,
    )


@dataclass(frozen=True, slots=True)
class EvalOptions:
    """Optional settings for a run: gate floors and model-usage accounting."""

    floors: Floors = field(default_factory=Floors)
    usage_tracker: UsageTracker | None = None
    price_per_1k: tuple[float, float] = (0.0, 0.0)  # (input, output) price per 1,000 tokens


def run_eval(
    workflow: IncidentWorkflow,
    retriever: HybridRetriever,
    cases: Sequence[EvalCase],
    options: EvalOptions | None = None,
) -> Report:
    """Run every case through the workflow and score it."""
    options = options or EvalOptions()
    usage_tracker, price_per_1k = options.usage_tracker, options.price_per_1k
    sources = {p.document_id: f"{p.title}. {p.content}" for p in retriever.playbooks}
    start = usage_tracker.snapshot() if usage_tracker else (0, 0, 0)
    results: list[CaseResult] = []
    for case in cases:
        began = time.perf_counter()
        briefing = workflow.run(case.snapshot, run_id=f"eval-{case.case_id}").briefing
        latency_ms = (time.perf_counter() - began) * 1000
        results.append(_evaluate(case, briefing, sources, latency_ms))

    usage = Usage()
    if usage_tracker:
        calls, tokens_in, tokens_out = (
            now - before for now, before in zip(usage_tracker.snapshot(), start, strict=True)
        )
        cost = tokens_in / 1000 * price_per_1k[0] + tokens_out / 1000 * price_per_1k[1]
        usage = Usage(calls, tokens_in, tokens_out, round(cost, 6))
    return summarize(results, usage, options.floors)
