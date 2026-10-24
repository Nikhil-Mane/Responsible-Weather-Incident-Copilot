"""LangGraph workflow: validate, assess, retrieve, draft, route, pause for a human."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from copilot.core.exceptions import ApprovalError, ProviderError
from copilot.core.logging_config import get_logger, reset_run_id, set_run_id
from copilot.domain.models import (
    ActionItem,
    ApprovalStatus,
    IncidentBriefing,
    Route,
    Severity,
    TraceStep,
    WeatherSnapshot,
)
from copilot.domain.rbac import (
    ALERT_RECIPIENT_ROLES,
    Permission,
    Role,
    allowed_playbook_ids,
    require,
)
from copilot.domain.rules import DEFAULT_RULES, RuleConfig, assess
from copilot.domain.validation import QualityPolicy, check_quality
from copilot.drafting.base import ACTION_TOPICS, DraftContext, Drafter
from copilot.drafting.rule_based import RuleBasedDrafter
from copilot.observability.observer import NoopObserver, Observer
from copilot.retrieval.hybrid import HybridRetriever, RetrievedPlaybook
from copilot.retrieval.query import build_query
from copilot.safety.guardrails import (
    alert_problems,
    enforce_action_citations,
    enforce_grounding,
    known_facts_text,
)
from copilot.workflow.state import Decision, RunResult, State

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    _Graph = CompiledStateGraph[Any, Any, Any, Any]

_LOG = get_logger(__name__)


_REVIEW_NODE = "human_review"


def make_checkpointer(path: Path | None) -> BaseCheckpointSaver[Any]:
    """In-memory checkpointing for tests, SQLite for durable pause and resume."""
    if path is None:
        return MemorySaver()
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


class IncidentWorkflow:
    """Builds and runs the incident graph. All collaborators are injected."""

    def __init__(  # noqa: PLR0913 - dependency-injection constructor; all but one are optional
        self,
        retriever: HybridRetriever,
        *,
        drafter: Drafter | None = None,
        policy: QualityPolicy | None = None,
        rules: RuleConfig = DEFAULT_RULES,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        clock: Callable[[], datetime] | None = None,
        observer: Observer | None = None,
    ) -> None:
        self._retriever = retriever
        self._drafter: Drafter = drafter or RuleBasedDrafter()
        self._fallback_drafter = RuleBasedDrafter()
        self._policy = policy or QualityPolicy()
        self._rules = rules
        self._clock = clock or (lambda: datetime.now(UTC))
        self._observer: Observer = observer or NoopObserver()
        self._graph: _Graph = self._build(checkpointer or MemorySaver())

    @property
    def observer(self) -> Observer:
        """The observer receiving finished runs."""
        return self._observer

    def flush(self) -> None:
        """Send any buffered observability data (call before a short-lived process exits)."""
        self._observer.flush()

    # ----- public API -----------------------------------------------------------------

    def run(
        self, snapshot: WeatherSnapshot, *, role: Role = Role.OPERATOR, run_id: str | None = None
    ) -> RunResult:
        """Assess one incident. Pauses at human review when the rules require it."""
        require(role, Permission.RUN_ASSESSMENT)
        identifier = run_id or uuid.uuid4().hex
        token = set_run_id(identifier)
        try:
            state: State = {
                "run_id": identifier,
                "role": role.value,
                "snapshot": snapshot.model_dump(mode="json"),
                "trace": [],
            }
            self._graph.invoke(state, self._config(identifier))
            result = self._result(identifier)
            self._observer.record_run(result.briefing, role.value)
            return result
        finally:
            reset_run_id(token)

    def status(self, run_id: str) -> RunResult:
        """Current briefing of an existing run."""
        return self._result(run_id)

    def decide(self, run_id: str, decision: Decision) -> RunResult:
        """Resume a paused run with a human decision. Only the incident commander may decide."""
        require(decision.role, Permission.APPROVE_ALERT)
        if not decision.actor.strip():
            msg = "a named approver is required"
            raise ApprovalError(msg)
        token = set_run_id(run_id)
        try:
            if _REVIEW_NODE not in self._graph.get_state(self._config(run_id)).next:
                msg = f"run '{run_id}' is not awaiting a decision"
                raise ApprovalError(msg)
            already = len(self._result(run_id).briefing.trace_detail)
            payload = {
                "approved": decision.approved,
                "actor": decision.actor,
                "role": decision.role.value,
                "note": decision.note,
            }
            self._graph.invoke(Command(resume=payload), self._config(run_id))
            result = self._result(run_id)
            self._observer.record_run(result.briefing, decision.role.value, since=already)
            return result
        finally:
            reset_run_id(token)

    # ----- graph construction -----------------------------------------------------------

    @staticmethod
    def _config(run_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": run_id}}

    def _build(self, checkpointer: BaseCheckpointSaver[Any]) -> _Graph:
        graph = StateGraph(State)
        graph.add_node("validate_input", self._validate_input)
        graph.add_node("hold_for_data_review", self._hold_for_data_review)
        graph.add_node("assess_hazards", self._assess_hazards)
        graph.add_node("retrieve_playbooks", self._retrieve_playbooks)
        graph.add_node("draft_actions", self._draft_actions)
        graph.add_node("draft_alert", self._draft_alert)
        graph.add_node("route_case", self._route_case)
        graph.add_node(_REVIEW_NODE, self._human_review)

        graph.add_edge(START, "validate_input")
        graph.add_conditional_edges(
            "validate_input",
            lambda s: "hold" if s.get("quality_issues") else "assess",
            {"hold": "hold_for_data_review", "assess": "assess_hazards"},
        )
        graph.add_edge("hold_for_data_review", END)
        graph.add_edge("assess_hazards", "retrieve_playbooks")
        graph.add_edge("retrieve_playbooks", "draft_actions")
        graph.add_conditional_edges(
            "draft_actions",
            lambda s: "alert" if s.get("flood_risk") else "route",
            {"alert": "draft_alert", "route": "route_case"},
        )
        graph.add_edge("draft_alert", "route_case")
        graph.add_conditional_edges(
            "route_case",
            lambda s: "review" if s["route"] == Route.HUMAN_REVIEW.value else "done",
            {"review": _REVIEW_NODE, "done": END},
        )
        graph.add_edge(_REVIEW_NODE, END)
        return graph.compile(checkpointer=checkpointer)

    # ----- nodes ------------------------------------------------------------------------

    def _step(self, name: str, status: str = "ok", **detail: object) -> dict[str, Any]:
        step = TraceStep(step=name, status=status, at=self._clock().isoformat(), detail=detail)
        return step.model_dump(mode="json")

    @staticmethod
    def _snapshot(state: State) -> WeatherSnapshot:
        return WeatherSnapshot.model_validate(state["snapshot"])

    def _allowed_ids(self, state: State) -> frozenset[str]:
        return allowed_playbook_ids(Role(state["role"]), self._retriever.playbooks)

    def _validate_input(self, state: State) -> State:
        snapshot = self._snapshot(state)
        report = check_quality(snapshot, self._clock(), self._policy)
        messages = [f"{i.code.value}: {i.message}" for i in report.issues]
        return {
            "quality_issues": messages,
            "trace": [
                self._step(
                    "validated_input",
                    "ok" if report.passed else "failed",
                    source_id=snapshot.source_id,
                    issues=list(report.codes),
                )
            ],
        }

    def _hold_for_data_review(self, state: State) -> State:
        snapshot = self._snapshot(state)
        query = build_query(
            assess(snapshot, self._rules),
            snapshot.event,
            quality_failed=True,
            allowed_ids=self._allowed_ids(state),
        )
        found = self._retriever.search(query)
        return {
            "hazards": [],
            "flood_risk": False,
            "severity": snapshot.severity.value,
            "review_reasons": list(state["quality_issues"]),
            "retrieved": [_retrieved_dict(r) for r in found],
            "action_items": [],
            "alert": None,
            "route": Route.DATA_REVIEW.value,
            "approval_status": ApprovalStatus.HOLD_FOR_DATA_REVIEW.value,
            "trace": [
                self._step("retrieved_playbooks", sources=[r.citation for r in found]),
                self._step("data_review_required", "held", reasons=state["quality_issues"]),
            ],
        }

    def _assess_hazards(self, state: State) -> State:
        result = assess(self._snapshot(state), self._rules)
        return {
            "hazards": [h.value for h in result.hazards],
            "flood_risk": result.flood_risk,
            "severity": result.severity.value,
            "review_reasons": list(result.review_reasons),
            "trace": [
                self._step(
                    "hazards_assessed",
                    severity=result.severity.value,
                    hazards=[h.value for h in result.hazards],
                    flood_risk=result.flood_risk,
                    rules_version=self._rules.version,
                )
            ],
        }

    def _retrieve_playbooks(self, state: State) -> State:
        snapshot = self._snapshot(state)
        result = assess(snapshot, self._rules)
        query = build_query(result, snapshot.event, allowed_ids=self._allowed_ids(state))
        found = self._retriever.search(query)
        reasons = list(state["review_reasons"])
        warnings = list(state.get("warnings", []))
        status = "ok"
        if result.hazards and not any(r.playbook.topic in ACTION_TOPICS for r in found):
            names = ", ".join(h.value for h in result.hazards)
            if result.flood_risk:
                # an alert will be drafted, so a person must see it has no guidance behind it
                reasons.append("no approved playbook matched the hazards")
            warnings.append(
                f"no approved operational playbook matched: {names}; no actions drafted"
            )
            status = "no_evidence"
        return {
            "retrieved": [_retrieved_dict(r) for r in found],
            "review_reasons": reasons,
            "warnings": warnings,
            "trace": [
                self._step(
                    "retrieved_playbooks",
                    status,
                    query=query.text,
                    topics=sorted(query.topics or ()),
                    sources=[r.citation for r in found],
                )
            ],
        }

    def _context(self, state: State) -> DraftContext:
        snapshot = self._snapshot(state)
        by_id = {p.document_id: p for p in self._retriever.playbooks}
        retrieved = tuple(
            RetrievedPlaybook(by_id[r["id"]], r["score"], r["rank"]) for r in state["retrieved"]
        )
        return DraftContext(snapshot, assess(snapshot, self._rules), retrieved)

    def _draft_actions(self, state: State) -> State:
        context = self._context(state)
        fallback_reason = ""
        try:
            drafted = self._drafter.draft_actions(context)
        except ProviderError as exc:
            fallback_reason = str(exc)
            _LOG.warning("drafter failed, using rule-based fallback: %s", exc)
            drafted = self._fallback_drafter.draft_actions(context)
        cited = enforce_action_citations(drafted, (r["id"] for r in state["retrieved"]))
        sources = {
            r.citation: r.playbook.title + ". " + r.playbook.content for r in context.playbooks
        }
        grounded = enforce_grounding(
            cited.kept, sources, known_facts=known_facts_text(context.snapshot)
        )
        reasons = list(state["review_reasons"])
        warnings = list(state.get("warnings", []))
        had_guidance = any(r.playbook.topic in ACTION_TOPICS for r in context.playbooks)
        if state["hazards"] and had_guidance and not grounded.kept:
            if state["flood_risk"]:
                reasons.append("no grounded actions could be drafted")
            warnings.append("no grounded actions could be drafted from the retrieved playbooks")
        return {
            "action_items": [a.model_dump(mode="json") for a in grounded.kept],
            "review_reasons": reasons,
            "warnings": warnings,
            "trace": [
                self._step(
                    "actions_drafted",
                    "fallback" if fallback_reason else "ok",
                    count=len(grounded.kept),
                    dropped_uncited=list(cited.dropped),
                    dropped_ungrounded=list(grounded.dropped),
                    fallback_reason=fallback_reason,
                )
            ],
        }

    def _draft_alert(self, state: State) -> State:
        context = self._context(state)
        actions = [ActionItem.model_validate(a) for a in state["action_items"]]
        fallback_reason = ""
        try:
            alert = self._drafter.draft_alert(context, actions)
        except ProviderError as exc:
            fallback_reason = str(exc)
            alert = self._fallback_drafter.draft_alert(context, actions)
        snapshot = context.snapshot
        problems = alert_problems(alert, event=snapshot.event, areas=snapshot.affected_areas)
        if problems:
            _LOG.warning("draft alert rejected by guardrail: %s", problems)
            alert = self._fallback_drafter.draft_alert(context, actions)
            fallback_reason = fallback_reason or "; ".join(problems)
        return {
            "alert": alert,
            "trace": [
                self._step(
                    "alert_drafted",
                    "fallback" if fallback_reason else "ok",
                    guardrail_problems=problems,
                    fallback_reason=fallback_reason,
                )
            ],
        }

    def _route_case(self, state: State) -> State:
        if state["review_reasons"]:
            route, approval, step = Route.HUMAN_REVIEW, ApprovalStatus.PENDING, "review_required"
        else:
            route = Route.OPERATIONAL_BRIEFING
            approval = ApprovalStatus.PENDING if state.get("alert") else ApprovalStatus.NOT_REQUIRED
            step = "briefing_ready"
        return {
            "route": route.value,
            "approval_status": approval.value,
            "trace": [self._step(step, reasons=state["review_reasons"], route=route.value)],
        }

    def _human_review(self, state: State) -> State:
        decision = interrupt(
            {
                "run_id": state["run_id"],
                "reasons": state["review_reasons"],
                "alert": state.get("alert"),
            }
        )
        approved = bool(decision["approved"])
        status = ApprovalStatus.APPROVED if approved else ApprovalStatus.HELD
        return {
            "approval_status": status.value,
            "trace": [
                self._step(
                    "commander_approved" if approved else "commander_held",
                    actor=decision["actor"],
                    role=decision["role"],
                    note=decision.get("note", ""),
                )
            ],
        }

    # ----- output -------------------------------------------------------------------------

    def _result(self, run_id: str) -> RunResult:
        snapshot = self._graph.get_state(self._config(run_id))
        if not snapshot.values:
            msg = f"unknown run '{run_id}'"
            raise ApprovalError(msg)
        state: State = snapshot.values  # type: ignore[assignment]
        return RunResult(
            briefing=self._briefing(state),
            awaiting_decision=_REVIEW_NODE in snapshot.next,
        )

    def _briefing(self, state: State) -> IncidentBriefing:
        snapshot = self._snapshot(state)
        items = tuple(ActionItem.model_validate(a) for a in state.get("action_items", []))
        steps = tuple(TraceStep.model_validate(t) for t in state["trace"])
        citations = ["weather_snapshot"]
        if snapshot.source_id:
            citations.append(snapshot.source_id)
        citations.extend(r["id"] for r in state.get("retrieved", []))
        alert = state.get("alert")
        return IncidentBriefing(
            rules_version=self._rules.version,
            run_id=state["run_id"],
            severity=Severity(state["severity"]),
            hazards=tuple(state.get("hazards", [])),
            actions=tuple(i.text for i in items),
            action_items=items,
            citations=tuple(dict.fromkeys(citations)),
            local_team_alert=alert,
            alert_recipients=tuple(r.value for r in ALERT_RECIPIENT_ROLES) if alert else (),
            route=Route(state["route"]),
            approval_status=ApprovalStatus(state["approval_status"]),
            review_reasons=tuple(state.get("review_reasons", [])),
            warnings=tuple(state.get("warnings", [])),
            generated_at=steps[-1].at if steps else "",
            trace=tuple(s.step for s in steps),
            trace_detail=steps,
        )


def _retrieved_dict(item: RetrievedPlaybook) -> dict[str, Any]:
    return {"id": item.citation, "score": item.score, "rank": item.rank}
