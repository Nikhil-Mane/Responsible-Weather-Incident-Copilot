"""Optional Langfuse tracing (the current, OpenTelemetry-based SDK, 5.x).

What is sent, all under one trace per incident (the trace ID is derived from the run ID):

* every workflow step (validation, hazards, retrieval, drafting, routing, the commander's decision),
* every CrewAI agent, task and model call, with prompts, outputs and token usage (`crewai_bridge`),
* every LangChain model call, through Langfuse's LangChain handler,
* the simulated send, and evaluation results as scores.

All calls are failure-isolated: Langfuse being down or misconfigured never affects a run.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from copilot.core.logging_config import get_logger, get_run_id, redact
from copilot.observability.crewai_bridge import CrewAIBridge
from copilot.observability.observer import NoopObserver, Observer

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from copilot.core.config import Settings
    from copilot.domain.models import IncidentBriefing
    from copilot.evaluation.report import Report

_LOG = get_logger(__name__)

_WARN_STATUSES = frozenset({"fallback", "no_evidence"})
_ERROR_STATUSES = frozenset({"failed", "held"})
_TRACE_NAME = "incident_run"

# Langfuse observation type for each workflow step (anything else is a plain span).
_STEP_TYPES = {
    "validated_input": "guardrail",
    "hazards_assessed": "evaluator",
    "retrieved_playbooks": "retriever",
    "actions_drafted": "chain",
    "alert_drafted": "chain",
}


def _clean(value: object) -> Any:
    """JSON-safe copy of a value with credential-looking strings redacted."""
    return json.loads(redact(json.dumps(value, default=str)))


def _propagate(**attributes: Any) -> Any:
    """Langfuse's trace-attribute context manager, imported lazily (the SDK is optional)."""
    from langfuse import propagate_attributes  # noqa: PLC0415 - optional dependency

    return propagate_attributes(**attributes)


class LangfuseObserver:
    """Sends runs, agents, model calls, sends and evaluations to Langfuse."""

    def __init__(
        self,
        client: Any,
        handler_factory: Callable[[str], Any] | None = None,
        propagate: Callable[..., Any] = _propagate,
    ) -> None:
        self._client = client
        self._handler_factory = handler_factory
        self._propagate = propagate
        self._bridge = CrewAIBridge(client, self._trace_id, self._attributes, _clean)

    # ----- helpers -----------------------------------------------------------------------

    def _trace_id(self, run_id: str) -> str:
        """Deterministic trace ID, so every part of a run lands in the same trace."""
        return str(self._client.create_trace_id(seed=run_id))

    def _attributes(self, run_id: str, **extra: Any) -> Any:
        """Trace-level attributes applied to the observations created inside the `with`."""
        return self._propagate(
            session_id=run_id, trace_name=_TRACE_NAME, metadata={"run_id": run_id}, **extra
        )

    # ----- the Observer contract -----------------------------------------------------------

    def record_run(self, briefing: IncidentBriefing, actor_role: str, *, since: int = 0) -> None:
        """One trace per run; one observation per workflow step from index `since` on."""
        try:
            self._record_run(briefing, actor_role, since)
        except Exception as exc:
            _LOG.warning("Langfuse record_run failed: %s", exc)

    def _record_run(self, briefing: IncidentBriefing, actor_role: str, since: int) -> None:
        run_id = briefing.run_id
        steps = briefing.trace_detail[since:]
        trace = {"trace_id": self._trace_id(run_id)}
        tags = [briefing.route.value, briefing.severity.value]
        with self._attributes(run_id, tags=tags):
            root = self._client.start_observation(
                trace_context=trace,
                name="incident_run" if since == 0 else "incident_decision",
                as_type="span",
                input={"run_id": run_id, "actor_role": actor_role},
                metadata={"rules_version": briefing.rules_version},
            )
            for step in steps:
                level = (
                    "ERROR"
                    if step.status in _ERROR_STATUSES
                    else "WARNING"
                    if step.status in _WARN_STATUSES
                    else "DEFAULT"
                )
                child = root.start_observation(
                    name=step.step,
                    as_type=_STEP_TYPES.get(step.step, "span"),
                    level=level,
                    status_message=step.status,
                    output=_clean(step.detail),
                    metadata={"at": step.at},
                )
                child.end()
            root.update(
                output=_clean(
                    {
                        "route": briefing.route.value,
                        "approval_status": briefing.approval_status.value,
                        "severity": briefing.severity.value,
                        "hazards": list(briefing.hazards),
                        "citations": list(briefing.citations),
                        "review_reasons": list(briefing.review_reasons),
                        "warnings": list(briefing.warnings),
                    }
                )
            )
            root.end()

    def record_send(self, run_id: str, receipt: Mapping[str, Any]) -> None:
        """The simulated send, as a tool observation. A suppressed duplicate is a warning."""
        try:
            duplicate = bool(receipt.get("duplicate"))
            with self._attributes(run_id):
                observation = self._client.start_observation(
                    trace_context={"trace_id": self._trace_id(run_id)},
                    name="simulated_send",
                    as_type="tool",
                    input={"run_id": run_id},
                    output=_clean(dict(receipt)),
                    level="WARNING" if duplicate else "DEFAULT",
                    status_message="duplicate suppressed" if duplicate else "sent (simulated)",
                )
                observation.end()
        except Exception as exc:
            _LOG.warning("Langfuse record_send failed: %s", exc)

    def record_eval(self, report: Report) -> None:
        """Aggregate metrics as scores on one evaluation trace, plus a score per case."""
        try:
            stamp = datetime.now(UTC).isoformat()
            summary_id = str(self._client.create_trace_id(seed=f"evaluation-{stamp}"))
            observation = self._client.start_observation(
                trace_context={"trace_id": summary_id},
                name="evaluation",
                as_type="span",
                output=_clean({"passed": report.passed, "gates": report.gates}),
            )
            observation.end()
            for name, value in report.metrics.items():
                self._client.create_score(
                    name=name, value=float(value), trace_id=summary_id, data_type="NUMERIC"
                )
            for result in report.results:
                self._client.create_score(
                    name="case_passed",
                    value=1.0 if result.passed else 0.0,
                    trace_id=self._trace_id(f"eval-{result.case_id}"),
                    data_type="NUMERIC",
                    comment="; ".join(result.problems) or None,
                )
        except Exception as exc:
            _LOG.warning("Langfuse record_eval failed: %s", exc)

    def llm_callbacks(self) -> list[Any]:
        """A LangChain handler that files model calls under the current run's trace."""
        if self._handler_factory is None:
            return []
        run_id = get_run_id()
        try:
            return [self._handler_factory(self._trace_id(run_id))]
        except Exception as exc:
            _LOG.warning("Langfuse LangChain handler unavailable: %s", exc)
            return []

    def instrument_agents(self) -> None:
        """Start tracing CrewAI agents, tasks and model calls."""
        try:
            self._bridge.attach()
        except Exception as exc:
            _LOG.warning("Langfuse could not attach to CrewAI: %s", exc)

    def flush(self) -> None:
        """Send buffered observations."""
        try:
            self._bridge.close_open()
            self._client.flush()
        except Exception as exc:
            _LOG.warning("Langfuse flush failed: %s", exc)


def _langchain_handler_factory(settings: Settings) -> Callable[[str], Any] | None:
    """Langfuse's LangChain handler, bound to a trace ID. None if it cannot be imported."""
    try:
        from langfuse.langchain import CallbackHandler  # noqa: PLC0415 - optional integration
    except ImportError:
        _LOG.info("Langfuse LangChain handler is not available; LangChain calls are not traced")
        return None
    public_key = settings.langfuse_public_key.get_secret_value()

    def make(trace_id: str) -> Any:
        return CallbackHandler(public_key=public_key, trace_context={"trace_id": trace_id})

    return make


def build_observer(settings: Settings) -> Observer:
    """Langfuse when configured and importable, otherwise the no-op observer."""
    if not settings.langfuse_enabled:
        return NoopObserver()
    try:
        from langfuse import Langfuse  # noqa: PLC0415 - optional dependency

        client = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            base_url=settings.langfuse_host or None,
        )
    except Exception as exc:
        _LOG.warning("Langfuse is configured but unavailable (%s); continuing without it", exc)
        return NoopObserver()
    return LangfuseObserver(client, _langchain_handler_factory(settings))
