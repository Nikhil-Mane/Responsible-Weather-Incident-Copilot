"""Turns CrewAI events into Langfuse observations: crew -> task -> agent -> model call.

CrewAI emits a start and a finish event for every crew, task, agent execution and model call,
on other threads but with the caller's context, so the incident's run ID is still known.
Each start event opens an observation (nested under the one named by `parent_event_id`);
the matching finish event closes it with the output and token usage.

Everything here is failure-isolated: a problem while observing must never reach CrewAI.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from copilot.core.logging_config import get_logger, get_run_id

if TYPE_CHECKING:
    from collections.abc import Callable

_LOG = get_logger(__name__)
_MAX_TEXT = 8_000
_MAX_LABEL = 60
_NO_RUN = "-"

# event class name -> (kind, Langfuse observation type)
_STARTS: dict[str, tuple[str, str]] = {
    "CrewKickoffStartedEvent": ("crew", "chain"),
    "TaskStartedEvent": ("task", "chain"),
    "AgentExecutionStartedEvent": ("agent", "agent"),
    "LLMCallStartedEvent": ("llm", "generation"),
}
_FINISHES = (
    "CrewKickoffCompletedEvent",
    "TaskCompletedEvent",
    "AgentExecutionCompletedEvent",
    "LLMCallCompletedEvent",
)
_FAILURES = (
    "CrewKickoffFailedEvent",
    "TaskFailedEvent",
    "LLMCallFailedEvent",
    "AgentExecutionErrorEvent",
)


def _text(value: Any) -> Any:
    """Keep payloads small and JSON-friendly."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[:_MAX_TEXT]
    if isinstance(value, list | tuple):
        return [_text(item) for item in value][:50]
    if isinstance(value, dict):
        return {str(k): _text(v) for k, v in list(value.items())[:50]}
    raw = getattr(value, "raw", None)
    return _text(raw if raw is not None else str(value))


def usage_details(usage: Any) -> dict[str, int] | None:
    """Map CrewAI/OpenAI token counts to Langfuse's `input` / `output` / `total`."""
    if not isinstance(usage, dict):
        return None
    mapping = (
        ("input", ("prompt_tokens", "input_tokens")),
        ("output", ("completion_tokens", "output_tokens")),
        ("total", ("total_tokens",)),
    )
    found: dict[str, int] = {}
    for target, names in mapping:
        for name in names:
            value = usage.get(name)
            if isinstance(value, int):
                found[target] = value
                break
    return found or None


def _role(event: Any) -> str:
    role = getattr(event, "agent_role", None) or getattr(
        getattr(event, "agent", None), "role", None
    )
    return str(role) if role else "agent"


def _label(value: Any, default: str) -> str:
    """First line of a name, shortened: CrewAI falls back to the whole prompt as a task name."""
    text = str(value or "").strip().splitlines()
    return (text[0][:_MAX_LABEL] if text else "") or default


def _name(kind: str, event: Any) -> str:
    if kind == "crew":
        return f"crew:{_label(getattr(event, 'crew_name', None), 'crew')}"
    if kind == "task":
        task = getattr(event, "task", None)
        raw = getattr(event, "task_name", None) or getattr(task, "name", None)
        return f"task:{_label(raw, 'task')}"
    if kind == "agent":
        return f"agent:{_role(event)}"
    return f"llm:{_role(event)}"


class CrewAIBridge:
    """Opens and closes Langfuse observations as CrewAI events arrive."""

    def __init__(
        self,
        client: Any,
        trace_id_for: Callable[[str], str],
        attributes: Callable[[str], Any],
        clean: Callable[[Any], Any],
    ) -> None:
        self._client = client
        self._trace_id_for = trace_id_for
        self._attributes = attributes
        self._clean = clean
        self._open: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.attached = False

    # ----- handlers ----------------------------------------------------------------------

    def on_event(self, event: Any) -> None:
        """Dispatch one CrewAI event. Never raises."""
        try:
            name = type(event).__name__
            if name in _STARTS:
                self._start(event, *_STARTS[name])
            elif name in _FINISHES:
                self._finish(event, failed=False)
            elif name in _FAILURES:
                self._finish(event, failed=True)
        except Exception as exc:
            _LOG.warning("Langfuse CrewAI observation failed: %s", exc)

    def _start(self, event: Any, kind: str, as_type: str) -> None:
        run_id = get_run_id()
        if run_id == _NO_RUN:
            return  # not part of an incident run
        with self._lock:
            parent = self._open.get(str(getattr(event, "parent_event_id", "") or ""))
        fields: dict[str, Any] = {"name": _name(kind, event), "as_type": as_type}
        if kind == "llm":
            fields["model"] = getattr(event, "model", None)
            fields["input"] = self._clean(_text(getattr(event, "messages", None)))
        elif kind == "agent":
            fields["input"] = self._clean(_text(getattr(event, "task_prompt", None)))
        fields["metadata"] = {"run_id": run_id, "agent_role": _role(event), "crewai_kind": kind}
        with self._attributes(run_id):
            if parent is not None:
                span = parent.start_observation(**fields)
            else:
                trace = {"trace_id": self._trace_id_for(run_id)}
                span = self._client.start_observation(trace_context=trace, **fields)
        with self._lock:
            self._open[str(event.event_id)] = span

    def _finish(self, event: Any, *, failed: bool) -> None:
        with self._lock:
            span = self._open.pop(str(getattr(event, "started_event_id", "") or ""), None)
        if span is None:
            return
        update: dict[str, Any] = {}
        if failed:
            update["level"] = "ERROR"
            update["status_message"] = str(getattr(event, "error", "failed"))[:500]
        else:
            output = getattr(event, "response", None)
            if output is None:
                output = getattr(event, "output", None)
            update["output"] = self._clean(_text(output))
            tokens = usage_details(getattr(event, "usage", None))
            if tokens:
                update["usage_details"] = tokens
        span.update(**update)
        span.end()

    def close_open(self) -> None:
        """End any observation whose finish event never arrived (for example on a crash)."""
        with self._lock:
            leftovers, self._open = list(self._open.values()), {}
        for span in leftovers:
            try:
                span.update(level="WARNING", status_message="no finish event received")
                span.end()
            except Exception as exc:
                _LOG.warning("could not close a Langfuse observation: %s", exc)

    # ----- registration ------------------------------------------------------------------

    def attach(self, bus: Any | None = None) -> bool:
        """Register for CrewAI's events once. Returns False when CrewAI is not installed."""
        if self.attached:
            return True
        try:
            import crewai.events as crewai_events  # noqa: PLC0415 - optional dependency
        except ImportError:
            _LOG.info("CrewAI is not installed; agent calls are not traced")
            return False
        target = bus or crewai_events.crewai_event_bus
        names = (*_STARTS, *_FINISHES, *_FAILURES)
        for name in names:
            event_type = _find_event_class(name)
            if event_type is None:
                continue
            target.on(event_type)(self._handler)
        self.attached = True
        return True

    def _handler(self, source: Any, event: Any) -> None:
        self.on_event(event)


def _find_event_class(name: str) -> Any | None:
    """Look an event class up by name across CrewAI's event modules."""
    import importlib  # noqa: PLC0415 - optional dependency

    for module in ("agent_events", "crew_events", "llm_events", "task_events"):
        try:
            found = getattr(importlib.import_module(f"crewai.events.types.{module}"), name, None)
        except ImportError:
            continue
        if found is not None:
            return found
    return None
