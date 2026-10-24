"""The CrewAI -> Langfuse bridge: nesting, outputs, token usage, failures and isolation."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from copilot.core.logging_config import reset_run_id, set_run_id
from copilot.observability.crewai_bridge import CrewAIBridge, _text, usage_details
from test_observability import FakeClient

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def bind_run_id(run_id: str) -> Iterator[None]:
    token = set_run_id(run_id)
    try:
        yield
    finally:
        reset_run_id(token)


def _event(kind: str, event_id: str, **fields: Any) -> Any:
    """An object whose class name matches a CrewAI event, as the bridge dispatches by name."""
    event_type = type(kind, (), {})
    event = event_type()
    event.event_id = event_id
    for key, value in fields.items():
        setattr(event, key, value)
    return event


@pytest.fixture
def bridge() -> tuple[CrewAIBridge, FakeClient]:
    client = FakeClient()
    made = CrewAIBridge(
        client, lambda run_id: f"trace-{run_id}", lambda run_id: nullcontext(), lambda v: v
    )
    return made, client


def test_events_outside_a_run_are_ignored(bridge) -> None:
    made, client = bridge
    made.on_event(_event("LLMCallStartedEvent", "e1", model="m", messages=[]))
    assert client.observations == []


def test_crew_task_agent_and_model_call_nest_and_close(bridge) -> None:
    made, client = bridge
    with bind_run_id("run-1"):
        made.on_event(_event("CrewKickoffStartedEvent", "c", crew_name="planning"))
        made.on_event(
            _event(
                "AgentExecutionStartedEvent",
                "a",
                parent_event_id="c",
                agent_role="Hazard Analyst",
                task_prompt="analyse",
            )
        )
        made.on_event(
            _event(
                "LLMCallStartedEvent",
                "l",
                parent_event_id="a",
                model="gpt",
                messages=[{"role": "user", "content": "hi"}],
            )
        )
        made.on_event(
            _event(
                "LLMCallCompletedEvent",
                "l2",
                started_event_id="l",
                response="done",
                usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            )
        )
        made.on_event(
            _event("AgentExecutionCompletedEvent", "a2", started_event_id="a", output="ok")
        )
        made.on_event(
            _event("CrewKickoffCompletedEvent", "c2", started_event_id="c", output="fine")
        )

    (crew,) = client.roots()
    assert crew.fields["name"] == "crew:planning"
    assert crew.fields["trace_context"] == {"trace_id": "trace-run-1"}
    (agent,) = crew.children
    assert agent.fields["name"] == "agent:Hazard Analyst"
    assert agent.fields["as_type"] == "agent"
    (call,) = agent.children
    assert call.fields["as_type"] == "generation"
    assert call.fields["model"] == "gpt"
    assert call.updates[-1]["output"] == "done"
    assert call.updates[-1]["usage_details"] == {"input": 10, "output": 4, "total": 14}
    assert crew.ended and agent.ended and call.ended


def test_a_failure_event_marks_the_observation_as_an_error(bridge) -> None:
    made, client = bridge
    with bind_run_id("run-2"):
        made.on_event(_event("LLMCallStartedEvent", "l", model="m", messages=[]))
        made.on_event(
            _event("LLMCallFailedEvent", "l2", started_event_id="l", error="rate limited")
        )
    (call,) = client.roots()
    assert call.updates[-1]["level"] == "ERROR"
    assert "rate limited" in call.updates[-1]["status_message"]
    assert call.ended


def test_observations_left_open_are_closed_with_a_warning(bridge) -> None:
    made, client = bridge
    with bind_run_id("run-3"):
        made.on_event(_event("TaskStartedEvent", "t", task_name="plan"))
    made.close_open()
    (task,) = client.roots()
    assert task.ended
    assert task.updates[-1]["level"] == "WARNING"
    made.close_open()  # nothing left, nothing happens


def test_a_broken_client_never_reaches_crewai(bridge) -> None:
    made, client = bridge

    def boom(**fields: Any) -> None:
        msg = "down"
        raise ConnectionError(msg)

    client.start_observation = boom  # type: ignore[method-assign]
    with bind_run_id("run-4"):
        made.on_event(_event("CrewKickoffStartedEvent", "c"))  # swallowed
    made.on_event(_event("SomethingElseEvent", "x"))  # unknown events are ignored
    made.on_event(_event("TaskCompletedEvent", "t2", started_event_id="never-started"))


def test_attach_registers_handlers_once() -> None:
    registered: list[Any] = []

    class Bus:
        def on(self, event_type: Any) -> Any:
            def register(handler: Any) -> Any:
                registered.append((event_type, handler))
                return handler

            return register

    made = CrewAIBridge(FakeClient(), str, lambda run_id: nullcontext(), lambda v: v)
    assert made.attach(Bus()) is True
    count = len(registered)
    assert count >= 4
    assert made.attach(Bus()) is True
    assert len(registered) == count


def test_the_registered_handler_forwards_events(bridge) -> None:
    made, client = bridge
    with bind_run_id("run-5"):
        made._handler(object(), _event("TaskStartedEvent", "t", task_name="plan"))
    assert client.roots()[0].fields["name"] == "task:plan"


def test_usage_details_maps_both_token_vocabularies() -> None:
    assert usage_details({"input_tokens": 3, "output_tokens": 2}) == {"input": 3, "output": 2}
    assert usage_details({"prompt_tokens": 1}) == {"input": 1}
    assert usage_details({}) is None
    assert usage_details("nope") is None


def test_text_keeps_payloads_small_and_json_friendly() -> None:
    assert len(_text("x" * 20_000)) == 8_000
    assert _text([1, "a", None]) == [1, "a", None]
    assert _text({"k": SimpleNamespace(raw="raw text")}) == {"k": "raw text"}
    assert isinstance(_text(object()), str)


def test_long_task_names_are_shortened_to_their_first_line(bridge) -> None:
    made, client = bridge
    with bind_run_id("run-6"):
        made.on_event(_event("TaskStartedEvent", "t", task_name="Write the alert.\n" + "x" * 500))
    assert client.roots()[0].fields["name"] == "task:Write the alert."
