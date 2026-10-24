"""CrewAI runtime access (optional: install the `agents` extra).

CrewAI telemetry is switched off before import: incident data must not leave the process
except to the configured model provider. No agent gets a tool, so none can send, notify or
dispatch anything.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from copilot.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from copilot.core.config import Settings

_TELEMETRY_OFF = {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "CREWAI_DISABLE_TRACKING": "true",
    # An explicit "no" to CrewAI's execution-trace collection. Without it, the first run offers
    # to collect and share traces (which contain prompts) and prints a panel into stdout.
    "CREWAI_TRACING_ENABLED": "false",
    # Deliberately NOT OTEL_SDK_DISABLED: Langfuse's SDK is built on OpenTelemetry, so that flag
    # would silently turn Langfuse tracing off.
}
_MAX_ITERATIONS = 2


def load_crewai() -> Any:
    """Import CrewAI lazily, with telemetry disabled first."""
    for key, value in _TELEMETRY_OFF.items():
        os.environ.setdefault(key, value)
    try:
        import crewai  # noqa: PLC0415 - lazy and optional
    except ImportError as exc:
        msg = "CrewAI is not installed; run: pip install 'weather-incident-copilot[agents]'"
        raise ConfigurationError(msg) from exc
    return crewai


def flush_events(timeout: float = 10.0) -> None:
    """Wait for CrewAI's event handlers, which run on other threads, to finish.

    Observers listen to these events, so call this after a crew finishes and before the run is
    recorded; otherwise the last agent or model-call events could arrive late.
    """
    try:
        from crewai.events import crewai_event_bus  # noqa: PLC0415 - lazy and optional
    except ImportError:
        return
    crewai_event_bus.flush(timeout=timeout)


def build_crew_llm(settings: Settings) -> Any | None:
    """CrewAI LLM for the Azure deployment, or `None` when Azure is not configured."""
    if not settings.azure_chat_enabled:
        return None
    crewai = load_crewai()
    return crewai.LLM(
        model=f"azure/{settings.azure_deployment}",
        api_key=settings.azure_api_key.get_secret_value(),
        base_url=settings.azure_endpoint,
        api_version=settings.azure_api_version,
        temperature=0.1,
        timeout=settings.llm_timeout_seconds,
    )


def make_agent(crewai: Any, llm: Any, role: str, goal: str, backstory: str) -> Any:
    """Create an agent with no tools, no delegation and a small iteration budget."""
    return crewai.Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        tools=[],  # no tools: roles can reason and write, nothing else
        allow_delegation=False,
        verbose=False,
        max_iter=_MAX_ITERATIONS,
    )
