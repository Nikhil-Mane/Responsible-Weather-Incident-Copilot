"""Wiring: build the workflow and its collaborators from settings."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from copilot.agents.crew.drafter import build_crew_drafter
from copilot.agents.factory import build_drafter
from copilot.core.exceptions import ConfigurationError
from copilot.core.logging_config import get_logger
from copilot.data.extract import extract_all
from copilot.data.store import RunStore
from copilot.domain.validation import QualityPolicy
from copilot.drafting.base import Drafter
from copilot.drafting.rule_based import RuleBasedDrafter
from copilot.evaluation.cases import EVAL_NOW
from copilot.llm.azure import build_embedder
from copilot.observability.langfuse import build_observer
from copilot.observability.observer import Observer
from copilot.retrieval.hybrid import HybridRetriever
from copilot.retrieval.playbooks import load_playbooks
from copilot.safety.approvals import ApprovalService
from copilot.safety.audit import AuditLog
from copilot.safety.notify import SimulatedNotifier
from copilot.workflow.graph import IncidentWorkflow, make_checkpointer

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from copilot.core.config import Settings

_LOG = get_logger(__name__)
_PLAYBOOKS_FILE = "operational_playbooks.json"


def ensure_generated(settings: Settings) -> None:
    """Create the derived JSON from the supplied PDFs when it is not there yet."""
    if (settings.generated_dir / _PLAYBOOKS_FILE).is_file():
        return
    _LOG.info("generated data missing; extracting from %s", settings.data_dir)
    try:
        extract_all(settings.data_dir, settings.generated_dir)
    except OSError as exc:
        msg = (
            f"cannot write the generated data to '{settings.generated_dir}': {exc}. "
            "Set COPILOT_GENERATED_DIR to a writable directory."
        )
        raise ConfigurationError(msg) from exc


def build_retriever(settings: Settings) -> HybridRetriever:
    """Load the approved playbooks. Uses Azure embeddings when configured, else TF-IDF."""
    ensure_generated(settings)
    return HybridRetriever(
        load_playbooks(settings.generated_dir / _PLAYBOOKS_FILE),
        embedder=build_embedder(settings),
    )


def select_drafter(settings: Settings, observer: Observer | None = None) -> Drafter:
    """Pick the drafter named by `COPILOT_DRAFTER`, failing clearly when it cannot run."""
    choice = settings.drafter
    callbacks = observer.llm_callbacks if observer else None
    if choice == "rules":
        return RuleBasedDrafter()
    if choice in {"lcel", "crew"} and not settings.azure_chat_enabled:
        msg = f"COPILOT_DRAFTER={choice} needs the AZURE_OPENAI_* settings"
        raise ConfigurationError(msg)
    if choice == "crew":
        crew = build_crew_drafter(settings)
        if crew is None:
            msg = "COPILOT_DRAFTER=crew needs the AZURE_OPENAI_* settings"
            raise ConfigurationError(msg)
        if observer:
            observer.instrument_agents()
        return crew
    # lcel, or auto: LCEL when configured, otherwise rules
    return build_drafter(settings, extra_callbacks=callbacks)


def build_workflow(
    settings: Settings, *, clock: Callable[[], datetime] | None = None
) -> IncidentWorkflow:
    """Create the workflow with durable SQLite checkpoints under the state directory."""
    observer = build_observer(settings)
    return IncidentWorkflow(
        build_retriever(settings),
        drafter=select_drafter(settings, observer),
        policy=QualityPolicy.from_settings(settings),
        checkpointer=make_checkpointer(settings.state_dir / "checkpoints.sqlite"),
        clock=clock,
        observer=observer,
    )


def build_eval_workflow(settings: Settings) -> tuple[IncidentWorkflow, HybridRetriever, Drafter]:
    """Workflow for evaluation: fixed clock, in-memory checkpoints, configured drafter."""
    retriever = build_retriever(settings)
    observer = build_observer(settings)
    drafter = select_drafter(settings, observer)
    workflow = IncidentWorkflow(
        retriever,
        drafter=drafter,
        policy=QualityPolicy.from_settings(settings),
        clock=lambda: EVAL_NOW,
        observer=observer,
    )
    return workflow, retriever, drafter


def build_store(settings: Settings) -> RunStore:
    """The store that keeps finished briefings for later review."""
    return RunStore(settings.state_dir / "runs")


def build_audit(settings: Settings, *, clock: Callable[[], datetime] | None = None) -> AuditLog:
    """The hash-chained audit log."""
    return AuditLog(settings.state_dir / "audit.jsonl", clock)


def build_approvals(
    settings: Settings, *, clock: Callable[[], datetime] | None = None
) -> ApprovalService:
    """Token service. Raises `ConfigurationError` when no approval secret is configured."""
    return ApprovalService(
        settings.approval_secret.get_secret_value(),
        ttl=timedelta(seconds=settings.approval_ttl_seconds),
        clock=clock,
    )


def build_notifier(
    settings: Settings, *, clock: Callable[[], datetime] | None = None
) -> SimulatedNotifier:
    """The simulated notification adapter (the only thing that can 'send')."""
    return SimulatedNotifier(
        settings.state_dir / "notifications",
        build_approvals(settings, clock=clock),
        build_audit(settings, clock=clock),
        clock,
    )
