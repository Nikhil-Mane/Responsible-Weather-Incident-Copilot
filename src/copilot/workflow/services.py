"""Application services shared by the CLI and the UI, so both behave identically.

Each function does one user-level action: run an incident, record the commander's decision
(and issue a token on approval), or perform the simulated send. They hold no state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from copilot.core.exceptions import ConfigurationError
from copilot.workflow.state import Decision

if TYPE_CHECKING:
    from copilot.data.store import RunStore
    from copilot.domain.models import WeatherSnapshot
    from copilot.domain.rbac import Role
    from copilot.safety.approvals import ApprovalService
    from copilot.safety.audit import AuditLog
    from copilot.safety.notify import SendReceipt, SimulatedNotifier
    from copilot.workflow.graph import IncidentWorkflow
    from copilot.workflow.state import RunResult


@dataclass(frozen=True, slots=True)
class DecisionOutcome:
    """Result of recording a commander's decision."""

    result: RunResult
    token: str | None = None
    token_ttl_seconds: int | None = None


def run_incident(  # noqa: PLR0913 - one call per user action; all arguments are required inputs
    workflow: IncidentWorkflow,
    store: RunStore,
    audit: AuditLog,
    snapshot: WeatherSnapshot,
    *,
    role: Role,
    run_id: str | None = None,
) -> RunResult:
    """Assess an incident, persist the briefing and write the audit entry."""
    result = workflow.run(snapshot, role=role, run_id=run_id)
    store.save(result.briefing)
    audit.append(
        "run_completed",
        run_id=result.briefing.run_id,
        actor=role.value,
        detail={
            "route": result.briefing.route.value,
            "severity": result.briefing.severity.value,
            "awaiting_decision": result.awaiting_decision,
        },
    )
    workflow.flush()
    return result


def decide_run(  # noqa: PLR0913 - one call per user action; all arguments are required inputs
    workflow: IncidentWorkflow,
    store: RunStore,
    audit: AuditLog,
    approvals: ApprovalService | None,
    run_id: str,
    *,
    actor: str,
    role: Role,
    approved: bool,
    note: str = "",
) -> DecisionOutcome:
    """Approve or hold a run. Approval also issues the expiring send token.

    Raises `AuthorizationError` unless the role may approve, and `ApprovalError` when the
    run is not in a state that can be approved.
    """
    current = workflow.status(run_id)
    token: str | None = None
    if approved:
        if approvals is None:  # fail before anything is recorded
            msg = "COPILOT_APPROVAL_SECRET must be set to approve and issue send tokens"
            raise ConfigurationError(msg)
        if current.awaiting_decision:
            current = workflow.decide(run_id, Decision(True, actor, role, note))
        token = approvals.issue(current.briefing, approver=actor, role=role)
    else:
        current = workflow.decide(run_id, Decision(False, actor, role, note))
    store.save(current.briefing)
    audit.append(
        "commander_approved" if approved else "commander_held",
        run_id=run_id,
        actor=actor,
        detail={"role": role.value, "note": note},
    )
    workflow.flush()
    ttl = int(approvals.ttl.total_seconds()) if approvals and token else None
    return DecisionOutcome(current, token, ttl)


def send_approved(
    workflow: IncidentWorkflow, notifier: SimulatedNotifier, run_id: str, token: str
) -> SendReceipt:
    """Perform the simulated send of an approved alert. Needs a valid token."""
    receipt = notifier.send(workflow.status(run_id).briefing, token)
    workflow.observer.record_send(run_id, asdict(receipt))
    return receipt
