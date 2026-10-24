"""Simulated notification adapter.

This is the notification boundary. It lives outside the agent workflow, requires a valid
approval token, is idempotent, and audits everything. It contains no network code: a
"send" only records that a message *would* have gone out. A real channel would implement
the same `send` contract behind the same checks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from copilot.core.exceptions import ApprovalError
from copilot.core.locking import FileLock
from copilot.domain.models import IncidentBriefing
from copilot.safety.approvals import ApprovalService, alert_hash

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from copilot.safety.audit import AuditLog

SIMULATION_NOTICE = "SIMULATED SEND: no external message was transmitted to anyone."


@dataclass(frozen=True, slots=True)
class SendReceipt:
    """Outcome of a send request."""

    run_id: str
    alert_hash: str
    approver: str
    recipients: tuple[str, ...]
    sent_at: str
    duplicate: bool
    simulated: bool = True
    notice: str = SIMULATION_NOTICE


class SimulatedNotifier:
    """Sends nothing, but enforces every rule a real sender would need."""

    def __init__(
        self,
        state_dir: Path,
        approvals: ApprovalService,
        audit: AuditLog,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._sent_file = state_dir / "sent.json"
        self._lock_file = state_dir / "sent.lock"
        self._approvals = approvals
        self._audit = audit
        self._clock = clock or (lambda: datetime.now(UTC))

    def send(self, briefing: IncidentBriefing, token: str) -> SendReceipt:
        """Record a simulated send of the approved draft alert, at most once per alert."""
        try:
            claims = self._approvals.verify(token, briefing)
        except ApprovalError as exc:
            self._audit.append("send_rejected", run_id=briefing.run_id, detail={"reason": str(exc)})
            raise

        alert = briefing.local_team_alert
        if alert is None:  # verify() already guarantees this; kept explicit, not an assert
            msg = "this run has no draft alert to send"
            raise ApprovalError(msg)
        key = f"{briefing.run_id}:{alert_hash(alert)}"
        with FileLock(self._lock_file):
            sent = self._load()
            if key in sent:
                original = SendReceipt(
                    **{**sent[key], "recipients": tuple(sent[key]["recipients"])}
                )
                duplicate = SendReceipt(**{**asdict(original), "duplicate": True})
                self._audit.append(
                    "send_duplicate_suppressed",
                    run_id=briefing.run_id,
                    actor=claims.approver,
                    detail={"idempotency_key": key},
                )
                return duplicate
            receipt = SendReceipt(
                run_id=briefing.run_id,
                alert_hash=claims.alert_hash,
                approver=claims.approver,
                recipients=briefing.alert_recipients,
                sent_at=self._clock().isoformat(),
                duplicate=False,
            )
            sent[key] = asdict(receipt)
            self._store(sent)
            self._audit.append(
                "alert_sent_simulated",
                run_id=briefing.run_id,
                actor=claims.approver,
                detail={
                    "idempotency_key": key,
                    "recipients": list(briefing.alert_recipients),
                    "simulated": True,
                },
            )
            return receipt

    def _load(self) -> dict[str, Any]:
        if not self._sent_file.is_file():
            return {}
        data = json.loads(self._sent_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def _store(self, sent: dict[str, Any]) -> None:
        self._sent_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._sent_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(sent, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self._sent_file)
