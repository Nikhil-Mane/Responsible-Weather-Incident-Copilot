"""Approval tokens, the simulated notifier, the audit chain and the file lock."""

from __future__ import annotations

import ast
import base64
import json
import os
import socket
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import copilot.safety.notify as notify_module
from conftest import NOW
from copilot.core.exceptions import (
    ApprovalError,
    AuditIntegrityError,
    AuthorizationError,
    ConfigurationError,
    LockTimeoutError,
)
from copilot.core.locking import FileLock
from copilot.domain.models import IncidentBriefing
from copilot.domain.rbac import Role
from copilot.safety.approvals import ApprovalService, alert_hash
from copilot.safety.audit import GENESIS, AuditLog
from copilot.safety.notify import SIMULATION_NOTICE, SimulatedNotifier
from copilot.workflow.graph import IncidentWorkflow
from copilot.workflow.state import Decision

SECRET = "s3cret-" * 6


class Clock:
    """A clock tests can move forward."""

    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def approvals(clock: Clock) -> ApprovalService:
    return ApprovalService(SECRET, ttl=timedelta(minutes=15), clock=clock)


@pytest.fixture
def workflow(real_retriever) -> IncidentWorkflow:
    return IncidentWorkflow(real_retriever, clock=lambda: NOW)


@pytest.fixture
def pending(workflow, make_snapshot) -> IncidentBriefing:
    return workflow.run(make_snapshot()).briefing


@pytest.fixture
def approved(workflow, pending) -> IncidentBriefing:
    decision = Decision(True, "J. Rivera", Role.INCIDENT_COMMANDER)
    return workflow.decide(pending.run_id, decision).briefing


@pytest.fixture
def audit(tmp_path: Path, clock: Clock) -> AuditLog:
    return AuditLog(tmp_path / "audit.jsonl", clock)


@pytest.fixture
def notifier(tmp_path: Path, approvals, audit, clock: Clock) -> SimulatedNotifier:
    return SimulatedNotifier(tmp_path / "notifications", approvals, audit, clock)


def _token(approvals: ApprovalService, briefing: IncidentBriefing) -> str:
    return approvals.issue(briefing, approver="J. Rivera", role=Role.INCIDENT_COMMANDER)


# ----- approval tokens -----------------------------------------------------------------


def test_only_a_commander_can_issue_a_token(approvals, approved) -> None:
    with pytest.raises(AuthorizationError):
        approvals.issue(approved, approver="Sam", role=Role.OPERATOR)


def test_an_approver_must_be_named(approvals, approved) -> None:
    with pytest.raises(ApprovalError, match="named approver"):
        approvals.issue(approved, approver=" ", role=Role.INCIDENT_COMMANDER)


def test_no_token_before_the_commander_decides(approvals, pending) -> None:
    with pytest.raises(ApprovalError, match="awaiting the commander"):
        _token(approvals, pending)


def test_no_token_after_a_hold(approvals, workflow, pending) -> None:
    held = workflow.decide(pending.run_id, Decision(False, "J. Rivera", Role.INCIDENT_COMMANDER))
    with pytest.raises(ApprovalError, match="held this run"):
        _token(approvals, held.briefing)


def test_no_token_for_data_review(approvals, workflow, make_snapshot) -> None:
    held = workflow.run(make_snapshot(source_confidence=0.2)).briefing
    with pytest.raises(ApprovalError):
        _token(approvals, held)


def test_no_token_when_there_is_no_alert(approvals, workflow, make_snapshot) -> None:
    heat = workflow.run(
        make_snapshot(
            severity="high",
            temperature_c=39,
            rainfall_mm_per_hour=0,
            river_level_m=2.0,
            flood_warning=False,
            river_rise_cm_per_hour=0,
        )
    ).briefing
    with pytest.raises(ApprovalError, match="no draft alert"):
        _token(approvals, heat)


def test_an_alert_without_a_review_can_be_approved_directly(
    approvals, workflow, make_snapshot
) -> None:
    moderate = workflow.run(
        make_snapshot(
            severity="high",
            rainfall_mm_per_hour=25,
            river_level_m=2.0,
            flood_warning=False,
            river_rise_cm_per_hour=0,
        )
    ).briefing
    claims = approvals.verify(_token(approvals, moderate), moderate)
    assert claims.approver == "J. Rivera"


def test_a_valid_token_verifies(approvals, approved) -> None:
    claims = approvals.verify(_token(approvals, approved), approved)
    assert claims.run_id == approved.run_id
    assert claims.alert_hash == alert_hash(approved.local_team_alert or "")
    assert claims.expires_at == NOW + timedelta(minutes=15)


def test_tokens_expire(approvals, approved, clock: Clock) -> None:
    token = _token(approvals, approved)
    clock.advance(minutes=15, seconds=1)
    with pytest.raises(ApprovalError, match="expired"):
        approvals.verify(token, approved)


def test_a_token_is_bound_to_its_run(approvals, approved) -> None:
    token = _token(approvals, approved)
    other = approved.model_copy(update={"run_id": "another-run"})
    with pytest.raises(ApprovalError, match="different run"):
        approvals.verify(token, other)


def test_editing_the_alert_after_approval_invalidates_the_token(approvals, approved) -> None:
    token = _token(approvals, approved)
    edited = approved.model_copy(update={"local_team_alert": "DRAFT: something else entirely"})
    with pytest.raises(ApprovalError, match="alert changed"):
        approvals.verify(token, edited)


def test_a_run_without_an_alert_cannot_be_verified(approvals, approved) -> None:
    token = _token(approvals, approved)
    with pytest.raises(ApprovalError, match="no draft alert"):
        approvals.verify(token, approved.model_copy(update={"local_team_alert": None}))


@pytest.mark.parametrize("bad", ["", "abc", "a.b", "a.b.c", "....", "e30.e30"])
def test_malformed_tokens_are_rejected(approvals, approved, bad: str) -> None:
    with pytest.raises(ApprovalError):
        approvals.verify(bad, approved)


def test_a_forged_signature_is_rejected(approvals, approved) -> None:
    payload, _signature = _token(approvals, approved).split(".")
    with pytest.raises(ApprovalError, match="signature"):
        approvals.verify(f"{payload}.AAAA", approved)


def test_a_token_from_another_secret_is_rejected(clock: Clock, approvals, approved) -> None:
    other = ApprovalService("a-completely-different-secret", clock=clock)
    with pytest.raises(ApprovalError, match="signature"):
        approvals.verify(_token(other, approved), approved)


def test_a_signed_token_from_a_non_commander_is_rejected(approvals, approved) -> None:
    claims = {
        "run_id": approved.run_id,
        "alert_hash": alert_hash(approved.local_team_alert or ""),
        "approver": "Sam",
        "role": "operator",
        "nonce": "n",
        "exp": int((NOW + timedelta(minutes=5)).timestamp()),
    }
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    with pytest.raises(ApprovalError, match="not issued by an incident commander"):
        approvals.verify(f"{payload}.{approvals._sign(payload)}", approved)


def test_a_missing_or_short_secret_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="at least 16"):
        ApprovalService("")
    with pytest.raises(ConfigurationError):
        ApprovalService("short")


# ----- simulated notifier ----------------------------------------------------------------


def test_an_approved_alert_is_sent_once_in_simulation(
    notifier, approvals, approved, tmp_path: Path, audit: AuditLog
) -> None:
    receipt = notifier.send(approved, _token(approvals, approved))

    assert receipt.duplicate is False
    assert receipt.simulated is True
    assert receipt.notice == SIMULATION_NOTICE
    assert receipt.recipients == ("response_team",)
    assert receipt.approver == "J. Rivera"
    events = [e["event"] for e in audit.entries()]
    assert events == ["alert_sent_simulated"]
    assert audit.verify() == 1


def test_sending_twice_is_idempotent(notifier, approvals, approved, audit: AuditLog) -> None:
    token = _token(approvals, approved)
    first = notifier.send(approved, token)
    second = notifier.send(approved, token)
    assert second.duplicate is True
    assert second.sent_at == first.sent_at
    assert [e["event"] for e in audit.entries()] == [
        "alert_sent_simulated",
        "send_duplicate_suppressed",
    ]


def test_a_second_approval_does_not_send_again(notifier, approvals, approved) -> None:
    notifier.send(approved, _token(approvals, approved))
    again = notifier.send(approved, _token(approvals, approved))
    assert again.duplicate is True


def test_an_invalid_token_sends_nothing_and_is_audited(
    notifier, approved, tmp_path: Path, audit: AuditLog
) -> None:
    with pytest.raises(ApprovalError):
        notifier.send(approved, "not-a-token")
    assert not (tmp_path / "notifications" / "sent.json").exists()
    assert [e["event"] for e in audit.entries()] == ["send_rejected"]


def test_an_expired_token_sends_nothing(notifier, approvals, approved, clock: Clock) -> None:
    token = _token(approvals, approved)
    clock.advance(hours=1)
    with pytest.raises(ApprovalError, match="expired"):
        notifier.send(approved, token)


def test_sending_works_with_the_network_disabled(
    notifier, approvals, approved, monkeypatch: pytest.MonkeyPatch
) -> None:
    def blocked(*_args: object, **_kwargs: object) -> None:
        msg = "network access attempted"
        raise AssertionError(msg)

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    assert notifier.send(approved, _token(approvals, approved)).simulated is True


def test_the_notifier_contains_no_network_code() -> None:
    source = Path(notify_module.__file__).read_text(encoding="utf-8")
    for forbidden in ("import socket", "requests", "httpx", "smtplib", "urllib", "http.client"):
        assert forbidden not in source


def test_concurrent_approvals_send_exactly_once(notifier, approvals, approved) -> None:
    token = _token(approvals, approved)
    receipts: list[bool] = []
    errors: list[BaseException] = []

    def attempt() -> None:
        try:
            receipts.append(notifier.send(approved, token).duplicate)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert receipts.count(False) == 1
    assert receipts.count(True) == 7


# ----- audit chain -----------------------------------------------------------------------


def test_audit_chain_links_entries(audit: AuditLog) -> None:
    first = audit.append("a", run_id="r1")
    second = audit.append("b", run_id="r1", actor="J. Rivera", detail={"k": 1})
    assert first["prev_hash"] == GENESIS
    assert second["prev_hash"] == first["hash"]
    assert audit.verify() == 2


def test_audit_detects_edited_content(audit: AuditLog, tmp_path: Path) -> None:
    audit.append("a", run_id="r1")
    audit.append("b", run_id="r1")
    path = tmp_path / "audit.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace('"event": "a"', '"event": "x"'))
    with pytest.raises(AuditIntegrityError, match="does not match its hash"):
        audit.verify()


def test_audit_detects_deleted_entries(audit: AuditLog, tmp_path: Path) -> None:
    for name in ("a", "b", "c"):
        audit.append(name, run_id="r1")
    path = tmp_path / "audit.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    with pytest.raises(AuditIntegrityError):
        audit.verify()


def test_audit_rejects_garbage_lines(audit: AuditLog, tmp_path: Path) -> None:
    audit.append("a", run_id="r1")
    with (tmp_path / "audit.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
    with pytest.raises(AuditIntegrityError, match="not valid JSON"):
        audit.verify()


def test_audit_redacts_secrets_in_details(audit: AuditLog) -> None:
    entry = audit.append("x", run_id="r", detail={"token": "abc123", "note": "api_key=zzz"})
    text = json.dumps(entry)
    assert "abc123" not in text
    assert "zzz" not in text


def test_an_empty_audit_log_verifies(audit: AuditLog) -> None:
    assert audit.verify() == 0
    assert audit.entries() == []


# ----- file lock -------------------------------------------------------------------------


def test_lock_is_exclusive_and_released(tmp_path: Path) -> None:
    path = tmp_path / "x.lock"
    with FileLock(path):
        assert path.exists()
        with pytest.raises(LockTimeoutError), FileLock(path, timeout=0.1):
            pass
    assert not path.exists()


def test_the_lock_survives_heavy_contention(tmp_path: Path) -> None:
    """Regression: on Windows a just-released lock file raised PermissionError, not
    FileExistsError, so a contending thread crashed instead of waiting."""
    path = tmp_path / "busy.lock"
    counter = {"value": 0}
    errors: list[BaseException] = []

    def work() -> None:
        for _ in range(60):
            try:
                with FileLock(path, timeout=20):
                    counter["value"] += 1
            except BaseException as exc:
                errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert counter["value"] == 16 * 60


def test_stale_locks_are_broken(tmp_path: Path) -> None:
    path = tmp_path / "x.lock"
    path.write_text("", encoding="utf-8")
    old = time.time() - 120
    os.utime(path, (old, old))
    with FileLock(path, stale_after=30):
        assert path.exists()


def test_clock_helper_uses_utc() -> None:
    assert datetime.now(UTC).tzinfo is UTC


def test_the_agent_workflow_cannot_reach_the_notifier() -> None:
    """Structural guard for the notification boundary.

    Only the wiring (`bootstrap`), the application services and the notifier itself may import
    `copilot.safety.notify`. Everything else, including the whole agent workflow, may not.
    """
    root = Path(notify_module.__file__).resolve().parents[1]  # src/copilot
    allowed = {
        root / "bootstrap.py",
        root / "workflow" / "services.py",
        root / "safety" / "notify.py",
    }
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(f"{node.module}.{a.name}" for a in node.names)]
            if any(
                name == "copilot.safety.notify" or name.endswith("safety.notify") for name in names
            ):
                offenders.append(str(path.relative_to(root)))
    assert offenders == []
