"""Signed, expiring approval tokens bound to one alert's exact content."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from copilot.core.exceptions import ApprovalError, ConfigurationError
from copilot.domain.models import ApprovalStatus, IncidentBriefing, Route
from copilot.domain.rbac import Permission, Role, require

if TYPE_CHECKING:
    from collections.abc import Callable

_MIN_SECRET_LENGTH = 16


def alert_hash(alert: str) -> str:
    """Content hash that an approval is bound to."""
    return hashlib.sha256(alert.encode("utf-8")).hexdigest()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


@dataclass(frozen=True, slots=True)
class TokenClaims:
    """What a verified token asserts."""

    run_id: str
    alert_hash: str
    approver: str
    role: str
    nonce: str
    expires_at: datetime


class ApprovalService:
    """Issues and verifies approval tokens. Only an incident commander can issue one."""

    def __init__(
        self,
        secret: str,
        *,
        ttl: timedelta = timedelta(minutes=15),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if len(secret) < _MIN_SECRET_LENGTH:
            msg = (
                f"COPILOT_APPROVAL_SECRET must be at least {_MIN_SECRET_LENGTH} characters "
                "to issue or verify approval tokens"
            )
            raise ConfigurationError(msg)
        self._key = secret.encode("utf-8")
        self._ttl = ttl
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def ttl(self) -> timedelta:
        """How long an issued token stays valid."""
        return self._ttl

    def issue(self, briefing: IncidentBriefing, *, approver: str, role: Role) -> str:
        """Create a token for the briefing's draft alert once it is cleared for approval."""
        require(role, Permission.APPROVE_ALERT)
        if not approver.strip():
            msg = "a named approver is required"
            raise ApprovalError(msg)
        alert = self._ready_alert(briefing)
        claims = {
            "run_id": briefing.run_id,
            "alert_hash": alert_hash(alert),
            "approver": approver.strip(),
            "role": role.value,
            "nonce": secrets.token_hex(8),
            "exp": int((self._clock() + self._ttl).timestamp()),
        }
        payload = _b64(json.dumps(claims, sort_keys=True).encode("utf-8"))
        return f"{payload}.{self._sign(payload)}"

    def verify(self, token: str, briefing: IncidentBriefing) -> TokenClaims:
        """Validate signature, expiry, run and alert content. Raises `ApprovalError`."""
        claims = self._decode(token)
        if self._clock().timestamp() >= claims["exp"]:
            msg = "approval token has expired"
            raise ApprovalError(msg)
        if claims["run_id"] != briefing.run_id:
            msg = "approval token was issued for a different run"
            raise ApprovalError(msg)
        if briefing.local_team_alert is None:
            msg = "this run has no draft alert to send"
            raise ApprovalError(msg)
        if not hmac.compare_digest(claims["alert_hash"], alert_hash(briefing.local_team_alert)):
            msg = "the alert changed after it was approved"
            raise ApprovalError(msg)
        if claims["role"] != Role.INCIDENT_COMMANDER.value:
            msg = "approval token was not issued by an incident commander"
            raise ApprovalError(msg)
        return TokenClaims(
            run_id=claims["run_id"],
            alert_hash=claims["alert_hash"],
            approver=claims["approver"],
            role=claims["role"],
            nonce=claims["nonce"],
            expires_at=datetime.fromtimestamp(claims["exp"], tz=UTC),
        )

    @staticmethod
    def _ready_alert(briefing: IncidentBriefing) -> str:
        """Return the draft alert if the run may be approved, otherwise raise."""
        alert = briefing.local_team_alert
        if alert is None:
            msg = "this run has no draft alert to approve"
            raise ApprovalError(msg)
        status = briefing.approval_status
        if briefing.route is Route.DATA_REVIEW or status is ApprovalStatus.HOLD_FOR_DATA_REVIEW:
            msg = "the run is held for data review; nothing can be approved"
            raise ApprovalError(msg)
        if status is ApprovalStatus.HELD:
            msg = "the commander held this run; it cannot be approved"
            raise ApprovalError(msg)
        if briefing.route is Route.HUMAN_REVIEW and status is not ApprovalStatus.APPROVED:
            msg = "the run is awaiting the commander's decision; record it first"
            raise ApprovalError(msg)
        return alert

    def _sign(self, payload: str) -> str:
        return _b64(hmac.new(self._key, payload.encode("ascii"), hashlib.sha256).digest())

    def _decode(self, token: str) -> dict[str, Any]:
        try:
            payload, signature = token.split(".")
            if not hmac.compare_digest(signature, self._sign(payload)):
                msg = "approval token signature is invalid"
                raise ApprovalError(msg)
            claims = json.loads(_unb64(payload))
            for key in ("run_id", "alert_hash", "approver", "role", "nonce", "exp"):
                if key not in claims:
                    msg = "approval token is incomplete"
                    raise ApprovalError(msg)
        except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
            msg = "approval token is malformed"
            raise ApprovalError(msg) from exc
        if not isinstance(claims["exp"], int):
            msg = "approval token is malformed"
            raise ApprovalError(msg)
        return dict(claims)
