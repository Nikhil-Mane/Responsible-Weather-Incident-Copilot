"""Role-based access control for playbooks, approvals and alert recipients.

Enforced in the functions that do the work (retrieval filtering, approval), not in any UI.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from copilot.core.exceptions import AuthorizationError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from copilot.retrieval.playbooks import Playbook


class Role(StrEnum):
    """People who interact with the copilot."""

    OPERATOR = "operator"
    RESPONSE_TEAM = "response_team"
    INCIDENT_COMMANDER = "incident_commander"
    ASSURANCE_REVIEWER = "assurance_reviewer"


class Permission(StrEnum):
    """Actions that can be allowed or denied."""

    RUN_ASSESSMENT = "run_assessment"
    VIEW_BRIEFING = "view_briefing"
    VIEW_AUDIT = "view_audit"
    APPROVE_ALERT = "approve_alert"


_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.OPERATOR: frozenset(
        {Permission.RUN_ASSESSMENT, Permission.VIEW_BRIEFING, Permission.VIEW_AUDIT}
    ),
    Role.RESPONSE_TEAM: frozenset({Permission.VIEW_BRIEFING}),
    Role.INCIDENT_COMMANDER: frozenset(
        {
            Permission.RUN_ASSESSMENT,
            Permission.VIEW_BRIEFING,
            Permission.VIEW_AUDIT,
            Permission.APPROVE_ALERT,
        }
    ),
    Role.ASSURANCE_REVIEWER: frozenset({Permission.VIEW_BRIEFING, Permission.VIEW_AUDIT}),
}

# None means every playbook. The response team does not need the data-quality standard.
_PLAYBOOK_TOPICS: dict[Role, frozenset[str] | None] = {
    Role.OPERATOR: None,
    Role.RESPONSE_TEAM: frozenset({"flood", "winter", "heat", "communications"}),
    Role.INCIDENT_COMMANDER: None,
    Role.ASSURANCE_REVIEWER: None,
}

ALERT_RECIPIENT_ROLES: tuple[Role, ...] = (Role.RESPONSE_TEAM,)


def parse_role(value: str) -> Role:
    """Parse a role name, raising `AuthorizationError` for unknown roles."""
    try:
        return Role(value.strip().lower())
    except ValueError as exc:
        valid = ", ".join(r.value for r in Role)
        msg = f"unknown role '{value}' (valid: {valid})"
        raise AuthorizationError(msg) from exc


def is_allowed(role: Role, permission: Permission) -> bool:
    """True when the role holds the permission."""
    return permission in _PERMISSIONS[role]


def require(role: Role, permission: Permission) -> None:
    """Raise `AuthorizationError` unless the role holds the permission."""
    if not is_allowed(role, permission):
        msg = f"role '{role.value}' is not permitted to {permission.value.replace('_', ' ')}"
        raise AuthorizationError(msg)


def allowed_playbook_ids(role: Role, playbooks: Iterable[Playbook]) -> frozenset[str]:
    """IDs of the playbooks the role may read."""
    topics = _PLAYBOOK_TOPICS[role]
    return frozenset(p.document_id for p in playbooks if topics is None or p.topic in topics)
