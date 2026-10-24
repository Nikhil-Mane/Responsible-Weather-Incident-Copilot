"""Streamlit UI for the Weather Incident Copilot.

Run: streamlit run app.py

A thin layer over `copilot.services`: the same functions the CLI calls. Nothing here can
send a message; the only "send" is the simulated adapter, and it needs a commander's token.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import streamlit as st

from copilot.bootstrap import (
    build_approvals,
    build_audit,
    build_eval_workflow,
    build_notifier,
    build_store,
    build_workflow,
)
from copilot.core.config import Settings, get_settings
from copilot.core.exceptions import AuditIntegrityError, ConfigurationError, CopilotError
from copilot.data.ingest import parse_incident
from copilot.data.store import RunStore
from copilot.domain.models import ApprovalStatus, IncidentBriefing, Route
from copilot.domain.rbac import Role
from copilot.evaluation.cases import EVAL_NOW, EvalCase, load_cases
from copilot.evaluation.runner import EvalOptions, run_eval
from copilot.safety.approvals import ApprovalService
from copilot.safety.audit import AuditLog
from copilot.safety.notify import SendReceipt, SimulatedNotifier
from copilot.workflow.graph import IncidentWorkflow
from copilot.workflow.services import decide_run, run_incident, send_approved

st.set_page_config(page_title="Weather Incident Copilot", page_icon="🌧️", layout="wide")

_UPLOAD = "Upload a JSON file"
_ROLES = [role.value for role in Role]


class MutableClock:
    """Lets the sidebar change what 'now' means without rebuilding the workflow."""

    def __init__(self) -> None:
        self.moment: datetime | None = None

    def __call__(self) -> datetime:
        return self.moment or datetime.now(UTC)


@dataclass(frozen=True)
class Services:
    """Everything the page needs, built once per process and state directory."""

    settings: Settings
    clock: MutableClock
    workflow: IncidentWorkflow
    store: RunStore
    audit: AuditLog
    approvals: ApprovalService | None
    notifier: SimulatedNotifier | None
    cases: list[EvalCase]


@st.cache_resource(show_spinner=False)
def get_services(state_key: str, secret_configured: bool) -> Services:
    """Build the services. The arguments are cache keys, so tests and users stay isolated."""
    del state_key, secret_configured  # only used to key the cache
    get_settings.cache_clear()
    settings = get_settings()
    clock = MutableClock()
    try:
        approvals: ApprovalService | None = build_approvals(settings)
        notifier: SimulatedNotifier | None = build_notifier(settings)
    except ConfigurationError:
        approvals, notifier = None, None
    workflow = build_workflow(settings, clock=clock)
    cases = load_cases(settings.generated_dir)
    return Services(
        settings, clock, workflow, build_store(settings), build_audit(settings), approvals,
        notifier, cases,
    )  # fmt: skip


def _state_key() -> tuple[str, bool]:
    settings = get_settings()
    return str(settings.state_dir.resolve()), bool(settings.approval_secret.get_secret_value())


def _banner(briefing: IncidentBriefing, receipt: SendReceipt | None) -> None:
    status = briefing.approval_status
    if receipt is not None:
        st.success(f"{receipt.notice}  (duplicate request: {receipt.duplicate})")
    elif briefing.route is Route.DATA_REVIEW:
        st.error("HELD FOR DATA REVIEW. No recommendations or alert were produced.")
    elif status is ApprovalStatus.PENDING:
        st.warning("NOT SENT. Awaiting incident-commander approval.")
    elif status is ApprovalStatus.APPROVED:
        st.info("Approved by the commander. Nothing is sent until the approval token is used.")
    elif status is ApprovalStatus.HELD:
        st.error("HELD by the commander. Nothing can be sent.")
    else:
        st.success("No alert drafted, so there is nothing to send.")


def _render_briefing(briefing: IncidentBriefing) -> None:
    left, middle, right = st.columns(3)
    left.metric("Severity", briefing.severity.value)
    middle.metric("Route", briefing.route.value)
    right.metric("Approval", briefing.approval_status.value)

    st.subheader("Hazards")
    st.write(", ".join(briefing.hazards) if briefing.hazards else "None detected.")

    st.subheader("Recommended readiness actions")
    if briefing.action_items:
        for item in briefing.action_items:
            st.markdown(f"- {item.text}  \n  _source: {', '.join(item.citations)}_")
    else:
        st.write("None.")

    st.subheader("Draft local-team alert")
    if briefing.local_team_alert:
        st.code(briefing.local_team_alert, language=None)
        st.caption("Recipients (draft): " + ", ".join(briefing.alert_recipients))
    else:
        st.write("No alert drafted.")

    if briefing.review_reasons:
        st.subheader("Why it needs review")
        for reason in briefing.review_reasons:
            st.markdown(f"- {reason}")
    for warning in briefing.warnings:
        st.warning(warning)

    st.subheader("Citations")
    st.write(", ".join(briefing.citations))

    st.subheader("Execution trace")
    st.dataframe(
        [{"step": s.step, "status": s.status, "at": s.at} for s in briefing.trace_detail],
        hide_index=True,
        width="stretch",
    )
    with st.expander("Step details"):
        for step in briefing.trace_detail:
            st.markdown(f"**{step.step}** ({step.status})")
            st.json(step.detail, expanded=False)
    st.download_button(
        "Download briefing JSON",
        briefing.model_dump_json(indent=2),
        file_name=f"{briefing.run_id}.json",
        mime="application/json",
    )


def _selected_snapshot(services: Services) -> tuple[Any, str] | None:
    options = [f"{c.case_id} · {c.name}" for c in services.cases] + [_UPLOAD]
    choice = st.selectbox("Incident", options, key="case")
    if choice == _UPLOAD:
        uploaded = st.file_uploader("Incident JSON", type=["json"], key="upload")
        if uploaded is None:
            return None
        snapshot = parse_incident(
            uploaded.getvalue(),
            max_bytes=services.settings.max_input_bytes,
            max_depth=services.settings.max_json_depth,
        )
        return snapshot, uploaded.name
    case = services.cases[options.index(choice)]
    return case.snapshot, case.case_id


def _apply_clock(services: Services) -> None:
    if st.session_state.get("real_time", False):
        services.clock.moment = None
        return
    raw = st.session_state.get("as_of", EVAL_NOW.isoformat())
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        st.sidebar.error("As-of time must be an ISO 8601 timestamp.")
        return
    if moment.tzinfo is None:
        st.sidebar.error("As-of time needs a timezone, e.g. +00:00.")
        return
    services.clock.moment = moment.astimezone(UTC)


def _incident_tab(services: Services) -> None:
    role = Role(st.session_state["role"])
    selection = _selected_snapshot(services)
    if selection is not None:
        snapshot, label = selection
        with st.expander("Incident input"):
            st.json(snapshot.model_dump(mode="json", exclude_none=True), expanded=False)
        if st.button("Run assessment", key="run", type="primary"):
            _apply_clock(services)
            try:
                result = run_incident(
                    services.workflow, services.store, services.audit, snapshot, role=role
                )
            except CopilotError as exc:
                st.error(str(exc))
            else:
                st.session_state.update(run_id=result.briefing.run_id, token=None, receipt=None)
                st.toast(f"Assessed {label}")

    run_id = st.session_state.get("run_id")
    if not run_id:
        st.info("Choose an incident and run the assessment.")
        return

    result = services.workflow.status(run_id)
    briefing = result.briefing
    st.divider()
    st.caption(f"Run {briefing.run_id}")
    _banner(briefing, st.session_state.get("receipt"))
    _decision_controls(services, result.awaiting_decision, briefing, role)
    _render_briefing(briefing)


def _decision_controls(
    services: Services, awaiting: bool, briefing: IncidentBriefing, role: Role
) -> None:
    actor = st.session_state.get("actor", "").strip()
    if awaiting:
        approve_col, hold_col = st.columns(2)
        if approve_col.button("Approve", key="approve", type="primary"):
            _decide(services, briefing, role, actor, approved=True)
        if hold_col.button("Hold", key="hold"):
            _decide(services, briefing, role, actor, approved=False)
        return

    can_issue = (
        briefing.local_team_alert is not None
        and briefing.approval_status in {ApprovalStatus.APPROVED, ApprovalStatus.PENDING}
        and services.approvals is not None
        and not st.session_state.get("token")
    )
    if can_issue and st.button("Issue send token (commander)", key="issue_token"):
        _decide(services, briefing, role, actor, approved=True)

    token = st.session_state.get("token")
    if token and st.session_state.get("receipt") is None:
        st.caption(
            "Approval token issued. It expires, is bound to this exact alert text, "
            "and the alert is sent at most once."
        )
        if st.button("Send alert (simulated)", key="send"):
            _send(services, briefing, token)


def _decide(
    services: Services, briefing: IncidentBriefing, role: Role, actor: str, *, approved: bool
) -> None:
    try:
        outcome = decide_run(
            services.workflow,
            services.store,
            services.audit,
            services.approvals,
            briefing.run_id,
            actor=actor,
            role=role,
            approved=approved,
        )
    except CopilotError as exc:
        st.error(str(exc))
        return
    st.session_state["token"] = outcome.token
    st.rerun()


def _send(services: Services, briefing: IncidentBriefing, token: str) -> None:
    if services.notifier is None:
        st.error("Set COPILOT_APPROVAL_SECRET to enable the simulated send.")
        return
    try:
        receipt = send_approved(services.workflow, services.notifier, briefing.run_id, token)
    except CopilotError as exc:
        st.error(str(exc))
        return
    st.session_state["receipt"] = receipt
    st.rerun()


def _evaluation_tab(services: Services) -> None:
    st.write(
        "Runs the 24 labeled scenarios, the supplied flood example and the data-quality "
        "failure paths through the same workflow, and checks route, alert, approval, "
        "citations and grounding."
    )
    if st.button("Run evaluation", key="run_eval"):
        workflow, retriever, drafter = build_eval_workflow(services.settings)
        options = EvalOptions(usage_tracker=getattr(drafter, "usage", None))
        with st.spinner("Evaluating..."):
            report = run_eval(
                workflow, retriever, load_cases(services.settings.generated_dir), options
            )
        st.session_state["report"] = report
    report = st.session_state.get("report")
    if report is None:
        return
    (st.success if report.passed else st.error)("PASS" if report.passed else "FAIL")
    st.dataframe(
        [{"metric": k, "value": round(v, 4)} for k, v in report.metrics.items()],
        hide_index=True,
        width="stretch",
    )
    st.dataframe(
        [
            {
                "case": r.case_id,
                "name": r.name,
                "kind": r.kind,
                "passed": r.passed,
                "route": r.actual_route,
                "problems": "; ".join(r.problems),
            }
            for r in report.results
        ],
        hide_index=True,
        width="stretch",
    )


def _audit_tab(services: Services) -> None:
    entries = services.audit.entries()
    st.write(f"{len(entries)} audit entries (hash-chained, append-only).")
    if st.button("Verify audit chain", key="verify_audit"):
        try:
            count = services.audit.verify()
        except AuditIntegrityError as exc:
            st.error(f"Audit log failed verification: {exc}")
        else:
            st.success(f"Chain intact ({count} entries).")
    st.dataframe(
        [
            {
                "seq": e["seq"],
                "at": e["at"],
                "event": e["event"],
                "run": e["run_id"],
                "actor": e["actor"],
            }
            for e in reversed(entries)
        ],
        hide_index=True,
        width="stretch",
    )


def main() -> None:
    """Render the page."""
    st.title("Weather Incident Copilot")
    st.caption(
        "Decision support for the control room. Alerts are drafts until the incident commander "
        "approves them. This app never sends a real message."
    )
    state_key, secret_configured = _state_key()
    services = get_services(state_key, secret_configured)

    with st.sidebar:
        st.header("Session")
        st.selectbox("Acting role", _ROLES, index=_ROLES.index(Role.OPERATOR.value), key="role")
        st.text_input("Your name (recorded on approvals)", key="actor")
        st.checkbox("Use the real current time", key="real_time")
        st.text_input(
            "Treat this as 'now'",
            value=EVAL_NOW.isoformat(),
            key="as_of",
            disabled=st.session_state.get("real_time", False),
            help="The supplied data is dated 2026-09-01, so freshness checks use this time.",
        )
        if services.approvals is None:
            st.warning("COPILOT_APPROVAL_SECRET is not set: approval tokens are disabled.")

    incident, evaluation, audit = st.tabs(["Incident briefing", "Evaluation", "Audit log"])
    with incident:
        _incident_tab(services)
    with evaluation:
        _evaluation_tab(services)
    with audit:
        _audit_tab(services)


main()
