"""Command-line interface.

Exit codes: 0 ok, 2 invalid input or usage, 3 held for data review, 4 awaiting commander
decision, 5 internal error, 6 not authorized, 7 audit log failed verification, 8 evaluation
gate failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from copilot.bootstrap import (
    build_approvals,
    build_audit,
    build_eval_workflow,
    build_notifier,
    build_store,
    build_workflow,
)
from copilot.core.config import Settings, get_settings
from copilot.core.exceptions import (
    ApprovalError,
    AuditIntegrityError,
    AuthorizationError,
    ConfigurationError,
    CopilotError,
    InputValidationError,
)
from copilot.core.logging_config import configure_logging, get_logger
from copilot.data.ingest import load_incident
from copilot.domain.models import Route
from copilot.domain.rbac import Role, parse_role
from copilot.evaluation.cases import load_cases
from copilot.evaluation.report import Floors, format_report, report_to_dict
from copilot.evaluation.runner import EvalOptions, run_eval
from copilot.workflow.services import decide_run, run_incident, send_approved

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from copilot.workflow.state import RunResult

EXIT_OK = 0
EXIT_INVALID = 2
EXIT_DATA_REVIEW = 3
EXIT_AWAITING = 4
EXIT_INTERNAL = 5
EXIT_FORBIDDEN = 6
EXIT_INTEGRITY = 7
EXIT_EVAL_FAILED = 8

_LOG = get_logger(__name__)


def _parse_as_of(value: str) -> datetime:
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as exc:
        msg = f"--as-of must be an ISO 8601 timestamp, got '{value}'"
        raise argparse.ArgumentTypeError(msg) from exc
    if moment.tzinfo is None:
        msg = "--as-of must include a timezone, for example 2026-09-01T09:05:00+00:00"
        raise argparse.ArgumentTypeError(msg)
    return moment.astimezone(UTC)


def _add_actor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_id")
    parser.add_argument("--actor", required=True, help="name of the person deciding")
    parser.add_argument(
        "--role", default=Role.INCIDENT_COMMANDER.value, help="role of the person deciding"
    )
    parser.add_argument("--note", default="", help="optional comment recorded in the audit trail")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(prog="copilot", description="Weather Incident Copilot")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="assess one incident JSON file")
    run.add_argument("incident", type=Path, help="path to the incident JSON")
    run.add_argument("--role", default=Role.OPERATOR.value, help="caller role (default: operator)")
    run.add_argument("--run-id", default=None, help="use a specific run ID")
    run.add_argument(
        "--as-of",
        type=_parse_as_of,
        default=None,
        help="treat this instant as 'now' for freshness checks (the supplied data is dated)",
    )

    status = commands.add_parser("status", help="show the briefing for a run")
    status.add_argument("run_id")

    approve = commands.add_parser(
        "approve", help="commander approves a run and receives an expiring send token"
    )
    _add_actor_arguments(approve)

    hold = commands.add_parser("hold", help="commander holds a run; nothing can be sent")
    _add_actor_arguments(hold)

    send = commands.add_parser("send", help="simulated send of an approved alert (needs a token)")
    send.add_argument("run_id")
    send.add_argument("--token", required=True, help="token printed by 'approve'")

    commands.add_parser("audit-verify", help="verify the audit log hash chain")

    evaluate = commands.add_parser(
        "eval", help="run the labeled scenarios and failure-path cases and report metrics"
    )
    evaluate.add_argument(
        "--report", type=Path, default=None, help="where to write the JSON report"
    )
    evaluate.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="uniform floor for gated metrics (default 1.0; lower it when evaluating a model)",
    )
    return parser


def _exit_code(result: RunResult) -> int:
    if result.awaiting_decision:
        return EXIT_AWAITING
    if result.briefing.route is Route.DATA_REVIEW:
        return EXIT_DATA_REVIEW
    return EXIT_OK


def _write_json(payload: object) -> None:
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")


def _emit(result: RunResult) -> None:
    sys.stdout.write(result.briefing.model_dump_json(indent=2) + "\n")


def _run(args: argparse.Namespace, settings: Settings) -> int:
    clock = (lambda: args.as_of) if args.as_of else None
    workflow = build_workflow(settings, clock=clock)
    snapshot = load_incident(
        args.incident, max_bytes=settings.max_input_bytes, max_depth=settings.max_json_depth
    )
    result = run_incident(
        workflow,
        build_store(settings),
        build_audit(settings),
        snapshot,
        role=parse_role(args.role),
        run_id=args.run_id,
    )
    _emit(result)
    return _exit_code(result)


def _status(args: argparse.Namespace, settings: Settings) -> int:
    result = build_workflow(settings).status(args.run_id)
    _emit(result)
    return _exit_code(result)


def _decide(args: argparse.Namespace, settings: Settings, *, approved: bool) -> int:
    approvals = build_approvals(settings) if approved else None  # fails early without a secret
    outcome = decide_run(
        build_workflow(settings),
        build_store(settings),
        build_audit(settings),
        approvals,
        args.run_id,
        actor=args.actor,
        role=parse_role(args.role),
        approved=approved,
        note=args.note,
    )
    output: dict[str, Any] = {
        "run_id": args.run_id,
        "approval_status": outcome.result.briefing.approval_status.value,
    }
    if outcome.token:
        output["token"] = outcome.token
        output["expires_in_seconds"] = outcome.token_ttl_seconds
        output["next"] = f"copilot send {args.run_id} --token <token>"
    _write_json(output)
    return EXIT_OK


def _send(args: argparse.Namespace, settings: Settings) -> int:
    receipt = send_approved(
        build_workflow(settings), build_notifier(settings), args.run_id, args.token
    )
    _write_json(asdict(receipt))
    return EXIT_OK


def _audit_verify(_args: argparse.Namespace, settings: Settings) -> int:
    count = build_audit(settings).verify()
    _write_json({"entries": count, "status": "intact"})
    return EXIT_OK


def _eval(args: argparse.Namespace, settings: Settings) -> int:
    if args.min_accuracy is not None and not 0.0 <= args.min_accuracy <= 1.0:
        msg = "--min-accuracy must be between 0 and 1"
        raise InputValidationError(msg)
    workflow, retriever, drafter = build_eval_workflow(settings)
    floors = Floors.uniform(args.min_accuracy) if args.min_accuracy is not None else Floors()
    options = EvalOptions(
        floors=floors,
        usage_tracker=getattr(drafter, "usage", None),
        price_per_1k=(settings.llm_price_input_per_1k, settings.llm_price_output_per_1k),
    )
    report = run_eval(workflow, retriever, load_cases(settings.generated_dir), options)
    workflow.observer.record_eval(report)
    workflow.flush()
    target = args.report or settings.state_dir / "eval" / "report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report_to_dict(report), indent=2) + "\n", encoding="utf-8")
    tmp.replace(target)
    sys.stdout.write(format_report(report) + f"\n  report: {target}\n")
    return EXIT_OK if report.passed else EXIT_EVAL_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    handlers: dict[str, Callable[[argparse.Namespace, Settings], int]] = {
        "run": _run,
        "status": _status,
        "approve": lambda a, s: _decide(a, s, approved=True),
        "hold": lambda a, s: _decide(a, s, approved=False),
        "send": _send,
        "audit-verify": _audit_verify,
        "eval": _eval,
    }
    try:
        settings = get_settings()
        configure_logging(settings.log_level, json_output=settings.log_json)
        return handlers[args.command](args, settings)
    except (InputValidationError, ConfigurationError, ApprovalError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return EXIT_INVALID
    except AuthorizationError as exc:
        sys.stderr.write(f"not authorized: {exc}\n")
        return EXIT_FORBIDDEN
    except AuditIntegrityError as exc:
        sys.stderr.write(f"audit log failed verification: {exc}\n")
        return EXIT_INTEGRITY
    except CopilotError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return EXIT_INTERNAL
    except Exception:
        _LOG.exception("unexpected failure")
        sys.stderr.write("internal error; see the log for details\n")
        return EXIT_INTERNAL


if __name__ == "__main__":
    sys.exit(main())
