"""CLI tests: exit codes and output, using a temporary state directory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import DATA_DIR, TRUSTED_FLOOD
from copilot import cli
from copilot.core.config import get_settings
from copilot.evaluation.cases import EVAL_NOW
from copilot.workflow.graph import IncidentWorkflow

AS_OF = "2026-09-01T09:05:00+00:00"


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, generated_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COPILOT_STATE_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("COPILOT_GENERATED_DIR", str(generated_dir))
    monkeypatch.setenv("COPILOT_DATA_DIR", str(DATA_DIR))
    monkeypatch.setenv("COPILOT_LOG_LEVEL", "ERROR")
    monkeypatch.setenv("COPILOT_APPROVAL_SECRET", "unit-test-secret-" * 3)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _write(tmp_path: Path, payload: dict[str, object], name: str = "incident.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_flood_run_awaits_the_commander(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        ["run", str(_write(tmp_path, TRUSTED_FLOOD)), "--as-of", AS_OF, "--run-id", "r1"]
    )
    briefing = json.loads(capsys.readouterr().out)
    assert code == cli.EXIT_AWAITING
    assert briefing["route"] == "human_review"
    assert briefing["approval_status"] == "pending_commander_approval"
    assert briefing["local_team_alert"].startswith("DRAFT:")
    assert (tmp_path / "var" / "runs" / "r1.json").is_file()


def test_status_reads_back_the_paused_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["run", str(_write(tmp_path, TRUSTED_FLOOD)), "--as-of", AS_OF, "--run-id", "r2"])
    capsys.readouterr()
    assert cli.main(["status", "r2"]) == cli.EXIT_AWAITING
    assert json.loads(capsys.readouterr().out)["run_id"] == "r2"


def test_stale_data_exits_with_data_review(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        ["run", str(_write(tmp_path, TRUSTED_FLOOD)), "--as-of", "2026-10-24T00:00:00+00:00"]
    )
    assert code == cli.EXIT_DATA_REVIEW
    assert json.loads(capsys.readouterr().out)["route"] == "data_review"


def test_invalid_input_exits_with_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = _write(tmp_path, {**TRUSTED_FLOOD, "rainfall_mm_per_hour": -5})
    assert cli.main(["run", str(bad)]) == cli.EXIT_INVALID
    assert "error:" in capsys.readouterr().err


def test_missing_file_exits_with_usage_error(tmp_path: Path) -> None:
    assert cli.main(["run", str(tmp_path / "nope.json")]) == cli.EXIT_INVALID


def test_unauthorized_role_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path, TRUSTED_FLOOD)
    assert (
        cli.main(["run", str(path), "--as-of", AS_OF, "--role", "response_team"])
        == cli.EXIT_FORBIDDEN
    )
    assert "not authorized" in capsys.readouterr().err


def test_unknown_role_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, TRUSTED_FLOOD)
    assert cli.main(["run", str(path), "--role", "admin"]) == cli.EXIT_FORBIDDEN


def test_unknown_run_status_is_an_error() -> None:
    assert cli.main(["status", "ghost"]) == cli.EXIT_INVALID


@pytest.mark.parametrize("value", ["yesterday", "2026-09-01T09:05:00"])
def test_as_of_must_be_an_aware_timestamp(value: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(["run", str(_write(tmp_path, TRUSTED_FLOOD)), "--as-of", value])


# ----- approval and simulated send ---------------------------------------------------------


def _run_flood(tmp_path: Path, run_id: str, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path, TRUSTED_FLOOD)
    assert cli.main(["run", str(path), "--as-of", AS_OF, "--run-id", run_id]) == cli.EXIT_AWAITING
    capsys.readouterr()


def _approve(run_id: str, capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    assert cli.main(["approve", run_id, "--actor", "J. Rivera"]) == cli.EXIT_OK
    return json.loads(capsys.readouterr().out)


def test_full_approval_flow_sends_once_in_simulation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run_flood(tmp_path, "flow1", capsys)
    approved = _approve("flow1", capsys)
    assert approved["approval_status"] == "approved"
    token = str(approved["token"])

    assert cli.main(["send", "flow1", "--token", token]) == cli.EXIT_OK
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["simulated"] is True
    assert receipt["duplicate"] is False
    assert "no external message" in receipt["notice"]

    assert cli.main(["send", "flow1", "--token", token]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["duplicate"] is True

    assert cli.main(["audit-verify"]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["status"] == "intact"
    assert cli.main(["status", "flow1"]) == cli.EXIT_OK


def test_nothing_can_be_sent_without_a_valid_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run_flood(tmp_path, "flow2", capsys)
    assert cli.main(["send", "flow2", "--token", "forged.token"]) == cli.EXIT_INVALID
    assert "error:" in capsys.readouterr().err


def test_a_held_run_cannot_be_approved(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run_flood(tmp_path, "flow3", capsys)
    assert cli.main(["hold", "flow3", "--actor", "J. Rivera", "--note", "wait"]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["approval_status"] == "held"
    assert cli.main(["approve", "flow3", "--actor", "J. Rivera"]) == cli.EXIT_INVALID


def test_only_a_commander_can_approve_or_hold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run_flood(tmp_path, "flow4", capsys)
    args = ["--actor", "Sam", "--role", "operator"]
    assert cli.main(["approve", "flow4", *args]) == cli.EXIT_FORBIDDEN
    assert cli.main(["hold", "flow4", *args]) == cli.EXIT_FORBIDDEN
    assert cli.main(["status", "flow4"]) == cli.EXIT_AWAITING


def test_approving_without_a_secret_is_a_configuration_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_flood(tmp_path, "flow5", capsys)
    monkeypatch.delenv("COPILOT_APPROVAL_SECRET")
    get_settings.cache_clear()
    assert cli.main(["approve", "flow5", "--actor", "J. Rivera"]) == cli.EXIT_INVALID
    assert "COPILOT_APPROVAL_SECRET" in capsys.readouterr().err
    assert cli.main(["status", "flow5"]) == cli.EXIT_AWAITING


def test_audit_tampering_is_detected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run_flood(tmp_path, "flow6", capsys)
    log = tmp_path / "var" / "audit.jsonl"
    log.write_text(log.read_text(encoding="utf-8").replace("human_review", "operational"))
    assert cli.main(["audit-verify"]) == cli.EXIT_INTEGRITY
    assert "failed verification" in capsys.readouterr().err


# ----- evaluation command ------------------------------------------------------------------


def test_eval_command_passes_and_writes_a_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "report.json"
    assert cli.main(["eval", "--report", str(target)]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "result: PASS" in out
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["metrics"]["cases"] == 38


def test_eval_command_fails_the_gate_on_a_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Empty:
        playbooks = ()

        def search(self, query):  # type: ignore[no-untyped-def]
            return []

    real = cli.build_eval_workflow

    def broken(settings):  # type: ignore[no-untyped-def]
        _workflow, retriever, drafter = real(settings)
        return IncidentWorkflow(Empty(), clock=lambda: EVAL_NOW), retriever, drafter  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "build_eval_workflow", broken)
    assert cli.main(["eval", "--report", str(tmp_path / "r.json")]) == cli.EXIT_EVAL_FAILED
    assert "result: FAIL" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["1.5", "-0.1"])
def test_eval_rejects_an_impossible_floor(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["eval", "--min-accuracy", value]) == cli.EXIT_INVALID
    assert "--min-accuracy" in capsys.readouterr().err
