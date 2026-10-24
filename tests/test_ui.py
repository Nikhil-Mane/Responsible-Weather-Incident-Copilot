"""Drive the Streamlit UI headlessly through the demo flows."""

from __future__ import annotations

from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from conftest import DATA_DIR
from copilot.core.config import get_settings

APP = Path(__file__).resolve().parent.parent / "app.py"
SECRET = "ui-test-secret-" * 3


@pytest.fixture(autouse=True)
def _environment(tmp_path: Path, generated_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COPILOT_STATE_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("COPILOT_GENERATED_DIR", str(generated_dir))
    monkeypatch.setenv("COPILOT_DATA_DIR", str(DATA_DIR))
    monkeypatch.setenv("COPILOT_LOG_LEVEL", "ERROR")
    monkeypatch.setenv("COPILOT_APPROVAL_SECRET", SECRET)
    get_settings.cache_clear()
    st.cache_resource.clear()
    yield
    get_settings.cache_clear()
    st.cache_resource.clear()


def _app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=60).run()


def _option(app: AppTest, prefix: str) -> str:
    return next(o for o in app.selectbox(key="case").options if o.startswith(prefix))


def _run_case(app: AppTest, prefix: str) -> AppTest:
    app.selectbox(key="case").select(_option(app, prefix))
    return app.button(key="run").click().run()


def _has_button(app: AppTest, key: str) -> bool:
    try:
        app.button(key=key)
    except KeyError:
        return False
    return True


def _messages(app: AppTest, kind: str) -> list[str]:
    return [element.value for element in getattr(app, kind)]


def test_the_page_loads_with_safe_defaults() -> None:
    app = _app()
    assert not app.exception
    assert app.title[0].value == "Weather Incident Copilot"
    assert app.sidebar.selectbox(key="role").value == "operator"
    assert "never sends a real message" in app.caption[0].value


def test_missing_secret_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COPILOT_APPROVAL_SECRET")
    get_settings.cache_clear()
    st.cache_resource.clear()
    app = _app()
    assert any("COPILOT_APPROVAL_SECRET" in w for w in _messages(app.sidebar, "warning"))


def test_flood_demo_end_to_end() -> None:
    app = _run_case(_app(), "EX-FLOOD")
    assert not app.exception
    assert any("NOT SENT" in w for w in _messages(app, "warning"))
    assert any("DRAFT:" in c.value for c in app.code)
    assert _has_button(app, "approve")

    app.sidebar.selectbox(key="role").select("operator")
    app.sidebar.text_input(key="actor").set_value("Sam")
    app.button(key="approve").click().run()
    assert any("not permitted" in e for e in _messages(app, "error"))

    app.sidebar.selectbox(key="role").select("incident_commander")
    app.sidebar.text_input(key="actor").set_value("J. Rivera")
    app = app.button(key="approve").click().run()
    assert not app.exception
    assert app.session_state["token"]
    assert any("Approved by the commander" in i for i in _messages(app, "info"))

    app = app.button(key="send").click().run()
    assert any("SIMULATED SEND" in s for s in _messages(app, "success"))
    assert app.session_state["receipt"].simulated is True


def test_holding_blocks_any_send() -> None:
    app = _run_case(_app(), "EX-FLOOD")
    app.sidebar.selectbox(key="role").select("incident_commander")
    app.sidebar.text_input(key="actor").set_value("J. Rivera")
    app = app.button(key="hold").click().run()
    assert any("HELD by the commander" in e for e in _messages(app, "error"))
    assert not _has_button(app, "send")
    assert app.session_state["token"] is None


def test_stale_data_is_held_for_review() -> None:
    app = _run_case(_app(), "FP-01")
    assert any("HELD FOR DATA REVIEW" in e for e in _messages(app, "error"))
    assert not _has_button(app, "approve")


def test_the_real_clock_makes_the_dated_sample_stale() -> None:
    app = _app()
    app.sidebar.checkbox(key="real_time").check()
    app = _run_case(app, "EX-FLOOD")
    assert any("HELD FOR DATA REVIEW" in e for e in _messages(app, "error"))


def test_a_bad_as_of_time_is_reported() -> None:
    app = _app()
    app.sidebar.text_input(key="as_of").set_value("yesterday")
    app = _run_case(app, "EX-FLOOD")
    assert any("ISO 8601" in e for e in _messages(app.sidebar, "error"))


def test_low_risk_heat_needs_no_approval() -> None:
    app = _run_case(_app(), "WX-018")
    assert not app.exception
    assert any("nothing to send" in s for s in _messages(app, "success"))
    assert not _has_button(app, "approve")


def test_the_evaluation_tab_reports_a_pass() -> None:
    app = _app()
    app = app.button(key="run_eval").click().run()
    assert not app.exception
    assert "PASS" in _messages(app, "success")


def test_the_audit_tab_verifies_the_chain() -> None:
    app = _run_case(_app(), "EX-FLOOD")
    app = app.button(key="verify_audit").click().run()
    assert any("Chain intact" in s for s in _messages(app, "success"))
