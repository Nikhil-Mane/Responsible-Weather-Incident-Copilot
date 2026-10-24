"""Wiring tests: drafter selection and retriever construction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from conftest import DATA_DIR
from copilot import bootstrap
from copilot.agents.crew.drafter import CrewDrafter
from copilot.agents.llm_drafter import LLMDrafter
from copilot.core.config import Settings
from copilot.core.exceptions import ConfigurationError
from copilot.drafting.rule_based import RuleBasedDrafter

AZURE = {
    "AZURE_OPENAI_API_KEY": "k-123456",
    "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
    "AZURE_OPENAI_API_VERSION": "2024-10-21",
    "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
}


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_rules_drafter_is_always_available() -> None:
    assert isinstance(
        bootstrap.select_drafter(_settings(COPILOT_DRAFTER="rules")), RuleBasedDrafter
    )
    azure_but_rules = _settings(COPILOT_DRAFTER="rules", **AZURE)
    assert isinstance(bootstrap.select_drafter(azure_but_rules), RuleBasedDrafter)


def test_auto_is_offline_without_azure_and_lcel_with_it() -> None:
    assert isinstance(bootstrap.select_drafter(_settings()), RuleBasedDrafter)
    assert isinstance(bootstrap.select_drafter(_settings(**AZURE)), LLMDrafter)


@pytest.mark.parametrize("choice", ["lcel", "crew"])
def test_explicit_model_drafters_need_azure(choice: str) -> None:
    with pytest.raises(ConfigurationError, match="AZURE_OPENAI"):
        bootstrap.select_drafter(_settings(COPILOT_DRAFTER=choice))


def test_lcel_choice_with_azure() -> None:
    assert isinstance(
        bootstrap.select_drafter(_settings(COPILOT_DRAFTER="lcel", **AZURE)), LLMDrafter
    )


def test_crew_choice_with_azure() -> None:
    pytest.importorskip("crewai")
    chosen = bootstrap.select_drafter(_settings(COPILOT_DRAFTER="crew", **AZURE))
    assert isinstance(chosen, CrewDrafter)


def test_crew_choice_reports_a_missing_crew(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "build_crew_drafter", lambda settings: None)
    with pytest.raises(ConfigurationError, match="crew needs"):
        bootstrap.select_drafter(_settings(COPILOT_DRAFTER="crew", **AZURE))


def test_unknown_drafter_names_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings(COPILOT_DRAFTER="magic")


def test_retriever_uses_the_configured_embedder(
    generated_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []

    class Spy:
        def embed(self, texts):  # type: ignore[no-untyped-def]
            seen.append(len(texts))
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(bootstrap, "build_embedder", lambda settings: Spy())
    retriever = bootstrap.build_retriever(_settings(COPILOT_GENERATED_DIR=str(generated_dir)))
    assert len(retriever.playbooks) == 5
    assert seen == [5]


def test_generated_data_is_extracted_when_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pymupdf")
    settings = _settings(
        COPILOT_DATA_DIR=str(DATA_DIR), COPILOT_GENERATED_DIR=str(tmp_path / "gen")
    )
    bootstrap.ensure_generated(settings)
    assert (tmp_path / "gen" / "operational_playbooks.json").is_file()
    bootstrap.ensure_generated(settings)  # second call is a no-op


def test_an_unwritable_generated_dir_is_reported_clearly(tmp_path) -> None:
    """Regression: a read-only container filesystem used to end in an unexplained stack trace."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("a file where a directory is expected", encoding="utf-8")
    settings = _settings(
        COPILOT_DATA_DIR=str(DATA_DIR), COPILOT_GENERATED_DIR=str(blocker / "generated")
    )
    with pytest.raises(ConfigurationError, match="COPILOT_GENERATED_DIR"):
        bootstrap.ensure_generated(settings)
