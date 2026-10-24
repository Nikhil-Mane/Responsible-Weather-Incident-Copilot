"""Tests for settings, logging redaction and the exception hierarchy."""

from __future__ import annotations

import json
import logging

import pytest
from pydantic import ValidationError

from copilot.core.config import Settings
from copilot.core.exceptions import (
    ApprovalError,
    AuthorizationError,
    ConfigurationError,
    CopilotError,
    DataQualityError,
    InputValidationError,
    ProviderError,
    RetrievalError,
)
from copilot.core.logging_config import configure_logging, redact, reset_run_id, set_run_id


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_defaults_are_safe_and_offline() -> None:
    settings = _settings()
    assert settings.env == "dev"
    assert settings.azure_chat_enabled is False
    assert settings.azure_embeddings_enabled is False
    assert settings.langfuse_enabled is False


def test_azure_enabled_only_when_all_values_present() -> None:
    partial = _settings(AZURE_OPENAI_API_KEY="k", AZURE_OPENAI_ENDPOINT="https://x")
    assert partial.azure_chat_enabled is False
    full = _settings(
        AZURE_OPENAI_API_KEY="k",
        AZURE_OPENAI_ENDPOINT="https://x",
        AZURE_OPENAI_API_VERSION="2024-06-01",
        AZURE_OPENAI_DEPLOYMENT="d",
    )
    assert full.azure_chat_enabled is True
    assert full.azure_embeddings_enabled is False


def test_secrets_never_appear_in_repr() -> None:
    settings = _settings(AZURE_OPENAI_API_KEY="super-secret-value")
    assert "super-secret-value" not in repr(settings)


def test_prod_requires_long_approval_secret() -> None:
    with pytest.raises(ValidationError):
        _settings(COPILOT_ENV="prod", COPILOT_APPROVAL_SECRET="short")
    ok = _settings(COPILOT_ENV="prod", COPILOT_APPROVAL_SECRET="x" * 32)
    assert ok.env == "prod"


def test_out_of_range_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings(COPILOT_MIN_SOURCE_CONFIDENCE=1.5)


def test_redact_masks_credentials() -> None:
    masked = redact('api_key=abc123 and "token": "xyz789"')
    assert "abc123" not in masked
    assert "xyz789" not in masked


def test_json_logs_carry_run_id_and_redact(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_output=True)
    token = set_run_id("run-42")
    try:
        logging.getLogger("test").info("calling with api_key=topsecret")
    finally:
        reset_run_id(token)
    line = capsys.readouterr().err.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["run_id"] == "run-42"
    assert "topsecret" not in payload["message"]


@pytest.mark.parametrize(
    "error",
    [
        ConfigurationError,
        InputValidationError,
        DataQualityError,
        RetrievalError,
        ProviderError,
        ApprovalError,
        AuthorizationError,
    ],
)
def test_all_errors_share_a_base(error: type[CopilotError]) -> None:
    assert issubclass(error, CopilotError)
