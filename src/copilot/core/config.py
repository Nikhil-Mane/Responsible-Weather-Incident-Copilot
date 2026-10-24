"""Typed application settings, loaded once from environment variables and `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from copilot.core.exceptions import ConfigurationError

_MIN_SECRET_LENGTH = 32


class Settings(BaseSettings):
    """All runtime configuration. Secrets use `SecretStr` so they never print."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    env: Literal["dev", "test", "prod"] = Field("dev", validation_alias="COPILOT_ENV")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", validation_alias="COPILOT_LOG_LEVEL"
    )
    log_json: bool = Field(True, validation_alias="COPILOT_LOG_JSON")

    drafter: Literal["auto", "rules", "lcel", "crew"] = Field(
        "auto", validation_alias="COPILOT_DRAFTER"
    )

    data_dir: Path = Field(Path("data"), validation_alias="COPILOT_DATA_DIR")
    generated_dir: Path = Field(Path("data/generated"), validation_alias="COPILOT_GENERATED_DIR")
    state_dir: Path = Field(Path("var"), validation_alias="COPILOT_STATE_DIR")

    approval_secret: SecretStr = Field(SecretStr(""), validation_alias="COPILOT_APPROVAL_SECRET")
    approval_ttl_seconds: int = Field(
        900, ge=30, le=86_400, validation_alias="COPILOT_APPROVAL_TTL_SECONDS"
    )

    max_input_bytes: int = Field(262_144, ge=1_024, validation_alias="COPILOT_MAX_INPUT_BYTES")
    max_json_depth: int = Field(12, ge=2, le=64, validation_alias="COPILOT_MAX_JSON_DEPTH")

    max_observation_age_minutes: int = Field(
        60, ge=1, validation_alias="COPILOT_MAX_OBSERVATION_AGE_MINUTES"
    )
    min_source_confidence: float = Field(
        0.7, ge=0.0, le=1.0, validation_alias="COPILOT_MIN_SOURCE_CONFIDENCE"
    )
    max_forecast_horizon_hours: int = Field(
        48, ge=1, validation_alias="COPILOT_MAX_FORECAST_HORIZON_HOURS"
    )

    azure_api_key: SecretStr = Field(SecretStr(""), validation_alias="AZURE_OPENAI_API_KEY")
    azure_endpoint: str = Field("", validation_alias="AZURE_OPENAI_ENDPOINT")
    azure_api_version: str = Field("", validation_alias="AZURE_OPENAI_API_VERSION")
    azure_deployment: str = Field("", validation_alias="AZURE_OPENAI_DEPLOYMENT")
    azure_embedding_deployment: str = Field(
        "", validation_alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    )
    llm_timeout_seconds: float = Field(30.0, gt=0, validation_alias="COPILOT_LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(3, ge=0, le=10, validation_alias="COPILOT_LLM_MAX_RETRIES")

    llm_price_input_per_1k: float = Field(
        0.0, ge=0, validation_alias="COPILOT_LLM_PRICE_INPUT_PER_1K"
    )
    llm_price_output_per_1k: float = Field(
        0.0, ge=0, validation_alias="COPILOT_LLM_PRICE_OUTPUT_PER_1K"
    )

    langfuse_public_key: SecretStr = Field(SecretStr(""), validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: SecretStr = Field(SecretStr(""), validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field("", validation_alias="LANGFUSE_HOST")

    @model_validator(mode="after")
    def _require_secret_in_prod(self) -> Settings:
        secret = self.approval_secret.get_secret_value()
        if self.env == "prod" and len(secret) < _MIN_SECRET_LENGTH:
            msg = (
                f"COPILOT_APPROVAL_SECRET must be at least {_MIN_SECRET_LENGTH} characters in prod"
            )
            raise ValueError(msg)
        return self

    @property
    def azure_chat_enabled(self) -> bool:
        """True when every setting needed for Azure chat completions is present."""
        return all(
            (
                self.azure_api_key.get_secret_value(),
                self.azure_endpoint,
                self.azure_api_version,
                self.azure_deployment,
            )
        )

    @property
    def azure_embeddings_enabled(self) -> bool:
        """True when Azure chat is configured and an embeddings deployment is named."""
        return self.azure_chat_enabled and bool(self.azure_embedding_deployment)

    @property
    def langfuse_enabled(self) -> bool:
        """True when Langfuse credentials are configured."""
        return bool(
            self.langfuse_public_key.get_secret_value()
            and self.langfuse_secret_key.get_secret_value()
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once. Raises `ConfigurationError` with a readable message on failure."""
    try:
        return Settings()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        msg = f"Invalid configuration: {problems}"
        raise ConfigurationError(msg) from exc
