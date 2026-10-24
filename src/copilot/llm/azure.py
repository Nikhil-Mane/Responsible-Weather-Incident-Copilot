"""Azure OpenAI chat and embeddings clients."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import SecretStr

from copilot.llm.resilience import CircuitBreaker, call_with_resilience

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.language_models import BaseChatModel

    from copilot.core.config import Settings

_TEMPERATURE = 0.1


def build_chat_model(settings: Settings) -> BaseChatModel | None:
    """Azure chat model, or `None` when Azure is not configured."""
    if not settings.azure_chat_enabled:
        return None
    return AzureChatOpenAI(
        azure_deployment=settings.azure_deployment,
        azure_endpoint=settings.azure_endpoint,
        api_version=settings.azure_api_version,
        api_key=SecretStr(settings.azure_api_key.get_secret_value()),
        temperature=_TEMPERATURE,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,  # retries are handled by call_with_resilience
    )


class AzureEmbedder:
    """`Embedder` backed by an Azure OpenAI embeddings deployment."""

    def __init__(
        self,
        client: Any,
        *,
        retries: int = 3,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._retries = retries
        self._breaker = breaker
        self._sleep = sleep

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts, raising `ProviderError` on failure."""
        batch = list(texts)
        return call_with_resilience(
            lambda: [list(map(float, v)) for v in self._client.embed_documents(batch)],
            retries=self._retries,
            breaker=self._breaker,
            sleep=self._sleep,
        )


def build_embedder(
    settings: Settings, breaker: CircuitBreaker | None = None
) -> AzureEmbedder | None:
    """Azure embedder, or `None` when no embeddings deployment is configured."""
    if not settings.azure_embeddings_enabled:
        return None
    client = AzureOpenAIEmbeddings(  # type: ignore[call-arg]  # aliases unknown to the mypy plugin
        azure_deployment=settings.azure_embedding_deployment,
        azure_endpoint=settings.azure_endpoint,
        api_version=settings.azure_api_version,
        api_key=SecretStr(settings.azure_api_key.get_secret_value()),
        timeout=settings.llm_timeout_seconds,
        max_retries=0,
    )
    return AzureEmbedder(client, retries=settings.llm_max_retries, breaker=breaker)
