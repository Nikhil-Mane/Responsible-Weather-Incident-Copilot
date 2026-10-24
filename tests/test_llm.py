"""Provider resilience: error mapping, retries, circuit breaker, Azure factories (no network)."""

from __future__ import annotations

import httpx
import openai
import pytest
from langchain_openai import AzureChatOpenAI

from copilot.core.config import Settings
from copilot.core.exceptions import ProviderError
from copilot.llm.azure import AzureEmbedder, build_chat_model, build_embedder
from copilot.llm.errors import classify_error
from copilot.llm.resilience import CircuitBreaker, call_with_resilience

_REQUEST = httpx.Request("POST", "https://example.openai.azure.com/")


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=_REQUEST)


def _no_sleep(_seconds: float) -> None:
    return None


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


AZURE = {
    "AZURE_OPENAI_API_KEY": "k-123456",
    "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
    "AZURE_OPENAI_API_VERSION": "2024-10-21",
    "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
}


# ----- error classification ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (openai.RateLimitError("slow down", response=_response(429), body=None), True),
        (openai.InternalServerError("oops", response=_response(500), body=None), True),
        (openai.APITimeoutError(request=_REQUEST), True),
        (openai.APIConnectionError(request=_REQUEST), True),
        (TimeoutError("late"), True),
        (ValueError("bad json"), True),
        (openai.AuthenticationError("no", response=_response(401), body=None), False),
        (openai.BadRequestError("nope", response=_response(400), body=None), False),
        (RuntimeError("something else"), False),
    ],
)
def test_error_classification(error: BaseException, retryable: bool) -> None:
    mapped = classify_error(error)
    assert isinstance(mapped, ProviderError)
    assert mapped.retryable is retryable


def test_content_filter_is_reported_plainly() -> None:
    error = openai.BadRequestError(
        "The response was filtered: content_filter", response=_response(400), body=None
    )
    assert "content filter" in str(classify_error(error))


def test_error_messages_never_leak_secrets() -> None:
    mapped = classify_error(RuntimeError("request failed with api_key=sk-very-secret"))
    assert "sk-very-secret" not in str(mapped)


def test_provider_errors_pass_through() -> None:
    original = ProviderError("x", retryable=True)
    assert classify_error(original) is original


# ----- retries ----------------------------------------------------------------------------


def test_transient_failures_are_retried_until_success() -> None:
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise openai.APITimeoutError(request=_REQUEST)
        return "ok"

    assert call_with_resilience(flaky, retries=3, sleep=_no_sleep) == "ok"
    assert calls["n"] == 3


def test_retries_are_bounded() -> None:
    calls = {"n": 0}

    def always_fails() -> str:
        calls["n"] += 1
        raise openai.RateLimitError("slow", response=_response(429), body=None)

    with pytest.raises(ProviderError, match="RateLimitError"):
        call_with_resilience(always_fails, retries=2, sleep=_no_sleep)
    assert calls["n"] == 3


def test_permanent_failures_are_not_retried() -> None:
    calls = {"n": 0}

    def rejected() -> str:
        calls["n"] += 1
        raise openai.AuthenticationError("bad key", response=_response(401), body=None)

    with pytest.raises(ProviderError, match="credentials"):
        call_with_resilience(rejected, retries=5, sleep=_no_sleep)
    assert calls["n"] == 1


def test_zero_retries_means_one_attempt() -> None:
    calls = {"n": 0}

    def once() -> str:
        calls["n"] += 1
        raise TimeoutError

    with pytest.raises(ProviderError):
        call_with_resilience(once, retries=0, sleep=_no_sleep)
    assert calls["n"] == 1


# ----- circuit breaker --------------------------------------------------------------------


class _Time:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_breaker_opens_then_recovers() -> None:
    clock = _Time()
    breaker = CircuitBreaker(failure_threshold=2, reset_after=30, clock=clock)
    assert breaker.is_open is False
    breaker.record_failure()
    assert breaker.is_open is False
    breaker.record_failure()
    assert breaker.is_open is True
    clock.now = 31
    assert breaker.is_open is False
    breaker.record_success()
    breaker.record_failure()
    assert breaker.is_open is False


def test_an_open_breaker_stops_calls_to_the_provider() -> None:
    breaker = CircuitBreaker(failure_threshold=1, reset_after=1_000)
    calls = {"n": 0}

    def failing() -> str:
        calls["n"] += 1
        raise openai.InternalServerError("down", response=_response(500), body=None)

    with pytest.raises(ProviderError):
        call_with_resilience(failing, retries=3, breaker=breaker, sleep=_no_sleep)
    first_round = calls["n"]
    assert breaker.is_open is True

    with pytest.raises(ProviderError, match="circuit breaker"):
        call_with_resilience(failing, retries=3, breaker=breaker, sleep=_no_sleep)
    assert calls["n"] == first_round


def test_success_resets_the_failure_count() -> None:
    breaker = CircuitBreaker(failure_threshold=2)
    breaker.record_failure()
    assert call_with_resilience(lambda: 1, retries=0, breaker=breaker) == 1
    breaker.record_failure()
    assert breaker.is_open is False


# ----- Azure factories --------------------------------------------------------------------


def test_no_chat_model_without_azure_settings() -> None:
    assert build_chat_model(_settings()) is None


def test_chat_model_is_configured_for_resilient_use() -> None:
    model = build_chat_model(_settings(**AZURE))
    assert isinstance(model, AzureChatOpenAI)
    assert model.temperature == 0.1
    assert model.max_retries == 0
    assert model.deployment_name == "gpt-4o"
    assert "k-123456" not in repr(model)


def test_embedder_needs_its_own_deployment() -> None:
    assert build_embedder(_settings(**AZURE)) is None
    embedder = build_embedder(_settings(**AZURE, AZURE_OPENAI_EMBEDDING_DEPLOYMENT="emb"))
    assert isinstance(embedder, AzureEmbedder)
    assert build_embedder(_settings()) is None


def test_azure_embedder_retries_and_converts_vectors() -> None:
    class Client:
        calls = 0

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            Client.calls += 1
            if Client.calls == 1:
                raise openai.APITimeoutError(request=_REQUEST)
            return [[1, 2] for _ in texts]

    embedder = AzureEmbedder(Client(), retries=2, sleep=_no_sleep)
    assert embedder.embed(["a", "b"]) == [[1.0, 2.0], [1.0, 2.0]]
