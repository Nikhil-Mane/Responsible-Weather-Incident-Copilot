"""Live Azure OpenAI checks. Skipped unless configured; run with: pytest -m azure.

These send the synthetic flood scenario to your Azure deployment and cost tokens.
"""

from __future__ import annotations

import pytest

from conftest import NOW
from copilot.agents.factory import build_drafter
from copilot.core.config import Settings
from copilot.domain.models import Route
from copilot.llm.azure import build_embedder
from copilot.workflow.graph import IncidentWorkflow

pytestmark = pytest.mark.azure


@pytest.fixture(scope="module")
def settings() -> Settings:
    loaded = Settings()
    if not loaded.azure_chat_enabled:
        pytest.skip("Azure OpenAI is not configured in the environment or .env")
    return loaded


def test_live_lcel_drafter_passes_the_guardrails(settings, real_retriever, make_snapshot) -> None:
    workflow = IncidentWorkflow(real_retriever, drafter=build_drafter(settings), clock=lambda: NOW)
    briefing = workflow.run(make_snapshot()).briefing

    assert briefing.route is Route.HUMAN_REVIEW
    assert briefing.local_team_alert is not None
    assert briefing.local_team_alert.startswith("DRAFT:")
    assert briefing.approval_status.value == "pending_commander_approval"
    for item in briefing.action_items:
        assert set(item.citations) <= set(briefing.citations)


def test_live_embeddings_return_vectors(settings) -> None:
    if not settings.azure_embeddings_enabled:
        pytest.skip("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is not configured")
    embedder = build_embedder(settings)
    assert embedder is not None
    vectors = embedder.embed(["flood response", "heat health"])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) > 100
