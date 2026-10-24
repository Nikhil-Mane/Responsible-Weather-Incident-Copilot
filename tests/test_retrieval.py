"""Retrieval tests: hybrid search, metadata filters, reranking, query building, metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from copilot.core.exceptions import RetrievalError
from copilot.domain.models import WeatherSnapshot
from copilot.domain.rules import assess
from copilot.retrieval.hybrid import HybridRetriever, RetrievalQuery
from copilot.retrieval.metrics import precision_at_k, recall_at_k, reciprocal_rank
from copilot.retrieval.playbooks import Playbook, load_playbooks, topic_for
from copilot.retrieval.query import build_query
from copilot.retrieval.rerank import CoverageReranker
from copilot.retrieval.tokenizer import tokenize

PLAYBOOKS = (
    Playbook(
        "PLAYBOOK-FLOOD-001",
        "Flood Response and Local Team Readiness",
        ("flood", "rainfall", "river", "drainage"),
        "For verified flood risk, inspect drainage hotspots, verify flood equipment readiness, "
        "coordinate approved diversions, and prepare a local-team readiness alert.",
        "flood",
    ),
    Playbook(
        "PLAYBOOK-WINTER-002",
        "Winter Route Resilience",
        ("snow", "ice", "blizzard", "visibility", "road"),
        "For heavy snow, low visibility, or icy roads, stage snow-clearing capability on "
        "priority routes.",
        "winter",
    ),
    Playbook(
        "PLAYBOOK-HEAT-003",
        "Heat Health and Workforce Protection",
        ("heat", "temperature", "cooling", "hydration"),
        "For extreme heat, prepare cooling-center readiness and coordinate welfare checks.",
        "heat",
    ),
    Playbook(
        "PLAYBOOK-COMMS-004",
        "Incident Communications Approval Standard",
        ("alert", "communication", "approval", "incident"),
        "Draft alerts must identify the verified event, affected area, required readiness "
        "action, and approval status. Do not send external notifications until approved.",
        "communications",
    ),
    Playbook(
        "PLAYBOOK-DATA-005",
        "Operational Data Quality Standard",
        ("source", "confidence", "timestamp", "freshness"),
        "Hold incomplete, stale, or low-confidence observations for data-quality review.",
        "data_quality",
    ),
)


@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    return HybridRetriever(PLAYBOOKS)


def _ids(results) -> list[str]:  # type: ignore[no-untyped-def]
    return [r.citation for r in results]


def test_flood_query_ranks_flood_playbook_first(retriever) -> None:
    results = retriever.search(RetrievalQuery("intense rainfall rising river level drainage"))
    assert _ids(results)[0] == "PLAYBOOK-FLOOD-001"
    assert results[0].rank == 1
    assert set(results[0].signals) == {"bm25", "vector", "vector_available", "fused", "rerank"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("heavy snowfall icy roads low visibility", "PLAYBOOK-WINTER-002"),
        ("extreme heat cooling center welfare", "PLAYBOOK-HEAT-003"),
        ("stale low confidence observation source", "PLAYBOOK-DATA-005"),
    ],
)
def test_each_hazard_finds_its_playbook(retriever, text: str, expected: str) -> None:
    assert _ids(retriever.search(RetrievalQuery(text)))[0] == expected


def test_metadata_topic_filter_excludes_other_topics(retriever) -> None:
    query = RetrievalQuery("road snow flood heat", topics=frozenset({"heat"}))
    assert _ids(retriever.search(query)) == ["PLAYBOOK-HEAT-003"]


def test_allowed_ids_filter_enforces_access(retriever) -> None:
    query = RetrievalQuery("flood drainage", allowed_ids=frozenset({"PLAYBOOK-WINTER-002"}))
    assert "PLAYBOOK-FLOOD-001" not in _ids(retriever.search(query))


def test_empty_topic_set_matches_nothing(retriever) -> None:
    assert retriever.search(RetrievalQuery("flood", topics=frozenset())) == []


def test_irrelevant_query_returns_nothing(retriever) -> None:
    assert retriever.search(RetrievalQuery("zzzz qqqq")) == []
    assert retriever.search(RetrievalQuery("   ")) == []


def test_top_k_limits_results_and_validates(retriever) -> None:
    assert len(retriever.search(RetrievalQuery("alert approval flood heat snow", top_k=2))) <= 2
    with pytest.raises(RetrievalError):
        retriever.search(RetrievalQuery("flood", top_k=0))


def test_results_are_deterministic(retriever) -> None:
    query = RetrievalQuery("flood alert approval readiness")
    assert _ids(retriever.search(query)) == _ids(retriever.search(query))


def test_embedder_is_swappable() -> None:
    class OneHot:
        def embed(self, texts):  # type: ignore[no-untyped-def]
            return [[1.0 if "flood" in t.lower() else 0.0, 0.1] for t in texts]

    results = HybridRetriever(PLAYBOOKS, embedder=OneHot()).search(RetrievalQuery("flood"))
    assert _ids(results)[0] == "PLAYBOOK-FLOOD-001"


def test_failing_embedder_degrades_to_lexical_search() -> None:
    class Boom:
        calls = 0

        def embed(self, texts):  # type: ignore[no-untyped-def]
            Boom.calls += 1
            if Boom.calls > 1:
                msg = "service down"
                raise TimeoutError(msg)
            return [[1.0, 0.0] for _ in texts]

    broken = HybridRetriever(PLAYBOOKS, embedder=Boom())
    results = broken.search(RetrievalQuery("flood drainage"))
    assert _ids(results)[0] == "PLAYBOOK-FLOOD-001"
    assert results[0].signals["vector_available"] == 0.0
    assert results[0].signals["vector"] == 0.0


def test_corpus_embedding_failure_degrades_to_lexical_search() -> None:
    class Down:
        def embed(self, texts):  # type: ignore[no-untyped-def]
            msg = "service down"
            raise TimeoutError(msg)

    results = HybridRetriever(PLAYBOOKS, embedder=Down()).search(RetrievalQuery("heat cooling"))
    assert _ids(results)[0] == "PLAYBOOK-HEAT-003"
    assert results[0].signals["vector_available"] == 0.0


def test_vector_signal_is_available_normally(retriever) -> None:
    result = retriever.search(RetrievalQuery("flood drainage"))[0]
    assert result.signals["vector_available"] == 1.0


def test_reranker_changes_order() -> None:
    class Prefer:
        def rerank(self, query, candidates):  # type: ignore[no-untyped-def]
            return [1.0 if "COMMS" in c.document_id else 0.0 for c in candidates]

    results = HybridRetriever(PLAYBOOKS, reranker=Prefer()).search(
        RetrievalQuery("flood alert approval readiness")
    )
    assert _ids(results)[0] == "PLAYBOOK-COMMS-004"


def test_coverage_reranker_prefers_better_coverage() -> None:
    scores = CoverageReranker().rerank("flood drainage river", [PLAYBOOKS[0], PLAYBOOKS[2]])
    assert scores[0] > scores[1]
    assert CoverageReranker().rerank("", PLAYBOOKS) == [0.0] * len(PLAYBOOKS)


def test_empty_playbook_set_is_rejected() -> None:
    with pytest.raises(RetrievalError):
        HybridRetriever(())


@pytest.mark.parametrize(
    ("document_id", "topic"),
    [
        ("PLAYBOOK-FLOOD-001", "flood"),
        ("PLAYBOOK-COMMS-004", "communications"),
        ("PLAYBOOK-DATA-005", "data_quality"),
        ("PLAYBOOK-NEW-009", "new"),
        ("weird", "weird"),
    ],
)
def test_topic_is_derived_from_the_id(document_id: str, topic: str) -> None:
    assert topic_for(document_id) == topic


def test_tokenizer_stems_and_drops_stopwords() -> None:
    assert tokenize("The flooding of roads") == ["flood", "road"]


def test_build_query_for_flood_adds_communications(make_snapshot) -> None:
    query = build_query(assess(make_snapshot()), "River flood warning")
    assert query.topics == frozenset({"flood", "communications"})


def test_build_query_for_heat_without_review_skips_communications(make_weather) -> None:
    query = build_query(assess(make_weather(severity="high", temperature_c=39)), "Heatwave")
    assert query.topics == frozenset({"heat"})


def test_build_query_for_severe_winter_adds_communications(make_weather) -> None:
    result = assess(make_weather(severity="severe", snowfall_rate_cm_per_hour=5))
    assert build_query(result, "Winter storm").topics == frozenset({"winter", "communications"})


def test_build_query_without_hazards_retrieves_nothing(make_weather, retriever) -> None:
    query = build_query(assess(make_weather()), "Clear")
    assert query.topics == frozenset()
    assert retriever.search(query) == []


def test_build_query_for_failed_quality_cites_the_data_standard(make_snapshot, retriever) -> None:
    query = build_query(assess(make_snapshot()), "x", quality_failed=True)
    assert _ids(retriever.search(query)) == ["PLAYBOOK-DATA-005"]


def test_full_flood_query_retrieves_flood_and_communications(make_snapshot, retriever) -> None:
    snapshot: WeatherSnapshot = make_snapshot()
    query = build_query(assess(snapshot), snapshot.event)
    assert set(_ids(retriever.search(query))) == {"PLAYBOOK-FLOOD-001", "PLAYBOOK-COMMS-004"}


def test_metrics() -> None:
    relevant = frozenset({"a", "b"})
    assert precision_at_k(["a", "x", "b"], relevant, 2) == 0.5
    assert recall_at_k(["a", "x", "b"], relevant, 3) == 1.0
    assert reciprocal_rank(["x", "b"], relevant) == 0.5
    assert reciprocal_rank(["x"], relevant) == 0.0
    assert precision_at_k([], relevant, 3) == 0.0
    assert recall_at_k([], frozenset(), 3) == 1.0
    assert reciprocal_rank([], frozenset()) == 1.0


def test_load_playbooks(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(
        json.dumps(
            [
                {
                    "document_id": "PLAYBOOK-FLOOD-001",
                    "title": "Flood",
                    "keywords": ["flood"],
                    "keywords_clipped": True,
                    "content": "inspect drainage",
                }
            ]
        ),
        encoding="utf-8",
    )
    loaded = load_playbooks(path)
    assert loaded[0].topic == "flood"


def test_load_playbooks_errors(tmp_path: Path) -> None:
    with pytest.raises(RetrievalError):
        load_playbooks(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(RetrievalError, match="no playbooks"):
        load_playbooks(bad)
    wrong = tmp_path / "wrong.json"
    wrong.write_text('[{"document_id": 1}]', encoding="utf-8")
    with pytest.raises(RetrievalError):
        load_playbooks(wrong)


def test_build_query_for_wind_targets_a_topic_with_no_playbook(make_weather, retriever) -> None:
    query = build_query(assess(make_weather(severity="high", wind_gust_kmh=60)), "High wind")
    assert query.topics == frozenset({"wind"})
    assert retriever.search(query) == []


def test_build_query_adds_topic_vocabulary_for_differently_worded_hazards(make_weather) -> None:
    query = build_query(
        assess(make_weather(severity="high", road_surface_temperature_c=45)), "heatwave_surface"
    )
    assert "cooling" in query.text
    assert "hot road surface" in query.text
