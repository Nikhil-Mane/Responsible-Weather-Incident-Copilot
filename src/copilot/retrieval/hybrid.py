"""Hybrid retrieval: BM25 plus vector search, rank fusion, then reranking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from copilot.core.exceptions import RetrievalError
from copilot.core.logging_config import get_logger
from copilot.retrieval.bm25 import Bm25Index
from copilot.retrieval.embeddings import Embedder, TfidfEmbedder
from copilot.retrieval.playbooks import Playbook
from copilot.retrieval.rerank import CoverageReranker, Reranker
from copilot.retrieval.tokenizer import tokenize

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_LOG = get_logger(__name__)


_RRF_K = 60


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """What to look for. `topics=None` means no metadata filter; an empty set matches nothing."""

    text: str
    topics: frozenset[str] | None = None
    allowed_ids: frozenset[str] | None = None
    top_k: int = 3


@dataclass(frozen=True, slots=True)
class RetrievedPlaybook:
    """A retrieved playbook with its score and the signals that produced it."""

    playbook: Playbook
    score: float
    rank: int
    signals: Mapping[str, float] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """The identifier used in briefings."""
        return self.playbook.document_id


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _ranks(scores: Mapping[int, float]) -> dict[int, int]:
    """Rank (1 = best) of every item with a positive score."""
    ordered = sorted((i for i, s in scores.items() if s > 0), key=lambda i: (-scores[i], i))
    return {index: position for position, index in enumerate(ordered, start=1)}


class HybridRetriever:
    """BM25 plus vector search, fused with reciprocal rank fusion, then reranked."""

    def __init__(
        self,
        playbooks: Sequence[Playbook],
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        if not playbooks:
            msg = "at least one playbook is required"
            raise RetrievalError(msg)
        self._playbooks = tuple(playbooks)
        corpus = [p.text for p in self._playbooks]
        self._bm25 = Bm25Index(corpus)
        self._embedder = embedder or TfidfEmbedder(corpus)
        self._vectors: list[list[float]] | None
        try:
            self._vectors = self._embedder.embed(corpus)
        except Exception as exc:  # remote embedder down: run lexical-only instead of failing
            _LOG.warning("playbook embedding failed; using lexical search only: %s", exc)
            self._vectors = None
        self._reranker = reranker or CoverageReranker()

    @property
    def playbooks(self) -> tuple[Playbook, ...]:
        """All indexed playbooks."""
        return self._playbooks

    def _embed_query(self, text: str) -> list[float] | None:
        """Embed the query, or return `None` so search degrades to lexical-only."""
        if self._vectors is None:
            return None
        try:
            return self._embedder.embed([text])[0]
        except Exception as exc:  # embedder may be a remote service
            _LOG.warning("query embedding failed; using lexical search only: %s", exc)
            return None

    def _candidates(self, query: RetrievalQuery) -> list[int]:
        return [
            i
            for i, playbook in enumerate(self._playbooks)
            if (query.topics is None or playbook.topic in query.topics)
            and (query.allowed_ids is None or playbook.document_id in query.allowed_ids)
        ]

    def search(self, query: RetrievalQuery) -> list[RetrievedPlaybook]:
        """Return up to `top_k` relevant playbooks, best first. May return an empty list."""
        if query.top_k < 1:
            msg = "top_k must be at least 1"
            raise RetrievalError(msg)
        candidates = self._candidates(query)
        if not candidates or not tokenize(query.text):
            return []

        query_vector = self._embed_query(query.text)
        lexical = {i: self._bm25.score(query.text, i) for i in candidates}
        semantic: dict[int, float] = {}
        if query_vector is not None and self._vectors is not None:
            semantic = {i: _cosine(query_vector, self._vectors[i]) for i in candidates}
        lexical_rank, semantic_rank = _ranks(lexical), _ranks(semantic)
        fused = {
            i: sum(1.0 / (_RRF_K + r[i]) for r in (lexical_rank, semantic_rank) if i in r)
            for i in candidates
        }
        relevant = [i for i in candidates if fused[i] > 0]
        if not relevant:
            return []

        rerank_scores = self._reranker.rerank(query.text, [self._playbooks[i] for i in relevant])
        reranked = dict(zip(relevant, rerank_scores, strict=True))
        # rerank score decides order; fused score breaks ties deterministically
        ordered = sorted(relevant, key=lambda i: (-reranked[i], -fused[i], i))[: query.top_k]
        return [
            RetrievedPlaybook(
                playbook=self._playbooks[i],
                score=round(reranked[i], 6),
                rank=position,
                signals={
                    "bm25": round(lexical[i], 6),
                    "vector": round(semantic.get(i, 0.0), 6),
                    "vector_available": 1.0 if semantic else 0.0,
                    "fused": round(fused[i], 6),
                    "rerank": round(reranked[i], 6),
                },
            )
            for position, i in enumerate(ordered, start=1)
        ]
