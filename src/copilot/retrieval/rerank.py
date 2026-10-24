"""Rerankers: the protocol, and the offline term-coverage implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from copilot.retrieval.playbooks import Playbook
from copilot.retrieval.tokenizer import tokenize

if TYPE_CHECKING:
    from collections.abc import Sequence


class Reranker(Protocol):
    """Reorders candidates for a query. Higher score is better."""

    def rerank(self, query: str, candidates: Sequence[Playbook]) -> list[float]:
        """Return one relevance score per candidate, in the same order."""
        ...


class CoverageReranker:
    """Offline reranker: share of query terms covered, weighting title and keyword matches."""

    def rerank(self, query: str, candidates: Sequence[Playbook]) -> list[float]:
        """Score each candidate by weighted query-term coverage."""
        terms = set(tokenize(query))
        if not terms:
            return [0.0] * len(candidates)
        scores: list[float] = []
        for candidate in candidates:
            title = set(tokenize(candidate.title))
            keywords = set(tokenize(" ".join(candidate.keywords)))
            body = set(tokenize(candidate.content))
            weighted = (
                2.0 * len(terms & title) + 1.5 * len(terms & keywords) + 1.0 * len(terms & body)
            )
            scores.append(weighted / (4.5 * len(terms)))
        return scores
