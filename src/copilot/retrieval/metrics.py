"""Retrieval relevance metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def precision_at_k(retrieved: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Share of the top-k retrieved items that are relevant."""
    top = retrieved[:k]
    return sum(1 for r in top if r in relevant) / len(top) if top else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Share of the relevant items found in the top-k. Defined as 1.0 when nothing is relevant."""
    if not relevant:
        return 1.0
    return sum(1 for r in retrieved[:k] if r in relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: frozenset[str]) -> float:
    """1 / rank of the first relevant item, or 0.0. Defined as 1.0 when nothing is relevant."""
    if not relevant:
        return 1.0
    for position, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0
