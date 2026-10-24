"""Embedders: the protocol, and the offline TF-IDF implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from sklearn.feature_extraction.text import TfidfVectorizer

from copilot.retrieval.tokenizer import tokenize

if TYPE_CHECKING:
    from collections.abc import Sequence


class Embedder(Protocol):
    """Turns texts into vectors. Implementations: TF-IDF offline, Azure OpenAI online."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text."""
        ...


class TfidfEmbedder:
    """Offline embedder: TF-IDF vectors fitted on the playbook corpus."""

    def __init__(self, corpus: Sequence[str]) -> None:
        self._vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        self._vectorizer.fit(corpus)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return L2-normalised TF-IDF vectors."""
        matrix = self._vectorizer.transform(list(texts))
        return [[float(x) for x in row] for row in matrix.toarray()]
