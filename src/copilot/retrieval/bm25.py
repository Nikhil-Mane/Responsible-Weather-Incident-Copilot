"""Okapi BM25 lexical index."""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

from copilot.retrieval.tokenizer import tokenize

if TYPE_CHECKING:
    from collections.abc import Sequence


class Bm25Index:
    """Okapi BM25 over the playbook corpus."""

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._docs = [tokenize(text) for text in corpus]
        self._k1, self._b = k1, b
        self._avg_len = sum(len(d) for d in self._docs) / max(len(self._docs), 1)
        doc_freq: Counter[str] = Counter()
        for doc in self._docs:
            doc_freq.update(set(doc))
        count = len(self._docs)
        self._idf = {
            term: math.log(1 + (count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def score(self, query: str, index: int) -> float:
        """BM25 score of document `index` for `query`."""
        doc = self._docs[index]
        if not doc:
            return 0.0
        frequencies = Counter(doc)
        total = 0.0
        for term in set(tokenize(query)):
            freq = frequencies.get(term, 0)
            if not freq:
                continue
            norm = freq + self._k1 * (1 - self._b + self._b * len(doc) / self._avg_len)
            total += self._idf[term] * freq * (self._k1 + 1) / norm
        return total
