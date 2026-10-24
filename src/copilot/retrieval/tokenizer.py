"""Tokenisation with light stemming, shared by lexical search and the guardrails."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


_TOKEN = re.compile(r"[a-z0-9]+")


_STOPWORDS = frozenset(
    {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it"}
    | {"of", "on", "or", "that", "the", "this", "to", "was", "with"}
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split and lightly stem so 'flooding' and 'floods' match 'flood'."""
    tokens: list[str] = []
    for raw in _TOKEN.findall(text.lower()):
        if raw in _STOPWORDS or len(raw) < 2:
            continue
        tokens.append(_stem(raw))
    return tokens


def _stem(word: str) -> str:
    for suffix in ("ing", "es", "s", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word
