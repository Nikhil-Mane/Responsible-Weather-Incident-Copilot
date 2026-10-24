"""Retrieval (RAG) over the approved playbooks.

Public names are re-exported here so callers can write `from copilot.retrieval import ...`.
"""

from copilot.retrieval.bm25 import Bm25Index
from copilot.retrieval.embeddings import Embedder, TfidfEmbedder
from copilot.retrieval.hybrid import HybridRetriever, RetrievalQuery, RetrievedPlaybook
from copilot.retrieval.metrics import precision_at_k, recall_at_k, reciprocal_rank
from copilot.retrieval.playbooks import Playbook, PlaybookRecord, load_playbooks, topic_for
from copilot.retrieval.query import COMMUNICATIONS_TOPIC, DATA_QUALITY_TOPIC, build_query
from copilot.retrieval.rerank import CoverageReranker, Reranker
from copilot.retrieval.tokenizer import tokenize

__all__ = [
    "COMMUNICATIONS_TOPIC",
    "DATA_QUALITY_TOPIC",
    "Bm25Index",
    "CoverageReranker",
    "Embedder",
    "HybridRetriever",
    "Playbook",
    "PlaybookRecord",
    "Reranker",
    "RetrievalQuery",
    "RetrievedPlaybook",
    "TfidfEmbedder",
    "build_query",
    "load_playbooks",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "tokenize",
    "topic_for",
]
