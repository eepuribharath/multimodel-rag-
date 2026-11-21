"""Reranker protocol — cross-encoder stage between retrieval and generation.

Implementations reorder the wide candidate set by joint query–document
relevance and cut to a small top-k. They must never add candidates: if the
right chunk is missing, fix retrieval, not the reranker.
"""

from typing import Protocol, runtime_checkable

from app.schemas import RetrievedItem


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self, query_text: str, items: list[RetrievedItem], top_k: int = 6
    ) -> list[RetrievedItem]: ...
