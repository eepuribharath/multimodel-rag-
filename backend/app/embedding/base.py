"""Embedder protocol — the only surface the rest of the pipeline may see.

Index-time and query-time vectors must come from the same implementation with
the same dim; retrieval must never receive a vector from a different model or
dimension.
"""

from typing import Protocol, runtime_checkable

from app.schemas import Chunk, Query, Vector


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed_chunks(self, chunks: list[Chunk]) -> list[Vector]: ...

    def embed_query(self, query: Query) -> Vector: ...
