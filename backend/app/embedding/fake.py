"""Deterministic offline embedder for tests and local dev.

Vectors are unit-norm and derived purely from content, so equal inputs embed
identically across runs — no network, no API key.
"""

import hashlib
import math
import random

from app.schemas import Chunk, Query, Vector


class FakeEmbedder:
    def __init__(self, dim: int = 3072):
        self.dim = dim

    def embed_chunks(self, chunks: list[Chunk]) -> list[Vector]:
        return [self._vector(f"{c.modality}:{c.content_ref}") for c in chunks]

    def embed_query(self, query: Query) -> Vector:
        return self._vector(f"query:{query.text or ''}:{query.media_path or ''}")

    def _vector(self, key: str) -> Vector:
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        rng = random.Random(seed)
        raw = [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]
