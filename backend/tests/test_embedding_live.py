"""Live smoke test against the real Gemini API.

Skipped automatically unless GEMINI_API_KEY is set in .env. Uses one tiny
request — safe for tight preview quotas.
"""

import pytest

from app.config import get_settings
from app.embedding import GeminiEmbedder, get_embedder
from app.schemas import Chunk, Query

settings = get_settings()

pytestmark = pytest.mark.skipif(
    not settings.gemini_api_key or settings.gemini_api_key == "paste-your-key-here",
    reason="GEMINI_API_KEY not configured in .env",
)


def test_real_embedding_roundtrip():
    embedder = get_embedder(settings)
    assert isinstance(embedder, GeminiEmbedder)

    chunk = Chunk(
        id="live-1",
        source_id="smoke",
        modality="text",
        content_ref="Reciprocal Rank Fusion combines dense and sparse rankings.",
        ordinal=0,
    )
    [chunk_vec] = embedder.embed_chunks([chunk])
    query_vec = embedder.embed_query(Query(text="What is RRF?"))

    assert len(chunk_vec) == settings.embed_dim
    assert len(query_vec) == settings.embed_dim
    # related text should be meaningfully more similar than orthogonal noise
    dot = sum(a * b for a, b in zip(chunk_vec, query_vec))
    assert dot > 0.3
