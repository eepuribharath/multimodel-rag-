"""Live spec 05 smoke tests against the real Gemini API (two small calls).

Skipped unless GEMINI_API_KEY is configured.
"""

import pytest

from app.config import get_settings
from app.generation import get_generator
from app.schemas import Chunk, RetrievedItem

settings = get_settings()

pytestmark = pytest.mark.skipif(
    not settings.gemini_api_key or settings.gemini_api_key == "paste-your-key-here",
    reason="GEMINI_API_KEY not configured in .env",
)


def item(i: int, text: str) -> RetrievedItem:
    return RetrievedItem(
        chunk=Chunk(
            id=f"live{i}", source_id="manual.pdf", modality="text",
            content_ref=text, ordinal=i, metadata={"page": i + 1},
        ),
        dense_score=None, sparse_score=None, rrf_score=1.0,
    )


ITEMS = [
    item(0, "The maximum torque for compressor XK-42817 is 5.5 Nm."),
    item(1, "Annual inspections are scheduled every March."),
]


def test_answerable_query_cites_sources():
    answer = get_generator(settings).generate(
        "What is the maximum torque for compressor XK-42817?", ITEMS
    )
    assert not answer.abstained
    assert "5.5" in answer.text
    assert answer.citations
    assert all(c.chunk_id in {"live0", "live1"} for c in answer.citations)


def test_out_of_corpus_query_abstains():
    answer = get_generator(settings).generate(
        "What is the boiling point of tungsten?", ITEMS
    )
    assert answer.abstained
    assert answer.citations == []
