"""Spec 03 acceptance criteria, against embedded in-memory Qdrant.

- upsert N chunks + query → items with populated rrf_score, sorted descending
- exact-identifier query ranks the matching chunk top even when semantically
  dissimilar (sparse leg works)
- paraphrased query with no lexical overlap still retrieves the right chunk
  (dense leg works)
- a text query can return an image chunk (cross-modal)
"""

import math
import random

import pytest
from qdrant_client import QdrantClient

from app.embedding import FakeEmbedder
from app.retrieval import VectorStore
from app.schemas import Chunk

DIM = 32


@pytest.fixture
def store() -> VectorStore:
    return VectorStore(
        client=QdrantClient(":memory:"), collection="test", dim=DIM
    )


def text_chunk(i: int, text: str, source_id: str = "doc-1") -> Chunk:
    return Chunk(
        id=f"chunk-{i}",
        source_id=source_id,
        modality="text",
        content_ref=text,
        ordinal=i,
    )


def unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def random_unit(seed: int) -> list[float]:
    rng = random.Random(seed)
    return unit([rng.gauss(0, 1) for _ in range(DIM)])


def near(vec: list[float], seed: int, eps: float = 0.05) -> list[float]:
    rng = random.Random(seed)
    return unit([x + eps * rng.gauss(0, 1) for x in vec])


class TestHybridSearch:
    def test_rrf_scores_populated_and_sorted(self, store):
        embedder = FakeEmbedder(dim=DIM)
        chunks = [
            text_chunk(i, f"passage about topic number {i}") for i in range(10)
        ]
        store.upsert(chunks, embedder.embed_chunks(chunks))

        items = store.hybrid_search(
            query_vector=random_unit(99), query_text="passage about topic", top_k=10
        )

        assert items
        scores = [item.rrf_score for item in items]
        assert all(s > 0 for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_sparse_leg_exact_identifier(self, store):
        chunks = [
            text_chunk(0, "routine maintenance procedure for the intake assembly"),
            text_chunk(1, "compressor part XK-42817 requires torque spec 4.2 Nm"),
            text_chunk(2, "annual inspection checklist for hydraulic systems"),
        ] + [text_chunk(3 + i, f"unrelated filler passage {i}") for i in range(7)]
        # dense vectors are random — no semantic signal, sparse must carry it
        store.upsert(chunks, [random_unit(i) for i in range(len(chunks))])

        items = store.hybrid_search(
            query_vector=random_unit(1234), query_text="XK-42817", top_k=10
        )

        assert items[0].chunk.id == "chunk-1"
        assert items[0].sparse_score is not None

    def test_dense_leg_paraphrase_no_lexical_overlap(self, store):
        target_vec = random_unit(7)
        chunks = [
            text_chunk(0, "the feline slept on the warm windowsill"),
            text_chunk(1, "quarterly financial results exceeded projections"),
            text_chunk(2, "installation guide for the water pump"),
        ]
        # chunk 0 is semantically near the query; the query shares zero tokens
        store.upsert(chunks, [target_vec, random_unit(8), random_unit(9)])

        items = store.hybrid_search(
            query_vector=near(target_vec, seed=42),
            query_text="cat napping in sunshine",
            top_k=10,
        )

        assert items[0].chunk.id == "chunk-0"
        assert items[0].dense_score is not None

    def test_cross_modal_text_query_returns_image(self, store):
        target_vec = random_unit(5)
        image = Chunk(
            id="img-1",
            source_id="slides.pdf",
            modality="image",
            content_ref="/media/figure3.png",
            ordinal=3,
            metadata={"caption": "architecture diagram of the ingestion pipeline"},
        )
        text = text_chunk(0, "meeting notes from the retrospective")
        store.upsert([image, text], [target_vec, random_unit(6)])

        items = store.hybrid_search(
            query_vector=near(target_vec, seed=43),
            query_text="ingestion pipeline architecture",
            top_k=5,
        )

        assert items[0].chunk.modality == "image"
        assert items[0].chunk.metadata["caption"].startswith("architecture")
        # sparse leg matched the caption surrogate too
        assert items[0].sparse_score is not None

    def test_metadata_filter(self, store):
        chunks = [
            text_chunk(0, "alpha content", source_id="doc-a"),
            text_chunk(1, "alpha content", source_id="doc-b"),
        ]
        store.upsert(chunks, [random_unit(1), random_unit(2)])

        items = store.hybrid_search(
            query_vector=random_unit(3),
            query_text="alpha content",
            filters={"source_id": "doc-b"},
        )

        assert [i.chunk.source_id for i in items] == ["doc-b"]

    def test_dim_mismatch_rejected(self, store):
        with pytest.raises(ValueError, match="dim"):
            store.upsert([text_chunk(0, "x")], [[0.1] * (DIM + 1)])

    def test_collection_dim_change_rejected(self):
        client = QdrantClient(":memory:")
        VectorStore(client=client, collection="c", dim=DIM)
        with pytest.raises(RuntimeError, match="re-index"):
            VectorStore(client=client, collection="c", dim=DIM * 2)
