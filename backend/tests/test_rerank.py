"""Spec 04 acceptance criteria.

- a relevant chunk sitting at rank 30 by rrf_score gets promoted into the
  returned top-k
- swapping LocalReranker for FakeReranker requires no change elsewhere
- non-text chunks are reranked via their text surrogate without error

Fast tests use a stub scorer; the slow test runs a real (small) cross-encoder.
"""

import pytest

from app.rerank import FakeReranker, LocalReranker, Reranker
from app.schemas import Chunk, RetrievedItem

RELEVANT = "The maximum torque for compressor XK-42817 is 5.5 Nm."


def item(i: int, text: str, modality: str = "text", metadata: dict | None = None) -> RetrievedItem:
    return RetrievedItem(
        chunk=Chunk(
            id=f"c{i}",
            source_id="doc",
            modality=modality,
            content_ref=text if modality == "text" else f"/media/{i}",
            ordinal=i,
            metadata=metadata or {},
        ),
        dense_score=None,
        sparse_score=None,
        rrf_score=1.0 / (60 + i + 1),  # descending with i, like real RRF output
    )


class KeywordScorer:
    """Stub cross-encoder: relevance = keyword presence."""

    def predict(self, pairs, batch_size=32):
        return [1.0 if "torque" in doc else 0.01 for _query, doc in pairs]


class TestLocalReranker:
    def test_rank_30_promoted_into_top_k(self):
        items = [item(i, f"unrelated filler passage number {i}") for i in range(40)]
        items[29] = item(29, RELEVANT)  # rank 30 by rrf_score

        reranker = LocalReranker(model=KeywordScorer())
        top = reranker.rerank("what torque does XK-42817 need?", items, top_k=6)

        assert len(top) == 6
        assert top[0].chunk.id == "c29"
        assert top[0].rerank_score is not None
        assert all(a.rerank_score >= b.rerank_score for a, b in zip(top, top[1:]))

    def test_non_text_chunks_use_surrogate(self):
        items = [
            item(0, "filler text passage"),
            item(1, "", modality="image", metadata={"caption": "torque wrench diagram"}),
            item(2, "", modality="audio", metadata={"transcript": "we discussed the schedule"}),
            item(3, "", modality="video"),  # no surrogate at all → scores as ""
        ]
        reranker = LocalReranker(model=KeywordScorer())
        top = reranker.rerank("torque specs", items, top_k=4)

        assert top[0].chunk.id == "c1"  # caption carried the signal
        assert {i.chunk.id for i in top} == {"c0", "c1", "c2", "c3"}  # no crash, none dropped

    def test_empty_candidates(self):
        assert LocalReranker(model=KeywordScorer()).rerank("q", [], top_k=6) == []


class TestSwappability:
    def test_both_satisfy_protocol(self):
        assert isinstance(FakeReranker(), Reranker)
        assert isinstance(LocalReranker(model=KeywordScorer()), Reranker)

    def test_fake_is_identity_with_cut(self):
        items = [item(i, f"passage {i}") for i in range(10)]
        top = FakeReranker().rerank("anything", items, top_k=6)
        assert [i.chunk.id for i in top] == [f"c{i}" for i in range(6)]
        assert all(i.rerank_score == i.rrf_score for i in top)


@pytest.mark.slow
class TestRealModel:
    def test_real_cross_encoder_promotes_relevant(self):
        # small English model (~90MB) so the test doesn't pull the 2.3GB default
        reranker = LocalReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        items = [
            item(0, "The cafeteria menu changes every Monday morning."),
            item(1, "Annual leave requests go through the HR portal."),
            item(2, "Parking permits are issued at the front desk."),
            item(3, RELEVANT),
            item(4, "The fire drill schedule is posted quarterly."),
        ]
        top = reranker.rerank(
            "what is the maximum torque for compressor XK-42817?", items, top_k=3
        )
        assert top[0].chunk.id == "c3"
