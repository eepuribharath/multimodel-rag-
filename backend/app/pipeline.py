"""End-to-end composition: ingest → embed → store, and
retrieve wide → rerank narrow → generate grounded.

This is the seam where Langfuse spans wrap each stage once observability is
configured; keep stage boundaries as single method calls.
"""

from pathlib import Path
from typing import Iterator

from app.config import Settings, get_settings
from app.embedding import Embedder, get_embedder
from app.generation import GatePlan, QueryGate, get_gate, get_generator, resolve_answer
from app.generation.generate import GeminiGenerator
from app.ingestion import ingest
from app.rerank import Reranker, get_reranker
from app.retrieval import VectorStore, get_vector_store
from app.schemas import Answer, Query, RetrievedItem

History = list[tuple[str, str]]  # (question, answer_text) pairs, oldest first


class RagPipeline:
    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        reranker: Reranker | None = None,
        generator: GeminiGenerator | None = None,
        gate: QueryGate | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder(self.settings)
        self.store = store or get_vector_store(self.settings)
        self.reranker = reranker or get_reranker(self.settings)
        self.generator = generator or get_generator(self.settings)
        self.gate = gate or get_gate(self.settings)

    def ingest_file(self, path: str, media_dir: Path | None = None) -> int:
        chunks = ingest(path, media_dir=media_dir)
        vectors = self.embedder.embed_chunks(chunks)
        self.store.upsert(chunks, vectors)
        return len(chunks)

    def retrieve_candidates(self, query_text: str) -> list[RetrievedItem]:
        """Stage 1: wide hybrid retrieval (dense + sparse, RRF-fused)."""
        query_vector = self.embedder.embed_query(Query(text=query_text))
        return self.store.hybrid_search(
            query_vector, query_text, top_k=self.settings.retrieve_top_k
        )

    def rerank_candidates(
        self, query_text: str, candidates: list[RetrievedItem]
    ) -> list[RetrievedItem]:
        """Stage 2: cross-encoder cut to the generation top-k."""
        return self.reranker.rerank(
            query_text, candidates, top_k=self.settings.rerank_top_k
        )

    def retrieve(self, query_text: str) -> list[RetrievedItem]:
        return self.rerank_candidates(query_text, self.retrieve_candidates(query_text))

    def plan(self, query_text: str, history: History | None = None) -> GatePlan:
        """Chitchat vs search, plus a standalone rewrite for follow-ups."""
        return self.gate.plan(query_text, history or [])

    def answer(self, query_text: str, history: History | None = None) -> Answer:
        plan = self.plan(query_text, history)
        if plan.kind == "chitchat":
            return self.generator.generate_chitchat(query_text, history)
        items = self.retrieve(plan.standalone)
        return self.generator.generate(query_text, items, history)

    def answer_stream(
        self, query_text: str, history: History | None = None
    ) -> tuple[Iterator[str], "_Finisher"]:
        """Streams raw deltas; call the finisher with the accumulated text to
        get the citation-checked Answer (the /query/stream route's shape)."""
        plan = self.plan(query_text, history)
        if plan.kind == "chitchat":
            return (
                self.generator.generate_chitchat_stream(query_text, history),
                _Finisher(items=None),
            )
        items = self.retrieve(plan.standalone)
        return self.generator.generate_stream(query_text, items, history), _Finisher(items)


class _Finisher:
    def __init__(self, items: list[RetrievedItem] | None):
        self.items = items  # None → chitchat stream, no citation check

    def __call__(self, full_text: str) -> Answer:
        from app.generation import chitchat_answer

        if self.items is None:
            return chitchat_answer(full_text)
        return resolve_answer(full_text, self.items)
