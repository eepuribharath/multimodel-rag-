"""Identity reranker: preserves retrieval order, for tests and offline dev."""

from app.schemas import RetrievedItem


class FakeReranker:
    def rerank(
        self, query_text: str, items: list[RetrievedItem], top_k: int = 6
    ) -> list[RetrievedItem]:
        # rrf_score doubles as rerank_score so downstream always has a sort key
        return [
            item.model_copy(update={"rerank_score": item.rrf_score})
            for item in items[:top_k]
        ]
