"""Local cross-encoder reranker via sentence-transformers.

Runs fully on-device (MPS on Apple Silicon), no account or per-query cost.
Non-text chunks are scored via their text surrogate (caption/transcript/
ocr/title) — raw media never reaches the cross-encoder.

The default model is downloaded from Hugging Face on first use and cached.
Managed APIs (Cohere, Voyage, Zerank) can drop in behind the same protocol
if spec 06's eval bake-off favors them.
"""

from app.schemas import RetrievedItem, chunk_match_text

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
MAX_DOC_TOKENS = 512  # per-document cap; the tokenizer truncates beyond this


class LocalReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        model=None,  # injectable for tests: anything with .predict(pairs)
    ):
        if model is None:
            from sentence_transformers import CrossEncoder  # heavy import

            model = CrossEncoder(model_name, max_length=MAX_DOC_TOKENS)
        self.model = model
        self.batch_size = batch_size

    def rerank(
        self, query_text: str, items: list[RetrievedItem], top_k: int = 6
    ) -> list[RetrievedItem]:
        if not items:
            return []
        pairs = [(query_text, chunk_match_text(item.chunk)) for item in items]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        rescored = [
            item.model_copy(update={"rerank_score": float(score)})
            for item, score in zip(items, scores)
        ]
        rescored.sort(key=lambda item: item.rerank_score, reverse=True)
        return rescored[:top_k]
