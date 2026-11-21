"""Local sentence-transformers embedder (no API key required)."""

from sentence_transformers import SentenceTransformer
from app.schemas import Chunk, Query, Vector, chunk_match_text


class LocalEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int = 384,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = dim

    def embed_chunks(self, chunks: list[Chunk]) -> list[Vector]:
        texts = [chunk_match_text(c) or "(no content)" for c in chunks]
        embeddings = self.model.encode(texts, batch_size=32, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    def embed_query(self, query: Query) -> Vector:
        text = query.text or ""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
