# Spec 03 — Retrieval (Qdrant hybrid + RRF)

## Goal
Given a query, return a wide candidate set combining dense semantic and sparse lexical matches, fused into a single ranked list.

## Module
`backend/app/retrieval/`

## Interface
```python
# schemas.py
class RetrievedItem(BaseModel):
    chunk: Chunk
    dense_score: float | None
    sparse_score: float | None
    rrf_score: float

# retrieval/store.py
class VectorStore:
    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> None: ...
    def hybrid_search(self, query_vector: Vector, query_text: str,
                      top_k: int = 100) -> list[RetrievedItem]: ...
```

## Constraints & decisions
- **Store:** Qdrant, one collection. Configure a named dense vector (size = `config.EMBED_DIM`, cosine) and a sparse vector (BM25/SPLADE-style) in the same collection so a single query hits both.
- **Fusion:** Reciprocal Rank Fusion, `k = 60`. Prefer Qdrant's server-side fusion query; if computing client-side: `rrf_score(d) = Σ_i 1/(k + rank_i(d))` over the dense and sparse result lists.
- **Retrieve wide.** Default `top_k = 100` candidates out of retrieval; the reranker (spec 04) narrows to 5–8. Recall here caps end-to-end quality — do not under-retrieve to save reranker cost.
- **Payload:** store the full `Chunk` (including citation metadata and `modality`) as the point payload so retrieved items are self-describing and cross-modal (a text query can return an image chunk).
- **Filtering:** support optional metadata filters (source_id, modality, date) passed through to Qdrant.

## Dependencies
`qdrant-client`. Local Qdrant via `docker-compose.yml`.

## Acceptance criteria
- Upserting N chunks and querying returns items with populated `rrf_score`, sorted descending.
- A query using an exact identifier that appears in one chunk ranks that chunk in the top results even if it is semantically dissimilar (proves the sparse leg works).
- A paraphrased query with no lexical overlap still retrieves the right chunk (proves the dense leg works).
- A text query can return an `image` or `audio` chunk (proves cross-modal retrieval).

## Claude Code notes
Stand up Qdrant from `docker-compose.yml` first. Seed with a `FakeEmbedder` so this stage is testable before the real embedder exists. Verify server-side hybrid query syntax against the installed `qdrant-client` version.
