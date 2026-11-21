# Spec 04 — Rerank (cross-encoder)

## Goal
Reorder the wide candidate set by joint query–document relevance and cut it to a small, high-precision top-k for generation. This is the single highest-leverage quality step (+5 to +15 NDCG@10 typical).

## Module
`backend/app/rerank/`

## Interface
```python
# rerank/base.py
class Reranker(Protocol):
    def rerank(self, query_text: str, items: list[RetrievedItem],
               top_k: int = 6) -> list[RetrievedItem]: ...

# rerank/cohere.py
class CohereReranker(Reranker): ...
```
`RetrievedItem` gains a `rerank_score: float | None` field.

## Constraints & decisions
- **Default model:** local cross-encoder `BAAI/bge-reranker-v2-m3` via sentence-transformers (decided 2026-07-02: no extra account/API key, runs on-device; torch was already a dependency via docling). Keep it behind the `Reranker` protocol so managed rerankers (Cohere Rerank 4, Zerank 2, Voyage Rerank 2.5) can drop in — the choice should be settled by eval (spec 06), not by default.
- **Input size:** rerank the top-50/100 from retrieval; return top 5–8. Reranking is text-relevance scored, so for non-text chunks pass a text surrogate (caption/transcript/OCR/title from `metadata`), not raw media.
- **Context caps:** respect the reranker's per-document token limit; truncate or use the chunk's text surrogate.
- **Latency budget:** reranking adds ~100–600ms for ~100 docs. Acceptable for a synchronous query; expose `top_k` and candidate count as config so it can be tuned.
- **Do not treat rerank as a retrieval fix.** If the right chunk is not in the candidate set, widen retrieval (spec 03) — the reranker cannot invent it.

## Dependencies
`cohere` (or the chosen provider's SDK).

## Acceptance criteria
- Given a candidate list where the most relevant chunk sits at rank 30 by `rrf_score`, the reranker promotes it into the returned top-k.
- Swapping `CohereReranker` for a `FakeReranker` (identity order) requires no change elsewhere.
- Non-text chunks are reranked via their text surrogate without error.

## Claude Code notes
Build the protocol + `FakeReranker` first so generation can be developed offline. Add the Cohere impl behind an API-key check that degrades to identity ordering if the key is absent (keeps local dev unblocked).
