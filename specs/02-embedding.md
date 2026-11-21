# Spec 02 — Embedding (Gemini Embedding 2 client)

## Goal
Turn `Chunk`s and queries into vectors in a single unified space, behind an interface thin enough that a preview→GA change is a one-file edit.

## Module
`backend/app/embedding/`

## Interface
```python
# embedding/base.py
class Embedder(Protocol):
    dim: int
    def embed_chunks(self, chunks: list[Chunk]) -> list[Vector]: ...
    def embed_query(self, query: Query) -> Vector: ...   # Query may be text OR media

# embedding/gemini.py
class GeminiEmbedder(Embedder):
    def __init__(self, model: str, dim: int, task_type: str): ...
```

`Vector = list[float]` of length `dim`.

## Constraints & decisions
- **Model:** Gemini Embedding 2 via the Gemini API (or Vertex AI). Verify the exact model string at `ai.google.dev` before wiring — do NOT hardcode a guessed ID. Put it in `config.py`.
- **Dimension is a permanent choice.** Default 3072. If you truncate via Matryoshka (1536/768) for storage/latency, that dimension is baked into the index — changing it later means a full re-index. Set once in `config.py`; assert `len(vector) == config.EMBED_DIM` on every call.
- **Same embedder, same dimension, both index-time and query-time.** The retrieval layer must never see a vector from a different model or dimension.
- **Respect batch limits:** ≤6 images per request; text ≤8192 tokens; audio ≤180s; video ≤120s; PDF ≤6 pages (verified at ai.google.dev 2026-07-02; constants in `config.py`). Batch text chunks up to the request cap; media respects per-item limits. This layer enforces limits and retries on rate-limit (exponential backoff) — preview quotas are tight.
- **Interleaved input** is supported (e.g. image + caption in one call) — expose it via `embed_query` so a multimodal query works, but keep the default path simple.
- **task_type** applies only to the gemini-embedding-001 family (confirmed 2026-07-02). Gemini Embedding 2 takes task instructions in the prompt text instead; the client accepts `task_type` for interface compatibility but does not send it.

## Dependencies
`google-genai` (or Vertex AI SDK), `tenacity` for backoff.

## Acceptance criteria
- `embed_query("...")` and `embed_chunks([...])` both return vectors of exactly `config.EMBED_DIM`.
- A batch of 20 text chunks is transparently split into request-sized batches and returns 20 vectors in order.
- Swapping `GeminiEmbedder` for a fake `Embedder` in tests requires no change to retrieval/ingestion.
- A simulated 429 triggers backoff-and-retry, not a crash.

## Claude Code notes
Define the `Embedder` protocol first and write retrieval/ingestion tests against a `FakeEmbedder`. Only then implement `GeminiEmbedder`. This keeps the preview API from blocking everything else.
