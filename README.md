# multimodal-rag

End-to-end multimodal RAG over text, images, audio, video, and PDFs, using Gemini Embedding 2 as a single unified embedder, hybrid retrieval, a cross-encoder reranker, and Gemini for grounded generation.

This repo is **spec-first**. Each stage has a contract in `specs/` that defines its interface, constraints, and acceptance criteria before any implementation exists. Hand one spec at a time to Claude Code; the contracts are written so stages can be built and tested in isolation.

## Stack

| Layer | Choice | Notes |
|---|---|---|
| Embeddings | Gemini Embedding 2 | Native multimodal, one vector space for all modalities. Preview. |
| Vector store | Qdrant | Native dense + sparse hybrid. Runs embedded via `qdrant-client` local mode (no Docker); swap `QDRANT_URL` to a server/cloud when scaling. |
| Lexical | BM25 (Qdrant sparse) | Fused with dense via RRF. |
| Reranker | Local cross-encoder (bge-reranker-v2-m3) | On-device via sentence-transformers, no API key. Managed APIs (Cohere/Voyage/Zerank) swappable behind the same interface (see `specs/04`). |
| Generation | Gemini 3.x Pro | Flash tier as a latency/cost fallback. |
| Backend | FastAPI + Pydantic | Thin custom pipeline, no heavyweight RAG framework. |
| Frontend | Vite + React + Tailwind | Visual layer built with the Impeccable skill (see `specs/07`). |
| Eval | Ragas | Faithfulness, answer relevance, context precision/recall. Gated in CI. |
| Observability | Langfuse | Traces retrieval → rerank → generation. |

## Repo layout

```
multimodal-rag/
├── README.md                 ← you are here
├── .env.example
├── docker-compose.yml        ← optional (Langfuse self-host); Qdrant runs embedded, no Docker needed
├── specs/
│   ├── 01-ingestion.md
│   ├── 02-embedding.md
│   ├── 03-retrieval.md
│   ├── 04-rerank.md
│   ├── 05-generation.md
│   ├── 06-eval.md
│   └── 07-frontend.md
├── backend/
│   ├── pyproject.toml
│   └── app/
│       ├── main.py           ← FastAPI entrypoint, wires the pipeline
│       ├── config.py         ← settings, model IDs, dimension, top-k values
│       ├── schemas.py        ← shared Pydantic models (Chunk, RetrievedItem, Answer...)
│       ├── ingestion/        ← parse + modality-aware chunking (spec 01)
│       ├── embedding/        ← Gemini Embedding 2 client behind an interface (spec 02)
│       ├── retrieval/        ← Qdrant hybrid search + RRF (spec 03)
│       ├── rerank/           ← reranker interface + Cohere impl (spec 04)
│       ├── generation/       ← grounded answer synthesis + citations (spec 05)
│       └── api/              ← /ingest, /query, /query/stream routes
├── eval/
│   ├── datasets/             ← golden Q/A + expected sources
│   └── run_eval.py           ← Ragas harness, CI-gated (spec 06)
└── frontend/                 ← Vite + React, built with Impeccable (spec 07)
```

## Shared constraints (read before touching any stage)

These are non-negotiable and are repeated in the specs that need them.

1. **Gemini Embedding 2 per-request limits.** One vectorization call caps at ~8,192 text tokens, 6 images, 120s of video, 180s of audio, and a 6-page PDF (verified at ai.google.dev 2026-07-02; mirrored in `config.py`). Chunking must respect these *per modality* — this is why ingestion is modality-aware, not one generic splitter.
2. **One embedder, one dimension, forever (per index).** Embeddings from different models — and different truncation dimensions of the same model — do not share a vector space. Migrating means a full re-index. Pick the dimension in `config.py` (3072 default; 1536/768 via Matryoshka truncation for storage/latency savings) and never mix.
3. **Same embedder at index time and query time.** No exceptions.
4. **Preview volatility.** Gemini Embedding 2 is in public preview. Wrap the embedding call behind the interface in `specs/02` so a GA rename or rate-limit change is a one-file edit. Verify the exact model string at `ai.google.dev` — do not hardcode a guessed ID.
5. **Hybrid first, rerank second, generate last.** Retrieval recall sets the ceiling; a reranker cannot recover a document that was never retrieved. Retrieve wide (top-50/100), rerank to a small top-k (5–8), then generate.
6. **Grounded or abstain.** Generation must cite source spans and refuse when context is insufficient. See `specs/05`.

## Building with Claude Code

- Feed **one spec per session**. Each spec is self-contained with its interface and acceptance criteria.
- Build order: `02 embedding` → `03 retrieval` → `01 ingestion` → `04 rerank` → `05 generation` → `06 eval` → `07 frontend`. (Embedding and retrieval first because everything downstream depends on their data shapes.)
- After each stage, run its acceptance criteria before moving on.
- Keep `schemas.py` as the single source of truth for data shapes crossing stage boundaries; if a spec and `schemas.py` disagree, fix the spec first.
