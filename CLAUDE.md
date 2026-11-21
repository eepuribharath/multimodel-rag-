# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

End-to-end multimodal RAG over text, images, audio, video, and PDFs. Pipeline: Gemini Embedding 2 (one unified embedder for all modalities) → Qdrant hybrid retrieval (dense + BM25 sparse, fused with RRF) → local cross-encoder rerank (bge-reranker-v2-m3 via sentence-transformers, swappable) → Gemini 3.x Pro grounded generation with citations. Backend is FastAPI + Pydantic; frontend is Vite + React + Tailwind; eval is Ragas (CI-gated); tracing is Langfuse.

## Spec-first

Each stage's contract — interface, constraints, acceptance criteria — lives in `specs/01`–`07`. All seven specs are implemented (as of 2026-07-02): specs 01–06 under `backend/app/` (tests in `backend/tests/`), eval harness in `eval/`, FastAPI routes in `app/api/routes.py` (/ingest upload, /query, /query/stream SSE emitting stage → evidence → delta → answer events, /media static mount), and the spec 07 frontend in `frontend/` (Vite + React + TS + Tailwind v4; SSE client in `src/api.ts`). Media queries (attaching an image/audio *as the question*) are not wired through the API yet — text queries only; the query-bar attach control ingests into the library instead.

Chat history is multi-chat (SQLite via `app/history.py` → `history.db` at repo root, gitignored; /chats CRUD routes; sidebar UI). Each query passes through a gate (`app/generation/gate.py`): social messages get a chitchat reply with no retrieval (`Answer.kind="chitchat"`); follow-ups are rewritten standalone via one flash call before retrieval; failures fall back to searching the raw question. Recent turns (`HISTORY_TURNS`, default 4) are included in the generation prompt. Note: flash models spend thinking tokens from `max_output_tokens` — keep generation budgets generous or replies truncate mid-sentence. Langfuse tracing deferred (needs a cloud account or Docker — user wants neither yet); `app/pipeline.py` stage boundaries are where spans go when it lands.

- Work **one spec per session**; read the spec fully before implementing.
- Build order: `02 embedding → 03 retrieval → 01 ingestion → 04 rerank → 05 generation → 06 eval → 07 frontend` (downstream stages depend on upstream data shapes).
- For each stage, define the protocol + a Fake implementation first (`FakeEmbedder`, `FakeReranker`) so stages are testable offline and preview-API outages don't block work.
- Run the spec's acceptance criteria before moving to the next stage.

## Environment: local venv only

All Python work happens inside the project-local virtualenv at `venv/` (`source venv/bin/activate`, or call `venv/bin/python` / `venv/bin/pip` directly). Never install packages globally or with `pip install --user` — every `pip`/`uv` install targets the project venv. Same rule for tooling (pytest, ruff, etc.). Frontend deps stay in the project's `node_modules`; no `npm install -g`.

## Commands

```bash
cp .env.example .env                       # then fill in API keys
venv/bin/pip install -e "backend[dev]"     # backend deps into the project venv
cd backend && ../venv/bin/python -m pytest # run backend tests
cd backend && ../venv/bin/python -m uvicorn app.main:app --reload --port 8000  # API server
cd frontend && npm run dev                 # UI at http://localhost:5173 (expects API on :8000)
venv/bin/python scripts/ingest_library.py  # ingest everything in library/ (works with server up or down)
cd frontend && npm run build               # type-check + production build
```

**No Docker.** The user does not want Docker. Qdrant runs embedded via `qdrant-client` local mode (`QDRANT_URL=local` → persists to `./qdrant_data`; `:memory:` in tests). For observability prefer Langfuse Cloud over the docker-compose Langfuse. `.env` exists with real keys — never print or commit it; tests needing the real Gemini API self-skip when the key is absent.

Run a single test: `../venv/bin/python -m pytest tests/test_embedding.py -k <name>` (from `backend/`).

Eval (live API, from repo root): `venv/bin/python eval/run_eval.py` — compares against `eval/baselines.json`, exits non-zero on regression (the CI gate). `--update-baselines` resets floors after a deliberate config change; `--full` adds Ragas answer metrics (needs `pip install ragas langchain-google-genai` and judge quota).

Default to `-m "not slow"` — the `slow` marker covers docling PDF tests (~90s, real layout models). Audio ingestion works on WAV without ffmpeg; mp3/m4a would need ffmpeg installed. PDF ingestion runs with `do_ocr=False` (born-digital PDFs only; docling 2.108's default OCR config also breaks on this platform).

## Non-negotiable constraints

These cut across stages and are repeated in the specs that need them:

1. **Gemini Embedding 2 per-request limits:** ~8,192 text tokens, 6 images, 128s video, 80s audio, 6 PDF pages. Chunking is modality-aware for this reason — no generic splitter.
2. **One embedder, one dimension, forever (per index).** `EMBED_DIM` (3072 default; 1536/768 via Matryoshka truncation) is permanent — changing it means a full re-index. Same embedder and dimension at index time and query time, no exceptions. Assert vector length on every call.
3. **Preview volatility.** Gemini Embedding 2 is in public preview. Verify exact model IDs at ai.google.dev — never hardcode a guessed ID. Keep the API behind the `Embedder` interface so a GA rename is a one-file edit.
4. **Hybrid first, rerank second, generate last.** Retrieve wide (top-100), rerank to a small top-k (5–8), then generate. A reranker cannot recover a document retrieval missed — if the right chunk isn't in the candidates, widen retrieval, don't tune the reranker.
5. **Grounded or abstain.** Every claim cites a `chunk_id` that was actually in the input items; citation integrity is a hard post-processing check, not a hope about model behavior. Insufficient context → `abstained = True`, never a fabricated answer.
6. **`schemas.py` is the single source of truth** for data shapes crossing stage boundaries (`Chunk`, `RetrievedItem`, `Answer`, `Citation`). If a spec and `schemas.py` disagree, fix the spec first.
7. **Ingestion fidelity:** deterministic chunk IDs (re-ingesting the same file yields identical IDs); preserve citation anchors (page, timestamp range, bbox) in metadata; never lowercase or strip exact identifiers — BM25 depends on them.
8. **Eval decides swaps.** Component choices (reranker, dimension, top-k) are settled by `eval/run_eval.py` numbers against `eval/baselines.json`, not defaults. CI fails on regression below baseline.

## Frontend note

Spec 07 defines structure, state, and data contracts only. The visual layer (layout, type, spacing, color, motion) is built with the Impeccable skill — keep them separate.
