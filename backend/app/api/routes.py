"""HTTP surface: /ingest, /query, /query/stream, /health.

/query/stream speaks SSE over a POST body (consumed with fetch +
ReadableStream, not EventSource). Event sequence mirrors the real pipeline so
the UI can show stage progress instead of an opaque spinner:

    event: stage      data: retrieving | reranking | generating
    event: delta      data: <text fragment, [n] markers included>
    event: answer     data: <Answer JSON, citation-checked>
    event: error      data: <message>
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import PROJECT_ROOT, get_settings
from app.generation import chitchat_answer, locator, resolve_answer
from app.history import HistoryStore
from app.pipeline import RagPipeline
from app.schemas import Answer, RetrievedItem, chunk_match_text

router = APIRouter()

MEDIA_DIR = PROJECT_ROOT / "media_store"
UPLOADS_DIR = MEDIA_DIR / "uploads"
SNIPPET_CHARS = 240


def _media_url(content_ref: str) -> str | None:
    """Chunks whose content lives under media_store are served at /media."""
    try:
        relative = Path(content_ref).resolve().relative_to(MEDIA_DIR.resolve())
    except ValueError:
        return None
    return f"/media/{relative}"


def _evidence_payload(
    top: list[RetrievedItem], candidates: list[RetrievedItem]
) -> dict:
    """Evidence cards for the reranked top-k, plus pre-rerank candidates for
    the retrieval inspector."""
    return {
        "items": [
            {
                "chunk_id": item.chunk.id,
                "source_id": item.chunk.source_id,
                "modality": item.chunk.modality,
                "locator": locator(item.chunk),
                "snippet": chunk_match_text(item.chunk)[:SNIPPET_CHARS],
                "media_url": _media_url(item.chunk.content_ref)
                if item.chunk.modality != "text"
                else None,
                "start_ms": item.chunk.metadata.get("start_ms"),
                "rrf_score": item.rrf_score,
                "rerank_score": item.rerank_score,
            }
            for item in top
        ],
        "pre_rerank": [
            {
                "chunk_id": item.chunk.id,
                "source_id": item.chunk.source_id,
                "modality": item.chunk.modality,
                "rrf_score": item.rrf_score,
            }
            for item in candidates[:10]
        ],
    }


def get_pipeline(request: Request) -> RagPipeline:
    # lazy singleton: the reranker model load (~seconds) happens on first
    # request, not at import, so the server boots fast and tests can inject
    if not hasattr(request.app.state, "pipeline"):
        request.app.state.pipeline = RagPipeline()
    return request.app.state.pipeline


def get_history(request: Request) -> HistoryStore:
    if not hasattr(request.app.state, "history"):
        request.app.state.history = HistoryStore()
    return request.app.state.history


class QueryRequest(BaseModel):
    question: str
    chat_id: int | None = None  # omitted → a new chat is created


def _resolve_chat(store: HistoryStore, chat_id: int | None) -> tuple[int, bool]:
    """Returns (chat_id, created)."""
    if chat_id is None:
        return store.create_chat()["id"], True
    if not store.chat_exists(chat_id):
        raise HTTPException(404, f"chat {chat_id} not found")
    return chat_id, False


class IngestResponse(BaseModel):
    source_id: str
    chunks: int


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/ingest")
def ingest_file(file: UploadFile, request: Request) -> IngestResponse:
    if not file.filename:
        raise HTTPException(400, "upload needs a filename")
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOADS_DIR / Path(file.filename).name
    destination.write_bytes(file.file.read())
    try:
        count = get_pipeline(request).ingest_file(str(destination))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
    return IngestResponse(source_id=destination.name, chunks=count)


@router.post("/chats")
def create_chat(request: Request) -> dict:
    return get_history(request).create_chat()


@router.get("/chats")
def list_chats(request: Request) -> list[dict]:
    return get_history(request).list_chats()


@router.get("/chats/{chat_id}")
def get_chat(chat_id: int, request: Request) -> list[dict]:
    store = get_history(request)
    if not store.chat_exists(chat_id):
        raise HTTPException(404, f"chat {chat_id} not found")
    return store.get_turns(chat_id)


@router.delete("/chats/{chat_id}", status_code=204)
def delete_chat(chat_id: int, request: Request) -> Response:
    get_history(request).delete_chat(chat_id)
    return Response(status_code=204)


@router.post("/query")
def query(body: QueryRequest, request: Request, response: Response) -> Answer:
    store = get_history(request)
    chat_id, _created = _resolve_chat(store, body.chat_id)
    history = store.recent_history(chat_id, n=get_settings().history_turns)
    answer = get_pipeline(request).answer(body.question, history)
    store.add_turn(chat_id, body.question, answer.model_dump_json())
    response.headers["X-Chat-Id"] = str(chat_id)
    return answer


def _sse(event: str, data: str) -> str:
    # SSE frames: one data: line per newline in the payload
    lines = "".join(f"data: {line}\n" for line in data.split("\n"))
    return f"event: {event}\n{lines}\n"


@router.post("/query/stream")
def query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
    pipeline = get_pipeline(request)
    store = get_history(request)
    chat_id, created = _resolve_chat(store, body.chat_id)
    history = store.recent_history(chat_id, n=get_settings().history_turns)

    def events():
        try:
            if created:
                yield _sse("chat", json.dumps({"chat_id": chat_id}))

            plan = pipeline.plan(body.question, history)
            if plan.kind == "chitchat":
                yield _sse("stage", "generating")
                collected: list[str] = []
                for delta in pipeline.generator.generate_chitchat_stream(
                    body.question, history
                ):
                    collected.append(delta)
                    yield _sse("delta", delta)
                answer = chitchat_answer("".join(collected))
                store.add_turn(chat_id, body.question, answer.model_dump_json())
                yield _sse("answer", answer.model_dump_json())
                return

            yield _sse("stage", "retrieving")
            candidates = pipeline.retrieve_candidates(plan.standalone)
            yield _sse("stage", "reranking")
            top = pipeline.rerank_candidates(plan.standalone, candidates)
            evidence = _evidence_payload(top, candidates)
            yield _sse("evidence", json.dumps(evidence))
            yield _sse("stage", "generating")
            collected = []
            for delta in pipeline.generator.generate_stream(
                body.question, top, history
            ):
                collected.append(delta)
                yield _sse("delta", delta)
            answer = resolve_answer("".join(collected), top)
            store.add_turn(
                chat_id, body.question, answer.model_dump_json(),
                evidence_json=json.dumps(evidence),
            )
            yield _sse("answer", answer.model_dump_json())
        except Exception as exc:  # surface, don't hang the stream
            yield _sse("error", json.dumps({"message": str(exc)}))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
