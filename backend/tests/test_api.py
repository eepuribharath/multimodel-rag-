"""API routes, offline: fake pipeline injected into app.state."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.embedding import FakeEmbedder
from app.generation import QueryGate
from app.generation.generate import GeminiGenerator
from app.history import HistoryStore
from app.main import create_app
from app.pipeline import RagPipeline
from app.rerank import FakeReranker
from app.retrieval import VectorStore
from app.schemas import Chunk

ANSWER_TEXT = "The torque is 5.5 Nm [1]."


class ScriptedClient:
    def __init__(self, text: str):
        self.models = SimpleNamespace(
            generate_content=lambda **kw: SimpleNamespace(text=text),
            generate_content_stream=lambda **kw: iter(
                [SimpleNamespace(text=piece) for piece in (text[: len(text) // 2], text[len(text) // 2 :])]
            ),
        )


@pytest.fixture
def client(tmp_path) -> TestClient:
    embedder = FakeEmbedder(dim=32)
    store = VectorStore(client=QdrantClient(":memory:"), collection="api", dim=32)
    chunks = [
        Chunk(id="k0", source_id="manual.pdf", modality="text",
              content_ref="Torque spec for XK-42817 is 5.5 Nm.", ordinal=0,
              metadata={"page": 3}),
    ]
    store.upsert(chunks, embedder.embed_chunks(chunks))
    pipeline = RagPipeline(
        embedder=embedder,
        store=store,
        reranker=FakeReranker(),
        generator=GeminiGenerator(model="m", client=ScriptedClient(ANSWER_TEXT)),
        gate=QueryGate(model="m", client=None),  # heuristic tiers only
    )
    app = create_app()
    app.state.pipeline = pipeline
    app.state.history = HistoryStore(db_path=tmp_path / "history.db")
    return TestClient(app)


def parse_sse(body: str) -> list[tuple[str, str]]:
    events = []
    for frame in body.strip().split("\n\n"):
        event, data = None, []
        for line in frame.split("\n"):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
        events.append((event, "\n".join(data)))
    return events


class TestRoutes:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_query_returns_cited_answer(self, client):
        response = client.post("/query", json={"question": "XK-42817 torque?"})
        assert response.status_code == 200
        answer = response.json()
        assert not answer["abstained"]
        assert answer["citations"][0]["chunk_id"] == "k0"
        assert answer["citations"][0]["locator"] == "p. 3"

    def test_query_stream_stages_deltas_answer(self, client):
        with client.stream(
            "POST", "/query/stream", json={"question": "XK-42817 torque?"}
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            events = parse_sse(response.read().decode())

        import json

        # no chat_id sent → server created one and announced it first
        assert events[0][0] == "chat"
        assert "chat_id" in json.loads(events[0][1])

        kinds = [e for e, _ in events]
        assert [d for e, d in events if e == "stage"] == [
            "retrieving", "reranking", "generating",
        ]
        evidence = json.loads(next(d for e, d in events if e == "evidence"))
        assert evidence["items"][0]["chunk_id"] == "k0"
        assert evidence["items"][0]["locator"] == "p. 3"
        assert evidence["items"][0]["snippet"].startswith("Torque spec")
        assert evidence["pre_rerank"]
        deltas = "".join(d for e, d in events if e == "delta")
        assert deltas == ANSWER_TEXT
        assert kinds[-1] == "answer"
        final = json.loads(events[-1][1])
        assert final["used_chunk_ids"] == ["k0"]

    def test_chats_lifecycle_and_turn_recording(self, client):
        chat = client.post("/chats").json()
        chat_id = chat["id"]

        response = client.post(
            "/query", json={"question": "XK-42817 torque?", "chat_id": chat_id}
        )
        assert response.status_code == 200
        assert response.headers["X-Chat-Id"] == str(chat_id)

        chats = client.get("/chats").json()
        assert chats[0]["id"] == chat_id
        assert chats[0]["title"] == "XK-42817 torque?"  # auto-titled
        assert chats[0]["turns"] == 1

        turns = client.get(f"/chats/{chat_id}").json()
        assert turns[0]["question"] == "XK-42817 torque?"
        assert turns[0]["answer"]["kind"] == "grounded"

        assert client.delete(f"/chats/{chat_id}").status_code == 204
        assert client.get(f"/chats/{chat_id}").status_code == 404

    def test_unknown_chat_id_404(self, client):
        response = client.post("/query", json={"question": "x?", "chat_id": 999})
        assert response.status_code == 404

    def test_stream_records_turn_with_evidence(self, client):
        chat_id = client.post("/chats").json()["id"]
        with client.stream(
            "POST", "/query/stream",
            json={"question": "XK-42817 torque?", "chat_id": chat_id},
        ) as response:
            events = parse_sse(response.read().decode())
        assert all(e != "chat" for e, _ in events)  # existing chat: no chat event
        turns = client.get(f"/chats/{chat_id}").json()
        assert len(turns) == 1
        assert turns[0]["evidence"]["items"][0]["chunk_id"] == "k0"

    def test_hi_is_chitchat_no_retrieval(self, client):
        import json

        with client.stream(
            "POST", "/query/stream", json={"question": "hi"}
        ) as response:
            events = parse_sse(response.read().decode())

        kinds = [e for e, _ in events]
        assert "evidence" not in kinds  # retrieval skipped entirely
        assert [d for e, d in events if e == "stage"] == ["generating"]
        final = json.loads(events[-1][1])
        assert final["kind"] == "chitchat"
        assert not final["abstained"]
        assert final["citations"] == []
        # recorded in history too
        chat_id = json.loads(events[0][1])["chat_id"]
        turns = client.get(f"/chats/{chat_id}").json()
        assert turns[0]["answer"]["kind"] == "chitchat"
        assert turns[0]["evidence"] is None

    def test_ingest_upload(self, client):
        response = client.post(
            "/ingest",
            files={"file": ("notes.md", b"# Title\n\nA short note about pumps.", "text/markdown")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["source_id"] == "notes.md"
        assert body["chunks"] >= 1

    def test_ingest_unsupported_type(self, client):
        response = client.post(
            "/ingest", files={"file": ("data.xyz", b"x", "application/octet-stream")}
        )
        assert response.status_code == 400

    def test_ingest_video_not_implemented(self, client):
        response = client.post(
            "/ingest", files={"file": ("clip.mp4", b"\x00", "video/mp4")}
        )
        assert response.status_code == 501
