"""Spec 05 acceptance criteria (offline: mock Gemini client).

- answerable query → non-empty text, abstained=False, ≥1 citation whose
  chunk_id is in the input set
- no supporting context → abstained=True, no fabrication
- every chunk_id in citations also appears in used_chunk_ids and input items
- citation integrity is a hard post-processing step (invalid markers repaired,
  citation-free answers become abstentions)
"""

from types import SimpleNamespace

from app.generation import ABSTAIN_TEXT, GeminiGenerator, locator, resolve_answer
from app.generation.generate import ABSTAIN_SENTINEL
from app.pipeline import RagPipeline
from app.rerank import FakeReranker
from app.embedding import FakeEmbedder
from app.retrieval import VectorStore
from app.schemas import Chunk, RetrievedItem

from qdrant_client import QdrantClient


def item(i: int, text: str, modality: str = "text", metadata: dict | None = None) -> RetrievedItem:
    return RetrievedItem(
        chunk=Chunk(
            id=f"c{i}",
            source_id="doc",
            modality=modality,
            content_ref=text if modality == "text" else f"/media/{i}",
            ordinal=i,
            metadata=metadata or {},
        ),
        dense_score=None,
        sparse_score=None,
        rrf_score=1.0,
    )


class ScriptedClient:
    """genai.Client stand-in returning a scripted response."""

    def __init__(self, text: str):
        self._text = text
        self.models = SimpleNamespace(
            generate_content=self._generate,
            generate_content_stream=self._stream,
        )

    def _generate(self, *, model, contents, config):
        self.last_prompt = contents
        return SimpleNamespace(text=self._text)

    def _stream(self, *, model, contents, config):
        for i in range(0, len(self._text), 5):
            yield SimpleNamespace(text=self._text[i : i + 5])


def generator(text: str) -> GeminiGenerator:
    return GeminiGenerator(model="test-model", client=ScriptedClient(text))


class TestResolveAnswer:
    def test_answerable_with_citations(self):
        items = [item(0, "irrelevant"), item(1, "The torque is 5.5 Nm.")]
        answer = resolve_answer("The maximum torque is 5.5 Nm [2].", items)

        assert not answer.abstained
        assert answer.text
        assert [c.chunk_id for c in answer.citations] == ["c1"]
        assert answer.used_chunk_ids == ["c1"]

    def test_invalid_marker_repaired(self):
        items = [item(0, "a"), item(1, "b")]
        answer = resolve_answer("Claim one [2]. Claim two [9].", items)

        assert "[9]" not in answer.text
        assert "[2]" in answer.text
        assert answer.used_chunk_ids == ["c1"]

    def test_sentinel_abstains(self):
        answer = resolve_answer(ABSTAIN_SENTINEL, [item(0, "x")])
        assert answer.abstained
        assert answer.text == ABSTAIN_TEXT
        assert answer.citations == []

    def test_citation_free_answer_becomes_abstention(self):
        answer = resolve_answer("Confident claim with no support.", [item(0, "x")])
        assert answer.abstained  # unverifiable text never ships as grounded

    def test_citations_subset_of_input(self):
        items = [item(i, f"passage {i}") for i in range(3)]
        answer = resolve_answer("A [1] and B [3] and again [1].", items)
        input_ids = {i.chunk.id for i in items}
        assert set(answer.used_chunk_ids) <= input_ids
        assert [c.chunk_id for c in answer.citations] == ["c0", "c2"]  # deduped, ordered


class TestLocator:
    def test_page(self):
        assert locator(item(0, "x", metadata={"page": 4}).chunk) == "p. 4"

    def test_timestamps(self):
        c = item(0, "x", "audio", {"start_ms": 750_000, "end_ms": 790_000}).chunk
        assert locator(c) == "12:30–13:10"

    def test_fallback_ordinal(self):
        assert locator(item(7, "x").chunk) == "§7"


class TestGeminiGenerator:
    def test_generate_returns_answer(self):
        items = [item(0, "The sky is blue.")]
        answer = generator("The sky is blue [1].").generate("what color?", items)
        assert not answer.abstained
        assert answer.used_chunk_ids == ["c0"]

    def test_empty_candidates_abstain_without_api_call(self):
        answer = generator("should never be used").generate("anything?", [])
        assert answer.abstained

    def test_stream_deltas_reassemble(self):
        text = "Streaming grounded answer [1]."
        gen = generator(text)
        deltas = list(gen.generate_stream("q", [item(0, "ctx")]))
        assert "".join(deltas) == text

    def test_context_includes_locator_and_surrogate(self):
        client = ScriptedClient("ok [1].")
        gen = GeminiGenerator(model="m", client=client)
        gen.generate(
            "q",
            [item(0, "", "image", {"caption": "pump diagram", "page": 2})],
        )
        assert "p. 2" in client.last_prompt
        assert "pump diagram" in client.last_prompt
        assert "modality: image" in client.last_prompt


class TestQueryGate:
    def test_social_messages_short_circuit(self):
        from app.generation import QueryGate, looks_social

        gate = QueryGate(model="m", client=None)
        for message in ("hi", "Hello!", "thanks", "who are you?", "good morning :)"):
            assert looks_social(message), message
            assert gate.plan(message, []).kind == "chitchat"

    def test_factual_question_searches_without_client(self):
        from app.generation import QueryGate

        gate = QueryGate(model="m", client=None)
        plan = gate.plan("What torque does XK-42817 need?", [])
        assert plan.kind == "search"
        assert plan.standalone == "What torque does XK-42817 need?"

    def test_follow_up_rewritten_via_client(self):
        from app.generation import QueryGate

        gate = QueryGate(
            model="m", client=ScriptedClient("How often should compressor XK-42817 be serviced?")
        )
        plan = gate.plan(
            "how often should it be serviced?",
            [("XK-42817 torque?", "5.5 Nm")],
        )
        assert plan.kind == "search"
        assert "XK-42817" in plan.standalone

    def test_client_chitchat_verdict(self):
        from app.generation import QueryGate

        gate = QueryGate(model="m", client=ScriptedClient("CHITCHAT"))
        plan = gate.plan("that was helpful, cheers mate", [("q", "a")])
        assert plan.kind == "chitchat"

    def test_gate_error_falls_back_to_search(self):
        from app.generation import QueryGate

        class ExplodingClient:
            class models:
                @staticmethod
                def generate_content(**kw):
                    raise RuntimeError("boom")

        gate = QueryGate(model="m", client=ExplodingClient())
        plan = gate.plan("what about the warranty?", [("q", "a")])
        assert plan.kind == "search"
        assert plan.standalone == "what about the warranty?"


class TestHistoryAwareGeneration:
    def test_history_block_in_prompt(self):
        client = ScriptedClient("ok [1].")
        gen = GeminiGenerator(model="m", client=client)
        gen.generate(
            "how often should it be serviced?",
            [item(0, "Service every 400 hours.")],
            history=[("XK-42817 torque?", "5.5 Nm.")],
        )
        assert "Previous conversation" in client.last_prompt
        assert "XK-42817 torque?" in client.last_prompt

    def test_chitchat_answer_shape(self):
        gen = GeminiGenerator(model="m", client=ScriptedClient("Hello! Ask me about your library."))
        answer = gen.generate_chitchat("hi")
        assert answer.kind == "chitchat"
        assert not answer.abstained
        assert answer.citations == []
        assert "Hello" in answer.text


class TestPipelineOffline:
    def test_end_to_end_with_fakes(self):
        embedder = FakeEmbedder(dim=32)
        store = VectorStore(client=QdrantClient(":memory:"), collection="t", dim=32)
        gen = GeminiGenerator(
            model="m", client=ScriptedClient("Torque is 5.5 Nm [1].")
        )
        pipeline = RagPipeline(
            embedder=embedder, store=store, reranker=FakeReranker(), generator=gen
        )

        chunks = [
            Chunk(id=f"k{i}", source_id="manual.pdf", modality="text",
                  content_ref=text, ordinal=i, metadata={"page": i})
            for i, text in enumerate(
                ["Torque spec for XK-42817 is 5.5 Nm.", "Unrelated maintenance notes."]
            )
        ]
        store.upsert(chunks, embedder.embed_chunks(chunks))

        answer = pipeline.answer("XK-42817 torque?")

        assert not answer.abstained
        assert len(answer.citations) == 1
        assert answer.citations[0].chunk_id in {"k0", "k1"}
        assert answer.citations[0].locator.startswith("p. ")
