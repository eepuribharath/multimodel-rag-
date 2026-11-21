"""Spec 02 acceptance criteria.

- embed_query and embed_chunks return vectors of exactly the configured dim
- a batch of 20 text chunks is transparently split into request-sized batches
  and returns 20 vectors in order
- FakeEmbedder is a drop-in for GeminiEmbedder (protocol conformance)
- a simulated 429 triggers backoff-and-retry, not a crash
"""

import hashlib
from types import SimpleNamespace

import pytest
from google.genai import errors

from app.embedding import Embedder, FakeEmbedder, GeminiEmbedder
from app.schemas import Chunk, Query

DIM = 64


def text_chunk(i: int, text: str | None = None) -> Chunk:
    return Chunk(
        id=f"chunk-{i}",
        source_id="doc-1",
        modality="text",
        content_ref=text or f"chunk text {i}",
        ordinal=i,
    )


def image_chunk(i: int, path: str) -> Chunk:
    return Chunk(
        id=f"img-{i}",
        source_id="doc-1",
        modality="image",
        content_ref=path,
        ordinal=i,
        metadata={"mime_type": "image/png"},
    )


class RecordingClient:
    """Mimics genai.Client.models.embed_content; one deterministic vector per
    Content, derived from its parts, so ordering is verifiable."""

    def __init__(self, dim: int = DIM, fail_first: int = 0, fail_code: int = 429):
        self.dim = dim
        self.calls: list[list] = []
        self.remaining_failures = fail_first
        self.fail_code = fail_code
        self.models = SimpleNamespace(embed_content=self._embed_content)

    def _embed_content(self, *, model, contents, config):
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise errors.APIError(
                self.fail_code, {"error": {"message": "quota exceeded"}}
            )
        self.calls.append(contents)
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=self._vector(content))
                for content in contents
            ]
        )

    def _vector(self, content):
        key = "|".join(
            part.text or (part.inline_data.mime_type + str(len(part.inline_data.data)))
            for part in content.parts
        )
        seed = hashlib.sha256(key.encode()).digest()
        return [seed[i % len(seed)] / 255.0 for i in range(self.dim)]


def make_embedder(client, **kwargs) -> GeminiEmbedder:
    kwargs.setdefault("batch_size", 8)
    return GeminiEmbedder(
        model="gemini-embedding-2",
        dim=DIM,
        client=client,
        backoff_base=0.001,
        backoff_max=0.01,
        **kwargs,
    )


class TestFakeEmbedder:
    def test_is_drop_in_for_protocol(self):
        assert isinstance(FakeEmbedder(dim=DIM), Embedder)
        assert isinstance(make_embedder(RecordingClient()), Embedder)

    def test_dims_and_determinism(self):
        fake = FakeEmbedder(dim=DIM)
        chunks = [text_chunk(i) for i in range(3)]
        vectors = fake.embed_chunks(chunks)
        assert [len(v) for v in vectors] == [DIM] * 3
        assert vectors == fake.embed_chunks(chunks)  # same input, same vectors
        query_vec = fake.embed_query(Query(text="what is RRF?"))
        assert len(query_vec) == DIM


class TestGeminiEmbedder:
    def test_batch_of_20_split_and_ordered(self):
        client = RecordingClient()
        embedder = make_embedder(client, batch_size=8)
        chunks = [text_chunk(i) for i in range(20)]

        vectors = embedder.embed_chunks(chunks)

        assert len(vectors) == 20
        assert [len(batch) for batch in client.calls] == [8, 8, 4]
        # order: each vector matches an independent single-chunk embedding
        for chunk, vec in zip(chunks, vectors):
            solo = make_embedder(RecordingClient()).embed_chunks([chunk])[0]
            assert vec == solo

    def test_images_capped_at_6_per_request(self, tmp_path):
        png = tmp_path / "img.png"
        png.write_bytes(b"\x89PNG fake bytes")
        client = RecordingClient()
        embedder = make_embedder(client, batch_size=32)

        vectors = embedder.embed_chunks(
            [image_chunk(i, str(png)) for i in range(7)]
        )

        assert len(vectors) == 7
        assert [len(batch) for batch in client.calls] == [6, 1]

    def test_mixed_modalities_preserve_order(self, tmp_path):
        png = tmp_path / "img.png"
        png.write_bytes(b"\x89PNG fake bytes")
        client = RecordingClient()
        embedder = make_embedder(client)
        chunks = [text_chunk(0), image_chunk(1, str(png)), text_chunk(2)]

        vectors = embedder.embed_chunks(chunks)

        assert len(vectors) == 3
        assert [len(batch) for batch in client.calls] == [1, 1, 1]
        assert vectors[0] != vectors[1]

    def test_query_vector_dim(self):
        embedder = make_embedder(RecordingClient())
        assert len(embedder.embed_query(Query(text="hello"))) == DIM

    def test_429_triggers_retry_not_crash(self):
        client = RecordingClient(fail_first=2)
        embedder = make_embedder(client)

        vectors = embedder.embed_chunks([text_chunk(0)])

        assert len(vectors) == 1
        assert client.remaining_failures == 0

    def test_non_retryable_error_fails_fast(self):
        client = RecordingClient(fail_first=1, fail_code=400)
        embedder = make_embedder(client)

        with pytest.raises(errors.APIError):
            embedder.embed_chunks([text_chunk(0)])
        assert client.calls == []  # no retry happened

    def test_wrong_dim_from_api_is_rejected(self):
        client = RecordingClient(dim=DIM + 1)
        embedder = make_embedder(client)

        with pytest.raises(RuntimeError, match="dim"):
            embedder.embed_chunks([text_chunk(0)])

    def test_requires_key_or_client(self):
        with pytest.raises(ValueError, match="api_key"):
            GeminiEmbedder(model="gemini-embedding-2", dim=DIM)
