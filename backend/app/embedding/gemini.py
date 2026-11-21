"""Gemini Embedding 2 client behind the Embedder protocol.

The model is in public preview; everything preview-specific (model ID, batch
limits, retry posture) is confined to this file so a GA change is a one-file
edit. task_type is only honored by gemini-embedding-001-family models — for
Embedding 2 the API expects task instructions inside the prompt text, so the
value is accepted (per the Embedder contract) but not sent.
"""

import mimetypes

from google import genai
from google.genai import types

from app.config import MAX_IMAGES_PER_REQUEST
from app.retrying import build_retryer
from app.schemas import Chunk, Query, Vector

_TASK_TYPE_MODEL_PREFIXES = ("gemini-embedding-001",)


class GeminiEmbedder:
    def __init__(
        self,
        model: str,
        dim: int,
        task_type: str | None = None,
        api_key: str | None = None,
        batch_size: int = 32,
        max_attempts: int = 6,
        backoff_base: float = 0.5,
        backoff_max: float = 30.0,
        client: genai.Client | None = None,
    ):
        if client is None and not api_key:
            raise ValueError(
                "GeminiEmbedder needs an api_key (GEMINI_API_KEY) or an injected client"
            )
        self.model = model
        self.dim = dim
        self.task_type = task_type
        self.batch_size = batch_size
        self._client = client or genai.Client(api_key=api_key)
        self._retryer = build_retryer(max_attempts, backoff_base, backoff_max)

    def embed_chunks(self, chunks: list[Chunk]) -> list[Vector]:
        vectors: list[Vector] = []
        for batch in self._batches(chunks):
            contents = [
                types.Content(parts=[self._chunk_part(c)]) for c in batch
            ]
            vectors.extend(self._embed(contents))
        return vectors

    def embed_query(self, query: Query) -> Vector:
        parts: list[types.Part] = []
        if query.text is not None:
            parts.append(types.Part.from_text(text=query.text))
        if query.media_path is not None:
            parts.append(
                types.Part.from_bytes(
                    data=self._read(query.media_path),
                    mime_type=query.media_mime_type,
                )
            )
        # One Content → one embedding; multiple parts are interleaved input.
        return self._embed([types.Content(parts=parts)])[0]

    def _embed(self, contents: list[types.Content]) -> list[Vector]:
        response = self._retryer(
            self._client.models.embed_content,
            model=self.model,
            contents=contents,
            config=self._config(),
        )
        vectors = [e.values for e in response.embeddings]
        if len(vectors) != len(contents):
            raise RuntimeError(
                f"expected {len(contents)} embeddings, got {len(vectors)}"
            )
        for v in vectors:
            if len(v) != self.dim:
                raise RuntimeError(
                    f"embedding has dim {len(v)}, index requires {self.dim} "
                    "— same model + dimension at index and query time, always"
                )
        return vectors

    def _config(self) -> types.EmbedContentConfig:
        kwargs: dict = {"output_dimensionality": self.dim}
        if self.task_type and self.model.startswith(_TASK_TYPE_MODEL_PREFIXES):
            kwargs["task_type"] = self.task_type
        return types.EmbedContentConfig(**kwargs)

    def _batches(self, chunks: list[Chunk]):
        """Split into request-sized batches, preserving order.

        Text batches up to batch_size; images up to 6 per request (API cap);
        audio/video/pdf go one per request — each already carries the
        per-request media budget on its own.
        """
        limits = {"text": self.batch_size, "image": min(MAX_IMAGES_PER_REQUEST, self.batch_size)}
        batch: list[Chunk] = []
        batch_modality: str | None = None
        for chunk in chunks:
            if chunk.modality not in limits:
                if batch:
                    yield batch
                    batch, batch_modality = [], None
                yield [chunk]
                continue
            if batch and (
                chunk.modality != batch_modality
                or len(batch) >= limits[batch_modality]
            ):
                yield batch
                batch = []
            batch.append(chunk)
            batch_modality = chunk.modality
        if batch:
            yield batch

    def _chunk_part(self, chunk: Chunk) -> types.Part:
        if chunk.modality == "text":
            return types.Part.from_text(text=chunk.content_ref)
        mime = chunk.metadata.get("mime_type") or (
            "application/pdf"
            if chunk.modality == "pdf_page"
            else mimetypes.guess_type(chunk.content_ref)[0]
        )
        if mime is None:
            raise ValueError(
                f"cannot determine mime type for chunk {chunk.id} "
                f"({chunk.content_ref}); set metadata['mime_type']"
            )
        return types.Part.from_bytes(data=self._read(chunk.content_ref), mime_type=mime)

    @staticmethod
    def _read(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()
