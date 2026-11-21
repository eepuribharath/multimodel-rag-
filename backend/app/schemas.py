"""Shared data shapes crossing stage boundaries.

Single source of truth: if a spec in specs/ disagrees with this file, the spec
gets fixed, not this file. Shapes are added as their stage is built.
"""

from typing import Literal

from pydantic import BaseModel, model_validator

Vector = list[float]

Modality = Literal["text", "image", "audio", "video", "pdf_page"]


class Chunk(BaseModel):
    id: str  # stable hash of (source_id, modality, ordinal)
    source_id: str  # original file identifier
    modality: Modality
    content_ref: str  # inline text OR path/URI to the media slice
    ordinal: int  # position within the source
    metadata: dict = {}  # page, timestamp range, bbox, title, etc.


class RetrievedItem(BaseModel):
    chunk: Chunk
    dense_score: float | None  # None if the chunk surfaced via one leg only
    sparse_score: float | None
    rrf_score: float
    rerank_score: float | None = None  # set by the rerank stage (spec 04)


# metadata keys tried, in order, as the text stand-in for media chunks
_SURROGATE_KEYS = ("caption", "transcript", "ocr", "title")


def chunk_match_text(chunk: Chunk) -> str:
    """Text used wherever a chunk must be matched lexically (BM25 sparse leg,
    cross-encoder rerank): the content itself for text chunks, the best
    available metadata surrogate for media chunks, else ""."""
    if chunk.modality == "text":
        return chunk.content_ref
    parts = []
    if chunk.metadata.get("title"):
        parts.append(str(chunk.metadata["title"]))
    if chunk.metadata.get("caption"):
        parts.append(str(chunk.metadata["caption"]))
    if chunk.metadata.get("ocr"):
        parts.append(str(chunk.metadata["ocr"]))
    if chunk.metadata.get("transcript"):
        parts.append(str(chunk.metadata["transcript"]))
    if parts:
        return " | ".join(parts)
    for key in _SURROGATE_KEYS:
        if chunk.metadata.get(key):
            return str(chunk.metadata[key])
    return ""


class Citation(BaseModel):
    chunk_id: str
    source_id: str
    locator: str  # "p. 4", "12:30–13:10", bbox, char span
    modality: str


class Answer(BaseModel):
    text: str
    citations: list[Citation]
    used_chunk_ids: list[str]
    abstained: bool
    # grounded = cited answer; abstained = insufficient context;
    # chitchat = social reply (greeting/thanks), no citations expected
    kind: Literal["grounded", "abstained", "chitchat"] = "grounded"


class Query(BaseModel):
    """A retrieval query: text, media, or both (interleaved multimodal)."""

    text: str | None = None
    media_path: str | None = None
    media_mime_type: str | None = None  # required if media_path is set

    @model_validator(mode="after")
    def _require_content(self) -> "Query":
        if self.text is None and self.media_path is None:
            raise ValueError("Query needs text, media_path, or both")
        if self.media_path is not None and self.media_mime_type is None:
            raise ValueError("media_mime_type is required with media_path")
        return self
