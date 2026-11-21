# Spec 01 — Ingestion (parse + modality-aware chunking)

## Goal
Turn raw source files into embeddable `Chunk`s that respect Gemini Embedding 2's per-request limits, preserving enough metadata to cite the source later.

## Module
`backend/app/ingestion/`

## Interface
```python
# schemas.py
class Chunk(BaseModel):
    id: str                      # stable hash of (source_id, modality, ordinal)
    source_id: str               # original file identifier
    modality: Literal["text", "image", "audio", "video", "pdf_page"]
    content_ref: str             # inline text OR path/URI to the media slice
    ordinal: int                 # position within the source
    metadata: dict               # page, timestamp range, bbox, title, etc.

# ingestion/pipeline.py
def ingest(path: str) -> list[Chunk]: ...
```

## Constraints & decisions
- **Chunk to fit the embedder, per modality:**
  - text → 512–1024 tokens, ~15% overlap, split on semantic boundaries (headings/paragraphs), never mid-sentence.
  - pdf → parse layout first (Docling or LlamaParse), keep tables and figures as their own chunks, and never exceed 6 pages per embed call.
  - image → one chunk per figure/image; batch of ≤6 per embed request handled by the embedding layer, not here.
  - audio → slices ≤80s, split on silence where possible (API cap is 180s/request; 80s is the retrieval-granularity target).
  - video → slices ≤120s (API cap, verified 2026-07-02).
- **Preserve citation anchors** in `metadata`: page number, char span, timestamp range, or bounding box. The generation stage cannot cite what ingestion did not record.
- **Exact-identifier fidelity:** do not lowercase or strip content that carries meaning for lexical match (codes, drug/compound names, dosages, part numbers). BM25 relies on it.
- **Deterministic IDs:** re-ingesting the same file must produce the same chunk IDs so re-indexing is idempotent.

## Dependencies
`docling` or `llama-parse` (PDF layout), `pydub`/`ffmpeg` (audio/video slicing), `tiktoken` or the Gemini tokenizer for token counts.

## Acceptance criteria
- A 40-page PDF with tables produces chunks where no chunk exceeds the embedder limits and every table/figure is its own chunk with a page anchor.
- Re-running `ingest()` on the same file yields byte-identical chunk IDs.
- A 5-minute audio file yields ≥4 chunks, each ≤80s, each with a start/end timestamp.

## Claude Code notes
Build the text and pdf paths first (highest value), stub audio/video behind the same `Chunk` interface. Write a fixture-based test with one small PDF, one image, one short audio clip.
