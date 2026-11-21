"""Spec 01 acceptance criteria.

- 40-page PDF with tables → no chunk exceeds embedder limits, every
  table/figure is its own chunk with a page anchor (marked slow: docling
  layout models run for real)
- re-running ingest() on the same file yields byte-identical chunk IDs
- 5-minute audio file → ≥4 chunks, each ≤80s, each with start/end timestamps
"""

import math
import struct
import wave
from pathlib import Path

import pytest

from app.ingestion import ingest
from app.ingestion.text import chunk_text, count_tokens

# --- fixture builders -------------------------------------------------------

WORDS = (
    "retrieval augmentation requires careful chunking of source material. "
    "Each passage must remain within the embedder token budget. "
    "Compressor part XK-42817 requires a torque of 4.2 Nm during assembly. "
    "The quarterly report showed unexpected divergence across regions. "
)


def long_markdown(sections: int = 12, sentences_per: int = 40) -> str:
    parts = []
    for s in range(sections):
        parts.append(f"## Section {s}\n\n")
        parts.append(
            " ".join(
                f"Sentence {s}-{i} explains that retrieval quality depends on "
                "precise chunk boundaries and stable identifiers such as XK-42817."
                for i in range(sentences_per)
            )
        )
        parts.append("\n\n")
    return "".join(parts)


def write_wav(path: Path, seconds: int, rate: int = 8000) -> None:
    """Tone bursts separated by 1s silences every ~20s so the slicer has
    natural cut points."""
    frames = bytearray()
    for i in range(seconds * rate):
        t = i / rate
        in_silence = (t % 20) > 19  # 1s silence at the end of every 20s
        amp = 0 if in_silence else int(12000 * math.sin(2 * math.pi * 440 * t))
        frames += struct.pack("<h", amp)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


def write_png(path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (64, 64), color=(200, 30, 30)).save(path)


def write_pdf(path: Path, pages: int = 40) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        Image as RLImage,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib import colors

    styles = getSampleStyleSheet()
    figure_png = path.parent / "embedded-figure.png"
    write_png(figure_png)

    story = []
    for page in range(pages):
        story.append(Paragraph(f"Chapter {page}", styles["Heading1"]))
        for i in range(4):
            story.append(Paragraph(f"Paragraph {page}-{i}. {WORDS}", styles["Normal"]))
            story.append(Spacer(1, 8))
        if page % 5 == 0:
            table = Table(
                [["Part", "Torque", "Qty"]]
                + [[f"XK-{page}{r}", f"{r}.2 Nm", str(r)] for r in range(4)]
            )
            table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
            story.append(table)
        if page == 3:
            story.append(RLImage(str(figure_png), width=120, height=120))
        if page < pages - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=letter).build(story)


# --- text chunker -----------------------------------------------------------

class TestChunkText:
    def test_budget_and_sentence_boundaries(self):
        chunks = chunk_text(long_markdown())
        assert len(chunks) > 1
        assert all(count_tokens(c) <= 1024 for c in chunks)
        # never mid-sentence: chunks end at a sentence end or a heading
        # (headings are semantic boundaries, allowed by spec 01)
        for c in chunks:
            assert c.rstrip()[-1] in ".!?" or c.splitlines()[-1].startswith("##")

    def test_overlap_carries_content(self):
        chunks = chunk_text(long_markdown())
        # the start of each subsequent chunk repeats the tail of the previous
        for a, b in zip(chunks, chunks[1:]):
            first_sentence = b.split(". ")[0][:80]
            assert first_sentence in a

    def test_exact_identifiers_untouched(self):
        chunks = chunk_text(long_markdown())
        assert any("XK-42817" in c for c in chunks)  # not lowercased/stripped


# --- ingest() dispatch ------------------------------------------------------

class TestIngest:
    def test_text_file_deterministic_ids(self, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text(long_markdown())
        first = ingest(str(f), media_dir=tmp_path / "media")
        second = ingest(str(f), media_dir=tmp_path / "media")
        assert [c.id for c in first] == [c.id for c in second]
        assert all(c.modality == "text" for c in first)
        assert [c.ordinal for c in first] == list(range(len(first)))

    def test_image_single_chunk(self, tmp_path):
        f = tmp_path / "diagram.png"
        write_png(f)
        chunks = ingest(str(f), media_dir=tmp_path / "media")
        assert len(chunks) == 1
        assert chunks[0].modality == "image"
        assert chunks[0].metadata["mime_type"] == "image/png"

    def test_five_minute_audio(self, tmp_path):
        f = tmp_path / "meeting.wav"
        write_wav(f, seconds=300)
        chunks = ingest(str(f), media_dir=tmp_path / "media")

        assert len(chunks) >= 4
        for c in chunks:
            assert c.modality == "audio"
            assert c.metadata["end_ms"] - c.metadata["start_ms"] <= 80_000
            assert Path(c.content_ref).exists()
        # contiguous, covering the full recording
        assert chunks[0].metadata["start_ms"] == 0
        assert chunks[-1].metadata["end_ms"] == 300_000
        for a, b in zip(chunks, chunks[1:]):
            assert a.metadata["end_ms"] == b.metadata["start_ms"]

    def test_video_is_stubbed(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"\x00")
        with pytest.raises(NotImplementedError):
            ingest(str(f))

    def test_unknown_type_rejected(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("x")
        with pytest.raises(ValueError, match="unsupported"):
            ingest(str(f))


# --- PDF (docling, real layout models) --------------------------------------

@pytest.mark.slow
class TestPdf:
    def test_40_page_pdf_with_tables(self, tmp_path):
        f = tmp_path / "manual.pdf"
        write_pdf(f, pages=40)

        chunks = ingest(str(f), media_dir=tmp_path / "media")
        again = ingest(str(f), media_dir=tmp_path / "media")

        assert [c.id for c in chunks] == [c.id for c in again]

        text_chunks = [c for c in chunks if c.modality == "text"]
        tables = [c for c in text_chunks if c.metadata.get("kind") == "table"]
        figures = [c for c in chunks if c.modality == "image"]

        assert all(count_tokens(c.content_ref) <= 1024 for c in text_chunks)
        assert len(tables) == 8  # every 5th page of 40
        assert all("page" in c.metadata for c in chunks)
        assert figures and all(Path(c.content_ref).exists() for c in figures)
        assert any("XK-" in c.content_ref for c in tables)  # identifiers intact
