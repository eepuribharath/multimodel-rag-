"""PDF ingestion: layout-aware parsing via Docling.

Text is chunked per page (keeps the page anchor exact for citations); every
table and figure becomes its own chunk. Figures are exported as PNGs into the
media store so image chunks have an embeddable, citable content_ref.
"""

from collections import defaultdict
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem, TextItem

from app.ingestion.ids import chunk_id
from app.ingestion.text import chunk_text
from app.schemas import Chunk

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter  # models take seconds to load; do it once per process
    if _converter is None:
        # do_ocr=False: born-digital PDFs only for now; docling 2.108's
        # default OCR config also fails on this platform (PP-OCRv6)
        options = PdfPipelineOptions(
            generate_picture_images=True, images_scale=2.0, do_ocr=False
        )
        _converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
    return _converter


def _page_no(item) -> int:
    return item.prov[0].page_no if item.prov else 0


def ingest_pdf(file: Path, source_id: str, media_dir: Path) -> list[Chunk]:
    document = _get_converter().convert(str(file)).document
    out_dir = media_dir / source_id
    chunks: list[Chunk] = []
    counters: dict[str, int] = defaultdict(int)
    page_texts: dict[int, list[str]] = defaultdict(list)

    def add(modality: str, content_ref: str, metadata: dict) -> None:
        ordinal = counters[modality]
        counters[modality] += 1
        chunks.append(
            Chunk(
                id=chunk_id(source_id, modality, ordinal),
                source_id=source_id,
                modality=modality,
                content_ref=content_ref,
                ordinal=ordinal,
                metadata={"original_path": str(file), **metadata},
            )
        )

    for item, _level in document.iterate_items():
        if isinstance(item, TableItem):
            add(
                "text",
                item.export_to_markdown(document),
                {"page": _page_no(item), "kind": "table"},
            )
        elif isinstance(item, PictureItem):
            image = item.get_image(document)
            if image is None:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            figure_path = out_dir / f"figure-{counters['image']:04d}.png"
            image.save(figure_path)
            add(
                "image",
                str(figure_path),
                {
                    "page": _page_no(item),
                    "kind": "figure",
                    "mime_type": "image/png",
                    "caption": item.caption_text(document) or "",
                },
            )
        elif isinstance(item, TextItem) and item.text.strip():
            page_texts[_page_no(item)].append(item.text)

    # ordered after tables/figures in ordinal space, but page anchors keep
    # citations exact; per-page chunking never crosses a page boundary
    for page in sorted(page_texts):
        for piece in chunk_text("\n\n".join(page_texts[page])):
            add("text", piece, {"page": page, "kind": "body"})

    return chunks
