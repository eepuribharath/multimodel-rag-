"""ingest(path) — dispatch by file type into modality-aware chunkers."""

import mimetypes
from pathlib import Path

from app.config import PROJECT_ROOT
from app.ingestion.audio import ingest_audio
from app.ingestion.ids import chunk_id
from app.ingestion.text import chunk_text
from app.schemas import Chunk

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv"}

_blip_processor = None
_blip_model = None


def _get_captioner():
    global _blip_processor, _blip_model
    if _blip_processor is None or _blip_model is None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return _blip_processor, _blip_model


def _caption_image(file: Path) -> str:
    try:
        from PIL import Image

        processor, model = _get_captioner()
        raw_image = Image.open(file).convert("RGB")
        # 1. Unconditional caption
        inputs = processor(raw_image, return_tensors="pt")
        out1 = model.generate(**inputs, max_new_tokens=60)
        c1 = processor.decode(out1[0], skip_special_tokens=True).strip()

        # 2. Conditional prompt for scene/setting
        inputs_scene = processor(raw_image, text="a photo of", return_tensors="pt")
        out2 = model.generate(**inputs_scene, max_new_tokens=60)
        c2 = processor.decode(out2[0], skip_special_tokens=True).strip()
        if c2.lower().startswith("a photo of "):
            c2 = c2[11:].strip()

        captions = []
        if c1:
            captions.append(c1)
        if c2 and c2.lower() not in c1.lower() and c1.lower() not in c2.lower():
            captions.append(c2)
        return "; ".join(captions)
    except Exception:
        return ""


def _ocr_image(file: Path) -> str:
    try:
        import rapidocr

        engine = rapidocr.RapidOCR()
        result, _ = engine(str(file))
        if result:
            return " ".join(line[1] for line in result).strip()
    except Exception:
        pass
    return ""


def ingest(path: str, media_dir: Path | None = None) -> list[Chunk]:
    """media_dir holds derived media (audio slices, extracted PDF figures);
    defaults to <project>/media_store."""
    file = Path(path).expanduser().resolve()
    if not file.is_file():
        raise FileNotFoundError(file)
    media_dir = media_dir or PROJECT_ROOT / "media_store"
    # source_id is the file name: stable across re-ingests of the same file,
    # which keeps chunk IDs (and therefore Qdrant upserts) idempotent
    source_id = file.name
    suffix = file.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        return _ingest_text(file, source_id)
    if suffix in IMAGE_SUFFIXES:
        return _ingest_image(file, source_id)
    if suffix in AUDIO_SUFFIXES:
        return ingest_audio(file, source_id, media_dir)
    if suffix == ".pdf":
        from app.ingestion.pdf import ingest_pdf  # docling import is heavy

        return ingest_pdf(file, source_id, media_dir)
    if suffix in VIDEO_SUFFIXES:
        raise NotImplementedError(
            "video ingestion is stubbed (spec 01): slice to ≤120s behind the "
            "same Chunk interface when needed"
        )
    raise ValueError(f"unsupported file type: {file.suffix} ({file})")


def _ingest_text(file: Path, source_id: str) -> list[Chunk]:
    pieces = chunk_text(file.read_text(encoding="utf-8"))
    return [
        Chunk(
            id=chunk_id(source_id, "text", i),
            source_id=source_id,
            modality="text",
            content_ref=piece,
            ordinal=i,
            metadata={"original_path": str(file)},
        )
        for i, piece in enumerate(pieces)
    ]


def _ingest_image(file: Path, source_id: str) -> list[Chunk]:
    mime = mimetypes.guess_type(str(file))[0] or "image/png"
    metadata = {"mime_type": mime, "title": file.stem}

    caption = _caption_image(file)
    if caption:
        metadata["caption"] = caption

    ocr_text = _ocr_image(file)
    if ocr_text:
        metadata["ocr"] = ocr_text

    return [
        Chunk(
            id=chunk_id(source_id, "image", 0),
            source_id=source_id,
            modality="image",
            content_ref=str(file),
            ordinal=0,
            metadata=metadata,
        )
    ]
