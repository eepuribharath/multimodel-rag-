"""Audio slicing: ≤80s per slice, cut at silence where possible.

80s is the retrieval-granularity target (the API itself allows 180s/request).
Slices are exported as WAV files next to the media store so content_ref
points at playable, embeddable segments with start/end timestamps preserved.
"""

from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_silence

from app.ingestion.ids import chunk_id
from app.schemas import Chunk

MAX_SLICE_MS = 80_000
MIN_SILENCE_MS = 400
SILENCE_MARGIN_DB = 16  # relative to segment's average loudness


def _cut_points(audio: AudioSegment) -> list[int]:
    """Slice boundaries: latest silence midpoint before each 80s budget."""
    silences = detect_silence(
        audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=int(audio.dBFS) - SILENCE_MARGIN_DB,
        seek_step=25,
    )
    midpoints = [(start + end) // 2 for start, end in silences]
    cuts = []
    pos = 0
    while len(audio) - pos > MAX_SLICE_MS:
        window_end = pos + MAX_SLICE_MS
        candidates = [m for m in midpoints if pos + MIN_SILENCE_MS < m <= window_end]
        cut = max(candidates) if candidates else window_end
        cuts.append(cut)
        pos = cut
    return cuts


def ingest_audio(path: Path, source_id: str, media_dir: Path) -> list[Chunk]:
    audio = AudioSegment.from_file(path)
    boundaries = [0, *_cut_points(audio), len(audio)]
    out_dir = media_dir / source_id
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for ordinal, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        slice_path = out_dir / f"audio-{ordinal:04d}.wav"
        audio[start:end].export(slice_path, format="wav")
        chunks.append(
            Chunk(
                id=chunk_id(source_id, "audio", ordinal),
                source_id=source_id,
                modality="audio",
                content_ref=str(slice_path),
                ordinal=ordinal,
                metadata={
                    "start_ms": start,
                    "end_ms": end,
                    "mime_type": "audio/wav",
                    "original_path": str(path),
                },
            )
        )
    return chunks
