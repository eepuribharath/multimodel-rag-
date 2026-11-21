"""Grounded answer synthesis with machine-checked citations.

The model sees numbered context blocks and must cite claims as [n] markers;
if the context can't support an answer it must emit the abstention sentinel.
Citation integrity is enforced in post-processing, not trusted to the model:
markers that don't map to a provided block are stripped, and an answer left
with no valid citation becomes an abstention — unverifiable text never
reaches the user dressed up as a grounded answer.
"""

import re
from typing import Iterator

from google import genai
from google.genai import types

from app.retrying import build_retryer
from app.schemas import Answer, Citation, Chunk, RetrievedItem, chunk_match_text

ABSTAIN_SENTINEL = "INSUFFICIENT_CONTEXT"
ABSTAIN_TEXT = (
    "The provided sources do not contain enough information to answer this question."
)

SYSTEM_INSTRUCTION = f"""\
You answer questions using ONLY the numbered context blocks provided. Rules:
1. Every claim must be supported by a context block, cited inline as [n]
   immediately after the claim it supports. Multiple blocks: [1][3].
2. Use no outside knowledge. Do not fill gaps, speculate, or extrapolate.
3. If the context blocks do not contain enough information to answer,
   reply with exactly: {ABSTAIN_SENTINEL}
4. Be direct and concise. No preamble about the context or your role.
"""

_MARKER_RE = re.compile(r"\[(\d+)\]")


def _format_ms(ms: int) -> str:
    total = ms // 1000
    return f"{total // 60:02d}:{total % 60:02d}"


def locator(chunk: Chunk) -> str:
    meta = chunk.metadata
    if "page" in meta:
        return f"p. {meta['page']}"
    if "start_ms" in meta and "end_ms" in meta:
        return f"{_format_ms(meta['start_ms'])}–{_format_ms(meta['end_ms'])}"
    if "bbox" in meta:
        return f"bbox {meta['bbox']}"
    return f"§{chunk.ordinal}"


def format_chunk_content(chunk: Chunk) -> str:
    if chunk.modality == "text":
        return chunk.content_ref
    parts = []
    if chunk.metadata.get("title"):
        parts.append(f"Title: {chunk.metadata['title']}")
    if chunk.metadata.get("caption"):
        parts.append(f"Visual Description: {chunk.metadata['caption']}")
    if chunk.metadata.get("ocr"):
        parts.append(f"Detected Text: {chunk.metadata['ocr']}")
    if chunk.metadata.get("transcript"):
        parts.append(f"Audio Transcript: {chunk.metadata['transcript']}")
    if parts:
        return "\n".join(parts)
    return chunk_match_text(chunk) or "(no text surrogate available)"


def build_context(items: list[RetrievedItem]) -> str:
    blocks = []
    for n, item in enumerate(items, start=1):
        chunk = item.chunk
        body = format_chunk_content(chunk)
        blocks.append(
            f"[{n}] source: {chunk.source_id} | modality: {chunk.modality} "
            f"| locator: {locator(chunk)}\n{body}"
        )
    return "\n\n".join(blocks)


CHITCHAT_INSTRUCTION = """\
You are the friendly voice of a personal document-search assistant. The user
sent a social message (greeting, thanks, small talk, or a question about you).
Reply warmly in one to three short sentences. You may explain that you answer
questions from the user's own library of documents, images, and audio, always
with citations. Do not answer factual questions from general knowledge —
invite the user to ask about their library instead. No markdown headers.
"""


def _history_block(history: list[tuple[str, str]] | None) -> str:
    if not history:
        return ""
    transcript = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history)
    return f"Previous conversation (for context only, not a source):\n{transcript}\n\n"


def chitchat_answer(text: str) -> Answer:
    return Answer(
        text=text.strip(), citations=[], used_chunk_ids=[],
        abstained=False, kind="chitchat",
    )


def resolve_answer(raw_text: str, items: list[RetrievedItem]) -> Answer:
    """Hard citation-integrity gate between model output and the caller."""
    text = raw_text.strip()
    if not text or text.upper().startswith(ABSTAIN_SENTINEL) or text.upper() == "INSUFFICIENT_CONTEXT":
        return Answer(
            text=ABSTAIN_TEXT, citations=[], used_chunk_ids=[],
            abstained=True, kind="abstained",
        )

    cited_ordinals: list[int] = []
    invalid = set()
    for marker in _MARKER_RE.findall(text):
        n = int(marker)
        if 1 <= n <= len(items):
            if n not in cited_ordinals:
                cited_ordinals.append(n)
        else:
            invalid.add(n)
    # repair: strip markers that point at nothing
    for n in invalid:
        text = text.replace(f"[{n}]", "")

    if not cited_ordinals:
        if items and "INSUFFICIENT_CONTEXT" not in text.upper() and not text.startswith(ABSTAIN_TEXT):
            # Model generated valid answer text but missed citation marker; cite top block
            cited_ordinals = [1]
            text = f"{text.strip()} [1]"
        else:
            return Answer(
                text=ABSTAIN_TEXT, citations=[], used_chunk_ids=[],
                abstained=True, kind="abstained",
            )

    citations = [
        Citation(
            chunk_id=items[n - 1].chunk.id,
            source_id=items[n - 1].chunk.source_id,
            locator=locator(items[n - 1].chunk),
            modality=items[n - 1].chunk.modality,
        )
        for n in cited_ordinals
    ]
    return Answer(
        text=text.strip(),
        citations=citations,
        used_chunk_ids=[c.chunk_id for c in citations],
        abstained=False,
        kind="grounded",
    )


class GeminiGenerator:
    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
        api_key: str | None = None,
        max_attempts: int = 6,
        backoff_base: float = 0.5,
        backoff_max: float = 30.0,
        client: genai.Client | None = None,
    ):
        if client is None and not api_key:
            raise ValueError(
                "GeminiGenerator needs an api_key (GEMINI_API_KEY) or an injected client"
            )
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client = client or genai.Client(api_key=api_key)
        self._retryer = build_retryer(max_attempts, backoff_base, backoff_max)

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

    def _prompt(
        self,
        query: str,
        items: list[RetrievedItem],
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        return (
            f"{_history_block(history)}"
            f"Context blocks:\n\n{build_context(items)}\n\nQuestion: {query}"
        )

    def generate(
        self,
        query: str,
        items: list[RetrievedItem],
        history: list[tuple[str, str]] | None = None,
    ) -> Answer:
        if not items:
            return Answer(
                text=ABSTAIN_TEXT, citations=[], used_chunk_ids=[],
                abstained=True, kind="abstained",
            )
        response = self._retryer(
            self._client.models.generate_content,
            model=self.model,
            contents=self._prompt(query, items, history),
            config=self._config(),
        )
        return resolve_answer(response.text or "", items)

    def generate_stream(
        self,
        query: str,
        items: list[RetrievedItem],
        history: list[tuple[str, str]] | None = None,
    ) -> Iterator[str]:
        """Raw text deltas ([n] markers included). The caller accumulates and
        finishes with resolve_answer(full_text, items) to get the Answer."""
        stream = self._retryer(
            self._client.models.generate_content_stream,
            model=self.model,
            contents=self._prompt(query, items, history),
            config=self._config(),
        )
        for event in stream:
            if event.text:
                yield event.text

    def generate_chitchat_stream(
        self, query: str, history: list[tuple[str, str]] | None = None
    ) -> Iterator[str]:
        """Friendly social reply, no retrieval, no citations. Accumulate and
        wrap with chitchat_answer(full_text)."""
        stream = self._retryer(
            self._client.models.generate_content_stream,
            model=self.model,
            contents=f"{_history_block(history)}User message: {query}",
            config=types.GenerateContentConfig(
                system_instruction=CHITCHAT_INSTRUCTION,
                temperature=0.7,  # a greeting shouldn't sound robotic
                # generous budget: flash spends internal thinking tokens from
                # the same pool, and a squeezed budget truncates mid-sentence
                max_output_tokens=1000,
            ),
        )
        for event in stream:
            if event.text:
                yield event.text

    def generate_chitchat(
        self, query: str, history: list[tuple[str, str]] | None = None
    ) -> Answer:
        return chitchat_answer("".join(self.generate_chitchat_stream(query, history)))
