"""Sentence-safe token-budgeted text chunking.

Targets 512–1024 tokens per chunk with ~15% overlap, splitting on semantic
boundaries (blank lines, headings) first and packing whole sentences — a
sentence is never cut in half. Content is passed through verbatim: no
lowercasing, no stripping of identifiers (BM25 depends on exact tokens).
"""

import re
from functools import lru_cache

import tiktoken

MAX_TOKENS = 1024
OVERLAP_RATIO = 0.15

# sentence end: terminal punctuation followed by whitespace + uppercase/digit,
# or end of block. Deliberately simple; block splitting does the heavy lifting.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_BLOCK_RE = re.compile(r"\n\s*\n|(?=\n#{1,6}\s)")


@lru_cache
def _encoding():
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def _split_oversized(sentence: str, max_tokens: int) -> list[str]:
    # last resort for a "sentence" longer than the whole budget (minified
    # blobs, giant lists): hard token windows, since no clean boundary exists
    tokens = _encoding().encode(sentence)
    return [
        _encoding().decode(tokens[i : i + max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]


def _sentences(text: str, max_tokens: int) -> list[str]:
    blocks = [b.strip() for b in _BLOCK_RE.split(text) if b and b.strip()]
    out: list[str] = []
    for block in blocks:
        for sentence in _SENTENCE_RE.split(block):
            if not sentence.strip():
                continue
            if count_tokens(sentence) > max_tokens:
                out.extend(_split_oversized(sentence, max_tokens))
            else:
                out.append(sentence)
        out.append("\n\n")  # preserve block boundary as glue marker
    return out[:-1] if out else []


def _flush(sentences: list[str]) -> str:
    return " ".join(s for s in sentences if s != "\n\n").strip()


def chunk_text(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    sentences = _sentences(text, max_tokens)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        if sentence == "\n\n":
            current.append(sentence)
            continue
        n = count_tokens(sentence)
        if current_tokens + n > max_tokens and current_tokens > 0:
            chunks.append(_flush(current))
            # carry trailing sentences (~15% of the budget) as overlap
            overlap: list[str] = []
            overlap_tokens = 0
            for prev in reversed([s for s in current if s != "\n\n"]):
                t = count_tokens(prev)
                if overlap_tokens + t > max_tokens * OVERLAP_RATIO:
                    break
                overlap.insert(0, prev)
                overlap_tokens += t
            current = overlap
            current_tokens = overlap_tokens
        current.append(sentence)
        current_tokens += n

    tail = _flush(current)
    if tail:
        chunks.append(tail)
    return chunks
