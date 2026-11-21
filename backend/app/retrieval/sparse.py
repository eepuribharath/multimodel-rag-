"""BM25-style sparse encoding for Qdrant's sparse vectors.

Client side computes term frequencies with BM25 saturation/length-normalization;
inverse document frequency is applied server-side by Qdrant via the IDF
modifier on the sparse vector config (query weights are therefore all 1.0).

Kept dependency-free on purpose: deterministic, offline, and transparent.
Tokens hash to stable uint32 indices, so encoders on different machines agree.
"""

import hashlib
import re
from collections import Counter

from qdrant_client import models

_TOKEN_RE = re.compile(r"\w+")


def _token_index(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big")


class Bm25SparseEncoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_doc_len: float = 256.0):
        self.k1 = k1
        self.b = b
        self.avg_doc_len = avg_doc_len

    def encode_document(self, text: str) -> models.SparseVector:
        # Lowercasing happens only here, symmetrically with queries — the
        # stored chunk content keeps exact identifiers untouched.
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return models.SparseVector(indices=[], values=[])
        counts = Counter(tokens)
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / self.avg_doc_len)
        weights = {
            _token_index(tok): tf * (self.k1 + 1) / (tf + norm)
            for tok, tf in counts.items()
        }
        indices = sorted(weights)
        return models.SparseVector(indices=indices, values=[weights[i] for i in indices])

    def encode_query(self, text: str) -> models.SparseVector:
        indices = sorted({_token_index(t) for t in _TOKEN_RE.findall(text.lower())})
        return models.SparseVector(indices=indices, values=[1.0] * len(indices))
