"""Deterministic chunk identity.

Re-ingesting the same file must yield byte-identical chunk IDs so re-indexing
is idempotent — Qdrant upserts overwrite instead of duplicating.
"""

import hashlib


def chunk_id(source_id: str, modality: str, ordinal: int) -> str:
    key = f"{source_id}:{modality}:{ordinal}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]
