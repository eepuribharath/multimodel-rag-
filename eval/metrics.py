"""Pure retrieval metrics, judged at source level: a retrieved chunk is
relevant iff its source_id is among the item's expected_source_ids."""

import math


def recall_at_k(retrieved_source_ids: list[str], expected: list[str], k: int) -> float:
    """Fraction of expected sources present anywhere in the top k."""
    if not expected:
        return 0.0
    top = set(retrieved_source_ids[:k])
    return sum(1 for s in expected if s in top) / len(expected)


def ndcg_at_k(retrieved_source_ids: list[str], expected: list[str], k: int) -> float:
    """Binary-relevance NDCG; each expected source counts once (first hit)."""
    if not expected:
        return 0.0
    remaining = set(expected)
    dcg = 0.0
    for rank, source in enumerate(retrieved_source_ids[:k], start=1):
        if source in remaining:
            dcg += 1.0 / math.log2(rank + 1)
            remaining.remove(source)
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
