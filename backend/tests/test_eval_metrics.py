"""Spec 06 metric math, offline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

from metrics import ndcg_at_k, recall_at_k  # noqa: E402


class TestRecall:
    def test_full_recall(self):
        assert recall_at_k(["a", "b", "c"], ["a", "c"], k=3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x", "y"], ["a", "c"], k=3) == 0.5

    def test_outside_k_does_not_count(self):
        assert recall_at_k(["x", "y", "z", "a"], ["a"], k=3) == 0.0


class TestNdcg:
    def test_perfect_ranking(self):
        assert ndcg_at_k(["a", "b"], ["a", "b"], k=6) == 1.0

    def test_relevant_at_bottom_scores_lower(self):
        high = ndcg_at_k(["a", "x", "y"], ["a"], k=3)
        low = ndcg_at_k(["x", "y", "a"], ["a"], k=3)
        assert high == 1.0
        assert 0 < low < high

    def test_duplicate_source_counts_once(self):
        # two chunks from the same expected source: second hit adds nothing
        once = ndcg_at_k(["a", "x"], ["a"], k=2)
        twice = ndcg_at_k(["a", "a"], ["a"], k=2)
        assert once == twice == 1.0

    def test_miss_is_zero(self):
        assert ndcg_at_k(["x", "y"], ["a"], k=2) == 0.0
