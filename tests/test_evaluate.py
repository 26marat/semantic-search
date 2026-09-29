import pandas as pd
import pytest

from evaluate import COHERENCE_K, MAX_K, evaluate_system, rank_of


def test_rank_of_finds_1_indexed_rank():
    results = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    assert rank_of(results, "b") == 2


def test_rank_of_returns_none_when_absent():
    results = [("a", 0.9), ("b", 0.8)]
    assert rank_of(results, "z") is None


class _FakeEngine:
    """Returns a fixed ranked list regardless of query, for testing the
    coherence@10 self-match exclusion in evaluate_system without a real
    retrieval engine.
    """

    def __init__(self, results):
        self._results = results

    def build(self):
        pass

    def search(self, query, k):
        return self._results[:k]


def test_coherence_excludes_the_query_own_entry():
    # Query paper "q1" (label A) is returned at rank 1 by itself, followed by
    # 9 more label-A docs and 10 label-B docs. If the self-match were left
    # in, coherence@10 would read 10/10; excluding it should read 9/10,
    # since one label-B doc slides into the top-10 window.
    fake_results = (
        [("q1", 1.0)]
        + [(f"a{i}", 0.9) for i in range(9)]
        + [(f"b{i}", 0.1) for i in range(10)]
    )
    assert len(fake_results) == MAX_K
    labels = {"q1": "A"} | {f"a{i}": "A" for i in range(9)} | {f"b{i}": "B" for i in range(10)}

    test_df = pd.DataFrame([{"id": "q1", "title": "Q1 Title", "label": "A"}])
    engine = _FakeEngine(fake_results)

    summary, per_query = evaluate_system("Fake", engine, test_df, labels)

    assert per_query[0]["rank"] == 1
    assert per_query[0]["coherence_at_10"] == pytest.approx(9 / COHERENCE_K)
    assert summary["coherence_at_10"] == pytest.approx(9 / COHERENCE_K)
    assert summary["recall@5"] == 1.0
    assert summary["mrr@20"] == 1.0


class _FakeEngineByQuery:
    """Like _FakeEngine, but returns a different fixed ranked list per query
    title, for testing recall@K/MRR behavior across more than one query.
    """

    def __init__(self, results_by_query):
        self._results_by_query = results_by_query

    def build(self):
        pass

    def search(self, query, k):
        return self._results_by_query[query][:k]


def test_mrr_and_recall_average_correctly_including_a_miss():
    # Documented design decision (README/evaluate.py docstring): a target
    # not found within MAX_K contributes 0 to MRR rather than a partial
    # score, and doesn't count toward Recall@K at any K. This exercises
    # that directly, plus a non-trivial (non-rank-1) hit, across 2 queries.
    results_by_query = {
        "Q1 Title": [(f"x{i}", 1.0) for i in range(2)]  # ranks 1-2: distractors
        + [("q1", 1.0)]  # rank 3: the target
        + [(f"y{i}", 1.0) for i in range(MAX_K - 3)],
        "Q2 Title": [(f"z{i}", 1.0) for i in range(MAX_K)],  # "q2" never appears
    }
    labels = {"q1": "A", "q2": "B"}
    test_df = pd.DataFrame(
        [
            {"id": "q1", "title": "Q1 Title", "label": "A"},
            {"id": "q2", "title": "Q2 Title", "label": "B"},
        ]
    )
    engine = _FakeEngineByQuery(results_by_query)

    summary, per_query = evaluate_system("Fake", engine, test_df, labels)

    assert per_query[0]["rank"] == 3
    assert per_query[1]["rank"] is None
    assert summary["recall@5"] == 0.5  # only q1 (rank 3) found within top 5
    assert summary["recall@20"] == 0.5  # q2 never found, even at the deepest cutoff
    assert summary["mrr@20"] == pytest.approx((1 / 3 + 0.0) / 2)
