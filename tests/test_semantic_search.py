import pytest

from semantic_search import CACHE_DIR, MODEL_NAME, SemanticSearch

DOCS = [
    {"id": "cats", "text": "Cats are small domesticated felines and popular pets."},
    {"id": "dogs", "text": "Dogs are loyal companions often called man's best friend."},
    {"id": "stocks", "text": "The stock market fell sharply after the earnings report."},
]


@pytest.fixture
def engine():
    e = SemanticSearch(DOCS, cache_key="pytest-tmp")
    e.build()
    yield e
    cache_file = CACHE_DIR / f"pytest-tmp-{MODEL_NAME}.npy"
    cache_file.unlink(missing_ok=True)


def test_semantically_similar_query_ranks_topically_related_doc_first(engine):
    results = engine.search("a household pet that meows", k=3)
    assert results[0][0] == "cats"


def test_search_returns_k_results_with_scores(engine):
    results = engine.search("finance news", k=2)
    assert len(results) == 2
    for _doc_id, score in results:
        assert isinstance(score, float)


def test_search_before_build_raises():
    e = SemanticSearch(DOCS, cache_key="pytest-tmp-unbuilt")
    with pytest.raises(RuntimeError):
        e.search("cats", k=3)
