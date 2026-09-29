from random_search import RandomSearch

DOCS = [{"id": f"d{i}", "text": f"document number {i}"} for i in range(10)]


def test_search_returns_k_ids_from_corpus():
    engine = RandomSearch(DOCS, seed=1)
    engine.build()
    results = engine.search("irrelevant query", k=5)
    assert len(results) == 5
    ids = {doc_id for doc_id, _score in results}
    assert ids <= {d["id"] for d in DOCS}
    assert len(ids) == 5  # no duplicates


def test_search_caps_k_at_corpus_size():
    engine = RandomSearch(DOCS, seed=1)
    engine.build()
    results = engine.search("q", k=100)
    assert len(results) == len(DOCS)


def test_same_seed_is_deterministic():
    a = RandomSearch(DOCS, seed=7)
    a.build()
    b = RandomSearch(DOCS, seed=7)
    b.build()
    assert a.search("q", k=5) == b.search("q", k=5)
