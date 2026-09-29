from bm25_search import BM25Search, _tokenize

DOCS = [
    {"id": "cats", "text": "Cats are small domesticated felines and popular pets."},
    {"id": "dogs", "text": "Dogs are loyal companions often called man's best friend."},
    {"id": "stocks", "text": "The stock market fell sharply after the earnings report."},
]


def test_tokenize_lowercases_and_strips_punctuation():
    assert _tokenize("Cats, Dogs! (2024)") == ["cats", "dogs", "2024"]


def test_exact_term_match_ranks_first():
    engine = BM25Search(DOCS)
    engine.build()
    results = engine.search("feline pets", k=3)
    assert results[0][0] == "cats"


def test_search_before_build_raises():
    engine = BM25Search(DOCS)
    try:
        engine.search("cats", k=3)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_k_limits_result_count():
    engine = BM25Search(DOCS)
    engine.build()
    assert len(engine.search("market", k=1)) == 1
