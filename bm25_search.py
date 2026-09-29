"""Lexical baseline retrieval using BM25 (Okapi variant)."""

import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Search:
    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.ids = [d["id"] for d in docs]
        self._bm25: BM25Okapi | None = None

    def build(self) -> None:
        tokenized_corpus = [_tokenize(d["text"]) for d in self.docs]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError("call build() before search()")
        scores = self._bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in top_idx]


if __name__ == "__main__":
    from data import load_splits, to_docs

    train_df, _ = load_splits()
    engine = BM25Search(to_docs(train_df))
    engine.build()

    for query in ["transformer attention mechanism", "graph neural networks"]:
        print(f"query: {query!r}")
        for doc_id, score in engine.search(query, k=5):
            title = train_df.loc[train_df["id"] == doc_id, "title"].iloc[0]
            print(f"  {score:.4f}  {doc_id}  {title[:80]}")
        print()
