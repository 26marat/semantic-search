"""Random baseline retrieval.

Sanity floor: if BM25 / embeddings don't clearly beat this, something is
broken. Ignores the query entirely and returns a random sample of doc ids.
"""

import random


class RandomSearch:
    def __init__(self, docs: list[dict], seed: int = 42):
        self.ids = [d["id"] for d in docs]
        self._rng = random.Random(seed)

    def build(self) -> None:
        pass  # nothing to precompute

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        sample = self._rng.sample(self.ids, min(k, len(self.ids)))
        return [(doc_id, 0.0) for doc_id in sample]


if __name__ == "__main__":
    from data import load_splits, to_docs

    train_df, _ = load_splits()
    engine = RandomSearch(to_docs(train_df))
    engine.build()

    for query in ["transformer attention mechanism", "graph neural networks"]:
        print(f"query: {query!r}")
        for doc_id, score in engine.search(query, k=5):
            title = train_df.loc[train_df["id"] == doc_id, "title"].iloc[0]
            print(f"  {score:.4f}  {doc_id}  {title[:80]}")
        print()
