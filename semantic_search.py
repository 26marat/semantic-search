"""Semantic retrieval using Sentence Transformer embeddings + FAISS.

Embeddings are L2-normalized so FAISS's IndexFlatIP (inner product) computes
cosine similarity.
"""

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).parent / "cache"


class SemanticSearch:
    def __init__(self, docs: list[dict], cache_key: str, model_name: str = MODEL_NAME):
        self.docs = docs
        self.ids = [d["id"] for d in docs]
        self.cache_key = cache_key
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None

    def build(self) -> None:
        self._model = SentenceTransformer(self.model_name)
        embeddings = self._load_or_encode()
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def _cache_path(self) -> Path:
        return CACHE_DIR / f"{self.cache_key}-{self.model_name}.npy"

    def _load_or_encode(self) -> np.ndarray:
        cache_path = self._cache_path()
        if cache_path.exists():
            return np.load(cache_path)

        texts = [d["text"] for d in self.docs]
        embeddings = self._model.encode(
            texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(embeddings)

        CACHE_DIR.mkdir(exist_ok=True)
        np.save(cache_path, embeddings)
        return embeddings

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("call build() before search()")
        query_emb = self._model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_emb)
        scores, idxs = self._index.search(query_emb, k)
        return [(self.ids[i], float(score)) for i, score in zip(idxs[0], scores[0]) if i != -1]


if __name__ == "__main__":
    from data import load_splits, to_docs

    train_df, _ = load_splits()
    engine = SemanticSearch(to_docs(train_df), cache_key="demo")
    engine.build()

    for query in ["transformer attention mechanism", "graph neural networks"]:
        print(f"query: {query!r}")
        for doc_id, score in engine.search(query, k=5):
            title = train_df.loc[train_df["id"] == doc_id, "title"].iloc[0]
            print(f"  {score:.4f}  {doc_id}  {title[:80]}")
        print()
