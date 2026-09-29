"""Qualitative error analysis: pick queries where BM25 clearly outperforms
semantic search, and vice versa, from evaluate.py's per-query output.

This script only *selects* the examples (by rank gap) and prints the raw
material (query, target paper, and what each system's top-1 actually was).
The "why" commentary is written by hand after inspecting the real abstracts -
that's the point of this section (see README): understanding failure modes,
not just computing a gap number.
"""

from pathlib import Path

import pandas as pd

from data import load_splits

RESULTS_DIR = Path(__file__).parent / "results"
NOT_FOUND_RANK = 21  # sentinel: worse than MAX_K=20 used in evaluate.py
N_EXAMPLES = 5


def load_comparison() -> pd.DataFrame:
    per_query = pd.read_csv(RESULTS_DIR / "per_query_results.csv")

    bm25 = per_query[per_query["system"] == "BM25"].set_index("query_id")
    semantic = per_query[per_query["system"] == "Semantic"].set_index("query_id")

    merged = pd.DataFrame(
        {
            "query_title": bm25["query_title"],
            "label": bm25["label"],
            "bm25_rank": bm25["rank"],
            "bm25_top1_id": bm25["top1_id"],
            "semantic_rank": semantic["rank"],
            "semantic_top1_id": semantic["top1_id"],
        }
    )
    merged["bm25_rank_filled"] = merged["bm25_rank"].fillna(NOT_FOUND_RANK)
    merged["semantic_rank_filled"] = merged["semantic_rank"].fillna(NOT_FOUND_RANK)
    merged["gap"] = merged["semantic_rank_filled"] - merged["bm25_rank_filled"]
    return merged


def describe(row: pd.Series, id_to_title: dict[str, str]) -> str:
    bm25_rank = "not in top 20" if pd.isna(row["bm25_rank"]) else f"rank {int(row['bm25_rank'])}"
    sem_rank = "not in top 20" if pd.isna(row["semantic_rank"]) else f"rank {int(row['semantic_rank'])}"
    bm25_top1 = id_to_title.get(row["bm25_top1_id"], "?")
    sem_top1 = id_to_title.get(row["semantic_top1_id"], "?")
    return (
        f"query: {row['query_title']!r}  [label={row['label']}]\n"
        f"  BM25:     {bm25_rank:15s}  top-1 returned: {bm25_top1[:90]!r}\n"
        f"  Semantic: {sem_rank:15s}  top-1 returned: {sem_top1[:90]!r}\n"
    )


def main() -> None:
    train_df, test_df = load_splits()
    id_to_title = dict(zip(train_df["id"], train_df["title"])) | dict(zip(test_df["id"], test_df["title"]))

    merged = load_comparison()

    bm25_wins = merged.sort_values("gap", ascending=False).head(N_EXAMPLES)
    semantic_wins = merged.sort_values("gap", ascending=True).head(N_EXAMPLES)

    print(f"=== BM25 outperforms Semantic (top {N_EXAMPLES} by rank gap) ===\n")
    for _, row in bm25_wins.iterrows():
        print(describe(row, id_to_title))

    print(f"=== Semantic outperforms BM25 (top {N_EXAMPLES} by rank gap) ===\n")
    for _, row in semantic_wins.iterrows():
        print(describe(row, id_to_title))

    bm25_wins.assign(direction="bm25_wins").to_csv(RESULTS_DIR / "error_analysis_bm25_wins.csv")
    semantic_wins.assign(direction="semantic_wins").to_csv(RESULTS_DIR / "error_analysis_semantic_wins.csv")
    print(f"Wrote selected examples to {RESULTS_DIR}/error_analysis_*.csv")

if __name__ == "__main__":
    main()
