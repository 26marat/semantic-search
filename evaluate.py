"""Evaluation harness for the three search systems (see README for details).

How it works: for each of the 412 test papers, search using that paper's
title and check whether its abstract comes back in the top 20 results.
The test papers are indexed using their abstract only (title removed), so
a system can't just match the query's exact words against itself; it has
to actually find the right paper from a differently-worded description.

We report two things:
- Recall@K and MRR@20: how often, and how high up, the right paper is
  found. Ranks are only checked down to position 20, so a paper that's
  ranked 35th counts the same as one that's missing entirely. This makes
  MRR@20 a conservative estimate, not a perfect score.
- Category coherence@10: a secondary, rougher signal. Of the top 10
  results, what fraction share the query paper's subject-area label. This
  is NOT a measure of relevance (many papers legitimately span more than
  one category) - see README for the caveat.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bm25_search import BM25Search
from data import load_splits, to_docs
from random_search import RandomSearch
from semantic_search import SemanticSearch

RESULTS_DIR = Path(__file__).parent / "results"
MAX_K = 20
RECALL_KS = (5, 10, 20)
COHERENCE_K = 10


def build_eval_corpus(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[list[dict], dict[str, str]]:
    """Eval corpus = train (full text) + test (abstract only). Returns the
    corpus and an id -> label lookup used for category coherence.
    """
    train_docs = to_docs(train_df, text_col="text")
    test_docs = to_docs(test_df, text_col="abstract")
    eval_docs = train_docs + test_docs
    labels = dict(zip(train_df["id"], train_df["label"])) | dict(zip(test_df["id"], test_df["label"]))
    return eval_docs, labels


def rank_of(results: list[tuple[str, float]], target_id: str) -> int | None:
    """1-indexed rank of target_id in results, or None if absent."""
    for rank, (doc_id, _score) in enumerate(results, start=1):
        if doc_id == target_id:
            return rank
    return None


def evaluate_system(name: str, engine, test_df: pd.DataFrame, labels: dict[str, str]) -> tuple[dict, list[dict]]:
    per_query = []
    for row in test_df.itertuples():
        results = engine.search(row.title, k=MAX_K)
        rank = rank_of(results, row.id)
        # Exclude the query's own (trivially same-label) entry so coherence@10
        # measures agreement among *other* results, not a guaranteed self-match.
        coherence_pool = [(doc_id, score) for doc_id, score in results if doc_id != row.id]
        top_labels = [labels.get(doc_id) for doc_id, _score in coherence_pool[:COHERENCE_K]]
        coherence = sum(1 for l in top_labels if l == row.label) / len(top_labels) if top_labels else 0.0
        per_query.append(
            {
                "system": name,
                "query_id": row.id,
                "query_title": row.title,
                "label": row.label,
                "rank": rank,
                "top1_id": results[0][0] if results else None,
                "coherence_at_10": coherence,
            }
        )

    n = len(per_query)
    recalls = {k: sum(1 for r in per_query if r["rank"] is not None and r["rank"] <= k) / n for k in RECALL_KS}
    mrr = sum((1.0 / r["rank"]) if r["rank"] else 0.0 for r in per_query) / n
    mean_coherence = sum(r["coherence_at_10"] for r in per_query) / n

    summary = {
        "system": name,
        **{f"recall@{k}": v for k, v in recalls.items()},
        "mrr@20": mrr,
        "coherence_at_10": mean_coherence,
    }
    return summary, per_query


def plot_results(summary_df: pd.DataFrame) -> None:
    metrics = ["recall@5", "recall@10", "recall@20", "mrr@20"]
    labels_display = ["Recall@5", "Recall@10", "Recall@20", "MRR@20"]
    systems = summary_df.index.tolist()
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, system in enumerate(systems):
        values = [summary_df.loc[system, m] for m in metrics]
        offsets = [x + (i - (len(systems) - 1) / 2) * width for x in range(len(metrics))]
        ax.bar(offsets, values, width=width, label=system)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels_display)
    ax.set_ylabel("Score")
    ax.set_title("Known-item retrieval: Random vs BM25 vs Semantic")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "recall_mrr_chart.png", dpi=150)
    plt.close(fig)


def main() -> None:
    train_df, test_df = load_splits()
    eval_docs, labels = build_eval_corpus(train_df, test_df)

    systems = [
        ("Random", RandomSearch(eval_docs)),
        ("BM25", BM25Search(eval_docs)),
        ("Semantic", SemanticSearch(eval_docs, cache_key="eval")),
    ]

    summaries = []
    all_per_query = []
    for name, engine in systems:
        print(f"[{name}] building index over {len(eval_docs)} docs...")
        engine.build()
        print(f"[{name}] evaluating on {len(test_df)} known-item queries...")
        summary, per_query = evaluate_system(name, engine, test_df, labels)
        summaries.append(summary)
        all_per_query.extend(per_query)

    RESULTS_DIR.mkdir(exist_ok=True)

    summary_df = pd.DataFrame(summaries).set_index("system")
    summary_df = summary_df[["recall@5", "recall@10", "recall@20", "mrr@20", "coherence_at_10"]]
    summary_df.to_csv(RESULTS_DIR / "comparison_table.csv")

    with open(RESULTS_DIR / "comparison_table.md", "w") as f:
        f.write("| System | Recall@5 | Recall@10 | Recall@20 | MRR@20 | Category coherence@10 (proxy, not relevance) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in summaries:
            f.write(
                f"| {row['system']} | {row['recall@5']:.3f} | {row['recall@10']:.3f} | "
                f"{row['recall@20']:.3f} | {row['mrr@20']:.3f} | {row['coherence_at_10']:.3f} |\n"
            )

    pd.DataFrame(all_per_query).to_csv(RESULTS_DIR / "per_query_results.csv", index=False)
    plot_results(summary_df)

    print()
    print(summary_df)
    print()
    print(f"Wrote comparison_table.{{csv,md}}, per_query_results.csv, recall_mrr_chart.png to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
