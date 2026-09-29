"""Dataset loading for the semantic search engine.

Loads Tiansve/arxiv-cs-subfield from Hugging Face. `train` is the searchable
corpus; `test` is reserved for known-item evaluation queries. The dataset also
ships a `validation` split, which this project does not use.
"""

import pandas as pd
from datasets import load_dataset

DATASET_NAME = "Tiansve/arxiv-cs-subfield"


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) as pandas DataFrames.

    Downloads and caches the dataset locally via `datasets` on first call.
    """
    ds = load_dataset(DATASET_NAME)
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()
    return train_df, test_df


def to_docs(df: pd.DataFrame, text_col: str = "text") -> list[dict]:
    """Convert a DataFrame into the `[{"id": ..., "text": ...}, ...]` shape
    every retrieval engine (`RandomSearch`, `BM25Search`, `SemanticSearch`)
    consumes. `text_col` lets callers index a different field per row set,
    e.g. abstract-only for eval-index test documents.
    """
    return [{"id": row.id, "text": getattr(row, text_col)} for row in df.itertuples()]


if __name__ == "__main__":
    train_df, test_df = load_splits()

    print(f"train: {len(train_df)} rows, test: {len(test_df)} rows")
    print()
    print("train label distribution:")
    print(train_df["label"].value_counts())
    print()
    print("test label distribution:")
    print(test_df["label"].value_counts())
    print()

    sample = train_df.iloc[0]
    expected_text = f"{sample['title']}. {sample['abstract']}"
    assert sample["text"] == expected_text, "text field is not title + '. ' + abstract as expected"
    print("Confirmed: text field == title + '. ' + abstract")
    print()
    print("Sample row:")
    print(sample[["id", "title", "label", "primary_category", "published"]])
