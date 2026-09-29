"""Streamlit demo: BM25 vs. semantic search, side by side.

Searches the demo index (train split, `title + ". " + abstract`) - the same
corpus a real user would search, as opposed to evaluate.py's eval index
which is specifically constructed for known-item scoring.
"""

import streamlit as st

from bm25_search import BM25Search
from data import load_splits, to_docs
from semantic_search import SemanticSearch

st.set_page_config(page_title="Semantic Search vs. BM25", layout="wide")


@st.cache_data(show_spinner="Loading dataset...")
def get_train_df():
    train_df, _ = load_splits()
    return train_df


@st.cache_resource(show_spinner="Building BM25 index...")
def get_bm25_engine(_docs):
    engine = BM25Search(_docs)
    engine.build()
    return engine


@st.cache_resource(show_spinner="Building semantic index (encoding ~3.3k abstracts, cached after first run)...")
def get_semantic_engine(_docs):
    engine = SemanticSearch(_docs, cache_key="demo")
    engine.build()
    return engine


train_df = get_train_df()
docs = to_docs(train_df)
bm25_engine = get_bm25_engine(docs)
semantic_engine = get_semantic_engine(docs)
id_to_row = train_df.set_index("id")

st.title("Semantic Search vs. BM25")
st.caption(
    "arXiv CS abstracts (train split, 3,292 papers). Compare lexical retrieval "
    "(BM25) against semantic retrieval (Sentence-Transformers + FAISS) side by side."
)

query = st.text_input("Search query", value="transformer attention mechanism")
k = st.slider("Number of results", min_value=3, max_value=20, value=5)


def render_results(engine, query: str, k: int) -> None:
    results = engine.search(query, k)
    if not results:
        st.write("No results.")
        return
    for doc_id, score in results:
        row = id_to_row.loc[doc_id]
        st.markdown(f"**{row['title']}**  \n`score={score:.3f}` &nbsp;·&nbsp; `{row['label']}`")
        with st.expander("abstract"):
            st.write(row["abstract"])


if query:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("BM25 (lexical)")
        render_results(bm25_engine, query, k)
    with col2:
        st.subheader("Semantic (embeddings)")
        render_results(semantic_engine, query, k)
else:
    st.info("Enter a query above to compare BM25 vs. semantic search results.")
