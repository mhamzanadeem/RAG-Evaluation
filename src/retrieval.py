"""
retrieval.py
Core retrieval primitive: embed a query and search the global VectorIndex.
All four pipelines call this module (directly or via variants).
"""

from typing import List, Tuple
import numpy as np

from src.corpus import VectorIndex, embed_texts


def retrieve(
    query: str,
    index: VectorIndex,
    top_k: int = 5,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> List[Tuple[str, float, dict]]:
    """
    Embed `query` and return top_k (chunk_text, score, metadata) from the index.
    """
    q_emb = embed_texts([query], model_name=embedding_model)[0]
    return index.retrieve(q_emb, top_k=top_k)


def retrieve_by_embedding(
    query_embedding: np.ndarray,
    index: VectorIndex,
    top_k: int = 5,
) -> List[Tuple[str, float, dict]]:
    """
    Search the index with a pre-computed query embedding.
    Used by HyDE (hypothetical-doc embedding) and RAG Fusion variants.
    """
    return index.retrieve(query_embedding, top_k=top_k)


def embed_query(
    query: str,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Return a 1-D numpy embedding for a single query string."""
    return embed_texts([query], model_name=embedding_model)[0]
