"""
basic_rag.py
Basic RAG pipeline: embed query → retrieve top-k chunks → generate answer.

This is the simplest baseline pipeline. No query expansion, no graph, no
confidence gating — just a straight vector search followed by LLM generation.

Used as a baseline to compare against the four advanced pipelines.
"""

from typing import Dict, Any

from src.corpus import CorpusIndex
from src.retrieval import retrieve
from src.generation import generate_answer


def run(
    query: str,
    index: CorpusIndex,
    top_k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """
    Run the Basic RAG pipeline.

    Flow:
        1. Embed the query using the embedding model.
        2. Retrieve top-k chunks from the global corpus index by cosine similarity.
        3. Pass query + retrieved chunks to the LLM and generate an answer.

    Args:
        query      : user question string
        index      : built/loaded CorpusIndex
        top_k      : number of chunks to retrieve
        model_name : sentence-transformers model (must match index build model)

    Returns:
        {
            "answer"          : str   — LLM-generated answer,
            "retrieved_chunks": list  — [{text, score, page_url, page_name, domain}],
            "pipeline"        : "basic_rag",
        }
    """
    # Step 1 + 2: Embed query and retrieve
    raw_chunks = retrieve(query, index, top_k=top_k, model_name=model_name)

    # Step 3: Generate answer from retrieved context
    answer = generate_answer(query, raw_chunks)

    # Format retrieved chunks for output / frontend display
    retrieved = [
        {
            "text":      chunk.get("text", ""),
            "score":     round(score, 4),
            "page_url":  chunk.get("page_url", ""),
            "page_name": chunk.get("page_name", ""),
            "domain":    chunk.get("domain", ""),
        }
        for chunk, score in raw_chunks
    ]

    return {
        "answer":           answer,
        "retrieved_chunks": retrieved,
        "pipeline":         "basic_rag",
    }