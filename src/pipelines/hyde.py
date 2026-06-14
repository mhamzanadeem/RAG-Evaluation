"""
hyde.py
HyDE (Hypothetical Document Embedding) pipeline.

Steps:
1. Prompt the LLM to write a short hypothetical document that would answer the query.
2. Embed the hypothetical document (not the query).
3. Retrieve top-k real chunks from the global index by similarity to this embedding.
4. Generate the final answer from the retrieved (real) context.
"""

from typing import Dict, Any, List, Tuple

import numpy as np

from src.corpus import VectorIndex, embed_texts
from src.retrieval import retrieve_by_embedding
from src.generation import generate_text, generate_answer


# ---------------------------------------------------------------------------
# Hypothetical document generation
# ---------------------------------------------------------------------------

HYDE_SYSTEM = (
    "You are a knowledgeable assistant. "
    "Write a short, factually plausible passage (1-2 paragraphs) that would be retrieved "
    "from a web encyclopedia or news article to answer the question below. "
    "Do NOT answer the question directly; write it as if it were part of a reference document."
)


def generate_hypothetical_doc(
    query: str,
    provider: str = None,
    max_tokens: int = 256,
) -> str:
    """Generate a hypothetical document passage for `query`."""
    prompt = f"Question: {query}\n\nWrite a relevant passage:"
    return generate_text(prompt, system_prompt=HYDE_SYSTEM, max_tokens=max_tokens, provider=provider)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run(
    query: str,
    index: VectorIndex,
    top_k: int = 5,
    embedding_model: str = "all-MiniLM-L6-v2",
    provider: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    HyDE pipeline.

    Returns:
        {
            "answer": str,
            "retrieved_chunks": [(text, score, meta), ...],
            "hypothetical_doc": str,
            "pipeline": "hyde",
        }
    """
    # 1. Generate hypothetical document
    hyp_doc = generate_hypothetical_doc(query, provider=provider)

    # 2. Embed the hypothetical document
    hyp_emb = embed_texts([hyp_doc], model_name=embedding_model)[0]

    # 3. Retrieve real chunks using the hypothetical embedding
    results = retrieve_by_embedding(hyp_emb, index, top_k=top_k)

    # 4. Generate answer from real retrieved chunks
    context = [(text, score) for text, score, _meta in results]
    answer = generate_answer(query, context, provider=provider)

    return {
        "answer": answer,
        "retrieved_chunks": results,
        "hypothetical_doc": hyp_doc,
        "pipeline": "hyde",
    }
