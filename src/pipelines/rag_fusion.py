"""
rag_fusion.py
RAG Fusion pipeline using Reciprocal Rank Fusion (RRF).

Steps:
1. Generate N query variants with the LLM.
2. Retrieve top-k chunks from the global index for EACH variant.
3. Merge the per-variant ranked lists using RRF.
4. Take the top-k fused chunks and generate a final answer.
"""

from typing import List, Tuple, Dict, Any

from src.corpus import VectorIndex
from src.retrieval import embed_query, retrieve_by_embedding
from src.generation import generate_text, generate_answer


# ---------------------------------------------------------------------------
# Query variant generation
# ---------------------------------------------------------------------------

VARIANT_SYSTEM = (
    "You are a search query rewriting assistant. "
    "Given a user question, output ONLY a numbered list of {n} diverse, "
    "semantically different reformulations of the question. "
    "Vary vocabulary, specificity, and phrasing. Output NOTHING else."
)


def generate_query_variants(query: str, n: int = 4, provider: str = None) -> List[str]:
    """Generate n alternative phrasings of `query`."""
    prompt = f"Original question: {query}\n\nGenerate {n} diverse reformulations:"
    system = VARIANT_SYSTEM.format(n=n)
    raw = generate_text(prompt, system_prompt=system, max_tokens=300, provider=provider)

    variants = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering like "1.", "1)", "-", "*"
        cleaned = line.lstrip("0123456789.-)*• ").strip()
        if cleaned and cleaned.lower() != query.lower():
            variants.append(cleaned)

    # Always include the original
    all_variants = [query] + variants[:n]
    return all_variants


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float, dict]]],
    k: int = 60,
) -> List[Tuple[str, float, dict]]:
    """
    Merge multiple ranked lists using RRF.
    RRF score for chunk c: sum over lists L of  1 / (k + rank_in_L(c))
    Returns a single ranked list sorted by descending RRF score.
    """
    scores: Dict[str, float] = {}
    chunk_meta: Dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, (text, _score, meta) in enumerate(ranked, start=1):
            scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank)
            chunk_meta[text] = meta

    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(text, score, chunk_meta[text]) for text, score in sorted_chunks]


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run(
    query: str,
    index: VectorIndex,
    top_k: int = 5,
    num_variants: int = 4,
    embedding_model: str = "all-MiniLM-L6-v2",
    provider: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    RAG Fusion pipeline.

    Returns:
        {
            "answer": str,
            "retrieved_chunks": [(text, score, meta), ...],
            "query_variants": [str, ...],
            "pipeline": "rag_fusion",
        }
    """
    # 1. Generate query variants
    variants = generate_query_variants(query, n=num_variants, provider=provider)

    # 2. Retrieve for each variant
    ranked_lists = []
    for v in variants:
        q_emb = embed_query(v, embedding_model=embedding_model)
        results = retrieve_by_embedding(q_emb, index, top_k=top_k)
        ranked_lists.append(results)

    # 3. RRF fusion
    fused = reciprocal_rank_fusion(ranked_lists)
    top_chunks = fused[:top_k]

    # 4. Generate answer
    context = [(text, score) for text, score, _meta in top_chunks]
    answer = generate_answer(query, context, provider=provider)

    return {
        "answer": answer,
        "retrieved_chunks": top_chunks,
        "query_variants": variants,
        "pipeline": "rag_fusion",
    }
