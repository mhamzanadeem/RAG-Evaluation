"""
crag.py
Corrective RAG (CRAG) pipeline.

Steps:
1. Retrieve top-k chunks from the global index.
2. Assess retrieval confidence with an LLM judge (or simple heuristic).
3. HIGH confidence  → generate answer from retrieved chunks (with citations).
   LOW confidence   → fall back to answering from the query alone (no retrieval),
                      or expand with more chunks.
4. Return answer with APA-style citations.
"""

from typing import Dict, Any, List, Tuple

from src.corpus import VectorIndex
from src.retrieval import embed_query, retrieve_by_embedding
from src.generation import generate_text, generate_answer


# ---------------------------------------------------------------------------
# Confidence assessment
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a retrieval quality judge. "
    "Given a question and a set of retrieved passages, respond with ONLY "
    "one of: HIGH, MEDIUM, or LOW. "
    "HIGH means the passages clearly contain information needed to answer the question. "
    "MEDIUM means the passages are somewhat relevant but incomplete. "
    "LOW means the passages are largely irrelevant or misleading."
)


def assess_confidence(
    query: str,
    chunks: List[Tuple[str, float, dict]],
    provider: str = None,
) -> Tuple[str, float]:
    """
    Ask the LLM to judge retrieval quality.
    Returns (label, numeric_score): label in {HIGH, MEDIUM, LOW},
    numeric_score in {1.0, 0.5, 0.0}.
    """
    passages = "\n\n".join(
        f"[{i+1}] {text}" for i, (text, _score, _meta) in enumerate(chunks)
    )
    prompt = (
        f"Question: {query}\n\n"
        f"Retrieved passages:\n{passages}\n\n"
        "Retrieval quality (HIGH / MEDIUM / LOW):"
    )
    raw = generate_text(prompt, system_prompt=JUDGE_SYSTEM, max_tokens=10, provider=provider)
    label = raw.strip().upper()
    if "HIGH" in label:
        return "HIGH", 1.0
    elif "MEDIUM" in label:
        return "MEDIUM", 0.5
    else:
        return "LOW", 0.0


# ---------------------------------------------------------------------------
# Citation-aware answer generation
# ---------------------------------------------------------------------------

CRAG_ANSWER_SYSTEM = (
    "You are a factual question-answering assistant. "
    "Answer the question accurately and concisely using ONLY the provided context passages. "
    "After your answer, add an 'References' section and list the sources used in APA style. "
    "Format references as:\n"
    "  [N] Author/Source Unknown. (n.d.). Retrieved from <URL>.\n"
    "Use the URL from the metadata. If no URL is available, write 'URL not available'.\n"
    "Do not hallucinate. If the context is insufficient, say so."
)


def generate_cited_answer(
    query: str,
    chunks: List[Tuple[str, float, dict]],
    provider: str = None,
) -> str:
    """Generate an answer with APA-style citations from retrieved chunks."""
    parts = []
    for i, (text, score, meta) in enumerate(chunks, 1):
        url = meta.get("page_url", "URL not available")
        name = meta.get("page_name", "")
        parts.append(f"[{i}] (relevance={score:.3f}) {text}\n   Source: {name} | {url}")

    context_str = "\n\n".join(parts)
    user_msg = (
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        "Answer (with References section):"
    )

    raw = generate_text(user_msg, system_prompt=CRAG_ANSWER_SYSTEM, max_tokens=600, provider=provider)
    return raw


FALLBACK_SYSTEM = (
    "You are a knowledgeable assistant. "
    "Answer the question as accurately as you can from your own knowledge. "
    "Keep the answer concise (1-3 sentences). "
    "Note at the end: '(Answer generated without retrieved context — low retrieval confidence.)'"
)


def generate_fallback_answer(query: str, provider: str = None) -> str:
    """Answer from LLM knowledge when retrieval confidence is low."""
    return generate_text(
        f"Question: {query}\n\nAnswer:",
        system_prompt=FALLBACK_SYSTEM,
        max_tokens=300,
        provider=provider,
    )


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run(
    query: str,
    index: VectorIndex,
    top_k: int = 5,
    embedding_model: str = "all-MiniLM-L6-v2",
    provider: str = None,
    confidence_threshold: str = "MEDIUM",  # "HIGH" | "MEDIUM" | "LOW"
    **kwargs,
) -> Dict[str, Any]:
    """
    CRAG pipeline.

    Returns:
        {
            "answer": str,
            "retrieved_chunks": [(text, score, meta), ...],
            "confidence_label": str,
            "confidence_score": float,
            "used_retrieval": bool,
            "pipeline": "crag",
        }
    """
    # 1. Retrieve
    q_emb = embed_query(query, embedding_model=embedding_model)
    results = retrieve_by_embedding(q_emb, index, top_k=top_k)

    # 2. Assess confidence
    conf_label, conf_score = assess_confidence(query, results, provider=provider)

    # 3. Conditional generation
    threshold_map = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.0}
    threshold_val = threshold_map.get(confidence_threshold.upper(), 0.5)

    used_retrieval = conf_score >= threshold_val

    if used_retrieval:
        answer = generate_cited_answer(query, results, provider=provider)
    else:
        # Low confidence: try expanding retrieval a bit, if still low, fall back
        if top_k < 10:
            expanded = retrieve_by_embedding(q_emb, index, top_k=10)
            conf2, _score2 = assess_confidence(query, expanded, provider=provider)
            if _score2 >= 0.5:
                answer = generate_cited_answer(query, expanded, provider=provider)
                results = expanded
                used_retrieval = True
            else:
                answer = generate_fallback_answer(query, provider=provider)
        else:
            answer = generate_fallback_answer(query, provider=provider)

    return {
        "answer": answer,
        "retrieved_chunks": results,
        "confidence_label": conf_label,
        "confidence_score": conf_score,
        "used_retrieval": used_retrieval,
        "pipeline": "crag",
    }
