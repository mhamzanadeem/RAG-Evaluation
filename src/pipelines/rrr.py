"""
rrr.py
Rewrite-Retrieve-Read (RRR) pipeline.

Reference: "Query Rewriting for Retrieval-Augmented Large Language Models"
           (Ma et al., 2023 — https://arxiv.org/abs/2305.14283)

Flow:
    1. REWRITE  — LLM rewrites the original query for maximum retrieval clarity.
                  The rewritten query is cleaner, more specific, and removes
                  ambiguity or colloquial phrasing that hurts embedding search.
    2. RETRIEVE — Embed the REWRITTEN query and retrieve top-k chunks from the
                  global corpus index.
    3. READ     — LLM reads the retrieved chunks and generates the final answer
                  using the ORIGINAL query as the question (so the answer stays
                  relevant to what the user actually asked).

Why it helps:
    Embedding models work best on clean, declarative statements. A user's raw
    question (e.g. "who won the thing with Federer in 07?") embeds differently
    from a clear query ("Roger Federer Wimbledon 2007 winner"). Rewriting bridges
    that gap without needing multiple retrieval passes.

Difference from HyDE:
    HyDE embeds a hypothetical *answer document*. RRR embeds a rewritten *query*.
    HyDE works better when the answer phrasing is predictable; RRR works better
    when the question itself is ambiguous or poorly worded.
"""

from typing import Dict, Any

from src.corpus import CorpusIndex
from src.retrieval import retrieve
from src.generation import generate_answer, _call_llm


# -----------------------------------------------------------------------
# Query rewriter
# -----------------------------------------------------------------------

def rewrite_query(original_query: str) -> str:
    """
    Use the LLM to rewrite the query for better retrieval.

    The rewritten query should be:
    - A clear, declarative noun-phrase or short statement
    - Free of pronouns, contractions, and ambiguous references
    - Specific enough to match relevant documents in a vector search

    Args:
        original_query : the raw user question

    Returns:
        Rewritten query string (falls back to original if LLM unavailable).
    """
    system = (
        "You are a search query optimization expert. "
        "Your task is to rewrite the given question into a clearer, more specific "
        "search query that will retrieve better results from a vector database. "
        "Rules:\n"
        "- Output ONLY the rewritten query, nothing else.\n"
        "- Do NOT answer the question.\n"
        "- Make it a concise noun phrase or declarative statement.\n"
        "- Expand acronyms, resolve pronouns, add relevant context.\n"
        "- Keep it under 20 words."
    )
    prompt = f"Original question: {original_query}\n\nRewritten search query:"

    rewritten = _call_llm(prompt=prompt, system=system, max_tokens=60)

    # Sanity check: if LLM returned something too long or seems like an answer,
    # fall back to the original query
    rewritten = rewritten.strip().strip('"').strip("'")
    if (
        not rewritten
        or len(rewritten) > 200
        or rewritten.lower().startswith(("the answer", "i don't", "i cannot", "[no"))
    ):
        return original_query

    return rewritten


# -----------------------------------------------------------------------
# Pipeline entry point
# -----------------------------------------------------------------------

def run(
    query: str,
    index: CorpusIndex,
    top_k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """
    Run the Rewrite-Retrieve-Read (RRR) pipeline.

    Args:
        query      : original user question
        index      : built/loaded CorpusIndex
        top_k      : number of chunks to retrieve
        model_name : embedding model name

    Returns:
        {
            "answer"           : str,
            "retrieved_chunks" : list of {text, score, page_url, page_name, domain},
            "original_query"   : str,
            "rewritten_query"  : str,
            "pipeline"         : "rrr",
        }
    """
    # ── Step 1: REWRITE ────────────────────────────────────────────────
    rewritten_query = rewrite_query(query)

    # ── Step 2: RETRIEVE (using rewritten query) ───────────────────────
    raw_chunks = retrieve(
        query=rewritten_query,
        index=index,
        top_k=top_k,
        model_name=model_name,
    )

    # ── Step 3: READ (generate answer using original query) ────────────
    # We pass the original query to the LLM so the answer matches what
    # the user actually asked, but retrieval used the cleaner rewritten form.
    answer = generate_answer(query, raw_chunks)

    # Format output
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
        "original_query":   query,
        "rewritten_query":  rewritten_query,
        "pipeline":         "rrr",
    }