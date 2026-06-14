"""
multi_query_rag.py
Multi-Query RAG: generate N query variants → retrieve for each → union/deduplicate
chunks → re-rank by frequency and score → generate answer.

Difference from RAG Fusion:
  - RAG Fusion uses Reciprocal Rank Fusion (RRF) to mathematically merge ranked lists.
  - Multi-Query RAG simply takes the UNION of all retrieved chunks, deduplicates them,
    re-ranks by (appearance_count, best_score), and picks the top-k.
    Simpler merge logic, faster, and easier to inspect.
"""

from collections import defaultdict
from typing import Dict, Any, List, Tuple

from src.corpus import CorpusIndex
from src.retrieval import retrieve_multi
from src.generation import generate_answer, generate_query_variants


def _merge_chunks(
    ranked_lists: List[List[Tuple[Dict, float]]],
    top_k: int = 5,
) -> List[Tuple[Dict, float]]:
    """
    Union + deduplicate chunks from multiple ranked lists.
    Re-rank by: (number of lists the chunk appeared in DESC, best score DESC).

    Args:
        ranked_lists : one ranked list per query variant
        top_k        : how many to return after merging

    Returns:
        List of (chunk_dict, best_score) sorted by frequency then score.
    """
    # key = chunk text (dedup), value = {chunk, best_score, count}
    seen: Dict[str, Dict] = {}

    for ranked_list in ranked_lists:
        for chunk, score in ranked_list:
            key = chunk["text"]
            if key not in seen:
                seen[key] = {"chunk": chunk, "best_score": score, "count": 1}
            else:
                seen[key]["count"] += 1
                if score > seen[key]["best_score"]:
                    seen[key]["best_score"] = score

    # Sort: more appearances first, then higher score first
    sorted_items = sorted(
        seen.values(),
        key=lambda x: (x["count"], x["best_score"]),
        reverse=True,
    )

    return [(item["chunk"], item["best_score"]) for item in sorted_items[:top_k]]


def run(
    query: str,
    index: CorpusIndex,
    top_k: int = 5,
    n_variants: int = 3,
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """
    Run the Multi-Query RAG pipeline.

    Flow:
        1. Generate n_variants rephrasings of the original query.
        2. Retrieve top-k chunks for EACH variant independently.
        3. Union all retrieved chunks; deduplicate by text.
        4. Re-rank: chunks appearing in more variant results ranked higher;
           ties broken by best cosine score.
        5. Pass top-k merged chunks + original query to LLM for answer generation.

    Args:
        query      : user question string
        index      : built/loaded CorpusIndex
        top_k      : final number of chunks to pass to LLM
        n_variants : number of query rephrasing variants to generate
        model_name : embedding model name

    Returns:
        {
            "answer"          : str,
            "retrieved_chunks": list of {text, score, page_url, page_name, domain, appearances},
            "query_variants"  : list of str (all queries used),
            "pipeline"        : "multi_query_rag",
        }
    """
    # Step 1: Generate query variants (includes original as first entry)
    all_queries = generate_query_variants(query, n=n_variants)

    # Step 2: Retrieve for each variant
    ranked_lists = retrieve_multi(
        queries=all_queries,
        index=index,
        top_k=top_k + 5,   # retrieve a few extra before merging
        model_name=model_name,
    )

    # Step 3 + 4: Union, deduplicate, re-rank
    merged_chunks = _merge_chunks(ranked_lists, top_k=top_k)

    # Step 5: Generate answer
    answer = generate_answer(query, merged_chunks)

    # Format output — include how many variant lists each chunk appeared in
    # (compute appearances from ranked_lists)
    text_to_count: Dict[str, int] = defaultdict(int)
    for rl in ranked_lists:
        seen_in_list = set()
        for chunk, _ in rl:
            t = chunk["text"]
            if t not in seen_in_list:
                text_to_count[t] += 1
                seen_in_list.add(t)

    retrieved = [
        {
            "text":        chunk.get("text", ""),
            "score":       round(score, 4),
            "page_url":    chunk.get("page_url", ""),
            "page_name":   chunk.get("page_name", ""),
            "domain":      chunk.get("domain", ""),
            "appearances": text_to_count.get(chunk.get("text", ""), 1),
        }
        for chunk, score in merged_chunks
    ]

    return {
        "answer":           answer,
        "retrieved_chunks": retrieved,
        "query_variants":   all_queries,
        "pipeline":         "multi_query_rag",
    }