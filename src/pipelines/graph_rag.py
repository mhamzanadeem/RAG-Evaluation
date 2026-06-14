"""
graph_rag.py
Graph RAG pipeline.

Graph construction (done once per session, cached):
  - Nodes  = corpus chunks
  - Edges  = cosine similarity > threshold between any two chunks
             (we use an approximate approach: for each chunk we only connect
              its top-M nearest neighbours to keep the graph sparse)

Retrieval (per query):
  1. Embed query, find seed nodes (top-k by vector similarity).
  2. Expand seed set by traversing the similarity graph (BFS, depth 1-2).
  3. Rank the expanded set by their similarity score and select top-k.
  4. Generate answer from the graph-expanded context.
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, Any, List, Tuple, Set

from src.corpus import VectorIndex, embed_texts
from src.retrieval import embed_query, retrieve_by_embedding
from src.generation import generate_answer


# ---------------------------------------------------------------------------
# Lazy graph cache (built once per process)
# ---------------------------------------------------------------------------

_graph_cache: Dict[int, Dict[int, float]] = {}  # adjacency: node_id -> {neighbour_id: sim}
_graph_index_id: int = None  # identity of the index we built the graph for


def _build_graph(
    index: VectorIndex,
    edge_threshold: float = 0.7,
    top_m_neighbours: int = 8,
) -> Dict[int, Dict[int, float]]:
    """
    Build a sparse similarity graph over all chunks in the index.
    Each node i is connected to its top_m_neighbours with cosine sim ≥ edge_threshold.

    We use the index's own retrieve() to find neighbours efficiently.
    """
    graph: Dict[int, Dict[int, float]] = defaultdict(dict)
    embs = index.embeddings  # (N, D)
    N = len(index)

    # Normalise once
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    normed = (embs / norms).astype(np.float32)

    # Build a name→id lookup
    chunk_to_id = {text: i for i, text in enumerate(index.chunks)}

    # For each node, find top-M neighbours via brute-force cosine
    # (for large corpora, sample a subset to avoid O(N^2) cost)
    MAX_NODES_FULL = 5000
    if N <= MAX_NODES_FULL:
        # Full pairwise (manageable for small/medium corpora)
        sims = normed @ normed.T  # (N, N)
        for i in range(N):
            row = sims[i]
            row[i] = -1.0  # exclude self
            neighbours = np.argpartition(row, -top_m_neighbours)[-top_m_neighbours:]
            for j in neighbours:
                sim = float(row[j])
                if sim >= edge_threshold:
                    graph[i][j] = sim
                    graph[j][i] = sim
    else:
        # Approximate: each node connects to its top_m_neighbours from the index
        for i, text in enumerate(index.chunks):
            q_emb = normed[i]
            # retrieve returns (text, score, meta); we need IDs
            top = index.retrieve(q_emb, top_k=top_m_neighbours + 1)
            for chunk_text, score, _meta in top:
                if chunk_text == text:
                    continue
                j = chunk_to_id.get(chunk_text)
                if j is not None and score >= edge_threshold:
                    graph[i][j] = score
                    graph[j][i] = score

    return dict(graph)


def _get_graph(index: VectorIndex, edge_threshold: float = 0.7, top_m: int = 8) -> Dict[int, Dict[int, float]]:
    global _graph_cache, _graph_index_id
    idx_id = id(index)
    if idx_id != _graph_index_id or not _graph_cache:
        print("Building similarity graph for Graph RAG …")
        _graph_cache = _build_graph(index, edge_threshold=edge_threshold, top_m_neighbours=top_m)
        _graph_index_id = idx_id
        print(f"Graph built: {len(_graph_cache)} nodes with edges.")
    return _graph_cache


# ---------------------------------------------------------------------------
# Graph-aware retrieval
# ---------------------------------------------------------------------------

def _chunk_id(index: VectorIndex, text: str) -> int:
    """Find the index of `text` in the corpus."""
    try:
        return index.chunks.index(text)
    except ValueError:
        return -1


def bfs_expand(
    seed_ids: List[int],
    graph: Dict[int, Dict[int, float]],
    depth: int = 1,
    max_nodes: int = 20,
) -> Set[int]:
    """Breadth-first expansion from seed nodes up to `depth` hops."""
    visited: Set[int] = set(seed_ids)
    queue = deque((node_id, 0) for node_id in seed_ids)
    while queue:
        node, d = queue.popleft()
        if d >= depth:
            continue
        for neighbour in graph.get(node, {}):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, d + 1))
            if len(visited) >= max_nodes:
                return visited
    return visited


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run(
    query: str,
    index: VectorIndex,
    top_k: int = 5,
    embedding_model: str = "all-MiniLM-L6-v2",
    provider: str = None,
    bfs_depth: int = 1,
    edge_threshold: float = 0.65,
    top_m_neighbours: int = 8,
    **kwargs,
) -> Dict[str, Any]:
    """
    Graph RAG pipeline.

    Returns:
        {
            "answer": str,
            "retrieved_chunks": [(text, score, meta), ...],
            "seed_chunks": [(text, score, meta), ...],
            "pipeline": "graph_rag",
        }
    """
    # 1. Vector seed retrieval
    q_emb = embed_query(query, embedding_model=embedding_model)
    seed_results = retrieve_by_embedding(q_emb, index, top_k=top_k)

    # 2. Build / load the graph
    graph = _get_graph(index, edge_threshold=edge_threshold, top_m=top_m_neighbours)

    # 3. Map seed chunks to node IDs
    seed_ids = []
    for text, _score, _meta in seed_results:
        nid = _chunk_id(index, text)
        if nid >= 0:
            seed_ids.append(nid)

    # 4. BFS expansion
    expanded_ids = bfs_expand(seed_ids, graph, depth=bfs_depth, max_nodes=top_k * 5)

    # 5. Score expanded nodes by cosine similarity to query
    expanded_chunks: List[Tuple[str, float, dict]] = []
    norms = np.linalg.norm(index.embeddings, axis=1) + 1e-10
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)
    for nid in expanded_ids:
        text = index.chunks[nid]
        emb = index.embeddings[nid] / norms[nid]
        score = float(np.dot(emb, q_norm))
        meta = index.metadata[nid] if nid < len(index.metadata) else {}
        expanded_chunks.append((text, score, meta))

    # Sort by score descending, take top_k
    expanded_chunks.sort(key=lambda x: x[1], reverse=True)
    top_chunks = expanded_chunks[:top_k]

    # 6. Generate answer
    context = [(text, score) for text, score, _meta in top_chunks]
    answer = generate_answer(query, context, provider=provider)

    return {
        "answer": answer,
        "retrieved_chunks": top_chunks,
        "seed_chunks": seed_results,
        "pipeline": "graph_rag",
    }
