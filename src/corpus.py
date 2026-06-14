"""
corpus.py
Build, save, and load a global embedding index over all page_snippets.

Index interface:
    index.retrieve(query_embedding, top_k) -> list of (chunk_text, score)
    build_index(dataset_path, ...) -> Index
    load_index(index_path, ...) -> Index
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

from src.data_loader import load_dataset


# ---------------------------------------------------------------------------
# Index class
# ---------------------------------------------------------------------------

class VectorIndex:
    """
    In-memory cosine-similarity index backed by numpy arrays.
    Optionally wraps FAISS if available (falls back to numpy silently).
    """

    def __init__(self):
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None  # shape (N, D)
        self.metadata: List[dict] = []  # per-chunk extra info (url, domain …)
        self._faiss_index = None

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add(self, texts: List[str], embeddings: np.ndarray, metadata: List[dict] = None):
        """Append texts and their embeddings to the index."""
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have equal length")
        self.chunks.extend(texts)
        if self.embeddings is None:
            self.embeddings = embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(texts))

    def build_faiss(self):
        """Build an optional FAISS flat IP index for faster search."""
        try:
            import faiss  # type: ignore
            d = self.embeddings.shape[1]
            # Normalize once so inner-product == cosine
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
            normed = (self.embeddings / norms).astype(np.float32)
            idx = faiss.IndexFlatIP(d)
            idx.add(normed)
            self._faiss_index = idx
            self._normed_embeddings = normed
            print("FAISS index built.")
        except ImportError:
            print("FAISS not available; using numpy cosine search.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, float, dict]]:
        """
        Return top_k (chunk_text, cosine_score, metadata) tuples.
        query_embedding: 1-D numpy array of shape (D,)
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        q = query_embedding.astype(np.float32)

        if self._faiss_index is not None:
            q_norm = q / (np.linalg.norm(q) + 1e-10)
            scores, indices = self._faiss_index.search(q_norm.reshape(1, -1), top_k)
            scores = scores[0]
            indices = indices[0]
            results = []
            for score, idx in zip(scores, indices):
                if idx < 0:
                    continue
                results.append((self.chunks[idx], float(score), self.metadata[idx]))
            return results

        # Numpy fallback
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        emb_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        normed = self.embeddings / emb_norms
        scores = normed @ q_norm  # (N,)

        top_indices = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            (self.chunks[i], float(scores[i]), self.metadata[i])
            for i in top_indices
        ]

    def __len__(self):
        return len(self.chunks)


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

_encoder = None


def get_encoder(model_name: str = "all-MiniLM-L6-v2"):
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {model_name}")
        _encoder = SentenceTransformer(model_name)
    return _encoder


def embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 256) -> np.ndarray:
    encoder = get_encoder(model_name)
    return encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------

def build_index(
    dataset_path: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    max_examples: int = None,
    batch_size: int = 256,
    use_faiss: bool = True,
) -> VectorIndex:
    """
    Read the CRAG dataset, collect all page_snippets as chunks,
    embed them, and return a populated VectorIndex.
    """
    print("Building global corpus from dataset …")
    texts: List[str] = []
    meta: List[dict] = []

    seen: set = set()  # deduplicate exact snippets

    for ex in tqdm(load_dataset(dataset_path, max_examples=max_examples), desc="Loading rows"):
        domain = ex.get("domain", "")
        for sr in ex.get("search_results", []):
            snippet = sr.get("page_snippet", "").strip()
            if snippet and snippet not in seen:
                seen.add(snippet)
                texts.append(snippet)
                meta.append({
                    "page_url": sr.get("page_url", ""),
                    "page_name": sr.get("page_name", ""),
                    "domain": domain,
                })

    print(f"Corpus size: {len(texts)} unique chunks.")

    index = VectorIndex()

    # Embed in one shot (sentence-transformers handles batching internally)
    print("Embedding chunks …")
    embeddings = embed_texts(texts, model_name=embedding_model, batch_size=batch_size)
    index.add(texts, embeddings, meta)

    if use_faiss:
        index.build_faiss()

    print(f"Index built with {len(index)} vectors.")
    return index


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_index(index: VectorIndex, index_path: str):
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Index saved to {index_path}")


def load_index(index_path: str) -> VectorIndex:
    if not Path(index_path).exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    print(f"Loading index from {index_path} …")
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    print(f"Index loaded: {len(index)} vectors.")
    return index


def get_or_build_index(
    index_path: str,
    dataset_path: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    max_examples: int = None,
    force_rebuild: bool = False,
) -> VectorIndex:
    """
    Load the index from disk if available; otherwise build and save it.
    """
    if not force_rebuild and Path(index_path).exists():
        return load_index(index_path)
    idx = build_index(
        dataset_path=dataset_path,
        embedding_model=embedding_model,
        max_examples=max_examples,
    )
    save_index(idx, index_path)
    return idx
