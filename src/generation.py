"""
generation.py
Thin wrapper around an LLM API for answer generation.
Uses local Ollama for text generation.
Configure via config/config.yaml.
"""

import os
from typing import List, Tuple

import requests


# ---------------------------------------------------------------------------
# Lazy client initialisation
# ---------------------------------------------------------------------------

_groq_client = None
_gemini_client = None
_config: dict = {}


def _load_config():
    global _config
    if _config:
        return _config
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.example.yaml")
    with open(cfg_path) as f:
        _config = yaml.safe_load(f) or {}
    return _config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    context_chunks: List[Tuple[str, float]],  # (text, score)
    system_prompt: str = None,
    max_tokens: int = 512,
    provider: str = None,
) -> str:
    """
    Generate an answer for `query` given retrieved `context_chunks`.

    context_chunks: list of (chunk_text, similarity_score)
    provider: "ollama" | None (auto-detect from config)
    """
    cfg = _load_config()
    if provider is None:
        provider = cfg.get("llm_provider", "ollama")

    # Build context string
    context_parts = []
    for i, (text, score) in enumerate(context_chunks, 1):
        context_parts.append(f"[{i}] (score={score:.3f}) {text}")
    context_str = "\n\n".join(context_parts)

    if system_prompt is None:
        system_prompt = (
            "You are a factual question-answering assistant. "
            "Answer the question concisely and accurately using only the provided context. "
            "Use the context to answer. If the answer is partially there, give your best answer based on it. "  
            "Do not hallucinate. Keep answers short (1-3 sentences)."
        )

    user_msg = (
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    if provider != "ollama":
        raise ValueError(f"Unsupported LLM provider: {provider}. This project is configured for ollama only.")
    return _generate_ollama(system_prompt, user_msg, max_tokens)


def generate_text(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 512,
    provider: str = None,
) -> str:
    """
    Raw text generation (used by HyDE for hypothetical doc, RAG Fusion for query variants, etc.)
    """
    cfg = _load_config()
    if provider is None:
        provider = cfg.get("llm_provider", "ollama")

    if provider != "ollama":
        raise ValueError(f"Unsupported LLM provider: {provider}. This project is configured for ollama only.")
    return _generate_ollama(system_prompt, prompt, max_tokens)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _generate_ollama(system_prompt: str, user_msg: str, max_tokens: int) -> str:
    cfg = _load_config()
    model = cfg.get("generation_model") or cfg.get("ollama_model", "llama3.2:3b")
    base_url = cfg.get("ollama_url", "http://localhost:11434")
    endpoint = f"{base_url.rstrip('/')}/api/chat"

    try:
        response = requests.post(
            endpoint,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        return str(content).strip()
    except Exception as e:
        return f"[Generation error: {e}]"
