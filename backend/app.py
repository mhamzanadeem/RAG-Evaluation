"""
backend/app.py
Flask + Flask-CORS backend for the React frontend.

Endpoints:
  GET  /api/health             — liveness check
  POST /api/query              — run a pipeline on a query
  GET  /api/samples            — return N sample queries from the dataset
  GET  /api/pipelines          — list available pipelines
"""

import os
import sys
import json
import yaml
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# Ensure project root is on the path when run from backend/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.corpus import get_or_build_index
from src.data_loader import load_dataset
from src.pipelines import PIPELINES, PIPELINE_NAMES

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Config + index (loaded once at startup)
# ---------------------------------------------------------------------------

def _load_config():
    for candidate in [
        ROOT / "config" / "config.yaml",
        ROOT / "config" / "config.example.yaml",
    ]:
        if candidate.exists():
            with open(candidate) as f:
                return yaml.safe_load(f) or {}
    return {}


CFG = _load_config()
INDEX = None


def get_index():
    global INDEX
    if INDEX is None:
        INDEX = get_or_build_index(
            index_path=CFG.get("index_path", str(ROOT / "index" / "global.pkl")),
            dataset_path=CFG.get("dataset_path", str(ROOT / "dataset" / "crag_task_1_and_2_dev_v4.jsonl")),
            embedding_model=CFG.get("embedding_model", "all-MiniLM-L6-v2"),
            max_examples=CFG.get("max_index_examples", None),
        )
    return INDEX


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/pipelines")
def list_pipelines():
    labels = {
        "rag_fusion": "RAG Fusion",
        "hyde": "HyDE",
        "crag": "CRAG",
        "graph_rag": "Graph RAG",
    }
    return jsonify([
        {"id": name, "label": labels.get(name, name)}
        for name in PIPELINE_NAMES
    ])


@app.route("/api/samples")
def get_samples():
    """Return up to N query samples from the dataset."""
    n = int(request.args.get("n", 10))
    dataset_path = CFG.get("dataset_path", str(ROOT / "dataset" / "crag_task_1_and_2_dev_v4.jsonl"))
    samples = []
    try:
        for i, ex in enumerate(load_dataset(dataset_path, max_examples=n * 3)):
            if len(samples) >= n:
                break
            samples.append({
                "query": ex.get("query", ""),
                "answer": ex.get("answer", ""),
                "domain": ex.get("domain", ""),
                "question_type": ex.get("question_type", ""),
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(samples)


@app.route("/api/query", methods=["POST"])
def query_pipeline():
    """
    POST body: { "query": "...", "pipeline": "rag_fusion|hyde|crag|graph_rag" }
    Returns: { answer, retrieved_chunks, pipeline, ... }
    """
    data = request.get_json(force=True) or {}
    query = data.get("query", "").strip()
    pipeline_name = data.get("pipeline", "rag_fusion").strip()

    if not query:
        return jsonify({"error": "query is required"}), 400
    if pipeline_name not in PIPELINES:
        return jsonify({"error": f"Unknown pipeline: {pipeline_name}"}), 400

    try:
        index = get_index()
        pipeline_fn = PIPELINES[pipeline_name]
        out = pipeline_fn(
            query=query,
            index=index,
            top_k=CFG.get("top_k", 5),
            embedding_model=CFG.get("embedding_model", "all-MiniLM-L6-v2"),
            provider=CFG.get("llm_provider", "ollama"),
        )

        # Serialise chunks
        chunks_serial = []
        for item in out.get("retrieved_chunks", []):
            text, score, meta = item
            chunks_serial.append({
                "text": text,
                "score": float(score),
                "page_url": meta.get("page_url", ""),
                "page_name": meta.get("page_name", ""),
                "domain": meta.get("domain", ""),
            })

        response = {
            "answer": out.get("answer", ""),
            "pipeline": pipeline_name,
            "retrieved_chunks": chunks_serial,
        }

        # Pipeline-specific extras
        if "query_variants" in out:
            response["query_variants"] = out["query_variants"]
        if "hypothetical_doc" in out:
            response["hypothetical_doc"] = out["hypothetical_doc"]
        if "confidence_label" in out:
            response["confidence_label"] = out["confidence_label"]
            response["confidence_score"] = float(out["confidence_score"])
            response["used_retrieval"] = out.get("used_retrieval", True)

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting RAG backend on port {port} …")
    print("Loading index (this may take a moment on first run) …")
    get_index()  # pre-warm
    app.run(host="0.0.0.0", port=port, debug=False)
