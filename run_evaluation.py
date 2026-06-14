"""
run_evaluation.py
Build (or load) the global index, run all four pipelines on the dev set,
and report accuracy per pipeline.

Usage:
    python run_evaluation.py [--config config/config.yaml] [--max-examples 200]

By default, Ollama LLM judge is enabled.
Use --no-llm-judge to disable it.
"""

import argparse
import json
import os
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from pathlib import Path

import yaml
from tqdm import tqdm

from src.corpus import get_or_build_index
from src.data_loader import iter_examples
from src.evaluation import compare_pipelines
from src.pipelines import PIPELINES


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def debug_ollama_access(ollama_base_url: str = "http://localhost:11434") -> bool:
    """Print a debug line indicating whether Ollama is reachable."""
    tags_url = f"{ollama_base_url.rstrip('/')}/api/tags"
    req = Request(tags_url, method="GET")
    try:
        with urlopen(req, timeout=5) as resp:
            status = getattr(resp, "status", 200)
            if status == 200:
                print(f"DEBUG: Ollama reachable at {tags_url} (HTTP {status})")
                return True
            print(f"DEBUG: Ollama not reachable at {tags_url} (HTTP {status})")
            return False
    except HTTPError as e:
        print(f"DEBUG: Ollama check failed at {tags_url} (HTTPError {e.code}: {e.reason})")
    except URLError as e:
        print(f"DEBUG: Ollama check failed at {tags_url} (URLError: {e.reason})")
    except Exception as e:
        print(f"DEBUG: Ollama check failed at {tags_url} ({type(e).__name__}: {e})")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate all RAG pipelines")
    parser.add_argument("--config", default="config/config.example.yaml")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit number of dev examples (for quick testing)")
    parser.add_argument("--output", default="results/evaluation_results.json",
                        help="Path to save JSON results")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Rebuild the vector index even if cached")
    parser.add_argument("--pipelines", nargs="+", default=None,
                        choices=list(PIPELINES.keys()),
                        help="Run only selected pipelines (default: all)")
    parser.add_argument("--no-llm-judge", action="store_true",
                    help="Disable Ollama LLM judge and use string matching only")
    parser.add_argument("--ollama-model", default="llama3",
                    help="Ollama model name to use as judge (default: llama3)")                    
    args = parser.parse_args()

    # Default behavior: use Ollama judge unless explicitly disabled.
    args.llm_judge = not args.no_llm_judge

    # Load config
    cfg_path = args.config
    if not Path(cfg_path).exists():
        cfg_path = "config/config.example.yaml"
    cfg = load_config(cfg_path)

    dataset_path = cfg.get("dataset_path", "dataset/crag_task_1_and_2_dev_v4.jsonl")
    index_path = cfg.get("index_path", "index/global.pkl")
    embedding_model = cfg.get("embedding_model", "all-MiniLM-L6-v2")
    top_k = cfg.get("top_k", 5)
    provider = cfg.get("llm_provider", "ollama")
    max_examples = 50

    if args.llm_judge:
        print("\n" + "="*60)
        print("DEBUG: OLLAMA CONNECTIVITY CHECK")
        print("="*60)
        debug_ollama_access(cfg.get("ollama_url", "http://localhost:11434"))

    # -----------------------------------------------------------------------
    # Step 1: Build / load index
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: Building / loading global index")
    print("="*60)
    index = get_or_build_index(
        index_path=index_path,
        dataset_path=dataset_path,
        embedding_model=embedding_model,
        max_examples=max_examples,
        force_rebuild=args.force_rebuild,
    )

    # -----------------------------------------------------------------------
    # Step 2: Collect dev examples
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: Loading dev examples")
    print("="*60)
    examples = list(iter_examples(dataset_path, max_examples=max_examples))
    print(f"Loaded {len(examples)} examples.")

    # -----------------------------------------------------------------------
    # Step 3: Run pipelines
    # -----------------------------------------------------------------------
    pipeline_names = args.pipelines or list(PIPELINES.keys())
    pipeline_results: dict = {name: [] for name in pipeline_names}

    for name in pipeline_names:
        pipeline_fn = PIPELINES[name]
        print(f"\n{'='*60}")
        print(f"Running pipeline: {name.upper()}")
        print("="*60)

        results = []
        errors = 0
        t0 = time.time()

        for query, answer, alt_ans, _search_results in tqdm(examples, desc=name):
            try:
                out = pipeline_fn(
                    query=query,
                    index=index,
                    top_k=top_k,
                    embedding_model=embedding_model,
                    provider=provider,
                )
                prediction = out.get("answer", "")
                chunks = out.get("retrieved_chunks", [])

                normalized_chunks = []
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        normalized_chunks.append({
                            "text": chunk.get("text", ""),
                            "score": float(chunk.get("score", 0.0)),
                            "meta": chunk.get("meta", {}),
                        })
                    else:
                        text, score, meta = chunk
                        normalized_chunks.append({
                            "text": text,
                            "score": float(score),
                            "meta": meta,
                        })

                result = {
                    "query": query,
                    "prediction": prediction,
                    "answer": answer,
                    "alt_ans": alt_ans,
                    "retrieved_chunks": normalized_chunks,
                }
                # Pipeline-specific extras
                for extra_key in ["confidence_label", "confidence_score", "query_variants",
                                  "hypothetical_doc", "used_retrieval"]:
                    if extra_key in out:
                        result[extra_key] = out[extra_key]

                results.append(result)

            except Exception as e:
                errors += 1
                results.append({
                    "query": query,
                    "prediction": "",
                    "answer": answer,
                    "alt_ans": alt_ans,
                    "error": str(e),
                })

        elapsed = time.time() - t0
        pipeline_results[name] = results
        print(f"  Done in {elapsed:.1f}s | errors: {errors}")

    # -----------------------------------------------------------------------
    # Step 4: Compute and display evaluation summary
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    summary = compare_pipelines(
        pipeline_results,
        threshold=0.3,
        use_llm_judge=args.llm_judge,
        ollama_model=args.ollama_model,
    )   
    rows = []
    for name, metrics in sorted(summary.items(), key=lambda x: -x[1].get("avg_score", 0.0)):
        acc = metrics["accuracy"]
        avg_score = metrics.get("avg_score", 0.0)
        total_score = metrics.get("total_score", 0)
        correct = metrics["correct"]
        total = metrics["total"]
        perfect = metrics.get("perfect", 0)
        acceptable = metrics.get("acceptable", 0)
        missing = metrics.get("missing", 0)
        incorrect = metrics.get("incorrect", 0)
        row = (
            f"  {name:<20}  avg_score={avg_score:.3f} total_score={total_score:+d}  "
            f"accuracy={acc:.3f} ({correct}/{total})  "
            f"[P={perfect} A={acceptable} M={missing} I={incorrect}]"
        )
        rows.append(row)
        print(row)

    # -----------------------------------------------------------------------
    # Step 5: Save results
    # -----------------------------------------------------------------------
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {
            "dataset_path": dataset_path,
            "embedding_model": embedding_model,
            "top_k": top_k,
            "max_examples": max_examples,
        },
        "summary": summary,
        "pipeline_results": pipeline_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
