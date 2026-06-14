"""
data_loader.py
Load the CRAG dataset and yield structured examples.
"""

import json
from pathlib import Path
from typing import Generator, Dict, Any, List, Tuple


def load_dataset(
    dataset_path: str,
    max_examples: int = None,
    split: int = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Iterate over the CRAG JSONL dataset.

    Yields dicts with keys:
        interaction_id, query_time, domain, question_type,
        static_or_dynamic, query, answer, alt_ans, split,
        popularity, search_results
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Download crag_task_1_and_2_dev_v4.jsonl.bz2 from the CRAG repo, "
            "decompress it, and place it in the dataset/ folder."
        )

    count = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Optional filter by split (0 = val, 1 = test)
            if split is not None and example.get("split") != split:
                continue

            yield example
            count += 1
            if max_examples and count >= max_examples:
                break


def iter_examples(
    dataset_path: str,
    max_examples: int = None,
    split: int = None,
) -> Generator[Tuple[str, str, List[str], List[Dict]], None, None]:
    """
    Convenience iterator that yields (query, answer, alt_ans, search_results).
    alt_ans is guaranteed to be a list.
    search_results is a list of dicts with keys:
        page_name, page_url, page_snippet, page_result, page_last_modified
    """
    for ex in load_dataset(dataset_path, max_examples=max_examples, split=split):
        query = ex.get("query", "")
        answer = ex.get("answer", "")
        alt_ans = ex.get("alt_ans", [])
        if not isinstance(alt_ans, list):
            alt_ans = [alt_ans] if alt_ans else []
        search_results = ex.get("search_results", [])
        yield query, answer, alt_ans, search_results


def get_snippets(search_results: List[Dict]) -> List[str]:
    """Extract non-empty page_snippet strings from a search_results list."""
    snippets = []
    for item in search_results:
        snippet = item.get("page_snippet", "").strip()
        if snippet:
            snippets.append(snippet)
    return snippets


def get_full_example(dataset_path: str, index: int) -> Dict[str, Any]:
    """Return a single example by 0-based line index (for frontend sample queries)."""
    for i, ex in enumerate(load_dataset(dataset_path)):
        if i == index:
            return ex
    raise IndexError(f"No example at index {index}")
