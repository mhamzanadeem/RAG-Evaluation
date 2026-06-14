"""
evaluation.py
Utilities for comparing predicted answers against gold answers.

Rubric labels:
    - perfect
    - acceptable
    - missing
    - incorrect

Score mapping:
    - perfect / acceptable -> 1
    - missing              -> 0
    - incorrect            -> -1
"""

import re
import string
from typing import List, Dict, Any, Tuple
import requests



def llm_judge_label(
    query: str,
    prediction: str,
    gold: str,
    ollama_model: str = "llama3",
    ollama_url: str = "http://localhost:11434/api/chat",
) -> str:
    """
    Use local Ollama as an LLM judge.
    Returns one of: perfect, acceptable, missing, incorrect.
    """
    system_prompt = (
        "You are a strict answer evaluator. "
        "Given a question, a ground-truth answer, and a predicted answer, "
        "return ONLY valid JSON like {\"label\": \"perfect\"}. "
        "Allowed labels: perfect, acceptable, missing, incorrect. "
        "perfect: fully correct, no hallucinated content. "
        "acceptable: useful and mostly correct, minor non-harmful errors allowed. "
        "missing: no concrete answer (e.g., I don't know / cannot find). "
        "incorrect: wrong or irrelevant answer. "
        "No extra text."
    )
    user_msg = (
        f"Question: {query}\n"
        f"Ground truth: {gold}\n"
        f"Prediction: {prediction}\n"
        "Is the prediction correct?"
    )
    try:
        resp = requests.post(
            ollama_url,
            json={
                "model": ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                "stream": False,
            },
            timeout=30,
        )
        content = resp.json()["message"]["content"]
        match = re.search(r'"label"\s*:\s*"(perfect|acceptable|missing|incorrect)"', content, re.I)
        if match:
            return match.group(1).lower()
    except Exception as e:
        print(f"[llm_judge] Error: {e}")
    return "incorrect"


MISSING_PATTERNS = [
    r"\bi\s+don'?t\s+know\b",
    r"\bi\s+do\s+not\s+know\b",
    r"\bi\s+am\s+not\s+sure\b",
    r"\bi\s+can'?t\s+find\b",
    r"\bi\s+cannot\s+find\b",
    r"\bi\s+don'?t\s+have\s+(that\s+)?information\b",
    r"\bsorry[,\s]+i\s+(can'?t|cannot|do\s+not)\b",
    r"\bno\s+information\b",
    r"\bnot\s+available\b",
]


def is_missing_answer(prediction: str) -> bool:
    text = _normalise(prediction)
    if not text:
        return True
    for pattern in MISSING_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


# ---------------------------------------------------------------------------
# Text normalisation (mirrors standard QA eval)
# ---------------------------------------------------------------------------

def _normalise(text) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = str(text)  # handles int/float answers like years
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _tokens(text: str) -> set:
    return set(_normalise(text).split())


# ---------------------------------------------------------------------------
# Individual match functions
# ---------------------------------------------------------------------------

def exact_match(prediction: str, gold: str) -> bool:
    return _normalise(prediction) == _normalise(gold)


def token_overlap(prediction: str, gold: str) -> float:
    """F1 token overlap score (standard in open-domain QA)."""
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _best_gold_overlap(prediction: str, all_gold: List[str]) -> Tuple[float, str]:
    best_score = -1.0
    best_gold = ""
    for gold in all_gold:
        score = token_overlap(prediction, gold)
        if score > best_score:
            best_score = score
            best_gold = gold
    return best_score, best_gold


def classify_prediction(
    prediction: str,
    answer: str,
    alt_ans: List[str],
    threshold: float = 0.3,
    use_llm_judge: bool = False,
    query: str = "",
    ollama_model: str = "llama3",
) -> Dict[str, Any]:
    """
    Classify a prediction into rubric labels and assign score.

    Returns dict with keys:
      label: perfect|acceptable|missing|incorrect
      score: 1|0|-1
      matched_gold: str
      overlap: float
    """
    if is_missing_answer(prediction):
        return {
            "label": "missing",
            "score": 0,
            "matched_gold": "",
            "overlap": 0.0,
        }

    all_gold = [answer] + (alt_ans if isinstance(alt_ans, list) else [])
    for gold in all_gold:
        if exact_match(prediction, gold):
            return {
                "label": "perfect",
                "score": 1,
                "matched_gold": gold,
                "overlap": 1.0,
            }

    best_overlap, best_gold = _best_gold_overlap(prediction, all_gold)
    if best_overlap >= threshold:
        return {
            "label": "acceptable",
            "score": 1,
            "matched_gold": best_gold,
            "overlap": best_overlap,
        }

    # LLM judge fallback
    if use_llm_judge and query:
        for gold in all_gold:
            label = llm_judge_label(query, prediction, gold, ollama_model=ollama_model)
            if label in {"perfect", "acceptable", "missing", "incorrect"}:
                score = 1 if label in {"perfect", "acceptable"} else 0 if label == "missing" else -1
                return {
                    "label": label,
                    "score": score,
                    "matched_gold": gold,
                    "overlap": best_overlap if best_overlap > 0 else 0.0,
                }

    return {
        "label": "incorrect",
        "score": -1,
        "matched_gold": best_gold,
        "overlap": max(0.0, best_overlap),
    }


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    results: List[Dict[str, Any]],
    threshold: float = 0.3,
    use_llm_judge: bool = False,
    ollama_model: str = "llama3",
) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
            "total_score": 0,
            "avg_score": 0.0,
            "perfect": 0,
            "acceptable": 0,
            "missing": 0,
            "incorrect": 0,
        }

    counts = {"perfect": 0, "acceptable": 0, "missing": 0, "incorrect": 0}
    total_score = 0

    for r in results:
        verdict = classify_prediction(
            r["prediction"],
            r["answer"],
            r.get("alt_ans", []),
            threshold,
            use_llm_judge=use_llm_judge,
            query=r.get("query", ""),
            ollama_model=ollama_model,
        )
        label = verdict["label"]
        counts[label] += 1
        total_score += verdict["score"]
        r["evaluation"] = verdict

    correct = counts["perfect"] + counts["acceptable"]

    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "total_score": total_score,
        "avg_score": total_score / total,
        "perfect": counts["perfect"],
        "acceptable": counts["acceptable"],
        "missing": counts["missing"],
        "incorrect": counts["incorrect"],
    }


def compare_pipelines(
    pipeline_results: Dict[str, List[Dict[str, Any]]],
    threshold: float = 0.3,
    use_llm_judge: bool = False,
    ollama_model: str = "llama3",
) -> Dict[str, Dict[str, Any]]:
    summary = {}
    for name, results in pipeline_results.items():
        summary[name] = evaluate_pipeline(results, threshold, use_llm_judge, ollama_model)
    return summary