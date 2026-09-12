"""
Evaluation metrics for Multi-Hop Question Answering:
- Exact Match (EM)
- Token F1 Score
- Retrieval Recall@k & Supporting Fact Recall
"""

from typing import List, Set, Tuple
import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Computes exact string match after standard normalization."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Computes token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_retrieval_metrics(
    retrieved_titles: List[str],
    ground_truth_titles: List[str]
) -> dict:
    """
    Computes passage retrieval precision, recall, and exact set match.
    """
    ret_set = set(retrieved_titles)
    gt_set = set(ground_truth_titles)

    if not gt_set:
        return {"recall": 1.0, "precision": 1.0, "full_coverage": 1.0}

    hits = len(ret_set.intersection(gt_set))
    recall = hits / len(gt_set)
    precision = hits / len(ret_set) if ret_set else 0.0
    full_coverage = 1.0 if hits == len(gt_set) else 0.0

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "full_coverage": round(full_coverage, 4)
    }
