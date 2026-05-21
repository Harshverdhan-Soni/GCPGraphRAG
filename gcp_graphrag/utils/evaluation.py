"""Evaluation framework: retrieval, context efficiency, generation quality."""
import re, logging
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0


def evidence_relevance_density(evidence, threshold=0.5):
    if not evidence: return 0.0
    return sum(1 for e in evidence if e.get("importance_score", 0) >= threshold) / len(evidence)


def redundancy_ratio(evidence, similarity_threshold=0.85):
    n = len(evidence)
    if n < 2: return 0.0
    def tokens(text): return set(re.split(r"\W+", text.lower()))
    pairs = n * (n - 1) // 2
    red   = sum(
        1 for i in range(n) for j in range(i + 1, n)
        if len(tokens(str(evidence[i].get("content",""))) &
               tokens(str(evidence[j].get("content","")))) /
           max(len(tokens(str(evidence[i].get("content",""))) |
                   tokens(str(evidence[j].get("content","")))), 1) >= similarity_threshold
    )
    return red / max(pairs, 1)


def token_budget_adherence(token_stats):
    used   = token_stats.get("tokens_used", 0)
    budget = token_stats.get("token_budget", 1)
    return min(used / max(budget, 1), 1.0)


def relevance_per_token(evidence, token_stats):
    used = token_stats.get("tokens_used", 1)
    if not evidence or used == 0: return 0.0
    return sum(e.get("importance_score", 0) for e in evidence) / used


def exact_match(prediction, ground_truth):
    return float(_norm(prediction) == _norm(ground_truth))


def f1_score(prediction, ground_truth):
    pred = _norm(prediction).split()
    gt   = _norm(ground_truth).split()
    common   = Counter(pred) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    p = num_same / len(pred)
    r = num_same / len(gt)
    return 2 * p * r / (p + r)


def rouge_l(prediction, reference):
    pred = _norm(prediction).split()
    ref  = _norm(reference).split()
    lcs  = _lcs(pred, ref)
    if lcs == 0: return 0.0
    p = lcs / len(pred)
    r = lcs / len(ref)
    return 2 * p * r / (p + r)


def evaluate_pipeline(prediction, ground_truth, retrieved_ids, relevant_ids,
                      evidence, token_stats, k=10):
    return {
        f"recall@{k}":             recall_at_k(retrieved_ids, relevant_ids, k),
        "evidence_density":        evidence_relevance_density(evidence),
        "redundancy_ratio":        redundancy_ratio(evidence),
        "budget_adherence":        token_budget_adherence(token_stats),
        "relevance_per_token":     relevance_per_token(evidence, token_stats),
        "context_reduction_ratio": token_stats.get("reduction_ratio", 0),
        "exact_match":             exact_match(prediction, ground_truth),
        "f1":                      f1_score(prediction, ground_truth),
        "rouge_l":                 rouge_l(prediction, ground_truth),
    }


def _norm(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _lcs(a, b):
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return int(dp[m][n])
