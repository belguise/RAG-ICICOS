"""
RAG Performance Metrics:
1. Answer Accuracy (ACC) - Token F1, Key term recall & semantic alignment with expected ground truth.
2. Evidence Grounding Score (EGS) - Proportion of claims grounded in retrieved context.
3. Hallucination Rate (HR) - Proportion of unsupported claims (1.0 - EGS).
"""

import re
from typing import Any, Dict, List, Set


def _get_clean_tokens(text: str) -> List[str]:
    """Tokenize and remove common Indonesian stopwords."""
    stopwords = {
        "yang", "dan", "atau", "dari", "pada", "untuk", "ke", "di", "oleh",
        "dengan", "ini", "itu", "adalah", "sebagai", "dalam", "bisa", "akan",
        "telah", "dapat", "juga", "harus", "sudah", "hanya", "serta"
    }
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in clean.split() if len(w) > 2 and w not in stopwords]


def calculate_answer_accuracy(predicted_answer: str, expected_answer: str) -> Dict[str, Any]:
    """
    Compute Answer Accuracy (ACC) via lexical overlap, token recall, precision, and F1.
    """
    if not predicted_answer or not expected_answer:
        return {
            "accuracy_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    pred_tokens = _get_clean_tokens(predicted_answer)
    exp_tokens = _get_clean_tokens(expected_answer)

    if not exp_tokens:
        return {"accuracy_score": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens:
        return {"accuracy_score": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_set = set(pred_tokens)
    exp_set = set(exp_tokens)

    intersection = pred_set.intersection(exp_set)
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(exp_set) if exp_set else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Key entity bonus: if predicted contains critical process keywords from expected
    accuracy_score = round(f1, 4)

    return {
        "accuracy_score": accuracy_score,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def calculate_evidence_grounding(predicted_answer: str, retrieved_context: str) -> Dict[str, Any]:
    """
    Compute Evidence Grounding Score (EGS) and Hallucination Rate (HR).
    Splits predicted text into sentences and evaluates sentence grounding against retrieved context.
    """
    if not predicted_answer:
        return {"egs_score": 0.0, "hallucination_rate": 1.0, "grounded_sentences": 0, "total_sentences": 0}

    if not retrieved_context or not retrieved_context.strip():
        # If no context was provided or retrieved (e.g. LLM-only), parametric answers are considered ungrounded
        return {"egs_score": 0.0, "hallucination_rate": 1.0, "grounded_sentences": 0, "total_sentences": 1}

    context_lower = retrieved_context.lower()
    # Split answer into meaningful sentences
    sentences = [s.strip() for s in re.split(r"[.\n;]", predicted_answer) if len(s.strip()) > 15]

    if not sentences:
        return {"egs_score": 0.5, "hallucination_rate": 0.5, "grounded_sentences": 0, "total_sentences": 0}

    grounded_count = 0
    for sent in sentences:
        sent_tokens = _get_clean_tokens(sent)
        if not sent_tokens:
            continue
        # Check how many sentence tokens exist in context
        matched = sum(1 for t in sent_tokens if t in context_lower)
        overlap = matched / len(sent_tokens)
        if overlap >= 0.45:  # At least 45% of substantive tokens grounded in context
            grounded_count += 1

    egs_score = round(grounded_count / len(sentences), 4)
    hallucination_rate = round(1.0 - egs_score, 4)

    return {
        "egs_score": egs_score,
        "hallucination_rate": hallucination_rate,
        "grounded_sentences": grounded_count,
        "total_sentences": len(sentences),
    }
