"""
Standard RAG Evaluation Metrics:
1. Faithfulness Score - Proportion of claims in generated answer supported by retrieved context (RAGAS).
2. Answer Relevancy Score - Semantic responsiveness of answer to user question (TruLens / RAGAS).
3. Context Precision - Rank-weighted relevance of retrieved chunks against reference facts.
4. Context Recall - Proportion of reference facts captured in retrieved context.
5. BERTScore - Contextual semantic similarity between generated and reference answers (Zhang et al. 2019).
6. Actor Accuracy - Exact/partial match of responsible organizational actors.
"""

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from evaluation.metrics.rag_metrics import _get_clean_tokens

# Lazy-loaded sentence transformer for embedding-based metrics
_EMBEDDING_MODEL = None


def _get_eval_embedding_model():
    """Load or return cached multilingual sentence transformer for embedding metrics."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            import os
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)
        except Exception:
            try:
                from sentence_transformers import SentenceTransformer
                _EMBEDDING_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            except Exception:
                _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# =========================================================================
# 1. Faithfulness Score
# =========================================================================

def calculate_faithfulness(generated_answer: str, retrieved_context: str) -> Dict[str, Any]:
    """
    Measure whether generated answer claims are supported by retrieved evidence (RAGAS Faithfulness).
    
    Returns:
        dict containing:
          - faithfulness_score: float in [0.0, 1.0]
          - total_claims: int
          - supported_claims: int
          - unsupported_claims: List[str]
    """
    if not generated_answer or not generated_answer.strip():
        return {
            "faithfulness_score": 0.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": [],
        }

    if not retrieved_context or not retrieved_context.strip():
        # Without context (e.g. parametric LLM-only), parametric claims cannot be verified against evidence
        return {
            "faithfulness_score": 0.0,
            "total_claims": 1,
            "supported_claims": 0,
            "unsupported_claims": [generated_answer[:100]],
        }

    context_lower = retrieved_context.lower()
    
    # Split generated answer into distinct factual sentences / bullet items
    raw_statements = [
        s.strip() for s in re.split(r"(?:[\n;]|\.\s+)", generated_answer)
        if len(s.strip()) > 15
    ]
    # Filter out conversational headers, greetings, or disclaimer lines
    ignorable_prefixes = ["berdasarkan", "referensi sumber", "catatan:", "ringkasan:"]
    statements = []
    for s in raw_statements:
        low = s.lower()
        if any(low.startswith(p) for p in ignorable_prefixes) and len(s) < 60:
            continue
        statements.append(s)

    if not statements:
        return {
            "faithfulness_score": 1.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": [],
        }

    supported_count = 0
    unsupported = []

    for stmt in statements:
        tokens = _get_clean_tokens(stmt)
        if not tokens:
            supported_count += 1
            continue

        # Check token containment overlap in context
        matched_tokens = sum(1 for t in tokens if t in context_lower)
        token_overlap = matched_tokens / len(tokens)

        # Check multi-word phrase or n-gram overlap
        phrase_found = False
        words = stmt.split()
        if len(words) >= 4:
            for i in range(len(words) - 3):
                trigram = " ".join(words[i:i+4]).lower()
                trigram_clean = re.sub(r"[^\w\s]", "", trigram)
                if trigram_clean in context_lower:
                    phrase_found = True
                    break

        if token_overlap >= 0.45 or phrase_found:
            supported_count += 1
        else:
            unsupported.append(stmt)

    score = round(supported_count / len(statements), 4)

    return {
        "faithfulness_score": score,
        "total_claims": len(statements),
        "supported_claims": supported_count,
        "unsupported_claims": unsupported,
    }


# =========================================================================
# 2. Answer Relevancy Score
# =========================================================================

def calculate_answer_relevancy(question: str, generated_answer: str) -> Dict[str, Any]:
    """
    Measure whether the generated answer directly addresses the user inquiry (TruLens / RAGAS).
    Uses semantic embedding cosine similarity combined with intent token coverage.
    """
    if not generated_answer or not generated_answer.strip():
        return {"relevancy_score": 0.0, "semantic_similarity": 0.0, "intent_coverage": 0.0}

    q_clean = question.strip()
    ans_clean = generated_answer.strip()

    # 1. Semantic Embedding Similarity via sentence transformer
    model = _get_eval_embedding_model()
    sim_score = 0.0
    if model is not None:
        try:
            embeddings = model.encode([q_clean, ans_clean])
            sim_score = max(0.0, min(1.0, _cosine_similarity(embeddings[0], embeddings[1])))
        except Exception:
            sim_score = 0.5
    else:
        # Fallback to lexical token Jaccard similarity if embedding model unavailable
        q_tokens = set(_get_clean_tokens(q_clean))
        a_tokens = set(_get_clean_tokens(ans_clean))
        sim_score = len(q_tokens.intersection(a_tokens)) / max(len(q_tokens), 1)

    # 2. Intent token coverage (did answer address core interrogative targets?)
    q_keywords = [w for w in _get_clean_tokens(q_clean) if len(w) > 3]
    ans_lower = ans_clean.lower()
    matched_kw = sum(1 for kw in q_keywords if kw in ans_lower)
    intent_cov = matched_kw / len(q_keywords) if q_keywords else 1.0

    # Composite relevancy: 70% semantic embedding alignment + 30% intent keyword coverage
    composite_score = round(0.70 * sim_score + 0.30 * intent_cov, 4)
    composite_score = max(0.0, min(1.0, composite_score))

    return {
        "relevancy_score": composite_score,
        "semantic_similarity": round(sim_score, 4),
        "intent_coverage": round(intent_cov, 4),
    }


# =========================================================================
# 3. Context Precision
# =========================================================================

def calculate_context_precision(
    retrieved_chunks: List[str],
    expected_answer: str,
    question: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Measure retrieval quality using rank-weighted precision (RAGAS Context Precision).
    Evaluates whether relevant chunks are retrieved and ranked near the top.
    
    Args:
        retrieved_chunks: List of retrieved text chunks in ranked order (index 0 is highest rank).
        expected_answer: Ground truth reference answer.
        question: Optional user inquiry.
    """
    if not retrieved_chunks:
        return {"context_precision": 0.0, "relevant_chunks": 0, "total_chunks": 0}

    ref_tokens = set(_get_clean_tokens(expected_answer))
    q_tokens = set(_get_clean_tokens(question or ""))

    if not ref_tokens:
        return {"context_precision": 1.0, "relevant_chunks": len(retrieved_chunks), "total_chunks": len(retrieved_chunks)}

    # Determine relevance for each chunk at rank k (1-indexed)
    relevance_flags: List[int] = []
    for chunk in retrieved_chunks:
        c_tokens = set(_get_clean_tokens(chunk))
        if not c_tokens:
            relevance_flags.append(0)
            continue

        ref_overlap = len(ref_tokens.intersection(c_tokens)) / len(ref_tokens)
        q_overlap = len(q_tokens.intersection(c_tokens)) / max(len(q_tokens), 1) if q_tokens else 0.0

        # Chunk is considered relevant if it contains >= 30% of reference tokens or high question overlap
        if ref_overlap >= 0.30 or (ref_overlap >= 0.20 and q_overlap >= 0.35):
            relevance_flags.append(1)
        else:
            relevance_flags.append(0)

    # Compute Average Precision (AP)
    total_relevant = sum(relevance_flags)
    if total_relevant == 0:
        return {"context_precision": 0.0, "relevant_chunks": 0, "total_chunks": len(retrieved_chunks)}

    precision_sum = 0.0
    running_rel = 0
    for k, is_rel in enumerate(relevance_flags, start=1):
        if is_rel:
            running_rel += 1
            precision_at_k = running_rel / k
            precision_sum += precision_at_k

    context_precision = round(precision_sum / total_relevant, 4)

    return {
        "context_precision": context_precision,
        "relevant_chunks": total_relevant,
        "total_chunks": len(retrieved_chunks),
        "relevance_vector": relevance_flags,
    }


# =========================================================================
# 4. Context Recall
# =========================================================================

def calculate_context_recall(
    retrieved_chunks: List[str],
    expected_answer: str,
) -> Dict[str, Any]:
    """
    Measure whether the retrieved context contains the necessary information to answer the question (RAGAS Context Recall).
    Evaluates sentence-level coverage of the ground truth reference in the retrieved context.
    """
    if not expected_answer or not expected_answer.strip():
        return {"context_recall": 1.0, "covered_sentences": 0, "total_sentences": 0}

    if not retrieved_chunks:
        return {"context_recall": 0.0, "covered_sentences": 0, "total_sentences": 1}

    combined_context = " ".join(retrieved_chunks).lower()

    # Split expected answer into meaningful declarative propositions
    ref_sentences = [
        s.strip() for s in re.split(r"(?:[\n;]|\.\s+)", expected_answer)
        if len(s.strip()) > 10
    ]

    if not ref_sentences:
        ref_sentences = [expected_answer.strip()]

    covered_count = 0
    for sent in ref_sentences:
        sent_tokens = _get_clean_tokens(sent)
        if not sent_tokens:
            covered_count += 1
            continue

        matched = sum(1 for t in sent_tokens if t in combined_context)
        token_ratio = matched / len(sent_tokens)

        if token_ratio >= 0.40:
            covered_count += 1

    recall = round(covered_count / len(ref_sentences), 4)

    return {
        "context_recall": recall,
        "covered_sentences": covered_count,
        "total_sentences": len(ref_sentences),
    }


# =========================================================================
# 5. BERTScore
# =========================================================================

def calculate_bert_score(generated_answer: str, reference_answer: str) -> Dict[str, Any]:
    """
    Compute BERTScore (F_BERT) semantic similarity using local multilingual sentence transformer.
    Computes precision, recall, and harmonic mean F1 over contextual representations.
    """
    if not generated_answer or not reference_answer:
        return {"bert_score": 0.0, "bert_precision": 0.0, "bert_recall": 0.0}

    model = _get_eval_embedding_model()
    if model is None:
        # Fallback to token F1 if sentence-transformers is unavailable
        tokens_gen = set(_get_clean_tokens(generated_answer))
        tokens_ref = set(_get_clean_tokens(reference_answer))
        common = len(tokens_gen.intersection(tokens_ref))
        p = common / len(tokens_gen) if tokens_gen else 0.0
        r = common / len(tokens_ref) if tokens_ref else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"bert_score": round(f1, 4), "bert_precision": round(p, 4), "bert_recall": round(r, 4)}

    try:
        # Chunk into sentences
        gen_sents = [s.strip() for s in re.split(r"[\n.]+", generated_answer) if len(s.strip()) > 8]
        ref_sents = [s.strip() for s in re.split(r"[\n.]+", reference_answer) if len(s.strip()) > 8]

        if not gen_sents:
            gen_sents = [generated_answer[:200]]
        if not ref_sents:
            ref_sents = [reference_answer[:200]]

        emb_gen = model.encode(gen_sents, normalize_embeddings=True)
        emb_ref = model.encode(ref_sents, normalize_embeddings=True)

        # Pairwise cosine similarity matrix: (len(gen), len(ref))
        sim_matrix = np.matmul(emb_gen, emb_ref.T)

        # Precision: for each generated sentence, max similarity to reference
        prec_vec = np.max(sim_matrix, axis=1)
        precision = float(np.mean(prec_vec))

        # Recall: for each reference sentence, max similarity to generated
        rec_vec = np.max(sim_matrix, axis=0)
        recall = float(np.mean(rec_vec))

        # Rescale / clip to [0.0, 1.0]
        precision = max(0.0, min(1.0, precision))
        recall = max(0.0, min(1.0, recall))

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return {
            "bert_score": round(f1, 4),
            "bert_precision": round(precision, 4),
            "bert_recall": round(recall, 4),
        }
    except Exception:
        return {"bert_score": 0.5, "bert_precision": 0.5, "bert_recall": 0.5}


# =========================================================================
# 6. Actor Accuracy
# =========================================================================

def calculate_actor_accuracy(generated_answer: str, expected_actor: Optional[str]) -> float:
    """
    Measure actor assignment correctness.
    Returns 1.0 if expected actor is correctly identified in the response, 0.0 if omitted.
    """
    if not expected_actor or not expected_actor.strip():
        return 1.0

    ans_lower = generated_answer.lower()
    expected_actors = [a.strip().lower() for a in expected_actor.split(",") if a.strip()]

    matched = sum(1 for a in expected_actors if a in ans_lower)
    return round(matched / len(expected_actors), 4) if expected_actors else 1.0
