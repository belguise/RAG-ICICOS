"""
Standardized RAG Literature Evaluation Metrics Package.

Established metrics for Q1/Q2 journal benchmark:
1. Retrieval Metrics:
   - Context Precision
   - Context Recall
2. Generation Metrics:
   - Faithfulness (RAGAS)
   - Answer Relevancy (TruLens / RAGAS)
3. Semantic Similarity:
   - BERTScore (Zhang et al.)
4. Reliability & Error Metrics:
   - Hallucination Rate (via Evidence Grounding)
   - Actor Role Accuracy
   - Router Accuracy

NOTE: Process Consistency Score (PCS) is archived in `evaluation.archive`
and excluded from primary standard literature metrics.
"""

from evaluation.metrics.rag_metrics import calculate_answer_accuracy, calculate_evidence_grounding
from evaluation.metrics.router_accuracy import calculate_router_metrics
from evaluation.metrics.standard_rag_metrics import (
    calculate_faithfulness,
    calculate_answer_relevancy,
    calculate_context_precision,
    calculate_context_recall,
    calculate_bert_score,
    calculate_actor_accuracy,
)

__all__ = [
    # Primary Standard Literature Metrics
    "calculate_context_precision",
    "calculate_context_recall",
    "calculate_faithfulness",
    "calculate_answer_relevancy",
    "calculate_bert_score",
    # Reliability & Auxiliary Validation
    "calculate_evidence_grounding",
    "calculate_answer_accuracy",
    "calculate_actor_accuracy",
    "calculate_router_metrics",
]
