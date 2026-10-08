"""
Router Agent Evaluation Metrics.
Evaluates classification fidelity for:
1. Knowledge Source Selection (SOP, BPMN, BOTH).
2. Intent Classification congruency.
"""

from typing import Any, Dict, Optional


def calculate_router_metrics(
    predicted_source: Optional[str],
    expected_source: str,
    predicted_intent: Optional[str] = None,
    expected_category: Optional[str] = None,
    predicted_secondary: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate accuracy and congruency of Router Agent decisions.
    """
    pred_src = (predicted_source or "").strip().upper()
    pred_sec = (predicted_secondary or "").strip().upper()
    exp_src = (expected_source or "").strip().upper()

    # Exact source match or dual-route coverage
    source_match = (pred_src == exp_src)
    if not source_match and exp_src == "BOTH" and pred_sec:
        source_match = True

    # Partial credit / lenient match if hybrid BOTH was dispatched or covered
    lenient_match = (
        source_match
        or (exp_src == "BOTH" and pred_src in ["SOP", "BPMN", "BOTH"])
        or (pred_src == "BOTH" and exp_src in ["SOP", "BPMN"])
    )

    # Category to intent congruency mapping
    intent_congruent = True
    if predicted_intent and expected_category:
        cat_lower = expected_category.lower()
        intent_lower = predicted_intent.lower()
        if "doc" in cat_lower and "doc" in intent_lower:
            intent_congruent = True
        elif "work" in cat_lower and ("work" in intent_lower or "proc" in intent_lower):
            intent_congruent = True
        elif "dec" in cat_lower and ("work" in intent_lower or "proc" in intent_lower):
            intent_congruent = True
        elif "hyb" in cat_lower:
            intent_congruent = True
        else:
            intent_congruent = (cat_lower in intent_lower or intent_lower in cat_lower)

    return {
        "source_correct": source_match,
        "source_lenient_correct": lenient_match,
        "intent_congruent": intent_congruent,
        "predicted_source": pred_src,
        "expected_source": exp_src,
    }
