"""
BPMN Process-Specific Evaluation Metrics for Enterprise Question Answering.

Provides procedural grounding metrics:
1. Path Accuracy (Process Path Accuracy - PPA):
   Evaluates coverage and sequential order preservation of business process activities.
2. Gateway Accuracy (Gateway Decision Accuracy - GDA):
   Evaluates conditional branching detection, decision path correctness, and terminal consequences.
3. Actor Accuracy (Actor Attribution Accuracy - AAA):
   Evaluates responsible organizational actor identification and swimlane attribution.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


def _clean_token_list(text: str) -> List[str]:
    """Tokenize and normalize text, omitting common Indonesian stop words."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    stop_words = {
        "yang", "dan", "atau", "dari", "pada", "untuk", "ke", "di", "oleh", "dengan",
        "ini", "itu", "adalah", "sebagai", "dalam", "bisa", "dapat", "harus", "akan",
        "jika", "apabila", "bila", "maka", "tersebut", "secara", "sudah", "telah"
    }
    return [w for w in clean.split() if len(w) > 2 and w not in stop_words]


def _match_step_in_answer(step_name: str, answer_lower: str) -> Tuple[bool, int]:
    """
    Check if a BPMN step is present in the answer text.
    Returns (is_matched, earliest_char_position).
    """
    # Remove prefix like 'Gateway: ' for matching
    clean_step = re.sub(r"^gateway:\s*", "", step_name, flags=re.IGNORECASE).strip()
    clean_step_lower = clean_step.lower()

    # 1. Direct exact substring match
    pos = answer_lower.find(clean_step_lower)
    if pos != -1:
        return True, pos

    # 2. Token overlap match (>= 50% content tokens)
    tokens = _clean_token_list(clean_step)
    if not tokens:
        return False, -1

    matched = [t for t in tokens if t in answer_lower]
    overlap = len(matched) / len(tokens)

    if overlap >= 0.5:
        earliest_pos = min(answer_lower.find(t) for t in matched if answer_lower.find(t) != -1)
        return True, earliest_pos

    return False, -1


# =========================================================================
# 1. Process Path Accuracy (PPA)
# =========================================================================

def calculate_path_accuracy(
    predicted_answer: str,
    expected_bpmn_path: List[str],
) -> Dict[str, Any]:
    """
    Compute Process Path Accuracy (PPA).
    Measures step coverage and sequence preservation of the BPMN execution path.

    Formula:
        Coverage = |Matched Steps| / |Expected Path Steps|
        Order Preservation = (Concordant step pairs) / (Total matched pairs)
        Path Accuracy = 0.6 * Coverage + 0.4 * Order Preservation
    """
    if not expected_bpmn_path or not predicted_answer or not predicted_answer.strip():
        return {
            "path_accuracy": 0.0,
            "coverage": 0.0,
            "order_preservation": 0.0,
            "matched_steps": [],
            "missing_steps": list(expected_bpmn_path) if expected_bpmn_path else [],
            "total_expected_steps": len(expected_bpmn_path) if expected_bpmn_path else 0,
            "total_matched_steps": 0,
        }

    ans_lower = predicted_answer.lower()
    matched_steps = []
    missing_steps = []
    matched_positions = []

    for step in expected_bpmn_path:
        is_matched, pos = _match_step_in_answer(step, ans_lower)
        if is_matched:
            matched_steps.append(step)
            matched_positions.append((pos, step))
        else:
            missing_steps.append(step)

    n_expected = len(expected_bpmn_path)
    n_matched = len(matched_steps)
    coverage = n_matched / n_expected if n_expected > 0 else 0.0

    # Order preservation calculation
    if n_matched <= 1:
        order_preservation = 1.0 if n_matched == 1 else 0.0
    else:
        # Expected step order indices
        step_idx_map = {step: idx for idx, step in enumerate(expected_bpmn_path)}
        # Sequence of indices sorted by appearance order in answer
        sorted_by_pos = sorted(matched_positions, key=lambda x: x[0])
        observed_order = [step_idx_map[step] for _, step in sorted_by_pos]

        # Count concordant pairs (i < j implies observed_order[i] < observed_order[j])
        total_pairs = 0
        concordant_pairs = 0
        for i in range(len(observed_order)):
            for j in range(i + 1, len(observed_order)):
                total_pairs += 1
                if observed_order[i] <= observed_order[j]:
                    concordant_pairs += 1

        order_preservation = concordant_pairs / total_pairs if total_pairs > 0 else 1.0

    path_accuracy = round(0.6 * coverage + 0.4 * order_preservation, 4)

    return {
        "path_accuracy": path_accuracy,
        "coverage": round(coverage, 4),
        "order_preservation": round(order_preservation, 4),
        "matched_steps": matched_steps,
        "missing_steps": missing_steps,
        "total_expected_steps": n_expected,
        "total_matched_steps": n_matched,
    }


# =========================================================================
# 2. Gateway Decision Accuracy (GDA)
# =========================================================================

def calculate_gateway_accuracy(
    predicted_answer: str,
    question: str,
    expected_answer: str,
    expected_bpmn_path: Optional[List[str]] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute Gateway Decision Accuracy (GDA).
    Evaluates whether conditional branching logic, decision conditions (XOR/OR),
    and branching outcomes are correctly identified and resolved.

    Returns:
        dict containing:
          - gateway_accuracy: float in [0.0, 1.0]
          - condition_identified: bool
          - outcome_correct: bool
          - detected_gateway: Optional[str]
    """
    if not predicted_answer or not predicted_answer.strip():
        return {
            "gateway_accuracy": 0.0,
            "condition_identified": False,
            "outcome_correct": False,
            "detected_gateway": None,
        }

    ans_lower = predicted_answer.lower()
    exp_lower = expected_answer.lower() if expected_answer else ""
    q_lower = question.lower() if question else ""

    # 1. Identify gateway elements from expected_bpmn_path if present
    gateway_steps = []
    if expected_bpmn_path:
        for s in expected_bpmn_path:
            if s.lower().startswith("gateway:") or "?" in s:
                gateway_steps.append(s)

    detected_gateway = gateway_steps[0] if gateway_steps else None

    # 2. Common decision conditions in ICICoS workflows:
    # - "sesuai" vs "tidak sesuai" / "tidak lengkap" / "revisi"
    # - "fix cost" vs "open va"
    # - "diterima" vs "ditolak"
    # - "lolos verifikasi" vs "belum lolos"
    condition_keywords = [
        ("tidak sesuai", ["tidak sesuai", "salah", "tidak cocok", "tidak valid"]),
        ("sesuai", ["sesuai", "cocok", "benar", "valid", "lengkap"]),
        ("fix cost", ["fix cost", "nominal pasti", "tetap"]),
        ("open va", ["open va", "nominal fleksibel", "terbuka"]),
        ("ditolak", ["ditolak", "reject", "revisi"]),
        ("diterima", ["diterima", "accept", "lolos"]),
    ]

    target_condition = None
    for cond_name, variants in condition_keywords:
        if any(v in q_lower for v in variants) or any(v in exp_lower for v in variants):
            target_condition = cond_name
            break

    # 3. Check condition identification in predicted answer
    condition_identified = False
    if target_condition:
        cond_variants = next(v for c, v in condition_keywords if c == target_condition)
        condition_identified = any(v in ans_lower for v in cond_variants)
    else:
        # General conditional terms check: "jika", "apabila", "kondisi", "gateway"
        condition_identified = any(k in ans_lower for k in ["jika", "apabila", "bila", "kondisi", "dalam hal"])

    # 4. Check outcome correctness against reference answer
    exp_tokens = _clean_token_list(expected_answer)
    matched_exp_tokens = [t for t in exp_tokens if t in ans_lower]
    outcome_overlap = len(matched_exp_tokens) / len(exp_tokens) if exp_tokens else 0.0
    outcome_correct = outcome_overlap >= 0.40

    # 5. Composite Gateway Accuracy:
    # 0.5 for condition detection + 0.5 for outcome correctness
    score = 0.0
    if condition_identified and outcome_correct:
        score = 1.0
    elif condition_identified:
        score = 0.5 + 0.5 * outcome_overlap
    elif outcome_correct:
        score = 0.4 + 0.4 * outcome_overlap
    else:
        score = 0.2 * outcome_overlap

    return {
        "gateway_accuracy": round(score, 4),
        "condition_identified": condition_identified,
        "outcome_correct": outcome_correct,
        "outcome_overlap": round(outcome_overlap, 4),
        "detected_gateway": detected_gateway,
    }


# =========================================================================
# 3. Actor Attribution Accuracy (AAA)
# =========================================================================

def calculate_actor_accuracy(
    predicted_answer: str,
    expected_actor: Optional[str],
) -> Dict[str, Any]:
    """
    Compute Actor Attribution Accuracy (AAA).
    Evaluates whether the organizational role/actor responsible for a process step
    is accurately attributed in the generated answer.

    Returns:
        dict containing:
          - actor_accuracy: float in [0.0, 1.0]
          - matched_actors: List[str]
          - missing_actors: List[str]
          - total_expected_actors: int
    """
    if not expected_actor or expected_actor.strip() in ["", "None", "Semua pihak"]:
        # If no specific actor is required, default to full credit
        return {
            "actor_accuracy": 1.0,
            "matched_actors": [],
            "missing_actors": [],
            "total_expected_actors": 0,
        }

    if not predicted_answer or not predicted_answer.strip():
        return {
            "actor_accuracy": 0.0,
            "matched_actors": [],
            "missing_actors": [expected_actor],
            "total_expected_actors": 1,
        }

    ans_lower = predicted_answer.lower()

    # Split composite actors like 'Keuangan Fakultas & Bendahara' or 'Author, Reviewer, Panitia'
    raw_actors = re.split(r"[,&/]\s*|\s+dan\s+", expected_actor)
    actors = [a.strip() for a in raw_actors if len(a.strip()) > 2]

    if not actors:
        actors = [expected_actor.strip()]

    matched = []
    missing = []

    # Common role abbreviations and synonyms in Indonesian enterprise context
    role_synonyms = {
        "bendahara": ["bendahara", "keuangan panitia", "treasurer"],
        "keuangan fakultas": ["keuangan fakultas", "bagian keuangan fsm", "keuangan fsm", "fakultas"],
        "keuangan universitas": ["keuangan universitas", "buku universitas", "rektorat"],
        "author": ["author", "penulis", "pemakalah", "peserta"],
        "reviewer": ["reviewer", "penelaah", "mitra bestari"],
        "panitia": ["panitia", "organizing committee", "oc"],
        "sekretaris": ["sekretaris", "sekretariat"],
        "pimpinan fakultas": ["pimpinan fakultas", "dekan", "wakil dekan"],
        "ieee": ["ieee", "ieee indonesia"],
    }

    for actor in actors:
        act_lower = actor.lower()
        # Direct match
        if act_lower in ans_lower:
            matched.append(actor)
            continue

        # Synonym match
        found = False
        for key, syns in role_synonyms.items():
            if key in act_lower:
                if any(s in ans_lower for s in syns):
                    matched.append(actor)
                    found = True
                    break
        if not found:
            # Check token overlap
            tokens = [t for t in _clean_token_list(act_lower) if len(t) > 3]
            if tokens and sum(1 for t in tokens if t in ans_lower) >= max(1, len(tokens) // 2):
                matched.append(actor)
            else:
                missing.append(actor)

    score = len(matched) / len(actors) if actors else 1.0

    return {
        "actor_accuracy": round(score, 4),
        "matched_actors": matched,
        "missing_actors": missing,
        "total_expected_actors": len(actors),
    }


# =========================================================================
# Unified BPMN Process Evaluation
# =========================================================================

def evaluate_bpmn_process(
    predicted_answer: str,
    expected_bpmn_path: Optional[List[str]],
    expected_actor: Optional[str],
    question: str = "",
    expected_answer: str = "",
    category: str = "workflow",
) -> Dict[str, Any]:
    """
    Comprehensive BPMN procedural evaluation combining:
    - Path Accuracy (PPA)
    - Gateway Accuracy (GDA)
    - Actor Accuracy (AAA)
    """
    path_res = calculate_path_accuracy(predicted_answer, expected_bpmn_path or [])
    gateway_res = calculate_gateway_accuracy(
        predicted_answer=predicted_answer,
        question=question,
        expected_answer=expected_answer,
        expected_bpmn_path=expected_bpmn_path,
        category=category,
    )
    actor_res = calculate_actor_accuracy(predicted_answer, expected_actor)

    return {
        "path_accuracy": path_res["path_accuracy"],
        "path_coverage": path_res["coverage"],
        "order_preservation": path_res["order_preservation"],
        "gateway_accuracy": gateway_res["gateway_accuracy"],
        "gateway_condition_identified": gateway_res["condition_identified"],
        "gateway_outcome_correct": gateway_res["outcome_correct"],
        "actor_accuracy": actor_res["actor_accuracy"],
        "matched_actors": actor_res["matched_actors"],
        "missing_actors": actor_res["missing_actors"],
    }
