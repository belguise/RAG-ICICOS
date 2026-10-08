"""
EXPERIMENTAL / ARCHIVED MODULE: Process Consistency Score (PCS).
Moved to `evaluation.archive.process_consistency_experimental`.

This file is maintained solely for backward compatibility with existing components
and is excluded from primary paper benchmarking.
"""

from evaluation.archive.process_consistency_experimental import (
    calculate_process_consistency_score,
    _levenshtein_distance,
    _match_step_in_text,
    _tokenize_indonesian,
)

__all__ = [
    "calculate_process_consistency_score",
]
