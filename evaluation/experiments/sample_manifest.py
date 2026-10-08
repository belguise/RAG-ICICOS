"""
Reproducible Sampling Manifest Generator.
Generates:
1. Primary Comparative Benchmark Cohort: N = 200 (50 per category: document_requirement, workflow, decision_gateway, hybrid)
2. Procedural Diagnostic Benchmark Cohort: N = 60 (20 per category: workflow, decision_gateway, hybrid)
   - Mathematically verified to be completely disjoint from the N = 200 cohort.

All selections are reproducible via a fixed random seed (seed = 42).
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
import random
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SampleManifest")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "evaluation" / "dataset" / "icicos_process_qa.json"
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def generate_manifest(seed: int = 42) -> Dict[str, Any]:
    logger.info(f"Loading full benchmark dataset from {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)

    total_corpus_size = len(dataset)
    logger.info(f"Full corpus size: {total_corpus_size} items")

    # Group by category
    by_category = defaultdict(list)
    for item in dataset:
        by_category[item["category"]].append(item)

    corpus_distribution = {cat: len(items) for cat, items in by_category.items()}
    logger.info(f"Corpus category distribution: {corpus_distribution}")

    # Deterministic shuffle within each category
    rng = random.Random(seed)
    shuffled_by_category = {}
    for cat in sorted(by_category.keys()):
        items = list(by_category[cat])
        # Sort first by ID to guarantee identical initial state across platforms
        items.sort(key=lambda x: x["id"])
        rng.shuffle(items)
        shuffled_by_category[cat] = items

    # 1. Primary Comparative Benchmark Cohort: N = 200 (50 per category)
    primary_categories = ["document_requirement", "workflow", "decision_gateway", "hybrid"]
    per_cat_primary = 50
    cohort_200 = []
    cohort_200_ids = []

    for cat in primary_categories:
        available = shuffled_by_category[cat]
        if len(available) < per_cat_primary:
            raise ValueError(f"Category {cat} has only {len(available)} items, less than required {per_cat_primary}")
        selected = available[:per_cat_primary]
        cohort_200.extend(selected)
        cohort_200_ids.extend([item["id"] for item in selected])

    # 2. Procedural Diagnostic Benchmark Cohort: N = 60 (20 per category across procedural categories)
    procedural_categories = ["workflow", "decision_gateway", "hybrid"]
    per_cat_procedural = 20
    cohort_60 = []
    cohort_60_ids = []

    for cat in procedural_categories:
        available = shuffled_by_category[cat]
        remaining = available[per_cat_primary:]
        if len(remaining) < per_cat_procedural:
            raise ValueError(f"Category {cat} has only {len(remaining)} remaining items, less than required {per_cat_procedural}")
        selected = remaining[:per_cat_procedural]
        cohort_60.extend(selected)
        cohort_60_ids.extend([item["id"] for item in selected])

    # Verification: check disjointness
    set_200 = set(cohort_200_ids)
    set_60 = set(cohort_60_ids)
    overlap = set_200.intersection(set_60)
    assert len(overlap) == 0, f"Disjointness invariant violated! Overlap detected: {overlap}"
    assert len(cohort_200) == 200, f"Primary cohort size mismatch: expected 200, got {len(cohort_200)}"
    assert len(cohort_60) == 60, f"Procedural diagnostic cohort size mismatch: expected 60, got {len(cohort_60)}"

    manifest = {
        "metadata": {
            "seed": seed,
            "total_corpus_size": total_corpus_size,
            "corpus_distribution": corpus_distribution,
            "primary_benchmark_n": len(cohort_200),
            "primary_benchmark_distribution": dict(Counter(item["category"] for item in cohort_200)),
            "procedural_diagnostic_n": len(cohort_60),
            "procedural_diagnostic_distribution": dict(Counter(item["category"] for item in cohort_60)),
            "overlap_count": len(overlap),
            "disjoint_verified": True,
            "total_sampled_unique": len(set_200.union(set_60)),
        },
        "primary_benchmark_ids": cohort_200_ids,
        "procedural_diagnostic_ids": cohort_60_ids,
        "primary_benchmark_items": cohort_200,
        "procedural_diagnostic_items": cohort_60,
    }

    # Save to disk
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for out_dir in [RESULTS_DIR, EVAL_RESULTS_DIR]:
        manifest_path = out_dir / "sampling_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved manifest to {manifest_path}")

    # Generate Markdown documentation report
    report_md = f"""# Benchmark Sampling Manifest & Reproducibility Audit

**Random Seed**: `{seed}`  
**Source Corpus**: `evaluation/dataset/icicos_process_qa.json` ($N = {total_corpus_size}$)  
**Disjoint Invariant Verified**: `{"PASSED" if len(overlap) == 0 else "FAILED"}` (Overlap = {len(overlap)})

---

## 1. Corpus vs. Cohort Allocation

| Category | Full Corpus ($N=338$) | Primary Benchmark ($N=200$) | Procedural Diagnostic ($N=60$) | Remaining Reserved |
| :--- | :---: | :---: | :---: | :---: |
| `document_requirement` | {corpus_distribution.get("document_requirement", 0)} | 50 | 0 | {corpus_distribution.get("document_requirement", 0) - 50} |
| `workflow` | {corpus_distribution.get("workflow", 0)} | 50 | 20 | {corpus_distribution.get("workflow", 0) - 70} |
| `decision_gateway` | {corpus_distribution.get("decision_gateway", 0)} | 50 | 20 | {corpus_distribution.get("decision_gateway", 0) - 70} |
| `hybrid` | {corpus_distribution.get("hybrid", 0)} | 50 | 20 | {corpus_distribution.get("hybrid", 0) - 70} |
| **Total** | **{total_corpus_size}** | **200** | **60** | **{total_corpus_size - 260}** |

---

## 2. Mathematical Disjointness Proof
- Primary cohort $S_{{200}} = \\{{q_1, \\dots, q_{{200}}\\}} \\subset \\mathcal{{D}}$
- Procedural diagnostic cohort $S_{{60}} = \\{{q'_{{1}}, \\dots, q'_{{60}}\\}} \\subset \\mathcal{{D}}$
- Strict disjointness: $S_{{200}} \\cap S_{{60}} = \\emptyset$
- Total evaluated queries without sample leakage: $|S_{{200}} \\cup S_{{60}}| = 260$ queries ($76.9\\%$ of corpus).

All query IDs and full items are locked in `sampling_manifest.json`.
"""
    with open(RESULTS_DIR / "sampling_manifest_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    return manifest


if __name__ == "__main__":
    generate_manifest(seed=42)
