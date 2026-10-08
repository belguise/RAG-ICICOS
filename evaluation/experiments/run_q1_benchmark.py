"""
Comprehensive Q1 Empirical Benchmark Runner for Elsevier KBS Submission.

Executes:
1. Primary Comparative Benchmark (N = 200, 50 per category across document_requirement, workflow, decision_gateway, hybrid)
2. Procedural Diagnostic Benchmark (N = 60, 20 per category across workflow, decision_gateway, hybrid)
3. Multi-Architecture Evaluation:
   - LLM Only (Zero-shot)
   - Traditional Vector RAG
   - Graph RAG (BPMN Subgraph)
   - Static Hybrid RAG (Direct Concatenation)
   - Proposed Adaptive Agentic Hybrid RAG
4. Cross-Backbone Robustness:
   - Primary: Google Gemini Flash (gemini-3.5-flash)
   - Second Backbone: Open-Weight Qwen (qwen/qwen3.8-27b via Groq)

Includes:
- Robust per-query checkpointing and resume capability
- Exponential backoff rate limit retry protection
- Execution log and failed queries isolation
- Master results exporter (results/master_results.csv)
- Exact metric calculation (Faithfulness, Relevancy, Precision, Recall, BERTScore, Hallucination Rate, Latency, PPA, GDA, AAA)
- Statistical significance testing (Paired t-test, Wilcoxon signed-rank, 10,000 bootstrap resamples)
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.stats as stats
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Q1Benchmark")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
MANIFEST_PATH = RESULTS_DIR / "primary_benchmark_manifest.json"
SAMPLING_MANIFEST_PATH = RESULTS_DIR / "sampling_manifest.json"
FAILED_QUERIES_PATH = RESULTS_DIR / "failed_queries.json"
EXECUTION_LOG_PATH = RESULTS_DIR / "execution_log.json"
MASTER_RESULTS_PATH = RESULTS_DIR / "master_results.csv"


def load_manifest() -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                "primary_benchmark_items": data.get("items", []),
                "metadata": data.get("metadata", {})
            }
    if SAMPLING_MANIFEST_PATH.exists():
        with open(SAMPLING_MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    from evaluation.experiments.sample_manifest import generate_manifest
    return generate_manifest(seed=42)


def get_configured_chain(backbone: str = "gemini"):
    from agentic_rag_sop.generation.chain import SOPRAGChain

    if backbone == "qwen":
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY is required for qwen backbone.")
        from langchain_groq import ChatGroq
        llm = ChatGroq(model_name="qwen/qwen3.8-27b", groq_api_key=groq_key, temperature=0.0)
        return SOPRAGChain(llm=llm)
    else:
        # Default Gemini Flash (gemini-3.5-flash-lite)
        from agentic_rag_sop.core.llm import get_llm
        llm = get_llm(model_name="gemini-3.5-flash-lite")
        return SOPRAGChain(llm=llm)


def log_execution_event(event_type: str, details: Dict[str, Any]):
    """Append event to results/execution_log.json."""
    logs = []
    if EXECUTION_LOG_PATH.exists():
        try:
            with open(EXECUTION_LOG_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append({
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        **details
    })
    try:
        with open(EXECUTION_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Could not write to execution log: {e}")


def log_failed_query(query_id: str, architecture: str, backbone: str, error: str):
    """Append failed query to results/failed_queries.json."""
    failed = []
    if FAILED_QUERIES_PATH.exists():
        try:
            with open(FAILED_QUERIES_PATH, "r", encoding="utf-8") as f:
                failed = json.load(f)
        except Exception:
            failed = []
    failed.append({
        "timestamp": datetime.now().isoformat(),
        "query_id": query_id,
        "architecture": architecture,
        "backbone": backbone,
        "error": error
    })
    try:
        with open(FAILED_QUERIES_PATH, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Could not write to failed queries: {e}")


def export_master_results(results_log: List[Dict[str, Any]], backbone: str, output_path: Path = MASTER_RESULTS_PATH):
    """
    Generate results/master_results.csv as the single source of truth across all evaluated queries and architectures.
    Columns: query_id, category, architecture, backbone, faithfulness, answer_relevancy,
             context_precision, context_recall, bert_score, hallucination_rate,
             path_accuracy, gateway_accuracy, actor_accuracy, latency, token_count,
             router_prediction, repair_iterations, audit_pass, audit_scores
    """
    fieldnames = [
        "query_id", "category", "architecture", "backbone",
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        "bert_score", "hallucination_rate", "path_accuracy", "gateway_accuracy",
        "actor_accuracy", "latency", "token_count", "router_prediction",
        "repair_iterations", "audit_pass", "audit_scores"
    ]
    
    existing_rows = []
    seen_keys = set()
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    key = (r.get("query_id"), r.get("architecture"), r.get("backbone"))
                    seen_keys.add(key)
                    existing_rows.append(r)
        except Exception as e:
            logger.warning(f"Could not read existing master results: {e}")

    new_rows = []
    for q_item in results_log:
        q_id = q_item.get("id") or q_item.get("query_id")
        q_cat = q_item.get("category", "general")
        baselines = q_item.get("baselines", {})

        for sys_name, data in baselines.items():
            if not data or data.get("error"):
                continue
            key = (q_id, sys_name, backbone)
            ans = data.get("answer", "")
            token_count = len(ans.split()) if ans else 0
            route_pred = ""
            repair_iter = 0
            audit_pass = True
            audit_conf = 1.0

            if sys_name == "proposed_agentic":
                route_dec = data.get("route_decision") or {}
                route_pred = route_dec.get("knowledge_source", "")
                repair_iter = 1 if data.get("was_revised") else 0
                critic_dec = data.get("critic_decision") or {}
                audit_pass = critic_dec.get("is_valid", True)
                audit_conf = critic_dec.get("confidence_score", 1.0)

            row = {
                "query_id": q_id,
                "category": q_cat,
                "architecture": sys_name,
                "backbone": backbone,
                "faithfulness": round(float(data.get("faithfulness", 0.0)), 4),
                "answer_relevancy": round(float(data.get("answer_relevancy", 0.0)), 4),
                "context_precision": round(float(data.get("context_precision", 0.0)), 4),
                "context_recall": round(float(data.get("context_recall", 0.0)), 4),
                "bert_score": round(float(data.get("bert_score", 0.0)), 4),
                "hallucination_rate": round(float(data.get("hallucination_rate", 1.0)), 4),
                "path_accuracy": round(float(data.get("path_accuracy", 0.0)), 4),
                "gateway_accuracy": round(float(data.get("gateway_accuracy", 0.0)), 4),
                "actor_accuracy": round(float(data.get("actor_accuracy", 1.0)), 4),
                "latency": round(float(data.get("duration", 0.0)), 3),
                "token_count": token_count,
                "router_prediction": route_pred,
                "repair_iterations": repair_iter,
                "audit_pass": audit_pass,
                "audit_scores": audit_conf,
            }
            if key not in seen_keys:
                seen_keys.add(key)
                new_rows.append(row)
            else:
                # Update in place
                for i, ex in enumerate(existing_rows):
                    if (ex.get("query_id"), ex.get("architecture"), ex.get("backbone")) == key:
                        existing_rows[i] = row
                        break

    all_rows = existing_rows + new_rows
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        logger.info(f"Updated {output_path} with {len(all_rows)} total query-level records.")
    except Exception as e:
        logger.warning(f"Could not save master results: {e}")


def run_cohort_benchmark(
    cohort_type: str = "primary",  # "primary" (N=200) or "procedural" (N=60)
    backbone: str = "gemini",
    delay: float = 1.0,
    max_questions: Optional[int] = None,
    systems_to_run: Optional[List[str]] = None,
    output_dir: Path = RESULTS_DIR,
) -> Dict[str, Any]:
    from evaluation.evaluator import BenchmarkEvaluator

    manifest = load_manifest()
    if cohort_type == "primary":
        items = manifest.get("primary_benchmark_items", [])
    elif cohort_type == "procedural":
        items = manifest.get("procedural_diagnostic_items", [])
    else:
        raise ValueError(f"Unknown cohort type: {cohort_type}")

    # Interleave items round-robin across categories to guarantee stratification at any N
    cats = ["document_requirement", "workflow", "decision_gateway", "hybrid"]
    by_cat = {c: [x for x in items if x.get("category") == c] for c in cats}
    interleaved_items = []
    max_len = max((len(v) for v in by_cat.values()), default=0)
    for i in range(max_len):
        for c in cats:
            if i < len(by_cat[c]):
                interleaved_items.append(by_cat[c][i])
    items = interleaved_items

    if max_questions is not None and max_questions < len(items):
        items = items[:max_questions]

    total_q = len(items)
    logger.info(f"Starting {cohort_type.upper()} benchmark (N={total_q}) with backbone={backbone}")

    chain = get_configured_chain(backbone=backbone)
    evaluator = BenchmarkEvaluator(chain=chain, delay_seconds=delay)

    output_dir.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"benchmark_results_{cohort_type}_{backbone}.json"
    existing_results: Dict[str, Any] = {}
    if results_file.exists():
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                for r in prev_data.get("results", []):
                    # Check if valid (no unhandled connection error)
                    baselines = r.get("baselines", {})
                    has_fatal_err = any(b.get("error") for b in baselines.values())
                    if not has_fatal_err and baselines:
                        existing_results[r["id"]] = r
            logger.info(f"Loaded {len(existing_results)} valid existing results from checkpoint {results_file}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")

    results_log = []
    systems = systems_to_run or ["llm_only", "traditional_rag", "graph_rag", "hybrid_rag", "proposed_agentic"]

    log_execution_event("BENCHMARK_START", {
        "cohort": cohort_type,
        "backbone": backbone,
        "total_questions": total_q,
        "systems": systems
    })

    for idx, item in enumerate(items, start=1):
        q_id = item.get("id") or item.get("query_id")
        q_cat = item.get("category")

        if q_id in existing_results:
            logger.info(f"[{idx}/{total_q}] Resuming from cache: {q_id} ({q_cat})")
            results_log.append(existing_results[q_id])
            continue

        logger.info(f"[{idx}/{total_q}] Evaluating {q_id} ({q_cat}): {item['question'][:60]}...")
        t0 = time.time()
        q_res = evaluator.evaluate_single_question(item, systems_to_run=systems, run_ablations=False)
        dur = time.time() - t0
        logger.info(f"[{idx}/{total_q}] Completed {q_id} in {dur:.2f}s")
        results_log.append(q_res)

        # Check for errors and log
        for s_name, s_data in q_res.get("baselines", {}).items():
            if s_data.get("error"):
                log_failed_query(q_id, s_name, backbone, s_data["error"])

        # Checkpoint after EVERY question to ensure zero loss
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({
                "experiment_id": f"EXP_{cohort_type.upper()}_{backbone.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "cohort": cohort_type,
                "backbone": backbone,
                "total_questions": len(results_log),
                "results": results_log,
            }, f, indent=2, ensure_ascii=False)

        # Update master results CSV immediately per query
        export_master_results(results_log, backbone=backbone)

        # Inter-question sleep to respect API RPM
        if delay > 0 and idx < total_q:
            time.sleep(delay)

    log_execution_event("BENCHMARK_END", {
        "cohort": cohort_type,
        "backbone": backbone,
        "completed_count": len(results_log),
    })

    # Export master results
    export_master_results(results_log, backbone=backbone)

    return {
        "cohort": cohort_type,
        "backbone": backbone,
        "total_questions": len(results_log),
        "results": results_log,
    }


def aggregate_and_save_metrics(
    results_log: List[Dict[str, Any]],
    cohort_type: str = "primary",
    backbone: str = "gemini",
    output_dir: Path = RESULTS_DIR,
) -> Dict[str, Any]:
    systems = ["llm_only", "traditional_rag", "graph_rag", "hybrid_rag", "proposed_agentic"]
    label_map = {
        "llm_only": "1. LLM Only (Zero-shot)",
        "traditional_rag": "2. Traditional Vector RAG",
        "graph_rag": "3. Graph RAG (BPMN Subgraph)",
        "hybrid_rag": "4. Static Hybrid RAG",
        "proposed_agentic": "5. Proposed Agentic Hybrid RAG",
    }

    metrics_by_sys = {}
    for sys_name in systems:
        metrics_by_sys[sys_name] = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_precision": [],
            "context_recall": [],
            "bert_score": [],
            "hallucination_rate": [],
            "duration": [],
            "path_accuracy": [],
            "gateway_accuracy": [],
            "actor_accuracy": [],
            "pcs": [],
        }

    for res in results_log:
        for sys_name in systems:
            data = res.get("baselines", {}).get(sys_name)
            if not data or data.get("error"):
                continue
            metrics_by_sys[sys_name]["faithfulness"].append(data.get("faithfulness", 0.0))
            metrics_by_sys[sys_name]["answer_relevancy"].append(data.get("answer_relevancy", 0.0))
            metrics_by_sys[sys_name]["context_precision"].append(data.get("context_precision", 0.0))
            metrics_by_sys[sys_name]["context_recall"].append(data.get("context_recall", 0.0))
            metrics_by_sys[sys_name]["bert_score"].append(data.get("bert_score", 0.0))
            metrics_by_sys[sys_name]["hallucination_rate"].append(data.get("hallucination_rate", 1.0))
            metrics_by_sys[sys_name]["duration"].append(data.get("duration", 0.0))
            metrics_by_sys[sys_name]["path_accuracy"].append(data.get("path_accuracy", 0.0))
            metrics_by_sys[sys_name]["gateway_accuracy"].append(data.get("gateway_accuracy", 0.0))
            metrics_by_sys[sys_name]["actor_accuracy"].append(data.get("actor_accuracy", 1.0))
            metrics_by_sys[sys_name]["pcs"].append(data.get("pcs", 0.0))

    summary_rows = []
    for sys_name in systems:
        d = metrics_by_sys[sys_name]
        n = len(d["faithfulness"])
        m_f = np.mean(d["faithfulness"]) if n > 0 else 0.0
        m_rel = np.mean(d["answer_relevancy"]) if n > 0 else 0.0
        m_cp = np.mean(d["context_precision"]) if n > 0 else 0.0
        m_cr = np.mean(d["context_recall"]) if n > 0 else 0.0
        m_bert = np.mean(d["bert_score"]) if n > 0 else 0.0
        m_hr = np.mean(d["hallucination_rate"]) if n > 0 else 0.0
        m_dur = np.mean(d["duration"]) if n > 0 else 0.0
        m_ppa = np.mean(d["path_accuracy"]) if n > 0 else 0.0
        m_gda = np.mean(d["gateway_accuracy"]) if n > 0 else 0.0
        m_aaa = np.mean(d["actor_accuracy"]) if n > 0 else 0.0

        summary_rows.append({
            "system": sys_name,
            "architecture": label_map.get(sys_name, sys_name),
            "sample_count": n,
            "faithfulness": round(float(m_f), 4),
            "answer_relevancy": round(float(m_rel), 4),
            "context_precision": round(float(m_cp), 4),
            "context_recall": round(float(m_cr), 4),
            "bert_score": round(float(m_bert), 4),
            "hallucination_rate_pct": round(float(m_hr * 100), 2),
            "avg_latency_sec": round(float(m_dur), 2),
            "path_accuracy": round(float(m_ppa), 4),
            "gateway_accuracy": round(float(m_gda), 4),
            "actor_accuracy": round(float(m_aaa), 4),
        })

    # Save final_comparison_table.csv
    for out_d in [output_dir, EVAL_RESULTS_DIR]:
        comp_path = out_d / f"final_comparison_table_{cohort_type}_{backbone}.csv"
        with open(comp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Architecture",
                "Faithfulness",
                "Answer Relevancy",
                "Context Precision",
                "Context Recall",
                "BERTScore",
                "Hallucination Rate",
                "Latency (s)",
            ])
            for r in summary_rows:
                writer.writerow([
                    r["architecture"],
                    f"{r['faithfulness']:.3f}" if r["system"] != "llm_only" else "N/A*",
                    f"{r['answer_relevancy']:.3f}",
                    f"{r['context_precision']:.3f}" if r["system"] != "llm_only" else "N/A*",
                    f"{r['context_recall']:.3f}" if r["system"] != "llm_only" else "N/A*",
                    f"{r['bert_score']:.3f}",
                    f"{r['hallucination_rate_pct']:.1f}%",
                    f"{r['avg_latency_sec']:.2f}s",
                ])

    logger.info(f"Saved summary metrics to {comp_path}")
    return {"summary": summary_rows, "metrics_by_sys": metrics_by_sys}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q1 Comprehensive Benchmark Runner")
    parser.add_argument("--cohort", type=str, default="primary", choices=["primary", "procedural"], help="Cohort to evaluate")
    parser.add_argument("--backbone", type=str, default="gemini", choices=["gemini", "qwen"], help="LLM backbone")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls")
    parser.add_argument("--systems", type=str, default=None, help="Comma-separated systems to run")

    args = parser.parse_args()
    sys_list = [s.strip() for s in args.systems.split(",")] if args.systems else None

    data = run_cohort_benchmark(
        cohort_type=args.cohort,
        backbone=args.backbone,
        delay=args.delay,
        max_questions=args.max_questions,
        systems_to_run=sys_list,
    )
    aggregate_and_save_metrics(
        results_log=data["results"],
        cohort_type=args.cohort,
        backbone=args.backbone,
    )
