"""
Research-Grade Agent Execution Trace Logger.
Persists standardized execution traces for every user query to enable:
1. Frontend Explainable Visualization
2. Experiment Reproducibility
3. Publication Methodology Auditing

Conforms strictly to the required schema:
{
  query_id,
  timestamp,
  original_query,
  rewritten_query,
  detected_intent,
  selected_source,
  router_reasoning,
  retrieved_sop_chunks,
  retrieved_bpmn_nodes,
  process_path,
  generated_answer,
  critic_result,
  confidence_score
}
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_default_trace_dir() -> Path:
    curr = Path(__file__).resolve()
    for p in curr.parents:
        if (p / "pyproject.toml").exists() or (p / "data").exists():
            return p / "data" / "traces"
    return curr.parents[3] / "data" / "traces"


class TraceLogger:
    """Manages recording, disk persistence, and indexing of research execution traces."""

    def __init__(self, trace_dir: Optional[Path] = None):
        self.trace_dir = trace_dir or get_default_trace_dir()
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def create_trace_id(self) -> str:
        """Generate human-readable timestamped trace identifier."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        rand = uuid.uuid4().hex[:6]
        return f"trace_{ts}_{rand}"

    def log_query_execution(
        self,
        original_query: str,
        rewritten_query: str,
        detected_intent: str,
        selected_source: str,
        router_reasoning: str,
        retrieved_sop_chunks: List[Dict[str, Any]],
        retrieved_bpmn_nodes: List[Dict[str, Any]],
        process_path: List[str],
        generated_answer: str,
        critic_result: Dict[str, Any],
        confidence_score: float,
        primary_source: Optional[str] = None,
        secondary_source: Optional[str] = None,
        query_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record a complete research execution trace and save to local JSON file.
        """
        trace_id = query_id or self.create_trace_id()
        timestamp = datetime.utcnow().isoformat() + "Z"

        trace_payload: Dict[str, Any] = {
            "query_id": trace_id,
            "timestamp": timestamp,
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "detected_intent": detected_intent,
            "selected_source": selected_source,
            "primary_source": primary_source or selected_source,
            "secondary_source": secondary_source,
            "router_reasoning": router_reasoning,
            "retrieved_sop_chunks": retrieved_sop_chunks,
            "retrieved_bpmn_nodes": retrieved_bpmn_nodes,
            "process_path": process_path,
            "generated_answer": generated_answer,
            "critic_result": critic_result,
            "confidence_score": round(confidence_score, 4),
        }

        if extra_metadata:
            trace_payload["metadata"] = extra_metadata

        # Persist to JSON file
        out_file = self.trace_dir / f"{trace_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(trace_payload, f, indent=2, ensure_ascii=False)

        return trace_payload

    # Alias for convenience
    log_trace = log_query_execution

    def get_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent traces sorted newest to oldest."""
        trace_files = sorted(
            list(self.trace_dir.glob("trace_*.json")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        traces = []
        for tf in trace_files[:limit]:
            try:
                with open(tf, "r", encoding="utf-8") as f:
                    traces.append(json.load(f))
            except Exception:
                continue
        return traces

    def get_trace_by_id(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific trace by query_id."""
        clean_id = query_id.strip()
        if not clean_id.endswith(".json"):
            target_path = self.trace_dir / f"{clean_id}.json"
        else:
            target_path = self.trace_dir / clean_id

        if target_path.exists():
            with open(target_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Fallback search across directory
        for tf in self.trace_dir.glob("*.json"):
            if clean_id in tf.stem:
                with open(tf, "r", encoding="utf-8") as f:
                    return json.load(f)

        return None


# Global singleton instance
_trace_logger = None


def get_trace_logger() -> TraceLogger:
    """Return singleton instance of TraceLogger."""
    global _trace_logger
    if _trace_logger is None:
        _trace_logger = TraceLogger()
    return _trace_logger
