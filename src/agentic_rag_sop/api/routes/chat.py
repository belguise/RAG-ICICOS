"""
Chat and session history route handlers.
Wraps SOPRAGChain and records agent execution logs to database.
"""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from agentic_rag_sop.db.database import get_db
from agentic_rag_sop.db import repository
from agentic_rag_sop.logging import get_trace_logger
from agentic_rag_sop.api.schemas import (
    ChatRequest,
    ChatResponse,
    SessionHistoryResponse,
)
from agentic_rag_sop.generation.chain import SOPRAGChain

router = APIRouter(prefix="/api/v1", tags=["Chat & Agent Execution"])

# Lazy singleton for the generation chain
_chain_instance = None


def get_rag_chain() -> SOPRAGChain:
    """Return singleton instance of SOPRAGChain."""
    global _chain_instance
    if _chain_instance is None:
        _chain_instance = SOPRAGChain()
    return _chain_instance


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Execute full ICICoS Agentic RAG QA generation.
    Returns generated answer with complete trace of QueryUnderstanding,
    Router, ContextFusion, and Critic Verification agents.
    """
    chain = get_rag_chain()

    # 1. Get or create session
    session = repository.get_or_create_session(
        db=db,
        session_id=request.session_id,
        title=request.question[:60],
    )

    # 2. Record incoming user message
    user_msg = repository.add_message(
        db=db,
        session_id=session.id,
        role="user",
        content=request.question,
    )

    # 3. Execute RAG Chain
    try:
        result = chain.invoke(
            question=request.question,
            k=request.k,
            document_type=request.document_type,
            actor=request.actor,
            process_name=request.process_name,
            use_query_agent=request.use_query_agent,
            use_router=request.use_router,
            use_critic=request.use_critic,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG execution failed: {str(e)}")

    # 4. Record assistant message
    asst_msg = repository.add_message(
        db=db,
        session_id=session.id,
        role="assistant",
        content=result["answer"],
    )

    # 5. Format and record agent execution trace logs
    trace_steps = []

    # Step 1: Query Understanding
    if request.use_query_agent:
        trace_steps.append({
            "agent_name": "QueryUnderstandingAgent",
            "status": "SUCCESS",
            "input": {"original_query": result["question"]},
            "output": {"rewritten_query": result.get("rewritten_query")},
        })

    # Step 2: Router Agent
    route_dec = result.get("route_decision")
    if request.use_router and route_dec:
        trace_steps.append({
            "agent_name": "RouterAgent",
            "status": "SUCCESS",
            "input": {"query": result.get("rewritten_query")},
            "output": {
                "intent": getattr(route_dec, "intent", str(route_dec)),
                "knowledge_source": getattr(route_dec, "knowledge_source", "SOP"),
                "reason": getattr(route_dec, "reason", ""),
            },
        })

    # Step 3: Retrieval
    trace_steps.append({
        "agent_name": result.get("selected_retriever", "SOP Retriever"),
        "status": "SUCCESS",
        "input": {"query": result.get("rewritten_query"), "k": request.k},
        "output": {"retrieved_count": len(result.get("sources", [])), "sources": result.get("sources", [])},
    })

    # Step 4: Context Fusion
    fused_ctx = result.get("fused_context")
    if fused_ctx:
        trace_steps.append({
            "agent_name": "ContextFusionLayer",
            "status": "SUCCESS",
            "input": {"domain": getattr(fused_ctx, "process_domain", "ICICoS")},
            "output": {
                "actors": getattr(fused_ctx, "actors", []),
                "workflow_steps_count": len(getattr(fused_ctx, "workflow_sequence", [])),
                "procedural_rules_count": len(getattr(fused_ctx, "procedural_rules", [])),
                "gateways_count": len(getattr(fused_ctx, "decision_gateways", [])),
            },
        })

    # Step 5: Critic Verification
    critic_dec = result.get("critic_decision")
    if request.use_critic and critic_dec:
        status_label = "REVISED" if result.get("was_revised") else ("PASSED" if critic_dec.is_valid else "FAILED")
        trace_steps.append({
            "agent_name": "CriticVerificationAgent",
            "status": status_label,
            "input": {"draft_answer": result.get("draft_answer")},
            "output": {
                "is_valid": critic_dec.is_valid,
                "confidence_score": critic_dec.confidence_score,
                "critique_summary": critic_dec.critique_summary,
                "was_revised": result.get("was_revised", False),
            },
        })

    repository.record_agent_logs(
        db=db,
        message_id=asst_msg.id,
        session_id=session.id,
        trace_steps=trace_steps,
    )

    # 6. Format API response
    router_data = None
    if route_dec:
        router_data = {
            "intent": getattr(route_dec, "intent", str(route_dec)),
            "knowledge_source": getattr(route_dec, "knowledge_source", "SOP"),
            "reason": getattr(route_dec, "reason", ""),
        }

    fusion_data = None
    if fused_ctx:
        fusion_data = {
            "process_domain": getattr(fused_ctx, "process_domain", "ICICoS"),
            "actors": getattr(fused_ctx, "actors", []),
            "workflow_sequence": getattr(fused_ctx, "workflow_sequence", []),
            "procedural_rules": getattr(fused_ctx, "procedural_rules", []),
            "decision_gateways": getattr(fused_ctx, "decision_gateways", []),
            "conflict_resolutions": getattr(fused_ctx, "conflict_resolutions", []),
            "synthesized_context": getattr(fused_ctx, "synthesized_context", ""),
        }

    critic_data = None
    if critic_dec:
        critic_data = {
            "is_valid": critic_dec.is_valid,
            "confidence_score": critic_dec.confidence_score,
            "critique_summary": critic_dec.critique_summary,
            "was_revised": result.get("was_revised", False),
            "suggested_revision_instructions": critic_dec.suggested_revision_instructions,
            "checks": {
                "factual_grounding": str(getattr(critic_dec.factual_grounding, "status", critic_dec.factual_grounding)),
                "domain_compliance": str(getattr(critic_dec.domain_compliance, "status", critic_dec.domain_compliance)),
                "workflow_correctness": str(getattr(critic_dec.workflow_correctness, "status", critic_dec.workflow_correctness)),
                "sop_correctness": str(getattr(critic_dec.sop_correctness, "status", critic_dec.sop_correctness)),
            },
        }

    # 6. Record standardized research execution trace
    trace_logger = get_trace_logger()
    all_sources = result.get("sources", [])
    sop_chunks = [s for s in all_sources if (s.get("document_type") or "sop").lower() == "sop"]
    bpmn_nodes = [s for s in all_sources if (s.get("document_type") or "").lower() == "bpmn"]
    process_path = getattr(fused_ctx, "workflow_sequence", []) if fused_ctx else []
    confidence_score = getattr(critic_dec, "confidence_score", 1.0) if critic_dec else 1.0

    trace_record = trace_logger.log_query_execution(
        original_query=result["question"],
        rewritten_query=result.get("rewritten_query", result["question"]),
        detected_intent=router_data["intent"] if router_data else "general_information",
        selected_source=router_data["knowledge_source"] if router_data else "SOP",
        router_reasoning=router_data["reason"] if router_data else "",
        retrieved_sop_chunks=sop_chunks,
        retrieved_bpmn_nodes=bpmn_nodes,
        process_path=process_path,
        generated_answer=result["answer"],
        critic_result=critic_data or {"is_valid": True, "critique_summary": "Auto-validated."},
        confidence_score=confidence_score,
        primary_source=router_data.get("knowledge_source", "SOP") if router_data else "SOP",
        secondary_source=router_data.get("secondary_source") if router_data else None,
        extra_metadata={"session_id": session.id, "message_id": asst_msg.id},
    )

    return ChatResponse(
        session_id=session.id,
        query_id=trace_record["query_id"],
        answer=result["answer"],
        original_query=result["question"],
        rewritten_query=result.get("rewritten_query", result["question"]),
        router_decision=router_data,
        selected_retriever=result.get("selected_retriever", "SOP Retriever"),
        retrieved_sources=all_sources,
        context_fusion=fusion_data,
        critic_result=critic_data,
        confidence_score=confidence_score,
        process_path=process_path,
    )


@router.get("/traces")
def get_traces_endpoint(limit: int = 50):
    """Retrieve recent research execution traces."""
    logger = get_trace_logger()
    return logger.get_traces(limit=limit)


@router.get("/traces/{query_id}")
def get_trace_detail_endpoint(query_id: str):
    """Retrieve a specific research execution trace by query_id."""
    logger = get_trace_logger()
    trace = logger.get_trace_by_id(query_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace '{query_id}' not found.")
    return trace


@router.get("/experiments/latest")
def get_latest_experiments_endpoint():
    """Serve benchmark metrics summary and comparative results for the research dashboard."""
    curr = Path(__file__).resolve()
    results_dir = None
    for p in curr.parents:
        if (p / "results").exists() or (p / "pyproject.toml").exists():
            results_dir = p / "results"
            break
    if not results_dir or not results_dir.exists():
        results_dir = curr.parents[4] / "results"
    
    summary_file = results_dir / "metrics_summary.csv"
    comparison_file = results_dir / "comparison_table.csv"
    json_file = results_dir / "experiment_results.json"

    metrics = []
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            metrics = list(reader)

    comparison = []
    if comparison_file.exists():
        with open(comparison_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            comparison = list(reader)

    experiment_meta = {}
    if json_file.exists():
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                experiment_meta = {
                    "experiment_id": data.get("experiment_id"),
                    "timestamp": data.get("timestamp"),
                    "total_questions": data.get("total_questions"),
                    "systems_evaluated": data.get("systems_evaluated"),
                }
        except Exception:
            pass

    return {
        "experiment_meta": experiment_meta,
        "metrics_summary": metrics,
        "comparison_table": comparison,
    }


@router.get("/history/{session_id}", response_model=SessionHistoryResponse)
def get_history_endpoint(
    session_id: str,
    db: Session = Depends(get_db),
) -> SessionHistoryResponse:
    """Retrieve complete chat session history including all agent execution logs."""
    history = repository.get_session_history(db=db, session_id=session_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    return SessionHistoryResponse.model_validate(history)
