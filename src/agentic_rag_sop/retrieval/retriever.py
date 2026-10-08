"""
SOP & BPMN Retriever Layer (Backward Compatibility Module).
Re-exports SOPRetriever from modular sop_retriever implementation.
"""

from agentic_rag_sop.retrieval.sop_retriever import SOPRetriever

__all__ = ["SOPRetriever"]
