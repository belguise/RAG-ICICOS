"""
Context Fusion Layer Module for Agentic RAG SOP & BPMN.
Unifies textual SOP requirements with BPMN workflow process topologies under
the strict ICICoS document domain.
"""

from agentic_rag_sop.fusion.context_fusion import ContextFusionLayer, FusedContext

__all__ = [
    "ContextFusionLayer",
    "FusedContext",
]
