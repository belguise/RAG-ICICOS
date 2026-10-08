"""
API routes package for ICICoS Agentic RAG.
"""

from agentic_rag_sop.api.routes.chat import router as chat_router
from agentic_rag_sop.api.routes.documents import router as documents_router
from agentic_rag_sop.api.routes.bpmn import router as bpmn_router

__all__ = [
    "chat_router",
    "documents_router",
    "bpmn_router",
]
