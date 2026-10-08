"""
API module for ICICoS Agentic RAG platform.
Provides FastAPI REST endpoints wrapping the AI agents and knowledge base.
"""

from agentic_rag_sop.api.main import app

__all__ = ["app"]
