"""
Main FastAPI Application Entrypoint.
Phase 7.1 Backend API Layer for ICICoS Agentic RAG Platform.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_rag_sop.config.settings import get_settings
from agentic_rag_sop.db.database import init_db, SessionLocal
from agentic_rag_sop.db.repository import sync_documents_if_empty
from agentic_rag_sop.api.routes import (
    chat_router,
    documents_router,
    bpmn_router,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown handler."""
    # 1. Initialize SQLite / PostgreSQL tables
    init_db()

    # 2. Sync indexed documents and chunks from ChromaDB if needed
    db = SessionLocal()
    try:
        sync_documents_if_empty(db)
    finally:
        db.close()

    yield


app = FastAPI(
    title="ICICoS Agentic RAG Platform API",
    version="1.0.0",
    description=(
        "Backend REST API for ICICoS Intelligent Process Assistant. "
        "Provides agentic question answering, document chunk inspection, "
        "and BPMN workflow topology exploration under document_domain='ICICoS'."
    ),
    lifespan=lifespan,
)

# Configure CORS for local Next.js development and production deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://*.vercel.app",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Route Handlers
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(bpmn_router)


@app.get("/", tags=["System"])
def root_endpoint():
    """Service status and meta information."""
    return {
        "service": "ICICoS Agentic RAG Backend API",
        "version": "1.0.0",
        "document_domain": "ICICoS",
        "status": "online",
        "docs_url": "/docs",
        "endpoints": {
            "chat": "POST /api/v1/chat",
            "history": "GET /api/v1/history/{session_id}",
            "documents": "GET /api/v1/documents",
            "chunks": "GET /api/v1/documents/{id}/chunks",
            "bpmn_processes": "GET /api/v1/bpmn/processes",
        },
    }


@app.get("/health", tags=["System"])
def health_endpoint():
    """Health check probe."""
    return {
        "status": "healthy",
        "document_domain": "ICICoS",
    }


def start_server():
    """Entrypoint function to run uvicorn server directly."""
    import uvicorn
    uvicorn.run(
        "agentic_rag_sop.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
    )


if __name__ == "__main__":
    start_server()
