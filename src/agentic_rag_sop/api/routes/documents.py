"""
Document metadata and chunk inspection route handlers.
Provides read access to indexed SOP documents and individual text chunks.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from agentic_rag_sop.db.database import get_db
from agentic_rag_sop.db import repository
from agentic_rag_sop.api.schemas import (
    DocumentListResponse,
    DocumentChunksResponse,
)

router = APIRouter(prefix="/api/v1", tags=["Documents & Knowledge Base"])


@router.get("/documents", response_model=DocumentListResponse)
def list_documents_endpoint(
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    """
    List all indexed SOP documents in the ICICoS knowledge base with chunk counts.
    """
    docs = repository.list_documents(db=db)
    return DocumentListResponse(
        total_documents=len(docs),
        documents=docs,
    )


@router.get("/documents/{id}/chunks", response_model=DocumentChunksResponse)
def get_document_chunks_endpoint(
    id: str,
    limit: int = Query(100, ge=1, le=500, description="Max chunks to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
) -> DocumentChunksResponse:
    """
    Retrieve all text chunks belonging to a specific document ID.
    Supports inspecting content, page numbers, and assigned actors.
    """
    chunks = repository.get_document_chunks(
        db=db,
        document_id=id,
        limit=limit,
        offset=offset,
    )

    if not chunks:
        # Check if document itself exists
        docs = repository.list_documents(db=db)
        matched = [d for d in docs if d["id"] == id]
        if not matched:
            raise HTTPException(status_code=404, detail=f"Document with ID '{id}' not found")

    return DocumentChunksResponse(
        document_id=id,
        total_chunks=len(chunks),
        chunks=chunks,
    )
