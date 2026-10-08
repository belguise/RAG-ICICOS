"""
Data Access Layer (Repository) for ICICoS RAG database operations.
Handles chat session persistence, agent audit logging, and document/chunk metadata.
"""

import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from agentic_rag_sop.db.models import (
    DocumentModel,
    DocumentChunkModel,
    ChatSessionModel,
    ChatMessageModel,
    AgentExecutionLogModel,
)


def get_or_create_session(
    db: Session,
    session_id: Optional[str] = None,
    title: Optional[str] = None,
) -> ChatSessionModel:
    """Retrieve existing session or create a new session."""
    if session_id:
        session = db.query(ChatSessionModel).filter(ChatSessionModel.id == session_id).first()
        if session:
            return session

    new_id = session_id or str(uuid.uuid4())
    session_title = title or "Sesi Percakapan ICICoS"
    session = ChatSessionModel(id=new_id, title=session_title, document_domain="ICICoS")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def add_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
) -> ChatMessageModel:
    """Save user or assistant message to database."""
    msg = ChatMessageModel(
        session_id=session_id,
        role=role,
        content=content,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def record_agent_logs(
    db: Session,
    message_id: str,
    session_id: str,
    trace_steps: List[Dict[str, Any]],
) -> None:
    """
    Record each step in the agent execution trace (QueryUnderstanding, Router,
    Retriever, ContextFusion, Critic).
    """
    for order, step in enumerate(trace_steps, 1):
        log = AgentExecutionLogModel(
            message_id=message_id,
            session_id=session_id,
            step_order=order,
            agent_name=step.get("agent_name", "UnknownAgent"),
            status=step.get("status", "SUCCESS"),
            input_payload=step.get("input"),
            output_payload=step.get("output"),
        )
        db.add(log)
    db.commit()


def get_session_history(
    db: Session,
    session_id: str,
) -> Optional[Dict[str, Any]]:
    """Retrieve all messages and associated agent logs for a session."""
    session = db.query(ChatSessionModel).filter(ChatSessionModel.id == session_id).first()
    if not session:
        return None

    messages_data = []
    for msg in session.messages:
        logs_data = []
        for log in msg.agent_logs:
            logs_data.append({
                "step_order": log.step_order,
                "agent_name": log.agent_name,
                "status": log.status,
                "input_payload": log.input_payload,
                "output_payload": log.output_payload,
                "created_at": log.created_at.isoformat() if log.created_at else None,
            })

        messages_data.append({
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
            "agent_logs": logs_data,
        })

    return {
        "session_id": session.id,
        "title": session.title,
        "document_domain": session.document_domain,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
        "messages": messages_data,
    }


def list_documents(db: Session) -> List[Dict[str, Any]]:
    """Query list of all indexed documents with chunk counts."""
    # Ensure ChromaDB chunks are synced to DB if empty
    sync_documents_if_empty(db)

    docs = db.query(DocumentModel).all()
    results = []
    for doc in docs:
        results.append({
            "id": doc.id,
            "document_name": doc.document_name,
            "file_type": doc.file_type,
            "document_domain": doc.document_domain,
            "category": doc.category,
            "total_pages": doc.total_pages,
            "chunk_count": doc.chunk_count,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
        })
    return results


def get_document_chunks(
    db: Session,
    document_id: str,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Query chunks belonging to a document."""
    sync_documents_if_empty(db)

    chunks = (
        db.query(DocumentChunkModel)
        .filter(DocumentChunkModel.document_id == document_id)
        .order_by(DocumentChunkModel.chunk_index)
        .offset(offset)
        .limit(limit)
        .all()
    )

    results = []
    for c in chunks:
        results.append({
            "id": c.id,
            "document_id": c.document_id,
            "chunk_index": c.chunk_index,
            "page_number": c.page_number,
            "process_name": c.process_name,
            "actor": c.actor,
            "content": c.content,
            "chroma_id": c.chroma_id,
        })
    return results


def sync_documents_if_empty(db: Session) -> None:
    """
    Populate documents and document_chunks tables from ChromaDB vector store
    if the documents table is currently empty.
    """
    count = db.query(DocumentModel).count()
    if count > 0:
        return

    try:
        import chromadb
        from agentic_rag_sop.config.settings import get_settings
        settings = get_settings()
        client = chromadb.PersistentClient(path=str(settings.persist_path))
        collection = client.get_collection(settings.CHROMA_COLLECTION_NAME)
        raw_data = collection.get(include=["documents", "metadatas"])

        docs_map: Dict[str, DocumentModel] = {}
        chunk_models: List[DocumentChunkModel] = []

        chroma_ids = raw_data.get("ids", [])
        documents = raw_data.get("documents", [])
        metadatas = raw_data.get("metadatas", [])

        for idx, (cid, text, meta) in enumerate(zip(chroma_ids, documents, metadatas)):
            meta = meta or {}
            doc_name = meta.get("document_name", "Unknown Document")
            # Create a clean document ID / slug
            doc_id = doc_name.replace(".md", "").replace(".pdf", "").replace(" ", "_").lower()

            if doc_id not in docs_map:
                doc_model = DocumentModel(
                    id=doc_id,
                    document_name=doc_name,
                    file_type=meta.get("document_type", "sop"),
                    document_domain=meta.get("document_domain", "ICICoS"),
                    category=meta.get("category", "ICICoS Operational Workflow"),
                    file_path=meta.get("source", doc_name),
                    total_pages=1,
                    chunk_count=0,
                )
                docs_map[doc_id] = doc_model

            docs_map[doc_id].chunk_count += 1

            chunk_id = f"{doc_id}_chunk_{meta.get('chunk_index', idx)}"
            chunk_model = DocumentChunkModel(
                id=chunk_id,
                document_id=doc_id,
                chunk_index=meta.get("chunk_index", idx),
                page_number=str(meta.get("page_number", "N/A")),
                process_name=meta.get("process_name", "General"),
                actor=meta.get("actor", "General"),
                content=text,
                chroma_id=cid,
            )
            chunk_models.append(chunk_model)

        for d in docs_map.values():
            db.add(d)
        for c in chunk_models:
            db.add(c)

        db.commit()
    except Exception:
        db.rollback()
