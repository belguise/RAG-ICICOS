"""
SQLAlchemy ORM models for ICICoS Agentic RAG.
Stores documents, chunks, chat sessions, messages, and agent execution audit logs.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Integer,
    Text,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship

from agentic_rag_sop.db.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChatSessionModel(Base):
    """Chat conversation session."""
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    title = Column(String(255), nullable=False, default="Percakapan Baru")
    document_domain = Column(String(50), default="ICICoS")
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

    messages = relationship("ChatMessageModel", back_populates="session", cascade="all, delete-orphan")


class ChatMessageModel(Base):
    """Individual message in a chat session."""
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user', 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=get_utc_now)

    session = relationship("ChatSessionModel", back_populates="messages")
    agent_logs = relationship("AgentExecutionLogModel", back_populates="message", cascade="all, delete-orphan")


class AgentExecutionLogModel(Base):
    """Detailed audit log for each AI agent step executed for a message."""
    __tablename__ = "agent_execution_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    message_id = Column(String(36), ForeignKey("chat_messages.id"), nullable=True)
    session_id = Column(String(36), nullable=True)
    step_order = Column(Integer, nullable=False, default=1)
    agent_name = Column(String(100), nullable=False)  # QueryUnderstanding, Router, Retriever, Fusion, Critic
    status = Column(String(50), nullable=False, default="SUCCESS")
    input_payload = Column(JSON, nullable=True)
    output_payload = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=get_utc_now)

    message = relationship("ChatMessageModel", back_populates="agent_logs")


class DocumentModel(Base):
    """Metadata for indexed SOP or BPMN documents."""
    __tablename__ = "documents"

    id = Column(String(100), primary_key=True)
    document_name = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False, default="sop")
    document_domain = Column(String(50), default="ICICoS")
    category = Column(String(100), default="ICICoS Operational Workflow")
    file_path = Column(String(500), nullable=False)
    total_pages = Column(Integer, default=1)
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=get_utc_now)

    chunks = relationship("DocumentChunkModel", back_populates="document", cascade="all, delete-orphan")


class DocumentChunkModel(Base):
    """Individual text chunk associated with an indexed document."""
    __tablename__ = "document_chunks"

    id = Column(String(150), primary_key=True)
    document_id = Column(String(100), ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False, default=0)
    page_number = Column(String(50), default="N/A")
    process_name = Column(String(255), default="General")
    actor = Column(String(255), default="General")
    content = Column(Text, nullable=False)
    chroma_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=get_utc_now)

    document = relationship("DocumentModel", back_populates="chunks")
