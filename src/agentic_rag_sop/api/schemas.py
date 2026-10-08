"""
Pydantic schemas for ICICoS Agentic RAG API requests and responses.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ==========================================
# Chat Schemas
# ==========================================

class ChatRequest(BaseModel):
    """Input payload for POST /api/v1/chat."""
    question: str = Field(..., description="User query or question", min_length=1)
    session_id: Optional[str] = Field(None, description="Optional conversation session ID")
    k: Optional[int] = Field(None, description="Number of document chunks to retrieve")
    document_type: Optional[str] = Field(None, description="Filter by document type (sop or bpmn)")
    actor: Optional[str] = Field(None, description="Filter by specific actor role")
    process_name: Optional[str] = Field(None, description="Filter by specific process name")
    use_query_agent: bool = Field(True, description="Enable Query Understanding Agent")
    use_router: bool = Field(True, description="Enable Router Agent")
    use_critic: bool = Field(True, description="Enable Critic Verification Agent")


class RouterDecisionSchema(BaseModel):
    """Router decision details."""
    intent: str
    knowledge_source: str
    reason: str


class SourceItemSchema(BaseModel):
    """Document citation item."""
    document_name: str
    document_type: str
    category: Optional[str] = "general"
    page_number: Any
    process_name: Optional[str] = "General"
    actor: Optional[str] = "General"
    chunk_index: Optional[int] = 0


class ContextFusionSchema(BaseModel):
    """Context fusion synthesis details."""
    process_domain: str = "ICICoS"
    actors: List[str] = Field(default_factory=list)
    workflow_sequence: List[str] = Field(default_factory=list)
    procedural_rules: List[str] = Field(default_factory=list)
    decision_gateways: List[str] = Field(default_factory=list)
    conflict_resolutions: List[str] = Field(default_factory=list)
    synthesized_context: Optional[str] = None


class CriticResultSchema(BaseModel):
    """Critic verification result."""
    is_valid: bool
    confidence_score: float
    critique_summary: str
    was_revised: bool = False
    suggested_revision_instructions: Optional[str] = None
    checks: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """
    Standardized response payload for POST /api/v1/chat.
    Conforms directly to the user requirements.
    """
    session_id: str = Field(description="Conversation session ID")
    query_id: Optional[str] = Field(None, description="Unique research execution trace ID")
    answer: str = Field(description="Final generated and verified answer")
    original_query: str = Field(description="Original user question")
    rewritten_query: str = Field(description="Normalized query from Query Understanding Agent")
    router_decision: Optional[Dict[str, Any]] = Field(None, description="Router Agent classification")
    selected_retriever: str = Field(description="Name of the retriever used")
    retrieved_sources: List[Dict[str, Any]] = Field(default_factory=list, description="List of source citations")
    context_fusion: Optional[Dict[str, Any]] = Field(None, description="Fused context details")
    critic_result: Optional[Dict[str, Any]] = Field(None, description="Critic verification outcome")
    confidence_score: Optional[float] = Field(1.0, description="Verification confidence score")
    process_path: List[str] = Field(default_factory=list, description="Extracted procedural path")


# ==========================================
# Document & Chunk Schemas
# ==========================================

class DocumentItemSchema(BaseModel):
    """Metadata for an indexed document."""
    id: str
    document_name: str
    file_type: str
    document_domain: str = "ICICoS"
    category: str
    total_pages: int
    chunk_count: int
    created_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response for GET /api/v1/documents."""
    total_documents: int
    documents: List[DocumentItemSchema]


class ChunkItemSchema(BaseModel):
    """Individual chunk details."""
    id: str
    document_id: str
    chunk_index: int
    page_number: str
    process_name: str
    actor: str
    content: str
    chroma_id: Optional[str] = None


class DocumentChunksResponse(BaseModel):
    """Response for GET /api/v1/documents/{id}/chunks."""
    document_id: str
    total_chunks: int
    chunks: List[ChunkItemSchema]


# ==========================================
# BPMN Process Schemas
# ==========================================

class BPMNProcessItemSchema(BaseModel):
    """Summary of a parsed BPMN process model."""
    process_id: str
    process_name: str
    document_domain: str = "ICICoS"
    category: str
    actors: List[str]
    total_nodes: int
    total_gateways: int
    total_flows: int


class BPMNProcessListResponse(BaseModel):
    """Response for GET /api/v1/bpmn/processes."""
    total_processes: int
    processes: List[BPMNProcessItemSchema]


# ==========================================
# History Schemas
# ==========================================

class AgentLogItemSchema(BaseModel):
    """Log entry for an agent execution step."""
    step_order: int
    agent_name: str
    status: str
    input_payload: Optional[Any] = None
    output_payload: Optional[Any] = None
    created_at: Optional[str] = None


class MessageItemSchema(BaseModel):
    """Message with trace logs in history."""
    id: str
    role: str
    content: str
    created_at: Optional[str] = None
    agent_logs: List[AgentLogItemSchema] = Field(default_factory=list)


class SessionHistoryResponse(BaseModel):
    """Response for GET /api/v1/history/{session_id}."""
    session_id: str
    title: str
    document_domain: str = "ICICoS"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    messages: List[MessageItemSchema] = Field(default_factory=list)
