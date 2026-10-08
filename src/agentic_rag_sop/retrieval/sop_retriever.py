"""
SOP Retriever Implementation.
Phase 2.3 Modular Architecture: Encapsulates ChromaDB vector search for textual SOP documents.
"""

from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever as LangChainBaseRetriever

from agentic_rag_sop.config.settings import get_settings
from agentic_rag_sop.core.vectorstore import VectorStoreManager
from agentic_rag_sop.retrieval.base import BaseKnowledgeRetriever


class SOPRetriever(BaseKnowledgeRetriever):
    """Retriever for Standard Operating Procedure (SOP) documents using ChromaDB vector search."""

    def __init__(self, vectorstore_manager: Optional[VectorStoreManager] = None):
        self.settings = get_settings()
        self.vectorstore_manager = vectorstore_manager or VectorStoreManager()

    @property
    def retriever_name(self) -> str:
        return "SOP Retriever"

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        document_type: Optional[str] = "sop",
        actor: Optional[str] = None,
        process_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Retrieve top-K matching SOP chunks from ChromaDB with metadata filtering.
        
        Args:
            query: Procedural search query.
            k: Number of chunks to retrieve (defaults to settings.TOP_K).
            document_type: Filter by document type (defaults to 'sop').
            actor: Filter by responsible role.
            process_name: Filter by specific business process.
            
        Returns:
            List of Document objects with preserved metadata:
            - document_name
            - document_type
            - process_name
            - actor
            - page_number
        """
        filter_dict: Dict[str, Any] = {}
        if document_type:
            filter_dict["document_type"] = document_type
        if actor:
            filter_dict["actor"] = actor
        if process_name:
            filter_dict["process_name"] = process_name

        filter_arg = filter_dict if filter_dict else None
        top_k = k or self.settings.TOP_K

        return self.vectorstore_manager.similarity_search(
            query=query,
            k=top_k,
            filter_dict=filter_arg,
        )

    def as_langchain_retriever(
        self,
        k: Optional[int] = None,
        search_type: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> LangChainBaseRetriever:
        """
        Convert to standard LangChain BaseRetriever for LCEL chains.
        Supports 'similarity' or 'mmr' (Maximal Marginal Relevance).
        """
        top_k = k or self.settings.TOP_K
        stype = search_type or self.settings.SEARCH_TYPE

        search_kwargs: Dict[str, Any] = {"k": top_k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        return self.vectorstore_manager.get_vectorstore().as_retriever(
            search_type=stype,
            search_kwargs=search_kwargs,
        )
