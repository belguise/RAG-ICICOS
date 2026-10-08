"""
Base Retriever Interface.
Phase 2.3 Modular Architecture: Defines abstract contract for knowledge retrievers
(SOPRetriever, BPMNRetriever, and future multi-modal retrievers).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document


class BaseKnowledgeRetriever(ABC):
    """Abstract interface defining required behavior for knowledge retrievers."""

    @property
    @abstractmethod
    def retriever_name(self) -> str:
        """Return the descriptive name of the retriever."""
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Retrieve relevant Document objects based on the input query.
        
        Args:
            query: Input user question or normalized search query.
            k: Maximum number of document chunks to return.
            **kwargs: Additional retriever-specific parameters.
            
        Returns:
            List of LangChain Document objects containing page_content and metadata.
        """
        pass

    def get_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a formatted context string suitable for LLM prompts.
        Default implementation formats standard metadata headers.
        """
        if not documents:
            return "Tidak ada dokumen relevan yang ditemukan."

        context_blocks = []
        for idx, doc in enumerate(documents, start=1):
            meta = doc.metadata
            header = (
                f"--- DOKUMEN [{idx}]: {meta.get('document_name', 'Unknown')} "
                f"(Tipe: {meta.get('document_type', 'sop')}, "
                f"Hal: {meta.get('page_number', 'N/A')}, "
                f"Proses: {meta.get('process_name', 'General')}, "
                f"Aktor: {meta.get('actor', 'General')}) ---"
            )
            context_blocks.append(f"{header}\n{doc.page_content.strip()}")

        return "\n\n".join(context_blocks)
