"""
ChromaDB Vector Store Manager.
Handles persistent collection management, document indexing, and similarity retrieval.
"""

from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from agentic_rag_sop.config.settings import get_settings
from agentic_rag_sop.core.embeddings import get_embeddings


class VectorStoreManager:
    """Manages ChromaDB vector database instance and operations."""

    def __init__(self, embeddings: Optional[Embeddings] = None):
        self.settings = get_settings()
        self.embeddings = embeddings or get_embeddings()
        self._vectorstore = None

    def get_vectorstore(self):
        """Lazy load and return persistent Chroma vector store."""
        if self._vectorstore is None:
            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma

            self._vectorstore = Chroma(
                collection_name=self.settings.CHROMA_COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(self.settings.persist_path),
            )
        return self._vectorstore

    def add_documents(self, documents: List[Document], batch_size: int = 100) -> int:
        """
        Add documents to Chroma in batches to prevent payload limits.
        Returns the total number of added documents.
        """
        if not documents:
            return 0

        vectorstore = self.get_vectorstore()
        total_added = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            vectorstore.add_documents(batch)
            total_added += len(batch)

        return total_added

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform cosine/l2 similarity search with optional metadata filtering."""
        vectorstore = self.get_vectorstore()
        top_k = k or self.settings.TOP_K
        return vectorstore.similarity_search(query, k=top_k, filter=filter_dict)

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search returning documents paired with relevance scores."""
        vectorstore = self.get_vectorstore()
        top_k = k or self.settings.TOP_K
        return vectorstore.similarity_search_with_relevance_scores(
            query, k=top_k, filter=filter_dict
        )

    def count(self) -> int:
        """Return total number of vectors in the collection."""
        store = self.get_vectorstore()
        try:
            return store._collection.count()
        except Exception:
            return 0

    def as_retriever(self, **kwargs):
        """Export as standard LangChain retriever."""
        return self.get_vectorstore().as_retriever(**kwargs)

    def reset(self) -> None:
        """Completely reset and clear all vectors from the Chroma collection."""
        store = self.get_vectorstore()
        try:
            store.delete_collection()
        except Exception:
            pass
        self._vectorstore = None
