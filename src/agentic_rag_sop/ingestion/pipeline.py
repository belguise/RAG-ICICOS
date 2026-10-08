"""
End-to-End Ingestion Pipeline.
Loads documents, enriches chunk metadata with the 7 required fields,
and stores embeddings into persistent ChromaDB.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from langchain_core.documents import Document

from agentic_rag_sop.config.settings import get_settings
from agentic_rag_sop.core.vectorstore import VectorStoreManager
from agentic_rag_sop.ingestion.chunker import ProcessAwareChunker
from agentic_rag_sop.ingestion.loader import UniversalDocumentLoader


class IngestionPipeline:
    """Orchestrates document loading, chunking, metadata enrichment, and vector storage."""

    def __init__(
        self,
        vectorstore_manager: Optional[VectorStoreManager] = None,
        loader: Optional[UniversalDocumentLoader] = None,
        chunker: Optional[ProcessAwareChunker] = None,
    ):
        self.settings = get_settings()
        self.vectorstore_manager = vectorstore_manager or VectorStoreManager()
        self.loader = loader or UniversalDocumentLoader()
        self.chunker = chunker or ProcessAwareChunker()

    def process_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a single file, chunk it, and enrich metadata."""
        docs = self.loader.load_file(file_path)
        if not docs:
            return []
        chunks = self.chunker.split_documents(docs)
        return chunks

    def process_directory(self, dir_path: Union[str, Path]) -> List[Document]:
        """Recursively load, chunk, and enrich all supported files in a directory."""
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        all_chunks: List[Document] = []
        supported_extensions = {".pdf", ".bpmn", ".xml", ".txt", ".md"}

        for item in path.rglob("*"):
            if item.is_file() and item.suffix.lower() in supported_extensions:
                file_chunks = self.process_file(item)
                all_chunks.extend(file_chunks)

        return all_chunks

    def run(
        self,
        source_path: Optional[Union[str, Path]] = None,
        reset: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the full ingestion pipeline and index into Chroma.
        
        Args:
            source_path: Path to directory or file to ingest.
            reset: If True, purges and resets the vector store collection prior to indexing.
        
        Returns summary statistics dictionary.
        """
        if reset:
            self.vectorstore_manager.reset()

        path = Path(source_path) if source_path else self.settings.raw_path

        if path.is_file():
            chunks = self.process_file(path)
            files_processed = 1
        else:
            chunks = self.process_directory(path)
            files_processed = len(set(c.metadata["document_name"] for c in chunks))

        if not chunks:
            return {
                "status": "warning",
                "message": f"No valid documents found in {path}",
                "files_processed": 0,
                "chunks_indexed": 0,
                "total_vectors_in_db": self.vectorstore_manager.count(),
            }

        # Index into ChromaDB
        added_count = self.vectorstore_manager.add_documents(chunks)
        total_vectors = self.vectorstore_manager.count()

        # Gather metadata statistics
        doc_types = Counter(c.metadata.get("document_type") for c in chunks)
        actors = Counter(c.metadata.get("actor") for c in chunks)
        processes = Counter(c.metadata.get("process_name") for c in chunks)

        return {
            "status": "success",
            "source_path": str(path),
            "files_processed": files_processed,
            "chunks_indexed": added_count,
            "total_vectors_in_db": total_vectors,
            "document_types": dict(doc_types),
            "top_actors": dict(actors.most_common(5)),
            "top_processes": dict(processes.most_common(5)),
        }
