"""
Universal Document Loader.
Supports PDF (via PyMuPDF), BPMN 2.0 XML (via BPMNXMLParser), and Markdown/Text.
"""

from pathlib import Path
from typing import List, Union
from langchain_core.documents import Document

from agentic_rag_sop.bpmn.parser import BPMNXMLParser


class UniversalDocumentLoader:
    """Loads documents from PDF, BPMN, TXT, and Markdown sources."""

    def __init__(self):
        self.bpmn_parser = BPMNXMLParser()

    def load_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Extract text and page metadata from PDF using PyMuPDF."""
        path = Path(file_path)
        import pymupdf

        documents: List[Document] = []
        doc = pymupdf.open(str(path))

        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                text = page.get_text("text").strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "document_name": path.name,
                                "document_type": "sop",
                                "source": str(path.resolve()),
                                "page_number": page_index + 1,
                            }
                        )
                    )
        finally:
            doc.close()

        return documents

    def load_bpmn(self, file_path: Union[str, Path]) -> List[Document]:
        """Parse structured BPMN 2.0 XML file into step-level Documents."""
        path = Path(file_path)
        return self.bpmn_parser.to_langchain_documents(path)

    def load_text(self, file_path: Union[str, Path]) -> List[Document]:
        """Load standard text or Markdown document."""
        path = Path(file_path)
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            return []

        return [
            Document(
                page_content=content,
                metadata={
                    "document_name": path.name,
                    "document_type": "sop" if "sop" in path.name.lower() else "general",
                    "source": str(path.resolve()),
                    "page_number": 1,
                }
            )
        ]

    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Route to appropriate loader based on file extension."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self.load_pdf(path)
        elif suffix in [".txt", ".md"]:
            return self.load_text(path)
        elif suffix in [".bpmn", ".bpm", ".xml"]:
            # BPMN parsing is reserved for future Agentic BPMN Retriever phase
            return []
        else:
            # Fallback text loader for other text formats
            try:
                return self.load_text(path)
            except Exception:
                return []

    def load_directory(self, dir_path: Union[str, Path]) -> List[Document]:
        """Recursively scan directory and load SOP documents (.pdf, .txt, .md)."""
        directory = Path(dir_path)
        if not directory.exists():
            return []

        all_docs: List[Document] = []
        # Phase 1: Only ingest SOP documents (.pdf, .txt, .md). Skip .bpmn/.bpm.
        supported_extensions = {".pdf", ".txt", ".md"}

        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                docs = self.load_file(file_path)
                all_docs.extend(docs)

        return all_docs
