"""
Process-Aware Chunker.
Splits text while preserving procedural continuity and enriching every chunk
with the 7 mandatory metadata attributes.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_rag_sop.config.settings import get_settings
from agentic_rag_sop.ingestion.metadata import enrich_chunk_metadata


class ProcessAwareChunker:
    """
    Chunks SOP and process documents while ensuring strict metadata preservation.
    Keeps BPMN atomic steps intact when appropriate.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # Text splitter tailored for procedural & regulatory text
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",      # Paragraphs / Articles
                "\n",        # Steps / List items
                ". ",        # Sentences
                "; ",        # Sub-clauses
                " ",         # Words
                ""
            ],
            is_separator_regex=False
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into metadata-enriched chunks."""
        chunked_docs: List[Document] = []
        global_chunk_idx = 0

        for doc in documents:
            doc_type = doc.metadata.get("document_type", "sop")

            # BPMN step documents are already atomic steps created by BPMNXMLParser
            if doc_type == "bpmn" and len(doc.page_content) <= self.chunk_size * 1.2:
                enriched_meta = enrich_chunk_metadata(
                    raw_metadata=doc.metadata,
                    chunk_text=doc.page_content,
                    chunk_index=global_chunk_idx,
                    default_document_type="bpmn",
                )
                chunked_docs.append(Document(page_content=doc.page_content, metadata=enriched_meta))
                global_chunk_idx += 1
                continue

            # Standard text / PDF splitting
            splits = self.splitter.split_text(doc.page_content)

            for split_text in splits:
                clean_text = split_text.strip()
                if not clean_text:
                    continue

                enriched_meta = enrich_chunk_metadata(
                    raw_metadata=doc.metadata,
                    chunk_text=clean_text,
                    chunk_index=global_chunk_idx,
                    default_document_type=doc_type,
                )

                chunked_docs.append(
                    Document(
                        page_content=clean_text,
                        metadata=enriched_meta
                    )
                )
                global_chunk_idx += 1

        return chunked_docs
