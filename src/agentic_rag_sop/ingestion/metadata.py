"""
Metadata schema and enrichment utilities for SOP and BPMN chunks.
Enforces the 7 required metadata fields:
- document_name
- document_type
- category
- process_name
- actor
- page_number
- chunk_index
"""

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Strict schema for chunk metadata preservation."""
    document_name: str = Field(description="Name of the source document file")
    document_type: str = Field(default="sop", description="Type of document, e.g., 'sop', 'bpmn', 'regulation'")
    category: str = Field(default="general", description="Domain or business category")
    process_name: str = Field(default="Standard Procedure", description="Name of the business process")
    department: str = Field(default="Panitia ICICoS / FSM", description="Responsible department or organizational unit")
    actor: str = Field(default="Semua Pihak", description="Primary actor, role, or swimlane responsible")
    page_number: Any = Field(default="1", description="Page number where the chunk originated")
    chunk_index: int = Field(default=0, description="Zero-based sequence index of the chunk")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with standard string/int types acceptable to ChromaDB."""
        data = self.model_dump()
        # Chroma requires metadata values to be str, int, float, or bool
        data["page_number"] = str(data["page_number"])
        data["chunk_index"] = int(data["chunk_index"])
        return data


# Common actor patterns in conference and academic SOPs
ACTOR_PATTERNS = [
    r"\b(Bendahara ICICoS|Bendahara)\b",
    r"\b(Ketua Panitia|Ketua)\b",
    r"\b(Sekretaris|Kesekretariatan)\b",
    r"\b(Author|Author of ICICoS 2026)\b",
    r"\b(Reviewer|Moderator|Keynote Speaker|Keynote)\b",
    r"\b(Keuangan Fakultas|Keuangan FSM|Subbagian Keuangan)\b",
    r"\b(Keuangan Universitas|Bagian Keuangan Universitas)\b",
    r"\b(Dekan|Wakil Dekan I|Wakil Dekan II)\b",
    r"\b(PIC Divisi|Divisi di ICICoS)\b",
    r"\b(Hotel|Rekanan|Vendor)\b",
    r"\b(Rektor|Wakil Rektor\s*[I-IV]?)\b",
    r"\b(Dosen Pembimbing|Dosen)\b",
    r"\b(Mahasiswa)\b",
]

# Common process/heading patterns
PROCESS_PATTERNS = [
    r"(?:BAB\s+[IVXLCDM]+\s*[:\-–]?\s*)([A-Z\s]{4,60})",
    r"(?:Pasal\s+\d+\s*[:\-–]?\s*)([^\n]{4,60})",
    r"(?:Prosedur|SOP|Alur)\s+([^\n\.,]{4,60})",
    r"(?:Kegiatan|Program)\s+([^\n\.,]{4,60})",
]


def detect_actors(text: str) -> List[str]:
    """Extract known role/actor mentions from text chunk."""
    found = []
    for pattern in ACTOR_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            cleaned = m.strip()
            if cleaned and cleaned not in found:
                found.append(cleaned)
    return found


def detect_process_name(text: str, fallback_title: str) -> str:
    """Attempt to detect specific process/chapter name or fall back to document title."""
    for pattern in PROCESS_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            proc = match.group(1).strip()
            # Clean trailing whitespace / symbols
            proc = re.sub(r"[\s\:\-]+$", "", proc)
            if len(proc) >= 3:
                return proc
    return fallback_title


def enrich_chunk_metadata(
    raw_metadata: Dict[str, Any],
    chunk_text: str,
    chunk_index: int,
    default_process_name: Optional[str] = None,
    default_category: str = "business_process",
    default_document_type: str = "sop",
) -> Dict[str, Any]:
    """
    Enrich raw chunk metadata to guarantee all 7 required fields are present and populated.
    """
    # 1. Document Name
    doc_name = (
        raw_metadata.get("document_name")
        or raw_metadata.get("source")
        or raw_metadata.get("file_name")
        or "Unknown_Document"
    )
    if "/" in doc_name or "\\" in doc_name:
        import os
        doc_name = os.path.basename(doc_name)

    # 2. Document Type
    doc_type = raw_metadata.get("document_type")
    if not doc_type:
        lower_name = doc_name.lower()
        if "bpmn" in lower_name or doc_name.endswith(".bpmn"):
            doc_type = "bpmn"
        elif "sop" in lower_name:
            doc_type = "sop"
        elif "peraturan" in lower_name or "sk" in lower_name:
            doc_type = "regulation"
        else:
            doc_type = default_document_type

    # 3. Category
    category = raw_metadata.get("category", default_category)

    # 4. Process Name
    clean_stem = doc_name.rsplit(".", 1)[0]
    fallback_proc = default_process_name or clean_stem
    process_name = raw_metadata.get("process_name") or detect_process_name(chunk_text, fallback_proc)

    # 5. Actor
    actor = raw_metadata.get("actor")
    if not actor:
        found_actors = detect_actors(chunk_text)
        actor = ", ".join(found_actors[:2]) if found_actors else "Semua Pihak / Terkait"

    # 6. Page Number
    page_number = raw_metadata.get("page", raw_metadata.get("page_number", "N/A"))
    if isinstance(page_number, int):
        # 1-indexed for human readability
        page_number = page_number + 1

    # 7. Chunk Index
    idx = raw_metadata.get("chunk_index", chunk_index)

    # Department
    department = raw_metadata.get("department", "Panitia ICICoS / FSM")

    # Validate with ChunkMetadata model
    validated = ChunkMetadata(
        document_name=str(doc_name),
        document_type=str(doc_type),
        category=str(category),
        process_name=str(process_name),
        department=str(department),
        actor=str(actor),
        page_number=str(page_number),
        chunk_index=int(idx),
    )

    return validated.to_dict()
