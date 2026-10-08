"""
Context Fusion Layer Implementation.
Phase 4: Unifies SOP textual knowledge and BPMN process graph topology under
the strict ICICoS document domain, applying explicit conflict resolution rules.
"""

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from agentic_rag_sop.agents.router import RouterDecision


class FusedContext(BaseModel):
    """
    Structured container for unified SOP and BPMN knowledge.
    Enforces the ICICoS domain boundary and explicit priority resolutions.
    """
    process_domain: str = "ICICoS"
    query: str
    intent: Optional[str] = None
    knowledge_source: Optional[str] = None
    sop_context: str = ""
    bpmn_context: str = ""
    actors: List[str] = Field(default_factory=list)
    workflow_sequence: List[str] = Field(default_factory=list)
    procedural_rules: List[str] = Field(default_factory=list)
    decision_gateways: List[str] = Field(default_factory=list)
    conflict_resolutions: List[str] = Field(default_factory=list)
    synthesized_context: str = ""

    def to_formatted_prompt(self) -> str:
        """Return the synthesized context representation for the LLM."""
        return self.synthesized_context


class ContextFusionLayer:
    """
    Modular Context Fusion Layer.
    Synthesizes SOP administrative/textual rules and BPMN topological process graphs
    into an integrated, conflict-resolved context block.
    
    Conflict & Priority Rules:
    1. BPMN Priority: Step execution order, sequential workflow transitions,
       actor handoffs, and decision gateway branches.
    2. SOP Priority: Administrative document checklists, required attachments,
       legal requirements, and financial/fee policies.
    """

    def __init__(self, domain: str = "ICICoS"):
        self.domain = domain

    def _extract_actors(
        self,
        sop_documents: List[Document],
        bpmn_documents: List[Document]
    ) -> List[str]:
        """Consolidate and deduplicate actor roles from both document types."""
        actor_set = set()

        # From BPMN documents
        for doc in bpmn_documents:
            actor_meta = doc.metadata.get("actor", "")
            if actor_meta and actor_meta != "Terkait":
                for part in re.split(r"[,;/]+", actor_meta):
                    cleaned = part.strip()
                    if cleaned and cleaned.lower() not in ["panitia icicos", "general", "terkait"]:
                        actor_set.add(cleaned)

        # From SOP documents
        for doc in sop_documents:
            actor_meta = doc.metadata.get("actor", "")
            if actor_meta and actor_meta != "Terkait":
                for part in re.split(r"[,;/]+", actor_meta):
                    cleaned = part.strip()
                    if cleaned and cleaned.lower() not in ["umum", "panitia", "terkait"]:
                        actor_set.add(cleaned)

        # Fallback if empty
        if not actor_set:
            actor_set.add("Panitia ICICoS")

        return sorted(list(actor_set))

    def _extract_workflow_sequence(
        self,
        bpmn_documents: List[Document],
        sop_documents: List[Document]
    ) -> List[str]:
        """
        Extract chronological workflow sequence.
        BPMN has absolute priority for sequence order and transitions.
        """
        sequence_steps = []

        # 1. Primary: Extract from BPMN documents (Topological reasoning)
        for doc in bpmn_documents:
            text = doc.page_content
            # Extract lines from [Struktur Lengkap Alur Proses] or sequence flow
            in_steps = False
            for line in text.split("\n"):
                line_s = line.strip()
                if "Langkah-langkah Proses:" in line_s or "[Struktur Lengkap Alur Proses]" in line_s:
                    in_steps = True
                    continue
                if in_steps:
                    if line_s.startswith("• Titik Keputusan") or line_s.startswith("===") or line_s.startswith("["):
                        if not line_s.startswith("[Analisis"):
                            in_steps = False
                            continue
                    match = re.match(r"^(\d+\.|\-|\•)\s*(.+)$", line_s)
                    if match:
                        step_text = match.group(2).strip()
                        if step_text and step_text not in sequence_steps:
                            sequence_steps.append(step_text)

            # Also check transition lines: [Analisis Alur Transisi Langkah]
            for line in text.split("\n"):
                line_s = line.strip()
                if line_s.startswith("• Titik/Tahapan Saat Ini:") or line_s.startswith("  - "):
                    cleaned = line_s.lstrip("•- ").strip()
                    if cleaned and cleaned not in sequence_steps:
                        sequence_steps.append(cleaned)

        # 2. Secondary: If no BPMN steps available, fall back to SOP numbered items
        if not sequence_steps:
            for doc in sop_documents:
                for line in doc.page_content.split("\n"):
                    line_s = line.strip()
                    match = re.match(r"^(\d+\.|\([a-z0-9]\))\s*(.+)$", line_s)
                    if match and len(match.group(2).strip()) > 10:
                        step_text = match.group(2).strip()
                        if step_text not in sequence_steps and len(sequence_steps) < 8:
                            sequence_steps.append(step_text)

        return sequence_steps

    def _extract_procedural_rules(self, sop_documents: List[Document]) -> List[str]:
        """
        Extract administrative rules, required documents, and policies from SOP.
        SOP has absolute priority for document checklists and financial rules.
        """
        rules = []
        rule_keywords = [
            "wajib", "harus", "dokumen", "lampiran", "syarat", "rekening",
            "virtual account", "bukti", "spj", "honor", "loa", "biaya",
            "pembayaran", "batas waktu", "ketentuan", "formulir", "surat tugas"
        ]

        for doc in sop_documents:
            for line in doc.page_content.split("\n"):
                line_s = line.strip()
                if not line_s or len(line_s) < 15:
                    continue
                # If line contains strong procedural markers
                lower = line_s.lower()
                if any(kw in lower for kw in rule_keywords):
                    clean_line = line_s.lstrip("-•* 1234567890.)")
                    if clean_line and clean_line not in rules and len(rules) < 12:
                        rules.append(clean_line)

        return rules

    def _extract_decision_gateways(self, bpmn_documents: List[Document]) -> List[str]:
        """Extract decision gateways and branching conditions from BPMN graphs."""
        gateways = []
        for doc in bpmn_documents:
            text = doc.page_content
            in_gw = False
            for line in text.split("\n"):
                line_s = line.strip()
                if "[Analisis Percabangan Keputusan" in line_s or "Percabangan Keputusan:" in line_s:
                    in_gw = True
                    continue
                if in_gw:
                    if line_s.startswith("===") or (line_s.startswith("[") and not line_s.startswith("[Analisis")):
                        in_gw = False
                        continue
                    if line_s.startswith("• Titik Keputusan:") or line_s.startswith("  - Jalur"):
                        cleaned = line_s.lstrip("•- ").strip()
                        if cleaned and cleaned not in gateways:
                            gateways.append(cleaned)
        return gateways

    def _build_synthesized_markdown(
        self,
        query: str,
        actors: List[str],
        sequence: List[str],
        rules: List[str],
        gateways: List[str],
        resolutions: List[str],
        has_sop: bool,
        has_bpmn: bool,
    ) -> str:
        """Construct a cohesive, authoritative Markdown synthesis for the LLM."""
        sections = []

        sections.append(f"### DOMAIN INFORMASI: {self.domain} (IEEE Conference Management)")
        sections.append(
            "> [!NOTE]\n"
            "> Seluruh konteks ini berada dalam lingkup operasional kepanitiaan dan pelaksanaan ICICoS. "
            "Bukan merupakan regulasi umum fakultas/universitas ataupun SK Dekan non-kepanitiaan."
        )

        # 1. Actors
        if actors:
            sections.append(f"**Aktor / Pihak Terkait ({len(actors)} teridentifikasi):**\n" + ", ".join(actors))

        # 2. Workflow Sequence (BPMN Priority)
        sections.append("\n#### 1. Sintesis Alur Kerja & Urutan Transisi (Otoritas Utama: BPMN)")
        if sequence:
            for idx, step in enumerate(sequence, 1):
                sections.append(f"{idx}. {step}")
        else:
            sections.append("*(Detail urutan langkah spesifik tidak terinci pada proses ini)*")

        # 3. Decision Gateways
        if gateways:
            sections.append("\n#### 2. Titik Keputusan & Jalur Percabangan (Otoritas Utama: BPMN Gateways)")
            for gw in gateways:
                sections.append(f"- {gw}")

        # 4. Procedural & Document Requirements (SOP Priority)
        sections.append("\n#### 3. Ketentuan Prosedural & Syarat Dokumen (Otoritas Utama: Dokumen SOP)")
        if rules:
            for rule in rules:
                sections.append(f"- {rule}")
        else:
            sections.append("*(Tidak ada rincian berkas/syarat khusus tambahan yang terdeteksi)*")

        # 5. Conflict Resolution Directives
        sections.append("\n#### 4. Pedoman Resolusi Prioritas Konflik")
        if resolutions:
            for res in resolutions:
                sections.append(f"• {res}")
        else:
            sections.append("• Tidak ditemukan kontradiksi antara urutan alur BPMN dan syarat dokumen SOP.")

        return "\n".join(sections)

    def fuse(
        self,
        sop_documents: List[Document],
        bpmn_documents: List[Document],
        query: str,
        route_decision: Optional[RouterDecision] = None,
    ) -> FusedContext:
        """
        Execute context fusion between SOP and BPMN knowledge sources.
        """
        # Format raw contexts
        sop_blocks = []
        for idx, doc in enumerate(sop_documents, 1):
            meta = doc.metadata
            hdr = f"--- [SOP {idx}: {meta.get('document_name', 'Dokumen')} | Hal. {meta.get('page_number', 'N/A')} | Aktor: {meta.get('actor', 'Terkait')}] ---"
            sop_blocks.append(f"{hdr}\n{doc.page_content.strip()}")
        raw_sop_context = "\n\n".join(sop_blocks) if sop_blocks else "Tidak ada dokumen teks SOP yang ditemukan."

        bpmn_blocks = []
        for idx, doc in enumerate(bpmn_documents, 1):
            meta = doc.metadata
            hdr = f"--- [BPMN {idx}: {meta.get('process_name', 'Workflow')} | Kategori: {meta.get('category', 'ICICoS Process')}] ---"
            bpmn_blocks.append(f"{hdr}\n{doc.page_content.strip()}")
        raw_bpmn_context = "\n\n".join(bpmn_blocks) if bpmn_blocks else "Tidak ada graf proses BPMN yang ditemukan."

        # Extract elements
        actors = self._extract_actors(sop_documents, bpmn_documents)
        sequence = self._extract_workflow_sequence(bpmn_documents, sop_documents)
        rules = self._extract_procedural_rules(sop_documents)
        gateways = self._extract_decision_gateways(bpmn_documents)

        # Formulate conflict resolutions
        resolutions = []
        has_sop = bool(sop_documents)
        has_bpmn = bool(bpmn_documents)

        if has_sop and has_bpmn:
            resolutions.append(
                "Prioritas Alur Kerja: Jika terdapat perbedaan urutan kronologis antara deskripsi teks SOP dan alur BPMN, "
                "urutan dan handoff aktor pada BPMN adalah sumber kebenaran mutlak (source of truth)."
            )
            resolutions.append(
                "Prioritas Persyaratan Dokumen: Jika BPMN hanya menyebutkan nama aktivitas (misal: 'Menyiapkan SPJ'), "
                "maka rincian berkas lampiran, NPWP, LoA, SPTJM, dan invoice wajib merujuk secara ketat pada teks SOP."
            )
            resolutions.append(
                "Prioritas Finansial: Besaran nominal tarif pendaftaran, ketentuan potongan pajak, dan mekanisme transfer bank "
                "mengacu pada petunjuk teknis SOP."
            )
        elif has_bpmn and not has_sop:
            resolutions.append(
                "Hanya graf BPMN yang tersedia: Alur urutan dan aktor sepenuhnya valid, namun kelengkapan dokumen administratif "
                "harus dicantumkan sesuai yang tertera pada aktivitas BPMN tanpa mengasumsikan syarat di luar konteks."
            )
        elif has_sop and not has_bpmn:
            resolutions.append(
                "Hanya dokumen SOP yang tersedia: Penjelasan langkah mengacu pada teks prosedural SOP yang ditemukan."
            )

        synthesized = self._build_synthesized_markdown(
            query=query,
            actors=actors,
            sequence=sequence,
            rules=rules,
            gateways=gateways,
            resolutions=resolutions,
            has_sop=has_sop,
            has_bpmn=has_bpmn,
        )

        intent = route_decision.intent if route_decision else None
        knowledge_source = route_decision.knowledge_source if route_decision else ("BOTH" if (has_sop and has_bpmn) else ("SOP" if has_sop else "BPMN"))

        return FusedContext(
            process_domain=self.domain,
            query=query,
            intent=intent,
            knowledge_source=knowledge_source,
            sop_context=raw_sop_context,
            bpmn_context=raw_bpmn_context,
            actors=actors,
            workflow_sequence=sequence,
            procedural_rules=rules,
            decision_gateways=gateways,
            conflict_resolutions=resolutions,
            synthesized_context=synthesized,
        )
