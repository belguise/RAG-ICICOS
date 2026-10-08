"""
Router Agent.
Phase 2.2 Agentic Layer: Classifies query intent and dynamically routes the query
to the appropriate knowledge source (SOP, BPMN, or BOTH).
"""

import json
import re
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from agentic_rag_sop.core.llm import get_llm


class RouterDecision(BaseModel):
    """Structured decision produced by Router Agent with weighted multi-source routing."""
    intent: Literal[
        "procedural_question",
        "workflow_question",
        "document_requirement",
        "payment_question",
        "general_information",
    ] = Field(description="Classified intent of the user query")
    knowledge_source: Literal["SOP", "BPMN", "BOTH"] = Field(
        description="Primary target knowledge source to handle the query"
    )
    secondary_source: Optional[Literal["SOP", "BPMN"]] = Field(
        default=None,
        description="Secondary supporting knowledge source"
    )
    reason: str = Field(description="Brief explanation justifying the routing decision")


ROUTER_PROMPT = """Anda adalah Router Agent AI untuk sistem Manajemen Pengetahuan SOP dan BPMN ICICoS.

FILOSOFI ROUTING PERUSAHAAN (ENTERPRISE ROUTING):
Proses operasional ICICoS membutuhkan SOP dan BPMN secara berdampingan.
BPMN menyediakan urutan alur proses (A -> B -> C), aktor, dan titik keputusan.
SOP menyediakan rincian aturan, syarat berkas lampiran, SLA/batas waktu, dan ketentuan finansial.
Oleh karena itu, satu sumber menjadi prioritas utama (knowledge_source), dan sumber lainnya menjadi pendukung (secondary_source), atau KEDUANYA (BOTH) jika pertanyaan kompleks.

TUGAS:
Analisis pertanyaan pengguna, tentukan:
1. INTENT: maksud utama pertanyaan.
2. KNOWLEDGE_SOURCE (Primary): "SOP", "BPMN", atau "BOTH".
3. SECONDARY_SOURCE: "BPMN" (jika primary SOP), "SOP" (jika primary BPMN), atau null (jika BOTH).
4. REASON: alasan singkat keputusan routing.

PANDUAN KLASIFIKASI:
- Dokumen / Syarat Berkas (misal: "Dokumen apa yang diperlukan?"):
  -> knowledge_source: "SOP", secondary_source: "BPMN"
- Alur Kerja / Urutan / Handoff Aktor (misal: "Siapa melakukan langkah berikutnya?"):
  -> knowledge_source: "BPMN", secondary_source: "SOP"
- Percabangan Keputusan / Hybrid (misal: "Apa yang terjadi jika ditolak?", "Jelaskan alur pembayaran termasuk batas waktunya"):
  -> knowledge_source: "BOTH", secondary_source: null

---
Pertanyaan: {query}

Berikan respon HANYA dalam format JSON yang valid persis seperti skema berikut tanpa teks tambahan:
{{
  "intent": "procedural_question | workflow_question | document_requirement | payment_question | general_information",
  "knowledge_source": "SOP | BPMN | BOTH",
  "secondary_source": "SOP | BPMN | null",
  "reason": "penjelasan singkat keputusan routing"
}}"""


class RouterAgent:
    """Agent that classifies intent and directs queries to optimal retrievers."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm or get_llm()
        self.prompt = PromptTemplate.from_template(ROUTER_PROMPT)
        self._chain = self.prompt | self.llm

    def _classify_heuristics(self, query: str) -> Optional[RouterDecision]:
        """Fast, deterministic rule-based pre-routing with complementary secondary support."""
        q = query.lower()

        # Check hybrid indicator: asks for flow/sequence AND administrative rules/time limits/docs
        has_flow_cue = any(w in q for w in ["alur", "urutan", "langkah", "tahap", "proses", "setelah", "sebelum"])
        has_rule_cue = any(w in q for w in ["batas waktu", "deadline", "lampiran", "syarat", "dokumen", "sla", "biaya", "tarif"])
        has_gateway_cue = any(w in q for w in ["jika", "apabila", "ditolak", "tidak sesuai", "gagal", "apa yang terjadi"])

        # 1. Hybrid or Gateway questions -> BOTH (Dual Primary)
        if (has_flow_cue and has_rule_cue) or has_gateway_cue or "hybrid" in q:
            return RouterDecision(
                intent="general_information" if not has_gateway_cue else "workflow_question",
                knowledge_source="BOTH",
                secondary_source=None,
                reason="Question requires unified BPMN sequence and SOP administrative criteria/gateway handling",
            )

        # 2. Document Requirements / Attachments -> SOP priority + BPMN support
        doc_keywords = ["lampiran", "syarat dokumen", "kelengkapan dokumen", "dokumen wajib", "berkas", "nota", "rundown", "undangan rapat", "materi presentasi"]
        if any(w in q for w in doc_keywords):
            return RouterDecision(
                intent="document_requirement",
                knowledge_source="SOP",
                secondary_source="BPMN",
                reason="Document requirements require SOP checklist with BPMN context support",
            )

        # 3. Pure Workflow / Sequence / Approval -> BPMN priority + SOP support
        if has_flow_cue or any(w in q for w in ["siapa yang", "siapa pihak", "approval", "persetujuan", "berikutnya", "handoff"]):
            return RouterDecision(
                intent="workflow_question",
                knowledge_source="BPMN",
                secondary_source="SOP",
                reason="Workflow inquiry requires BPMN process sequencing with SOP textual support",
            )

        # 4. Payment / Bank / Virtual Account questions -> SOP priority + BPMN support
        payment_keywords = ["bayar", "pembayaran", "transfer", "mbanking", "virtual account", "va author", "va sponsor", "rekening bni"]
        if any(w in q for w in payment_keywords):
            return RouterDecision(
                intent="payment_question",
                knowledge_source="SOP",
                secondary_source="BPMN",
                reason="Payment questions require SOP banking instructions with BPMN workflow support",
            )

        # 5. Standard SOP policy questions -> SOP priority + BPMN support
        if any(w in q for w in ["sop", "aturan", "kebijakan", "ketentuan", "timeout"]):
            return RouterDecision(
                intent="procedural_question",
                knowledge_source="SOP",
                secondary_source="BPMN",
                reason="Procedural policies require SOP rules with BPMN process support",
            )

        return None

    def route(self, query: str) -> RouterDecision:
        """
        Classify intent and select knowledge source for the given query.
        Uses fast heuristic matching first, with Gemini LLM classification as primary fallback.
        """
        clean_query = query.strip()
        if not clean_query:
            return RouterDecision(
                intent="general_information",
                knowledge_source="BOTH",
                secondary_source=None,
                reason="Empty query defaulted to BOTH",
            )

        # Fast deterministic classification
        heuristic_decision = self._classify_heuristics(clean_query)
        if heuristic_decision:
            return heuristic_decision

        try:
            response = self._chain.invoke({"query": clean_query})
            content = response.content if hasattr(response, "content") else str(response)

            if isinstance(content, list):
                text_parts = [item.get("text", "") if isinstance(item, dict) else str(item) for item in content]
                content = " ".join(text_parts)

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                data = json.loads(content)

            intent = data.get("intent", "general_information").lower()
            if intent not in RouterDecision.model_fields["intent"].annotation.__args__:
                intent = "general_information"

            ks = data.get("knowledge_source", "BOTH").upper()
            if ks not in RouterDecision.model_fields["knowledge_source"].annotation.__args__:
                ks = "BOTH"

            sec = data.get("secondary_source")
            if sec:
                sec = sec.upper()
                if sec not in ["SOP", "BPMN"]:
                    sec = "BPMN" if ks == "SOP" else ("SOP" if ks == "BPMN" else None)
            else:
                sec = "BPMN" if ks == "SOP" else ("SOP" if ks == "BPMN" else None)

            reason = data.get("reason", "Determined by Gemini Router Agent")

            return RouterDecision(
                intent=intent,
                knowledge_source=ks,
                secondary_source=sec,
                reason=reason,
            )
        except Exception as e:
            # Safe fallback: always route to BOTH so neither source is missed
            return RouterDecision(
                intent="general_information",
                knowledge_source="BOTH",
                secondary_source=None,
                reason=f"Routing fallback to BOTH due to: {str(e)}",
            )
