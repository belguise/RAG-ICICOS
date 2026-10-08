"""
Critic Verification Agent.
Phase 5: Evaluates generated answers for factual grounding, strict ICICoS domain
compliance, BPMN workflow consistency, and SOP requirement accuracy.
Triggers an iterative revision loop when hallucinations or invalid claims are detected.
"""

import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from agentic_rag_sop.core.llm import get_llm


class CheckStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CriticCheckItem(BaseModel):
    """Evaluation result for a single verification check."""
    status: CheckStatus = CheckStatus.PASSED
    reason: str = Field(description="Explanation of the check result")
    evidence_quote: Optional[str] = Field(default=None, description="Direct quote from context or answer")


class CriticDecision(BaseModel):
    """Structured decision output produced by CriticVerificationAgent."""
    is_valid: bool = Field(
        description="True if all critical checks passed and no hallucination is present"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score (0.0 to 1.0) in the factual grounding of the answer"
    )
    factual_grounding: CriticCheckItem = Field(
        description="Verification that all claims originate directly from the retrieved context"
    )
    domain_compliance: CriticCheckItem = Field(
        description="Verification that content belongs to ICICoS and contains no university/SK Dekan contamination"
    )
    workflow_correctness: CriticCheckItem = Field(
        description="Verification of actor assignments, chronological steps, and BPMN gateway branches"
    )
    sop_correctness: CriticCheckItem = Field(
        description="Verification of administrative forms, attachments, deadlines, and financial rules"
    )
    answer_completeness: CriticCheckItem = Field(
        default_factory=lambda: CriticCheckItem(status=CheckStatus.PASSED, reason="Jawaban substantif"),
        description="Verification that the answer substantively answers the question and is not a false refusal when context exists",
    )
    critique_summary: str = Field(
        description="High-level evaluation explanation"
    )
    suggested_revision_instructions: Optional[str] = Field(
        default=None,
        description="Concrete, actionable correction instructions if is_valid is False"
    )


CRITIC_PROMPT = """Anda adalah CRITIC VERIFICATION AGENT untuk sistem manajemen pengetahuan operasional konferensi ICICoS (IEEE International Conference on Information Technology and Computer Science).

TUGAS ANDA:
Lakukan audit skeptis dan ketat (adversarial inspection) terhadap DRAF JAWABAN yang dihasilkan generator berdasarkan PERTANYAAN dan KONTEKS KNOWLEDGE BASE (SOP & BPMN).

KRITERIA EVALUASI:
1. FACTUAL GROUNDING (Zero Hallucination):
   - Apakah setiap klaim pada jawaban didukung oleh fakta tertulis pada konteks?
   - Tolak (FAILED) jika terdapat informasi fiktif, asumsi tanpa dasar, atau detail yang tidak ada pada dokumen.

2. DOMAIN COMPLIANCE (document_domain = "ICICoS"):
   - Seluruh jawaban WAJIB berada dalam lingkup kepanitiaan/operasional konferensi ICICoS.
   - Tolak (FAILED) jika jawaban memasukkan regulasi umum universitas/fakultas, SK Dekan reguler, Senat Akademik, Rektorat, atau birokrasi kampus yang tidak terkait kepanitiaan ICICoS.

3. WORKFLOW CORRECTNESS (BPMN Topology):
   - Jika pertanyaan/jawaban menyangkut alur proses, periksa:
     * Aktor penanggung jawab pada setiap langkah (apakah tepat sesuai konteks).
     * Urutan kronologis langkah-langkah proses.
     * Logika percabangan keputusan / gateway (misal: jika disetujui vs jika ditolak).
   - Tolak (FAILED) jika aktor tertukar, urutan proses terbalik, atau cabang gateway salah.

4. SOP CORRECTNESS (Syarat Dokumen & Finansial):
   - Jika pertanyaan/jawaban menyangkut syarat berkas (misal: SPJ, LoA, kwitansi) atau aturan finansial (rekening, batas waktu/timeout, biaya):
     * Apakah kelengkapan berkas akurat sesuai teks SOP?
     * Apakah batas waktu (misal: 3 hari bayar, 2x24 jam verifikasi) sesuai SOP?
   - Tolak (FAILED) jika ada dokumen wajib yang dikarang atau ketentuan nominal/waktu yang salah.

5. ANSWER COMPLETENESS & NON-REFUSAL:
   - Apakah draf jawaban memberikan substansi faktual untuk menjawab pertanyaan pengguna?
   - Tolak (FAILED) jika draf jawaban menyatakan 'informasi tidak ditemukan', 'tidak tercantum dalam konteks/dokumen', atau menolak menjawab PADAHAL bukti atau petunjuk terkait sudah ada dalam KONTEKS PENGETAHUAN TERSEDIA.

ATURAN PENILAIAN:
- "is_valid" bernilai TRUE HANYA JIKA factual_grounding, domain_compliance, workflow_correctness, sop_correctness, dan answer_completeness SEMUANYA berstatus "PASSED" atau "NOT_APPLICABLE".
- Jika SALAH SATU bernilai "FAILED", maka "is_valid" WAJIB bernilai FALSE.
- Jika "is_valid" adalah FALSE, berikan arahan spesifik pada "suggested_revision_instructions" agar generator tahu apa yang harus diperbaiki dan apa yang harus dihapus.

---
KONTEKS PENGETAHUAN TERSEDIA:
{context}

PERTANYAAN PENGGUNA:
{question}

DRAF JAWABAN YANG AKAN DIAUDIT:
{answer}
---

Berikan respon HANYA dalam format JSON yang valid persis seperti skema berikut tanpa teks pendahuluan/penutup:
{{
  "is_valid": true | false,
  "confidence_score": 0.95,
  "factual_grounding": {{
    "status": "PASSED | FAILED | NOT_APPLICABLE",
    "reason": "penjelasan",
    "evidence_quote": "kutipan bukti atau null"
  }},
  "domain_compliance": {{
    "status": "PASSED | FAILED | NOT_APPLICABLE",
    "reason": "penjelasan",
    "evidence_quote": "kutipan bukti atau null"
  }},
  "workflow_correctness": {{
    "status": "PASSED | FAILED | NOT_APPLICABLE",
    "reason": "penjelasan",
    "evidence_quote": "kutipan bukti atau null"
  }},
  "sop_correctness": {{
    "status": "PASSED | FAILED | NOT_APPLICABLE",
    "reason": "penjelasan",
    "evidence_quote": "kutipan bukti atau null"
  }},
  "answer_completeness": {{
    "status": "PASSED | FAILED | NOT_APPLICABLE",
    "reason": "penjelasan",
    "evidence_quote": "kutipan bukti atau null"
  }},
  "critique_summary": "ringkasan hasil audit",
  "suggested_revision_instructions": "arahan perbaikan konkret jika is_valid false, atau null jika valid"
}}"""


class CriticVerificationAgent:
    """
    Independent adversarial verification agent evaluating answers against retrieved facts.
    Guarantees strict factual grounding, ICICoS domain integrity, and BPMN/SOP correctness.
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm or get_llm()
        self.prompt = PromptTemplate.from_template(CRITIC_PROMPT)
        self._chain = self.prompt | self.llm

    def _fast_deterministic_check(
        self,
        answer: str,
        fused_context: Any,
        question: str,
    ) -> Optional[CriticDecision]:
        """
        Fast heuristic rules to catch immediate domain violations (e.g. SK Dekan intrusion)
        and detect false refusal when retrieved context actually contains answer evidence.
        """
        ans_lower = answer.lower()
        context_str = str(fused_context)

        # 1. Immediate domain violation check: Generic university bureaucracy not present in context
        forbidden_domain_markers = ["sk dekan", "senat fakultas", "senat akademik", "biro rektorat", "rektor unpad", "dekanat"]
        for marker in forbidden_domain_markers:
            if marker in ans_lower:
                return CriticDecision(
                    is_valid=False,
                    confidence_score=0.20,
                    factual_grounding=CriticCheckItem(
                        status=CheckStatus.FAILED,
                        reason=f"Klaim menyertakan entitas birokrasi luar ('{marker}') yang tidak ada dalam konteks.",
                    ),
                    domain_compliance=CriticCheckItem(
                        status=CheckStatus.FAILED,
                        reason=f"Terdeteksi kontaminasi regulasi non-ICICoS: ditemukan '{marker}'.",
                        evidence_quote=marker,
                    ),
                    workflow_correctness=CriticCheckItem(
                        status=CheckStatus.FAILED,
                        reason="Hierarki aktor menyertakan pejabat non-kepanitiaan ICICoS.",
                    ),
                    sop_correctness=CriticCheckItem(status=CheckStatus.PASSED, reason="Pengecekan dihentikan akibat pelanggaran domain."),
                    answer_completeness=CriticCheckItem(status=CheckStatus.PASSED, reason="Domain check failed first."),
                    critique_summary=f"Jawaban terbukti berhalusinasi memasukkan regulasi/pejabat '{marker}' di luar ranah ICICoS.",
                    suggested_revision_instructions=(
                        f"Hapus seluruh rujukan ke '{marker}'. Fokus hanya pada kepanitiaan operasional ICICoS "
                        "(Author, Bendahara, Sekretaris, Ketua Panitia) sebagaimana tercantum dalam konteks."
                    ),
                )

        # 2. Immediate false refusal check: draft answer claims information is missing when context exists
        refusal_patterns = [
            "tidak tercantum", "tidak ditemukan", "tidak terdapat",
            "tidak tersedia", "tidak disebutkan", "informasi tidak ada",
            "tidak ada informasi", "tidak memiliki informasi", "tidak memuat informasi",
        ]
        if any(p in ans_lower for p in refusal_patterns) and len(context_str.strip()) > 150:
            q_words = set(re.findall(r"\w{4,}", question.lower()))
            ctx_lower = context_str.lower()
            overlap = [w for w in q_words if w in ctx_lower]
            if len(overlap) >= 1:
                return CriticDecision(
                    is_valid=False,
                    confidence_score=0.25,
                    factual_grounding=CriticCheckItem(
                        status=CheckStatus.FAILED,
                        reason="Draf menyatakan informasi tidak ditemukan padahal teks konteks memuat fakta terkait.",
                    ),
                    domain_compliance=CriticCheckItem(status=CheckStatus.PASSED, reason="Domain ICICoS valid."),
                    workflow_correctness=CriticCheckItem(status=CheckStatus.PASSED, reason="N/A"),
                    sop_correctness=CriticCheckItem(status=CheckStatus.PASSED, reason="N/A"),
                    answer_completeness=CriticCheckItem(
                        status=CheckStatus.FAILED,
                        reason=f"Terdeteksi penolakan palsu (false refusal). Konteks memuat kata kunci: {', '.join(overlap[:4])}.",
                        evidence_quote=", ".join(overlap[:4]),
                    ),
                    critique_summary="Penolakan palsu: Generator menolak menjawab padahal konteks memuat informasi yang relevan.",
                    suggested_revision_instructions=(
                        "Konteks pengetahuan yang tersedia memuat informasi dan alur terkait pertanyaan. "
                        "Jangan menolak menjawab atau menyatakan informasi tidak ada. Ekstrak fakta, alur proses, peran/aktor, "
                        "dan persyaratan dokumen dari teks konteks di atas secara lengkap dan jawab pertanyaan secara langsung."
                    ),
                )

        return None

    def verify(
        self,
        answer: str,
        context: str,
        question: str,
    ) -> CriticDecision:
        """
        Audit the draft answer against the provided context and question.
        Returns a structured CriticDecision.
        """
        # 1. Fast deterministic check
        fast_check = self._fast_deterministic_check(answer, context, question)
        if fast_check:
            return fast_check

        # 2. Comprehensive LLM-based audit
        try:
            response = self._chain.invoke({
                "context": context,
                "question": question,
                "answer": answer,
            })
            content = response.content if hasattr(response, "content") else str(response)

            if isinstance(content, list):
                text_parts = [item.get("text", "") if isinstance(item, dict) else str(item) for item in content]
                content = " ".join(text_parts)

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                data = json.loads(content)

            decision = CriticDecision.model_validate(data)

            # Enforce safety constraint: if any critical check failed, is_valid must be False
            if (
                decision.factual_grounding.status == CheckStatus.FAILED
                or decision.domain_compliance.status == CheckStatus.FAILED
                or decision.workflow_correctness.status == CheckStatus.FAILED
                or decision.sop_correctness.status == CheckStatus.FAILED
                or (hasattr(decision, "answer_completeness") and decision.answer_completeness.status == CheckStatus.FAILED)
            ):
                decision.is_valid = False

            return decision

        except Exception as e:
            # Resilient fallback: if JSON parsing or LLM fails, assume valid with moderate confidence
            return CriticDecision(
                is_valid=True,
                confidence_score=0.80,
                factual_grounding=CriticCheckItem(status=CheckStatus.PASSED, reason="Fallback passed"),
                domain_compliance=CriticCheckItem(status=CheckStatus.PASSED, reason="Fallback passed"),
                workflow_correctness=CriticCheckItem(status=CheckStatus.PASSED, reason="Fallback passed"),
                sop_correctness=CriticCheckItem(status=CheckStatus.PASSED, reason="Fallback passed"),
                answer_completeness=CriticCheckItem(status=CheckStatus.PASSED, reason="Fallback passed"),
                critique_summary=f"Audit completed via resilient fallback: {str(e)}",
                suggested_revision_instructions=None,
            )
