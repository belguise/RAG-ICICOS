"""
Critic Verification Agent.
Phase 5: Evaluates generated answers for factual grounding, strict ICICoS domain
compliance, BPMN workflow consistency, and SOP requirement accuracy.
Triggers an iterative revision loop when hallucinations or invalid claims are detected.

Guarantees:
- Strict verification statuses: VERIFIED, REJECTED, UNVERIFIED, VERIFICATION_ERROR.
- Fail-safe exception handling: Errors NEVER falsely pass or return is_valid=True.
- Full 5-dimensional compliance auditing (Factual, Domain, Workflow, SOP, Completeness).
"""

from enum import Enum
import json
import logging
import re
import traceback
from typing import Any, Dict, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from agentic_rag_sop.core.llm import get_llm

logger = logging.getLogger("CriticVerificationAgent")


class CheckStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class VerificationStatus(str, Enum):
    """Explicit system verification status reflecting the state of audit."""
    VERIFIED = "VERIFIED"                     # All 5 dimensions passed with high confidence
    REJECTED = "REJECTED"                     # One or more checks failed; actionable critique provided
    UNVERIFIED = "UNVERIFIED"                 # Reached max repair attempts without satisfying critic
    VERIFICATION_ERROR = "VERIFICATION_ERROR" # Infrastructure/LLM error occurred; answer uncertified


class CriticCheckItem(BaseModel):
    """Evaluation result for a single verification check."""
    status: CheckStatus = CheckStatus.PASSED
    reason: str = Field(description="Explanation of the check result")
    evidence_quote: Optional[str] = Field(default=None, description="Direct quote from context or answer")


class CriticDecision(BaseModel):
    """Structured decision output produced by CriticVerificationAgent."""
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.REJECTED,
        description="Explicit system verification status (VERIFIED, REJECTED, UNVERIFIED, VERIFICATION_ERROR)"
    )
    is_valid: bool = Field(
        description="True strictly if verification_status is VERIFIED and all critical checks passed"
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
    failed_checks: List[str] = Field(
        default_factory=list,
        description="List of verification dimension names that failed evaluation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostic execution metadata"
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
     * Logika percabangan keputusan / gateway (misal: jika disetujui vs jika ditolak/tidak lengkap).
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
        answer: Optional[str],
        fused_context: Any,
        question: Optional[str],
    ) -> Optional[CriticDecision]:
        """
        Fast heuristic checks to catch immediate boundary violations, empty text,
        domain intrusion (e.g. SK Dekan), and false refusals.
        """
        # 1. Empty Answer Check
        if not answer or not isinstance(answer, str) or not answer.strip():
            return CriticDecision(
                verification_status=VerificationStatus.REJECTED,
                is_valid=False,
                confidence_score=0.0,
                factual_grounding=CriticCheckItem(
                    status=CheckStatus.FAILED,
                    reason="Jawaban kosong atau tidak valid.",
                ),
                domain_compliance=CriticCheckItem(status=CheckStatus.FAILED, reason="Jawaban kosong."),
                workflow_correctness=CriticCheckItem(status=CheckStatus.FAILED, reason="Jawaban kosong."),
                sop_correctness=CriticCheckItem(status=CheckStatus.FAILED, reason="Jawaban kosong."),
                answer_completeness=CriticCheckItem(status=CheckStatus.FAILED, reason="Jawaban kosong."),
                critique_summary="Penolakan instan: Draf jawaban kosong.",
                suggested_revision_instructions="Hasilkan jawaban substantif berdasarkan konteks yang tersedia.",
                failed_checks=["factual_grounding", "domain_compliance", "workflow_correctness", "sop_correctness", "answer_completeness"],
            )

        # 2. Empty Context Check
        context_str = str(fused_context) if fused_context else ""
        if not context_str.strip() or "Tidak ada konteks dokumen" in context_str:
            return CriticDecision(
                verification_status=VerificationStatus.REJECTED,
                is_valid=False,
                confidence_score=0.0,
                factual_grounding=CriticCheckItem(
                    status=CheckStatus.FAILED,
                    reason="Tidak ada konteks eksternal yang tersedia untuk memverifikasi klaim (zero context).",
                ),
                domain_compliance=CriticCheckItem(status=CheckStatus.PASSED, reason="Konteks kosong."),
                workflow_correctness=CriticCheckItem(status=CheckStatus.PASSED, reason="Konteks kosong."),
                sop_correctness=CriticCheckItem(status=CheckStatus.PASSED, reason="Konteks kosong."),
                answer_completeness=CriticCheckItem(status=CheckStatus.FAILED, reason="Konteks tidak memuat bukti."),
                critique_summary="Penolakan instan: Konteks pengetahuan kosong sehingga jawaban tidak dapat diground.",
                suggested_revision_instructions="Tunjukkan bahwa bukti tidak tersedia pada pangkalan data operasional.",
                failed_checks=["factual_grounding", "answer_completeness"],
            )

        ans_lower = answer.lower()

        # 3. Immediate Domain Violation Check: Generic campus bureaucracy not in ICICoS scope
        forbidden_domain_markers = ["sk dekan", "senat fakultas", "senat akademik", "biro rektorat", "rektor unpad", "dekanat"]
        for marker in forbidden_domain_markers:
            if marker in ans_lower:
                return CriticDecision(
                    verification_status=VerificationStatus.REJECTED,
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
                    failed_checks=["factual_grounding", "domain_compliance", "workflow_correctness"],
                )

        # 4. Immediate False Refusal Check
        refusal_patterns = [
            "tidak tercantum", "tidak ditemukan", "tidak terdapat",
            "tidak tersedia", "tidak disebutkan", "informasi tidak ada",
            "tidak ada informasi", "tidak memiliki informasi", "tidak memuat informasi",
        ]
        if any(p in ans_lower for p in refusal_patterns) and len(context_str.strip()) > 150:
            q_clean = question or ""
            q_words = set(re.findall(r"\w{4,}", q_clean.lower()))
            ctx_lower = context_str.lower()
            overlap = [w for w in q_words if w in ctx_lower]
            if len(overlap) >= 1:
                return CriticDecision(
                    verification_status=VerificationStatus.REJECTED,
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
                    failed_checks=["factual_grounding", "answer_completeness"],
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
        Returns a structured CriticDecision with explicit VerificationStatus.
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
            if hasattr(response, "content") and not isinstance(response.content, (re.Pattern, type(re))):
                content = response.content
            else:
                content = str(response)

            if isinstance(content, list):
                text_parts = [item.get("text", "") if isinstance(item, dict) else str(item) for item in content]
                content = " ".join(text_parts)
            elif not isinstance(content, str):
                content = str(content)

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                data = json.loads(content)

            decision = CriticDecision.model_validate(data)

            # Determine failed checks across all 5 verification dimensions
            failed_dims: List[str] = []
            for dim_name, check_item in [
                ("factual_grounding", decision.factual_grounding),
                ("domain_compliance", decision.domain_compliance),
                ("workflow_correctness", decision.workflow_correctness),
                ("sop_correctness", decision.sop_correctness),
                ("answer_completeness", decision.answer_completeness),
            ]:
                if check_item.status == CheckStatus.FAILED:
                    failed_dims.append(dim_name)

            decision.failed_checks = failed_dims

            # Strict Safety Gate: is_valid is True ONLY if NO dimension failed
            if failed_dims:
                decision.is_valid = False
                decision.verification_status = VerificationStatus.REJECTED
            else:
                decision.is_valid = True
                decision.verification_status = VerificationStatus.VERIFIED

            return decision

        except Exception as e:
            logger.error(f"Error during Critic verification: {str(e)}\n{traceback.format_exc()}")
            # Fail-Safe Error Handling: NEVER mark as valid or verified on error
            return CriticDecision(
                verification_status=VerificationStatus.VERIFICATION_ERROR,
                is_valid=False,
                confidence_score=0.0,
                factual_grounding=CriticCheckItem(
                    status=CheckStatus.FAILED,
                    reason=f"Audit failed due to exception: {type(e).__name__}: {str(e)}",
                ),
                domain_compliance=CriticCheckItem(status=CheckStatus.FAILED, reason="Audit incomplete due to error."),
                workflow_correctness=CriticCheckItem(status=CheckStatus.FAILED, reason="Audit incomplete due to error."),
                sop_correctness=CriticCheckItem(status=CheckStatus.FAILED, reason="Audit incomplete due to error."),
                answer_completeness=CriticCheckItem(status=CheckStatus.FAILED, reason="Audit incomplete due to error."),
                critique_summary=f"Critic verification raised an infrastructure error: {str(e)}",
                suggested_revision_instructions=None,
                failed_checks=["factual_grounding", "domain_compliance", "workflow_correctness", "sop_correctness", "answer_completeness"],
                metadata={"error": str(e), "error_type": type(e).__name__},
            )
