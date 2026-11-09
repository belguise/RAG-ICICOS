"""
Deterministic Forensic Regression Tests for Critic Confidence Threshold Gate.
Validates the manuscript certification definition:
  AuditPass = (all 5 critical dimensions not FAILED) AND (confidence_score >= CRITIC_CONFIDENCE_THRESHOLD)

Mandatory test coverage (Section B4):
1. Confidence = 0.30, all dimensions passed -> REJECTED (is_valid=False)
2. Confidence = 0.849, all dimensions passed -> REJECTED (is_valid=False)
3. Confidence = 0.850, all dimensions passed -> VERIFIED (is_valid=True)
4. Confidence = 0.950, all dimensions passed -> VERIFIED (is_valid=True)
5. Confidence = 0.950 with one FAILED dimension -> REJECTED (is_valid=False)
6. Invalid or missing confidence -> REJECTED (fail-safe)
7. Provider exception -> VERIFICATION_ERROR (never VERIFIED)
8. Repair status propagation through generation chain (repair succeeds)
9. Maximum repair-attempt handling (budget exhausted -> UNVERIFIED)
"""

import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.documents import Document

from agentic_rag_sop.agents.critic import (
    CriticVerificationAgent,
    CriticDecision,
    CheckStatus,
    VerificationStatus,
)
from agentic_rag_sop.generation.chain import SOPRAGChain
from agentic_rag_sop.fusion.context_fusion import FusedContext


class FakeChatModel(BaseChatModel):
    """LangChain-compliant fake chat model for deterministic unit testing."""
    responses: List[str] = []
    raise_exception: Optional[Exception] = None
    call_count: int = 0

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_count += 1
        if self.raise_exception:
            raise self.raise_exception

        if not self.responses:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

        idx = min(self.call_count - 1, len(self.responses) - 1)
        resp = self.responses[idx]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=resp))])

    @property
    def _llm_type(self) -> str:
        return "fake_chat_model"


def _make_critic_json(
    is_valid: bool = True,
    confidence: Any = 0.95,
    factual_status: CheckStatus = CheckStatus.PASSED,
    domain_status: CheckStatus = CheckStatus.PASSED,
    workflow_status: CheckStatus = CheckStatus.PASSED,
    sop_status: CheckStatus = CheckStatus.PASSED,
    completeness_status: CheckStatus = CheckStatus.PASSED,
    summary: str = "Audit completed.",
    instructions: Optional[str] = None,
) -> str:
    """Helper generating valid JSON response string from Critic LLM."""
    payload = {
        "is_valid": is_valid,
        "confidence_score": confidence,
        "factual_grounding": {"status": factual_status.value, "reason": "Factual check", "evidence_quote": None},
        "domain_compliance": {"status": domain_status.value, "reason": "Domain check", "evidence_quote": None},
        "workflow_correctness": {"status": workflow_status.value, "reason": "Workflow check", "evidence_quote": None},
        "sop_correctness": {"status": sop_status.value, "reason": "SOP check", "evidence_quote": None},
        "answer_completeness": {"status": completeness_status.value, "reason": "Completeness check", "evidence_quote": None},
        "critique_summary": summary,
        "suggested_revision_instructions": instructions,
    }
    return json.dumps(payload)


class TestCriticThresholdRegression(unittest.TestCase):
    """Regression test suite for Critic certification confidence gate."""

    def setUp(self):
        self.question = "Bagaimana alur pengajuan insentif publikasi ICICoS dan siapa yang mencairkan dana?"
        self.context = (
            "SOP ICICoS: Pemakalah mengunggah naskah dan bukti pembayaran. "
            "Sekretariat memeriksa kelengkapan berkas. "
            "Jika berkas dinyatakan sesuai, bendahara mencairkan dana insentif sebesar Rp 2.500.000."
        )
        self.valid_draft = (
            "Berdasarkan SOP ICICoS, pemakalah mengunggah naskah dan bukti pembayaran, "
            "kemudian sekretariat memeriksa kelengkapan berkas. Apabila sesuai, "
            "bendahara mencairkan dana insentif sebesar Rp 2.500.000."
        )

    def test_01_confidence_030_all_passed_rejected(self):
        """1. Verify confidence = 0.30 with all dimensions PASSED is REJECTED."""
        json_resp = _make_critic_json(confidence=0.30)
        critic = CriticVerificationAgent(llm=FakeChatModel(responses=[json_resp]), confidence_threshold=0.85)
        decision = critic.verify(answer=self.valid_draft, context=self.context, question=self.question)
        self.assertFalse(decision.is_valid)
        self.assertEqual(decision.verification_status, VerificationStatus.REJECTED)
        self.assertIn("confidence_threshold", decision.failed_checks)

    def test_02_confidence_0849_all_passed_rejected(self):
        """2. Verify confidence = 0.849 (< 0.850 boundary) is REJECTED."""
        json_resp = _make_critic_json(confidence=0.849)
        critic = CriticVerificationAgent(llm=FakeChatModel(responses=[json_resp]), confidence_threshold=0.85)
        decision = critic.verify(answer=self.valid_draft, context=self.context, question=self.question)
        self.assertFalse(decision.is_valid)
        self.assertEqual(decision.verification_status, VerificationStatus.REJECTED)
        self.assertIn("confidence_threshold", decision.failed_checks)

    def test_03_confidence_0850_all_passed_verified(self):
        """3. Verify confidence = 0.850 (exact boundary) is VERIFIED."""
        json_resp = _make_critic_json(confidence=0.850)
        critic = CriticVerificationAgent(llm=FakeChatModel(responses=[json_resp]), confidence_threshold=0.85)
        decision = critic.verify(answer=self.valid_draft, context=self.context, question=self.question)
        self.assertTrue(decision.is_valid)
        self.assertEqual(decision.verification_status, VerificationStatus.VERIFIED)
        self.assertEqual(len(decision.failed_checks), 0)

    def test_04_confidence_0950_all_passed_verified(self):
        """4. Verify confidence = 0.950 with all dimensions PASSED is VERIFIED."""
        json_resp = _make_critic_json(confidence=0.950)
        critic = CriticVerificationAgent(llm=FakeChatModel(responses=[json_resp]), confidence_threshold=0.85)
        decision = critic.verify(answer=self.valid_draft, context=self.context, question=self.question)
        self.assertTrue(decision.is_valid)
        self.assertEqual(decision.verification_status, VerificationStatus.VERIFIED)

    def test_05_confidence_0950_one_failed_dimension_rejected(self):
        """5. Verify confidence = 0.950 with one FAILED dimension is REJECTED."""
        json_resp = _make_critic_json(
            confidence=0.950,
            workflow_status=CheckStatus.FAILED,
            instructions="Koreksi urutan transisi pada alur proses."
        )
        critic = CriticVerificationAgent(llm=FakeChatModel(responses=[json_resp]), confidence_threshold=0.85)
        decision = critic.verify(answer=self.valid_draft, context=self.context, question=self.question)
        self.assertFalse(decision.is_valid)
        self.assertEqual(decision.verification_status, VerificationStatus.REJECTED)
        self.assertIn("workflow_correctness", decision.failed_checks)

    def test_06_invalid_or_missing_confidence(self):
        """6. Verify missing or non-numeric confidence fails safely."""
        # Missing confidence (None)
        json_resp1 = _make_critic_json(confidence=None)
        critic1 = CriticVerificationAgent(llm=FakeChatModel(responses=[json_resp1]), confidence_threshold=0.85)
        dec1 = critic1.verify(answer=self.valid_draft, context=self.context, question=self.question)
        self.assertFalse(dec1.is_valid)
        self.assertEqual(dec1.verification_status, VerificationStatus.REJECTED)

        # String non-numeric confidence
        json_resp2 = _make_critic_json(confidence="high")
        critic2 = CriticVerificationAgent(llm=FakeChatModel(responses=[json_resp2]), confidence_threshold=0.85)
        dec2 = critic2.verify(answer=self.valid_draft, context=self.context, question=self.question)
        self.assertFalse(dec2.is_valid)
        self.assertEqual(dec2.verification_status, VerificationStatus.REJECTED)

    def test_07_provider_exception(self):
        """7. Verify provider exception returns VERIFICATION_ERROR and is_valid=False."""
        failing_llm = FakeChatModel()
        failing_llm.raise_exception = RuntimeError("Google API 503 Service Unavailable")
        critic = CriticVerificationAgent(llm=failing_llm, confidence_threshold=0.85)
        decision = critic.verify(answer=self.valid_draft, context=self.context, question=self.question)
        self.assertFalse(decision.is_valid)
        self.assertEqual(decision.verification_status, VerificationStatus.VERIFICATION_ERROR)
        self.assertEqual(decision.confidence_score, 0.0)

    def test_08_repair_status_propagation_through_generation_chain(self):
        """8. Verify self-repair loop triggers on low confidence and certifies on high confidence."""
        # Generator: draft 1, revised draft 2
        gen_llm = FakeChatModel(responses=["Draft 1", "Revised certified Draft 2"])

        # Critic: Pass 0 has confidence 0.70 (< 0.85) -> REJECTED
        # Pass 1 has confidence 0.92 (>= 0.85) -> VERIFIED
        c_pass0 = _make_critic_json(confidence=0.70, instructions="Tingkatkan rincian bukti SOP.")
        c_pass1 = _make_critic_json(confidence=0.92, summary="Semua checks passed.")
        critic_llm = FakeChatModel(responses=[c_pass0, c_pass1])

        critic_agent = CriticVerificationAgent(llm=critic_llm, confidence_threshold=0.85)
        chain = SOPRAGChain(llm=gen_llm, critic_agent=critic_agent, max_repair_iterations=2)

        # Mock retrieval and fusion
        mock_doc = Document(page_content=self.context, metadata={"document_name": "SOP_ICICOS.pdf"})
        chain.sop_retriever.retrieve = lambda *args, **kwargs: [mock_doc]
        chain.bpmn_retriever.retrieve = lambda *args, **kwargs: []
        chain.fusion_layer.fuse = lambda *args, **kwargs: FusedContext(
            query=self.question,
            synthesized_context=self.context,
            sop_context=self.context,
            bpmn_context="",
            sources=[mock_doc],
        )

        res = chain.invoke(question=self.question, use_critic=True)
        self.assertTrue(res["was_revised"])
        self.assertEqual(res["repair_iterations"], 1)
        self.assertEqual(res["final_verification_status"], VerificationStatus.VERIFIED.value)
        self.assertTrue(res["critic_decision"].is_valid)
        self.assertEqual(res["answer"], "Revised certified Draft 2")

    def test_09_maximum_repair_attempt_handling(self):
        """9. Verify exhausting repair budget without passing results in UNVERIFIED."""
        gen_llm = FakeChatModel(responses=["Draft 1", "Draft 2", "Draft 3"])
        # Both initial and revision audits fail
        c_fail0 = _make_critic_json(is_valid=False, confidence=0.50, instructions="Koreksi nama aktor.")
        c_fail1 = _make_critic_json(is_valid=False, confidence=0.60, instructions="Masih kurang akurat.")
        critic_llm = FakeChatModel(responses=[c_fail0, c_fail1])

        critic_agent = CriticVerificationAgent(llm=critic_llm, confidence_threshold=0.85)
        chain = SOPRAGChain(llm=gen_llm, critic_agent=critic_agent, max_repair_iterations=1)

        mock_doc = Document(page_content=self.context, metadata={"document_name": "SOP_ICICOS.pdf"})
        chain.sop_retriever.retrieve = lambda *args, **kwargs: [mock_doc]
        chain.bpmn_retriever.retrieve = lambda *args, **kwargs: []
        chain.fusion_layer.fuse = lambda *args, **kwargs: FusedContext(
            query=self.question,
            synthesized_context=self.context,
            sop_context=self.context,
            bpmn_context="",
            sources=[mock_doc],
        )

        res = chain.invoke(question=self.question, use_critic=True)
        self.assertTrue(res["was_revised"])
        self.assertEqual(res["repair_iterations"], 1)
        self.assertEqual(res["final_verification_status"], VerificationStatus.UNVERIFIED.value)
        self.assertFalse(res["critic_decision"].is_valid)


if __name__ == "__main__":
    unittest.main()
