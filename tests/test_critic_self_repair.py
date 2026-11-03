"""
Unit Test Suite for Critic Verification Agent and Bounded Self-Repair Loop.
Tests all 12 required scenarios using LangChain-compliant FakeChatModel fixtures.

Scenarios:
1. Correct draft accepted immediately (0 revisions, VERIFIED)
2. Unsupported draft rejected (REJECTED)
3. One successful revision (1 revision, re-verified as VERIFIED)
4. Two successful revisions (2 revisions, re-verified as VERIFIED)
5. Two failed revisions (2 revisions, terminated as UNVERIFIED)
6. Invalid Critic JSON (VERIFICATION_ERROR, is_valid=False)
7. LLM timeout / exception (VERIFICATION_ERROR, is_valid=False)
8. Empty retrieved context (Deterministic rejection, is_valid=False)
9. Incorrect gateway condition (workflow_correctness=FAILED)
10. Incorrect actor attribution (workflow_correctness=FAILED)
11. Incorrect process sequence (workflow_correctness=FAILED)
12. Unsupported revision content (re-verification rejects, bounded loop terminates)
"""

import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any, List, Optional

# Add project root and src to path
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
    CriticCheckItem,
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


def _make_critic_json_response(
    is_valid: bool = True,
    confidence: float = 0.95,
    factual_status: CheckStatus = CheckStatus.PASSED,
    domain_status: CheckStatus = CheckStatus.PASSED,
    workflow_status: CheckStatus = CheckStatus.PASSED,
    sop_status: CheckStatus = CheckStatus.PASSED,
    completeness_status: CheckStatus = CheckStatus.PASSED,
    summary: str = "Audit passed.",
    instructions: str = None,
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


class TestCriticVerificationAndSelfRepair(unittest.TestCase):
    """Forensic verification test suite for Critic and Bounded Self-Repair."""

    def setUp(self):
        """Set up standard question and context fixtures."""
        self.question = "Bagaimana alur pengajuan insentif publikasi ICICoS dan siapa yang mencairkan dana?"
        self.context = (
            "SOP ICICoS: Pemakalah mengunggah naskah dan bukti pembayaran. "
            "Sekretariat memeriksa kelengkapan berkas. "
            "Jika berkas dinyatakan sesuai, bendahara mencairkan dana insentif sebesar Rp 2.500.000."
        )

    def _create_mock_chain(self, gen_responses: List[str], critic_responses: List[str], max_repairs: int = 2):
        """Construct a SOPRAGChain with deterministic FakeChatModels."""
        gen_llm = FakeChatModel(responses=gen_responses)
        critic_llm = FakeChatModel(responses=critic_responses)
        critic_agent = CriticVerificationAgent(llm=critic_llm)

        chain = SOPRAGChain(
            llm=gen_llm,
            critic_agent=critic_agent,
            max_repair_iterations=max_repairs,
        )

        # Mock retrieval and fusion layers to return realistic static fixtures
        mock_doc = Document(page_content=self.context, metadata={"document_name": "SOP_ICICOS.pdf"})
        chain.sop_retriever.retrieve = lambda *args, **kwargs: [mock_doc]
        chain.bpmn_retriever.retrieve = lambda *args, **kwargs: []

        fake_fused = FusedContext(
            query=self.question,
            synthesized_context=self.context,
            sop_context=self.context,
            bpmn_context="",
            sources=[mock_doc],
        )
        chain.fusion_layer.fuse = lambda *args, **kwargs: fake_fused

        return chain

    # =========================================================================
    # Test 1: Correct Draft Accepted Immediately (0 revisions)
    # =========================================================================
    def test_01_correct_draft_accepted_immediately(self):
        """A compliant draft must be certified as VERIFIED on attempt 0 without triggering repair."""
        valid_json = _make_critic_json_response(is_valid=True, confidence=0.95)
        critic = CriticVerificationAgent(llm=FakeChatModel(responses=[valid_json]))

        draft = (
            "Pemakalah mengunggah naskah dan bukti bayar. Sekretariat memverifikasi berkas. "
            "Bila sesuai, bendahara mencairkan dana Rp 2.500.000."
        )
        decision = critic.verify(answer=draft, context=self.context, question=self.question)

        self.assertTrue(decision.is_valid)
        self.assertEqual(decision.verification_status, VerificationStatus.VERIFIED)
        self.assertEqual(len(decision.failed_checks), 0)
        self.assertIsNone(decision.suggested_revision_instructions)

    # =========================================================================
    # Test 2: Unsupported Draft Rejected (Fast check)
    # =========================================================================
    def test_02_unsupported_draft_rejected(self):
        """Draft containing out-of-domain intrusion must be immediately rejected by fast check."""
        critic = CriticVerificationAgent(llm=FakeChatModel(responses=["{}"]))

        draft = "Berdasarkan SK Dekan Fakultas, dana dicairkan melalui biro rektorat."
        decision = critic.verify(answer=draft, context=self.context, question=self.question)

        self.assertFalse(decision.is_valid)
        self.assertEqual(decision.verification_status, VerificationStatus.REJECTED)
        self.assertIn("domain_compliance", decision.failed_checks)
        self.assertIsNotNone(decision.suggested_revision_instructions)

    # =========================================================================
    # Test 3: One Successful Revision (1 revision pass)
    # =========================================================================
    def test_03_one_successful_revision(self):
        """Draft fails initial audit, generator revises once, and re-verification certifies as VERIFIED."""
        critic_responses = [
            _make_critic_json_response(
                is_valid=False,
                confidence=0.40,
                workflow_status=CheckStatus.FAILED,
                summary="Aktor salah: dekan bukan bendahara.",
                instructions="Ganti dekan dengan bendahara sesuai SOP.",
            ),
            _make_critic_json_response(
                is_valid=True,
                confidence=0.95,
                summary="Audit perbaikan berhasil. Aktor dan alur valid.",
            ),
        ]
        gen_responses = [
            "Draf Awal: Dekan mencairkan dana.",
            "Draf Revisi 1: Bendahara mencairkan dana.",
        ]

        chain = self._create_mock_chain(gen_responses, critic_responses, max_repairs=2)
        result = chain.invoke(question=self.question, use_router=False, use_query_agent=False)

        self.assertTrue(result["was_revised"])
        self.assertEqual(result["repair_iterations"], 1)
        self.assertEqual(result["final_verification_status"], "VERIFIED")
        self.assertEqual(result["answer"], "Draf Revisi 1: Bendahara mencairkan dana.")
        self.assertEqual(len(result["revision_trace"]), 2)
        self.assertEqual(result["revision_trace"][0]["action"], "initial_verification")
        self.assertEqual(result["revision_trace"][1]["action"], "revision_and_reverification")

    # =========================================================================
    # Test 4: Two Successful Revisions (2 revision passes)
    # =========================================================================
    def test_04_two_successful_revisions(self):
        """Draft fails twice, passes on second revision within N_max=2 budget."""
        critic_responses = [
            _make_critic_json_response(
                is_valid=False,
                confidence=0.30,
                workflow_status=CheckStatus.FAILED,
                summary="Aktor tertukar.",
                instructions="Perbaiki aktor.",
            ),
            _make_critic_json_response(
                is_valid=False,
                confidence=0.50,
                sop_status=CheckStatus.FAILED,
                summary="Nominal uang salah.",
                instructions="Ganti nominal menjadi 2.500.000.",
            ),
            _make_critic_json_response(
                is_valid=True,
                confidence=0.95,
                summary="Semua aspek terpenuhi.",
            ),
        ]
        gen_responses = [
            "Draf Awal",
            "Draf Revisi 1",
            "Draf Revisi 2 (Sempurna)",
        ]

        chain = self._create_mock_chain(gen_responses, critic_responses, max_repairs=2)
        result = chain.invoke(question=self.question, use_router=False, use_query_agent=False)

        self.assertTrue(result["was_revised"])
        self.assertEqual(result["repair_iterations"], 2)
        self.assertEqual(result["final_verification_status"], "VERIFIED")
        self.assertEqual(result["answer"], "Draf Revisi 2 (Sempurna)")
        self.assertEqual(len(result["revision_trace"]), 3)

    # =========================================================================
    # Test 5: Two Failed Revisions (Exhausted Budget)
    # =========================================================================
    def test_05_two_failed_revisions_budget_exhaustion(self):
        """When 2 revisions fail, loop must terminate gracefully with status UNVERIFIED and not loop infinitely."""
        critic_responses = [
            _make_critic_json_response(is_valid=False, confidence=0.30, factual_status=CheckStatus.FAILED, instructions="Perbaiki fakta."),
            _make_critic_json_response(is_valid=False, confidence=0.40, factual_status=CheckStatus.FAILED, instructions="Fakta masih salah."),
            _make_critic_json_response(is_valid=False, confidence=0.45, factual_status=CheckStatus.FAILED, instructions="Fakta tetap salah."),
        ]
        gen_responses = [
            "Draf Awal",
            "Draf Revisi 1 (Masih Salah)",
            "Draf Revisi 2 (Masih Salah)",
        ]

        chain = self._create_mock_chain(gen_responses, critic_responses, max_repairs=2)
        result = chain.invoke(question=self.question, use_router=False, use_query_agent=False)

        self.assertTrue(result["was_revised"])
        self.assertEqual(result["repair_iterations"], 2)
        self.assertEqual(result["final_verification_status"], "UNVERIFIED")
        self.assertFalse(result["critic_decision"].is_valid)

    # =========================================================================
    # Test 6: Invalid Critic JSON
    # =========================================================================
    def test_06_invalid_critic_json(self):
        """Malformed JSON from Critic LLM must yield VERIFICATION_ERROR and is_valid=False."""
        critic_llm = FakeChatModel(responses=["<html>502 Bad Gateway: Proxy Error</html>"])
        critic = CriticVerificationAgent(llm=critic_llm)

        decision = critic.verify(answer="Draf jawaban", context=self.context, question=self.question)

        self.assertFalse(decision.is_valid)
        self.assertEqual(decision.verification_status, VerificationStatus.VERIFICATION_ERROR)
        self.assertEqual(decision.confidence_score, 0.0)
        self.assertIn("factual_grounding", decision.failed_checks)

    # =========================================================================
    # Test 7: LLM Timeout / API Exception
    # =========================================================================
    def test_07_llm_timeout_exception(self):
        """API timeout or network exception must yield VERIFICATION_ERROR and is_valid=False."""
        critic_llm = FakeChatModel(responses=[], raise_exception=TimeoutError("HTTP 408 Request Timeout after 30s"))
        critic = CriticVerificationAgent(llm=critic_llm)

        decision = critic.verify(answer="Draf jawaban", context=self.context, question=self.question)

        self.assertFalse(decision.is_valid)
        self.assertEqual(decision.verification_status, VerificationStatus.VERIFICATION_ERROR)
        self.assertEqual(decision.confidence_score, 0.0)
        self.assertEqual(decision.metadata.get("error_type"), "TimeoutError")
        self.assertIn("HTTP 408", decision.critique_summary)

    # =========================================================================
    # Test 8: Empty Retrieved Context
    # =========================================================================
    def test_08_empty_retrieved_context(self):
        """Empty context cannot ground claims; must trigger immediate rejection."""
        critic = CriticVerificationAgent(llm=FakeChatModel(responses=["{}"]))

        for empty_ctx in ["", "  ", "Tidak ada konteks dokumen SOP maupun graf BPMN yang relevan ditemukan."]:
            decision = critic.verify(answer="Jawaban apapun", context=empty_ctx, question=self.question)
            self.assertFalse(decision.is_valid)
            self.assertEqual(decision.verification_status, VerificationStatus.REJECTED)
            self.assertEqual(decision.confidence_score, 0.0)

    # =========================================================================
    # Test 9: Incorrect Gateway Condition
    # =========================================================================
    def test_09_incorrect_gateway_condition(self):
        """Answer claiming disbursement occurs when documents are invalid must fail workflow check."""
        json_resp = _make_critic_json_response(
            is_valid=False,
            confidence=0.35,
            workflow_status=CheckStatus.FAILED,
            summary="Percabangan gateway terbalik: dana dicairkan jika berkas ditolak.",
            instructions="Koreksi gateway: dana hanya cair jika berkas valid.",
        )
        critic = CriticVerificationAgent(llm=FakeChatModel(responses=[json_resp]))

        inverted_gateway_draft = "Jika berkas ditolak atau tidak lengkap, bendahara tetap mencairkan dana insentif."
        decision = critic.verify(answer=inverted_gateway_draft, context=self.context, question=self.question)

        self.assertFalse(decision.is_valid)
        self.assertEqual(decision.verification_status, VerificationStatus.REJECTED)
        self.assertIn("workflow_correctness", decision.failed_checks)

    # =========================================================================
    # Test 10: Incorrect Actor Attribution
    # =========================================================================
    def test_10_incorrect_actor_attribution(self):
        """Answer attributing financial disbursement to Author instead of Bendahara must fail."""
        json_resp = _make_critic_json_response(
            is_valid=False,
            confidence=0.40,
            workflow_status=CheckStatus.FAILED,
            summary="Aktor penanggung jawab keliru: pemakalah tidak mencairkan dana sendiri.",
            instructions="Ubah aktor pencairan dana menjadi Bendahara.",
        )
        critic = CriticVerificationAgent(llm=FakeChatModel(responses=[json_resp]))

        wrong_actor_draft = "Pemakalah memeriksa berkas registrasi lalu pemakalah mencairkan dana untuk dirinya sendiri."
        decision = critic.verify(answer=wrong_actor_draft, context=self.context, question=self.question)

        self.assertFalse(decision.is_valid)
        self.assertIn("workflow_correctness", decision.failed_checks)

    # =========================================================================
    # Test 11: Incorrect Process Sequence
    # =========================================================================
    def test_11_incorrect_process_sequence(self):
        """Answer stating funds are disbursed before registration must fail workflow precedence."""
        json_resp = _make_critic_json_response(
            is_valid=False,
            confidence=0.30,
            workflow_status=CheckStatus.FAILED,
            summary="Urutan kronologis terbalik: pencairan mendahului pengunggahan naskah.",
            instructions="Urutkan proses secara kronologis: unggah naskah -> verifikasi berkas -> pencairan dana.",
        )
        critic = CriticVerificationAgent(llm=FakeChatModel(responses=[json_resp]))

        scrambled_seq_draft = "Pertama bendahara mencairkan dana insentif, setelah itu baru author mengunggah naskah."
        decision = critic.verify(answer=scrambled_seq_draft, context=self.context, question=self.question)

        self.assertFalse(decision.is_valid)
        self.assertIn("workflow_correctness", decision.failed_checks)

    # =========================================================================
    # Test 12: Unsupported Revision Content
    # =========================================================================
    def test_12_unsupported_revision_content(self):
        """If a revision introduces new hallucinated claims, Critic rejects revision and loop proceeds."""
        critic_responses = [
            _make_critic_json_response(
                is_valid=False,
                confidence=0.35,
                workflow_status=CheckStatus.FAILED,
                summary="Aktor salah.",
                instructions="Koreksi aktor.",
            ),
            _make_critic_json_response(
                is_valid=False,
                confidence=0.20,
                factual_status=CheckStatus.FAILED,
                summary="Revisi memuat halusinasi baru: dana diberikan dalam bentuk Bitcoin.",
                instructions="Hapus Bitcoin, gunakan nominal rupiah sesuai SOP.",
            ),
            _make_critic_json_response(
                is_valid=True,
                confidence=0.95,
                summary="Revisi 2 valid.",
            ),
        ]
        gen_responses = [
            "Draf Awal (Aktor Salah)",
            "Draf Revisi 1 (Aktor Benar tapi Halusinasi Bitcoin)",
            "Draf Revisi 2 (Bersih dan Valid)",
        ]

        chain = self._create_mock_chain(gen_responses, critic_responses, max_repairs=2)
        result = chain.invoke(question=self.question, use_router=False, use_query_agent=False)

        self.assertTrue(result["was_revised"])
        self.assertEqual(result["repair_iterations"], 2)
        self.assertEqual(result["final_verification_status"], "VERIFIED")
        self.assertEqual(result["answer"], "Draf Revisi 2 (Bersih dan Valid)")


if __name__ == "__main__":
    unittest.main()
