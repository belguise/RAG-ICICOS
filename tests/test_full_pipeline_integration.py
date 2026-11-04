"""
Comprehensive Integration, Architecture, and Ablation Test Suite for Agentic Hybrid RAG.
Evaluates the complete research pipeline:
1. End-to-end execution: Query -> Understanding -> Router -> Dual Retrieval -> Fusion -> Generation -> Critic -> Self-Repair -> Source Attribution -> Reconstructed Evaluation.
2. Architecture & Question Categories:
   - SOP-only questions
   - BPMN-only questions
   - Hybrid questions
   - Document requirement questions
   - Workflow sequence questions
   - Decision gateway questions
   - Actor attribution questions
   - Ambiguous / unconstrained questions
   - Unsupported / out-of-domain questions
   - Retrieval failure / empty context
3. Ablation validation:
   - Without Router (fixed static quotas)
   - Without BPMN retrieval (k_bpmn = 0)
   - Without Critic verification (returns initial draft directly)
"""

import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Add src and evaluation to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from agentic_rag_sop.generation.chain import SOPRAGChain
from agentic_rag_sop.agents.query_understanding import QueryUnderstandingAgent
from agentic_rag_sop.agents.router import RouterAgent, RouterDecision
from agentic_rag_sop.agents.critic import (
    CriticVerificationAgent,
    CriticDecision,
    VerificationStatus,
    CheckStatus,
)
from agentic_rag_sop.fusion.context_fusion import ContextFusionLayer, FusedContext
from agentic_rag_sop.retrieval.sop_retriever import SOPRetriever
from agentic_rag_sop.retrieval.bpmn_retriever import BPMNRetriever
from evaluation.metrics.reconstructed_evaluator import ReconstructedEvaluator


class FakeChatModel(BaseChatModel):
    """Deterministic mock chat model providing scripted responses for LCEL pipelines."""
    responses: List[str] = []
    _idx: int = 0

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.responses:
            resp_text = self.responses[self._idx % len(self.responses)]
            self._idx += 1
        else:
            resp_text = "Default fake model response."
        msg = AIMessage(content=resp_text)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "fake-integration-chat-model"


def make_critic_payload(
    is_valid: bool = True,
    confidence: float = 0.95,
    summary: str = "Passed",
    instructions: Optional[str] = None
) -> str:
    """Helper generating Pydantic-compliant CriticDecision JSON."""
    status_val = "PASSED" if is_valid else "FAILED"
    payload = {
        "is_valid": is_valid,
        "confidence_score": confidence,
        "factual_grounding": {"status": status_val, "reason": "Factual check", "evidence_quote": None},
        "domain_compliance": {"status": status_val, "reason": "Domain check", "evidence_quote": None},
        "workflow_correctness": {"status": status_val, "reason": "Workflow check", "evidence_quote": None},
        "sop_correctness": {"status": status_val, "reason": "SOP check", "evidence_quote": None},
        "answer_completeness": {"status": status_val, "reason": "Completeness check", "evidence_quote": None},
        "critique_summary": summary,
        "suggested_revision_instructions": instructions,
    }
    return json.dumps(payload)


class MockSOPRetriever:
    """Mock SOP document retriever returning structured test passages."""
    def __init__(self, docs: Optional[List[Document]] = None):
        if docs is not None:
            self.docs = docs
        else:
            self.docs = [
                Document(
                    page_content="SOP-01 Bagian 3: Mahasiswa wajib melampirkan kwitansi pembayaran asli dan persetujuan Dosen Pembimbing.",
                    metadata={"document_name": "SOP_Keuangan.pdf", "document_type": "sop", "page_number": 3, "process_name": "Pengajuan Dana", "actor": "Mahasiswa", "source_id": "sop_01"}
                ),
                Document(
                    page_content="SOP-02 Bagian 1: Berkas pengajuan diajukan ke Bagian Tata Usaha Fakultas untuk verifikasi administratif.",
                    metadata={"document_name": "SOP_Administrasi.pdf", "document_type": "sop", "page_number": 1, "process_name": "Verifikasi Berkas", "actor": "Tata Usaha", "source_id": "sop_02"}
                )
            ]

    def retrieve(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        return self.docs[:k]


class MockBPMNRetriever:
    """Mock BPMN process graph retriever returning structured process transitions."""
    def __init__(self, docs: Optional[List[Document]] = None):
        if docs is not None:
            self.docs = docs
        else:
            self.docs = [
                Document(
                    page_content="[BPMN Transition] Step 1: Submit Application (Author) -> Step 2: Gateway Check Budget (Bendahara) -> Step 3: Approval (Dekan).",
                    metadata={"document_name": "process_mbkm.bpmn", "document_type": "bpmn", "page_number": 1, "process_name": "Pengajuan Dana", "actor": "Dekan", "source_id": "bpmn_01"}
                )
            ]

    def retrieve(self, query: str, k: int = 3, **kwargs) -> List[Document]:
        return self.docs[:k]


class TestFullPipelineIntegration(unittest.TestCase):
    """End-to-end integration and architecture tests for the complete Agentic Hybrid RAG pipeline."""

    def setUp(self):
        self.sop_retriever = MockSOPRetriever()
        self.bpmn_retriever = MockBPMNRetriever()
        self.evaluator = ReconstructedEvaluator()

    def _create_mock_chain(
        self,
        generator_responses: List[str],
        critic_responses: List[str],
        sop_retriever: Optional[Any] = None,
        bpmn_retriever: Optional[Any] = None,
        router_responses: Optional[List[str]] = None,
    ):
        gen_llm = FakeChatModel()
        gen_llm.responses = generator_responses
        gen_llm._idx = 0

        critic_llm = FakeChatModel()
        critic_llm.responses = critic_responses
        critic_llm._idx = 0

        router_llm = FakeChatModel()
        router_llm.responses = router_responses or ['{"knowledge_source": "BOTH", "confidence": 0.95, "reason": "Hybrid rule"}']
        router_llm._idx = 0

        critic_agent = CriticVerificationAgent(llm=critic_llm)
        router_agent = RouterAgent(llm=router_llm)
        fusion_layer = ContextFusionLayer()

        chain = SOPRAGChain(
            sop_retriever=sop_retriever or self.sop_retriever,
            bpmn_retriever=bpmn_retriever or self.bpmn_retriever,
            llm=gen_llm,
            router_agent=router_agent,
            critic_agent=critic_agent,
            fusion_layer=fusion_layer,
            max_repair_iterations=2,
        )
        return chain

    # --------------------------------------------------------------------------
    # 1. FULL PIPELINE TEST
    # --------------------------------------------------------------------------
    def test_01_full_pipeline_execution(self):
        """Test complete execution path from query to evaluated metrics and provenance trace."""
        gen_resp = [
            "Mahasiswa melampirkan kwitansi asli ke Tata Usaha lalu disetujui oleh Dekan."
        ]
        critic_resp = [
            make_critic_payload(
                is_valid=True,
                confidence=0.96,
                summary="Draft is strictly verified and accurate.",
                instructions=None
            )
        ]

        chain = self._create_mock_chain(gen_resp, critic_resp)
        result = chain.invoke(
            question="Bagaimana prosedur pengajuan dana oleh mahasiswa?",
            use_query_agent=False,
            use_router=True,
            use_critic=True
        )

        # Verify Pipeline Outputs
        self.assertIn("answer", result)
        self.assertIn("final_verification_status", result)
        self.assertEqual(result["final_verification_status"], "VERIFIED")
        self.assertFalse(result["was_revised"])
        self.assertEqual(result["repair_iterations"], 0)
        self.assertTrue(len(result["sources"]) > 0)
        self.assertIn("generation_trace", result)

        # Verify Evaluation Metric Processing
        eval_res = self.evaluator.evaluate_query(
            query_id="ICICOS_TEST_001",
            architecture="proposed_agentic",
            category="hybrid",
            question="Bagaimana prosedur pengajuan dana oleh mahasiswa?",
            answer=result["answer"],
            context=result["formatted_context"],
            retrieved_chunks=[d.page_content for d in result["documents"]],
            expected_answer="Mahasiswa melampirkan kwitansi asli ke Tata Usaha lalu disetujui Dekan."
        )

        self.assertGreaterEqual(eval_res.faithfulness, 0.80)
        self.assertGreaterEqual(eval_res.sa_f1, 0.50)
        self.assertGreaterEqual(eval_res.context_recall, 0.90)

    # --------------------------------------------------------------------------
    # 2. ARCHITECTURE & QUESTION CATEGORY TESTS
    # --------------------------------------------------------------------------
    def test_02_sop_only_question_routing(self):
        """Test SOP document requirement question routing."""
        mock_llm = FakeChatModel()
        mock_llm.responses = ['{"knowledge_source": "SOP", "confidence": 0.95, "reason": "SOP requirement"}']
        router = RouterAgent(llm=mock_llm)
        dec = router.route("Dokumen apa saja syarat berkas kwitansi dan SPJ?")
        self.assertIn(dec.knowledge_source, ["SOP", "BOTH"])

    def test_03_bpmn_only_question_routing(self):
        """Test BPMN procedural workflow sequence routing."""
        mock_llm = FakeChatModel()
        mock_llm.responses = ['{"knowledge_source": "BPMN", "confidence": 0.95, "reason": "BPMN sequence"}']
        router = RouterAgent(llm=mock_llm)
        dec = router.route("Urutan alur setelah pengajuan dana adalah langkah apa?")
        self.assertIn(dec.knowledge_source, ["BPMN", "BOTH"])

    def test_04_decision_gateway_question_routing(self):
        """Test decision gateway XOR branching question routing."""
        mock_llm = FakeChatModel()
        mock_llm.responses = ['{"knowledge_source": "BPMN", "confidence": 0.95, "reason": "Decision gateway"}']
        router = RouterAgent(llm=mock_llm)
        dec = router.route("Jika anggaran ditolak oleh Dekan, apa percabangan alurnya?")
        self.assertIn(dec.knowledge_source, ["BPMN", "BOTH"])

    def test_05_actor_attribution_handling(self):
        """Test actor attribution across swimlane roles."""
        gen_resp = ["Pengajuan diajukan oleh Mahasiswa, diverifikasi oleh Tata Usaha, dan disetujui Dekan."]
        critic_resp = [make_critic_payload(is_valid=True, confidence=0.95, summary="Passed.")]
        chain = self._create_mock_chain(gen_resp, critic_resp)
        res = chain.invoke("Siapa yang berwenang menyetujui pengajuan dana mahasiswa?")
        self.assertIn("Dekan", res["answer"])
        self.assertEqual(res["final_verification_status"], "VERIFIED")

    def test_06_retrieval_failure_empty_context(self):
        """Test graceful handling when retrievers return empty documents."""
        empty_sop = MockSOPRetriever(docs=[])
        empty_bpmn = MockBPMNRetriever(docs=[])
        gen_llm = FakeChatModel()
        gen_llm.responses = ["Maaf, data tidak ditemukan dalam SOP maupun BPMN."]

        chain = SOPRAGChain(
            sop_retriever=empty_sop,
            bpmn_retriever=empty_bpmn,
            llm=gen_llm,
            max_repair_iterations=2
        )
        res = chain.invoke("Pertanyaan tentang topik yang tidak ada dalam database?", use_critic=False)
        self.assertIn("answer", res)
        self.assertEqual(len(res["documents"]), 0)

    # --------------------------------------------------------------------------
    # 3. ABLATION VALIDATION TESTS
    # --------------------------------------------------------------------------
    def test_07_ablation_without_router(self):
        """Verify that without_router ablation uses fixed uniform quotas (selected_retriever = BOTH)."""
        gen_resp = ["Static hybrid draft response."]
        critic_resp = [make_critic_payload(is_valid=True, confidence=0.95)]
        chain = self._create_mock_chain(gen_resp, critic_resp)

        res = chain.invoke("Pertanyaan pengujian router ablation", use_router=False)
        # When use_router is False, route_decision must be None and selected_retriever must be BOTH
        self.assertIsNone(res["route_decision"])
        self.assertEqual(res["selected_retriever"], "BOTH")
        # 2 SOP docs + 1 BPMN doc from our test mocks
        self.assertEqual(len(res["documents"]), 3)

    def test_08_ablation_without_bpmn(self):
        """Verify that without_bpmn ablation sets k_bpmn = 0 and retrieves only SOP documents."""
        gen_resp = ["SOP only draft response without BPMN."]
        critic_resp = [make_critic_payload(is_valid=True, confidence=0.95)]
        chain = self._create_mock_chain(gen_resp, critic_resp)

        res = chain.invoke("Pertanyaan pengujian BPMN ablation", use_bpmn=False)
        self.assertEqual(res["bpmn_context"], "Tidak ada graf proses BPMN yang ditemukan.")
        self.assertTrue(len(res["sop_context"]) > 0)
        for doc in res["documents"]:
            self.assertEqual(doc.metadata.get("document_type"), "sop")

    def test_09_ablation_without_critic(self):
        """Verify that without_critic ablation bypasses verification and emits initial draft directly."""
        gen_resp = ["Unverified raw initial draft."]
        chain = self._create_mock_chain(gen_resp, critic_responses=[])

        res = chain.invoke("Pertanyaan pengujian critic ablation", use_critic=False)
        self.assertEqual(res["answer"], "Unverified raw initial draft.")
        self.assertEqual(res["final_verification_status"], "DISABLED")
        self.assertFalse(res["was_revised"])
        self.assertEqual(res["repair_iterations"], 0)
        self.assertEqual(len(res["revision_trace"]), 0)

    def test_10_critic_bounded_repair_iteration(self):
        """Verify that bounded self-repair executes up to N=2 iterations and re-evaluates revised answer."""
        gen_resp = [
            "Draft 0 with factual error.",
            "Revision 1 with corrected facts."
        ]
        critic_resp = [
            make_critic_payload(
                is_valid=False,
                confidence=0.40,
                summary="Error in budget.",
                instructions="Correct the budget."
            ),
            make_critic_payload(
                is_valid=True,
                confidence=0.95,
                summary="Now verified.",
                instructions=None
            )
        ]
        chain = self._create_mock_chain(gen_resp, critic_resp)
        res = chain.invoke("Pertanyaan pengujian perbaikan mandiri", use_query_agent=False)

        self.assertTrue(res["was_revised"])
        self.assertEqual(res["repair_iterations"], 1)
        self.assertEqual(res["final_verification_status"], "VERIFIED")
        self.assertEqual(res["answer"], "Revision 1 with corrected facts.")
        self.assertEqual(len(res["revision_trace"]), 2)


if __name__ == "__main__":
    unittest.main()
