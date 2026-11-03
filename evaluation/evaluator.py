"""
ICICoS Process QA Benchmark Evaluator.
Executes 5 comparative architectures on identical benchmark questions:
1. Baseline 1: LLM Only (Zero-shot parametric Gemini Flash)
2. Baseline 2: Traditional Vector RAG (ChromaDB dense semantic search)
3. Baseline 3: Graph RAG (BPMN Subgraph Path Reasoning)
4. Baseline 4: Hybrid RAG (Direct Vector + BPMN Concatenation without agent reasoning)
5. Proposed Model: Agentic Hybrid RAG (Full pipeline: QUA + Router + Dual Retrieval + Fusion + Critic)

Also provides Ablation Study execution:
- Ablation 1: Agentic RAG without Router
- Ablation 2: Agentic RAG without BPMN Retriever
- Ablation 3: Agentic RAG without Critic
"""

import time
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from agentic_rag_sop.generation.chain import SOPRAGChain
from evaluation.archive.process_consistency_experimental import calculate_process_consistency_score
from evaluation.metrics.rag_metrics import calculate_answer_accuracy, calculate_evidence_grounding
from evaluation.metrics.router_accuracy import calculate_router_metrics
from evaluation.metrics.bpmn_process_metrics import evaluate_bpmn_process
from evaluation.metrics.standard_rag_metrics import (
    calculate_faithfulness,
    calculate_answer_relevancy,
    calculate_context_precision,
    calculate_context_recall,
    calculate_bert_score,
    calculate_actor_accuracy,
)

logger = logging.getLogger(__name__)


def _evaluate_system_output(
    answer: str,
    context: str,
    chunks: List[str],
    question: str,
    expected_ans: str,
    expected_path: List[str],
    expected_actor: Optional[str],
    category: str = "general",
) -> Dict[str, Any]:
    """Unified metric evaluation across retrieval, generation, and process dimensions."""
    pcs_data = calculate_process_consistency_score(answer, expected_path, expected_actor)
    acc_data = calculate_answer_accuracy(answer, expected_ans)
    egs_data = calculate_evidence_grounding(answer, context)
    faith_data = calculate_faithfulness(answer, context)
    relevancy_data = calculate_answer_relevancy(question, answer)
    precision_data = calculate_context_precision(chunks, expected_ans, question)
    recall_data = calculate_context_recall(chunks, expected_ans)
    bert_data = calculate_bert_score(answer, expected_ans)
    bpmn_proc = evaluate_bpmn_process(
        predicted_answer=answer,
        expected_bpmn_path=expected_path,
        expected_actor=expected_actor,
        question=question,
        expected_answer=expected_ans,
        category=category,
    )

    return {
        "faithfulness": faith_data["faithfulness_score"],
        "answer_relevancy": relevancy_data["relevancy_score"],
        "context_precision": precision_data["context_precision"],
        "context_recall": recall_data["context_recall"],
        "bert_score": bert_data["bert_score"],
        "path_accuracy": bpmn_proc["path_accuracy"],
        "gateway_accuracy": bpmn_proc["gateway_accuracy"],
        "actor_accuracy": bpmn_proc["actor_accuracy"],
        "path_coverage": bpmn_proc["path_coverage"],
        "order_preservation": bpmn_proc["order_preservation"],
        "pcs": pcs_data["pcs_score"],
        "accuracy": acc_data["accuracy_score"],
        "precision": acc_data["precision"],
        "recall": acc_data["recall"],
        "egs": egs_data["egs_score"],
        "hallucination_rate": egs_data["hallucination_rate"],
    }


class BenchmarkEvaluator:
    """
    Evaluator engine comparing Baseline 1 (LLM Only), Baseline 2 (Traditional RAG),
    Baseline 3 (Graph RAG), Baseline 4 (Hybrid RAG), and Proposed Agentic Hybrid RAG.
    """

    def __init__(
        self,
        chain: Optional[SOPRAGChain] = None,
        delay_seconds: float = 1.0,
    ):
        self.chain = chain or SOPRAGChain()
        self.delay_seconds = delay_seconds

    def _invoke_with_retry(self, fn, max_retries: int = 3, initial_wait: float = 3.0):
        """Execute a callable with exponential backoff on rate-limit and transient connection errors."""
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(k in err_str for k in ["429", "quota", "rate", "exhausted", "too many requests"])
                is_transient = any(k in err_str for k in ["connection", "503", "timeout", "temporarily unavailable", "service unavailable"])
                if (is_rate_limit or is_transient) and attempt < max_retries - 1:
                    wait_time = (initial_wait * (2 ** attempt)) + (15.0 if is_rate_limit else 2.0)
                    logger.warning(f"Transient error on attempt {attempt+1}/{max_retries}: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    raise e

    def _safe_invoke_llm_only(self, question: str) -> Dict[str, Any]:
        """Execute Baseline 1: LLM Only (Zero-shot)."""
        start_t = time.time()
        try:
            def _call():
                return self.chain._core_chain.invoke({
                    "context": "Tidak ada dokumen referensi eksternal. Jawablah sesuai pengetahuan umum.",
                    "question": question,
                })
            answer = self._invoke_with_retry(_call)
            duration = round(time.time() - start_t, 3)
            return {"answer": answer, "context": "", "chunks": [], "duration": duration, "error": None}
        except Exception as e:
            logger.error(f"Error in LLM Only baseline: {e}")
            return {"answer": "", "context": "", "chunks": [], "duration": round(time.time() - start_t, 3), "error": str(e)}

    def _safe_invoke_traditional_rag(self, question: str) -> Dict[str, Any]:
        """Execute Baseline 2: Traditional Vector RAG (ChromaDB only, no agents)."""
        start_t = time.time()
        try:
            def _call():
                return self.chain.invoke(
                    question=question,
                    use_query_agent=False,
                    use_router=False,
                    use_critic=False,
                )
            res = self._invoke_with_retry(_call)
            duration = round(time.time() - start_t, 3)
            chunks = [d.page_content for d in res.get("documents", [])]
            return {
                "answer": res["answer"],
                "context": res.get("sop_context", ""),
                "chunks": chunks,
                "duration": duration,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error in Traditional RAG baseline: {e}")
            return {"answer": "", "context": "", "chunks": [], "duration": round(time.time() - start_t, 3), "error": str(e)}

    def _safe_invoke_graph_rag(self, question: str) -> Dict[str, Any]:
        """Execute Baseline 3: Graph RAG (BPMN Subgraph reasoning only)."""
        start_t = time.time()
        try:
            bpmn_docs = self.chain.bpmn_retriever.retrieve(query=question, k=4)
            bpmn_context = "\n\n".join(doc.page_content for doc in bpmn_docs)
            ctx_block = bpmn_context if bpmn_context else "Tidak ada proses BPMN terkait."
            def _call():
                return self.chain._core_chain.invoke({
                    "context": ctx_block,
                    "question": question,
                })
            answer = self._invoke_with_retry(_call)
            duration = round(time.time() - start_t, 3)
            chunks = [doc.page_content for doc in bpmn_docs]
            return {
                "answer": answer,
                "context": bpmn_context,
                "chunks": chunks,
                "duration": duration,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error in Graph RAG baseline: {e}")
            return {"answer": "", "context": "", "chunks": [], "duration": round(time.time() - start_t, 3), "error": str(e)}

    def _safe_invoke_hybrid_rag(self, question: str) -> Dict[str, Any]:
        """Execute Baseline 4: Hybrid RAG (Direct Vector + BPMN Concatenation without agent reasoning)."""
        start_t = time.time()
        try:
            sop_docs = self.chain.sop_retriever.retrieve(query=question, k=3)
            bpmn_docs = self.chain.bpmn_retriever.retrieve(query=question, k=3)
            sop_context = "\n\n".join(doc.page_content for doc in sop_docs)
            bpmn_context = "\n\n".join(doc.page_content for doc in bpmn_docs)
            combined_context = f"[SOP DOCUMENTATION]\n{sop_context}\n\n[BPMN BUSINESS PROCESS]\n{bpmn_context}"
            def _call():
                return self.chain._core_chain.invoke({
                    "context": combined_context,
                    "question": question,
                })
            answer = self._invoke_with_retry(_call)
            duration = round(time.time() - start_t, 3)
            chunks = [doc.page_content for doc in sop_docs + bpmn_docs]
            return {
                "answer": answer,
                "context": combined_context,
                "chunks": chunks,
                "duration": duration,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error in Hybrid RAG baseline: {e}")
            return {"answer": "", "context": "", "chunks": [], "duration": round(time.time() - start_t, 3), "error": str(e)}

    def _safe_invoke_proposed_agentic(self, question: str) -> Dict[str, Any]:
        """Execute Proposed Model: Agentic Hybrid RAG (Full pipeline)."""
        start_t = time.time()
        try:
            def _call():
                return self.chain.invoke(
                    question=question,
                    use_query_agent=True,
                    use_router=True,
                    use_critic=True,
                )
            res = self._invoke_with_retry(_call)
            duration = round(time.time() - start_t, 3)
            route = res.get("route_decision")
            critic = res.get("critic_decision")
            chunks = [d.page_content for d in res.get("documents", [])]
            return {
                "answer": res["answer"],
                "draft_answer": res.get("draft_answer"),
                "context": res.get("formatted_context", res.get("fused_text", "")),
                "chunks": chunks,
                "rewritten_query": res.get("rewritten_query"),
                "route_decision": {
                    "intent": route.intent if route else None,
                    "knowledge_source": route.knowledge_source if route else None,
                    "secondary_source": getattr(route, "secondary_source", None),
                    "reason": getattr(route, "reason", ""),
                } if route else None,
                "critic_decision": {
                    "is_valid": getattr(critic, "is_valid", True) if critic else True,
                    "confidence_score": getattr(critic, "confidence_score", 1.0) if critic else 1.0,
                    "critique_summary": getattr(critic, "critique_summary", "") if critic else "",
                    "suggested_revision_instructions": getattr(critic, "suggested_revision_instructions", None) if critic else None,
                } if critic else None,
                "was_revised": res.get("was_revised", False),
                "duration": duration,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error in Proposed Agentic baseline: {e}")
            return {
                "answer": "",
                "draft_answer": "",
                "context": "",
                "chunks": [],
                "rewritten_query": "",
                "route_decision": None,
                "critic_decision": None,
                "was_revised": False,
                "duration": round(time.time() - start_t, 3),
                "error": str(e),
            }

    # =====================================================================
    # Ablation Study Invocations
    # =====================================================================

    def _safe_invoke_ablation_no_router(self, question: str) -> Dict[str, Any]:
        """Ablation 1: Agentic RAG without Router (fixed dual retrieval, no adaptive routing)."""
        start_t = time.time()
        try:
            rewritten_query = question
            if self.chain.query_agent is None:
                from agentic_rag_sop.agents.query_understanding import QueryUnderstandingAgent
                self.chain.query_agent = QueryUnderstandingAgent(llm=self.chain.llm)
            rewritten_query = self.chain.query_agent.rewrite_query(question)

            sop_docs = self.chain.sop_retriever.retrieve(query=rewritten_query, k=4)
            bpmn_docs = self.chain.bpmn_retriever.retrieve(query=rewritten_query, k=3)
            fused_context = self.chain.fusion_layer.fuse(
                sop_documents=sop_docs,
                bpmn_documents=bpmn_docs,
                query=rewritten_query,
                route_decision=None,
            )
            from agentic_rag_sop.generation.prompts import format_fused_context_block
            formatted_context = format_fused_context_block(fused_context)
            draft_answer = self.chain._core_chain.invoke({
                "context": formatted_context,
                "question": rewritten_query,
            })
            if self.chain.critic_agent is None:
                from agentic_rag_sop.agents.critic import CriticVerificationAgent
                self.chain.critic_agent = CriticVerificationAgent(llm=self.chain.llm)
            critic_decision = self.chain.critic_agent.verify(
                answer=draft_answer,
                context=formatted_context,
                question=rewritten_query,
            )
            final_answer = draft_answer
            if not critic_decision.is_valid and critic_decision.suggested_revision_instructions:
                final_answer = self.chain._revision_chain.invoke({
                    "context": formatted_context,
                    "critique_instructions": critic_decision.suggested_revision_instructions,
                    "draft_answer": draft_answer,
                    "question": rewritten_query,
                })
            duration = round(time.time() - start_t, 3)
            chunks = [d.page_content for d in sop_docs + bpmn_docs]
            return {
                "answer": final_answer,
                "context": formatted_context,
                "chunks": chunks,
                "duration": duration,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error in Ablation No-Router: {e}")
            return {"answer": "", "context": "", "chunks": [], "duration": round(time.time() - start_t, 3), "error": str(e)}

    def _safe_invoke_ablation_no_bpmn(self, question: str) -> Dict[str, Any]:
        """Ablation 2: Agentic RAG without BPMN Retriever (textual SOP only)."""
        start_t = time.time()
        try:
            rewritten_query = question
            if self.chain.query_agent is None:
                from agentic_rag_sop.agents.query_understanding import QueryUnderstandingAgent
                self.chain.query_agent = QueryUnderstandingAgent(llm=self.chain.llm)
            rewritten_query = self.chain.query_agent.rewrite_query(question)

            sop_docs = self.chain.sop_retriever.retrieve(query=rewritten_query, k=6)
            fused_context = self.chain.fusion_layer.fuse(
                sop_documents=sop_docs,
                bpmn_documents=[],
                query=rewritten_query,
                route_decision=None,
            )
            from agentic_rag_sop.generation.prompts import format_fused_context_block
            formatted_context = format_fused_context_block(fused_context)
            draft_answer = self.chain._core_chain.invoke({
                "context": formatted_context,
                "question": rewritten_query,
            })
            if self.chain.critic_agent is None:
                from agentic_rag_sop.agents.critic import CriticVerificationAgent
                self.chain.critic_agent = CriticVerificationAgent(llm=self.chain.llm)
            critic_decision = self.chain.critic_agent.verify(
                answer=draft_answer,
                context=formatted_context,
                question=rewritten_query,
            )
            final_answer = draft_answer
            if not critic_decision.is_valid and critic_decision.suggested_revision_instructions:
                final_answer = self.chain._revision_chain.invoke({
                    "context": formatted_context,
                    "critique_instructions": critic_decision.suggested_revision_instructions,
                    "draft_answer": draft_answer,
                    "question": rewritten_query,
                })
            duration = round(time.time() - start_t, 3)
            chunks = [d.page_content for d in sop_docs]
            return {
                "answer": final_answer,
                "context": formatted_context,
                "chunks": chunks,
                "duration": duration,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error in Ablation No-BPMN: {e}")
            return {"answer": "", "context": "", "chunks": [], "duration": round(time.time() - start_t, 3), "error": str(e)}

    def _safe_invoke_ablation_no_critic(self, question: str) -> Dict[str, Any]:
        """Ablation 3: Agentic RAG without Critic (unverified generation)."""
        start_t = time.time()
        try:
            res = self.chain.invoke(
                question=question,
                use_query_agent=True,
                use_router=True,
                use_critic=False,
            )
            duration = round(time.time() - start_t, 3)
            chunks = [d.page_content for d in res.get("documents", [])]
            return {
                "answer": res["draft_answer"],
                "context": res.get("formatted_context", res.get("fused_text", "")),
                "chunks": chunks,
                "duration": duration,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error in Ablation No-Critic: {e}")
            return {"answer": "", "context": "", "chunks": [], "duration": round(time.time() - start_t, 3), "error": str(e)}

    # =====================================================================
    # Main Question Evaluator
    # =====================================================================

    def evaluate_single_question(
        self,
        item: Dict[str, Any],
        systems_to_run: Optional[List[str]] = None,
        run_ablations: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate a single question across requested systems and optional ablation variants.
        """
        if systems_to_run is None:
            systems_to_run = ["llm_only", "traditional_rag", "graph_rag", "hybrid_rag", "proposed_agentic"]

        q_id = item.get("id") or item.get("query_id", "")
        question = item["question"]
        expected_ans = item.get("expected_answer", "")
        expected_path = item.get("expected_bpmn_path", [])
        expected_actor = item.get("expected_actor")
        expected_src = item.get("expected_source", "SOP")
        category = item.get("category", "general")

        question_result: Dict[str, Any] = {
            "id": q_id,
            "question": question,
            "category": category,
            "expected_source": expected_src,
            "expected_actor": expected_actor,
            "expected_process": item.get("expected_process"),
            "baselines": {},
            "ablations": {},
        }

        # 1. Baseline 1: LLM Only
        if "llm_only" in systems_to_run:
            out1 = self._safe_invoke_llm_only(question)
            m1 = _evaluate_system_output(
                answer=out1["answer"],
                context=out1["context"],
                chunks=out1["chunks"],
                question=question,
                expected_ans=expected_ans,
                expected_path=expected_path,
                expected_actor=expected_actor,
                category=category,
            )
            question_result["baselines"]["llm_only"] = {
                "answer": out1["answer"],
                "duration": out1["duration"],
                "error": out1["error"],
                **m1,
            }
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)

        # 2. Baseline 2: Traditional Vector RAG
        if "traditional_rag" in systems_to_run:
            out2 = self._safe_invoke_traditional_rag(question)
            m2 = _evaluate_system_output(
                answer=out2["answer"],
                context=out2["context"],
                chunks=out2["chunks"],
                question=question,
                expected_ans=expected_ans,
                expected_path=expected_path,
                expected_actor=expected_actor,
                category=category,
            )
            question_result["baselines"]["traditional_rag"] = {
                "answer": out2["answer"],
                "duration": out2["duration"],
                "error": out2["error"],
                **m2,
            }
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)

        # 3. Baseline 3: Graph RAG
        if "graph_rag" in systems_to_run:
            out3 = self._safe_invoke_graph_rag(question)
            m3 = _evaluate_system_output(
                answer=out3["answer"],
                context=out3["context"],
                chunks=out3["chunks"],
                question=question,
                expected_ans=expected_ans,
                expected_path=expected_path,
                expected_actor=expected_actor,
                category=category,
            )
            question_result["baselines"]["graph_rag"] = {
                "answer": out3["answer"],
                "duration": out3["duration"],
                "error": out3["error"],
                **m3,
            }
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)

        # 4. Baseline 4: Hybrid RAG (Direct Vector + BPMN Concatenation)
        if "hybrid_rag" in systems_to_run:
            out_hyb = self._safe_invoke_hybrid_rag(question)
            m_hyb = _evaluate_system_output(
                answer=out_hyb["answer"],
                context=out_hyb["context"],
                chunks=out_hyb["chunks"],
                question=question,
                expected_ans=expected_ans,
                expected_path=expected_path,
                expected_actor=expected_actor,
                category=category,
            )
            question_result["baselines"]["hybrid_rag"] = {
                "answer": out_hyb["answer"],
                "duration": out_hyb["duration"],
                "error": out_hyb["error"],
                **m_hyb,
            }
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)

        # 5. Proposed Model: Agentic Hybrid RAG
        if "proposed_agentic" in systems_to_run:
            out4 = self._safe_invoke_proposed_agentic(question)
            m4 = _evaluate_system_output(
                answer=out4["answer"],
                context=out4["context"],
                chunks=out4["chunks"],
                question=question,
                expected_ans=expected_ans,
                expected_path=expected_path,
                expected_actor=expected_actor,
                category=category,
            )
            router_eval = None
            if out4.get("route_decision"):
                router_eval = calculate_router_metrics(
                    predicted_source=out4["route_decision"].get("knowledge_source"),
                    expected_source=expected_src,
                    predicted_intent=out4["route_decision"].get("intent"),
                    expected_category=category,
                    predicted_secondary=out4["route_decision"].get("secondary_source"),
                )

            question_result["baselines"]["proposed_agentic"] = {
                "answer": out4["answer"],
                "draft_answer": out4.get("draft_answer"),
                "rewritten_query": out4.get("rewritten_query"),
                "route_decision": out4.get("route_decision"),
                "critic_decision": out4.get("critic_decision"),
                "was_revised": out4.get("was_revised", False),
                "router_eval": router_eval,
                "duration": out4["duration"],
                "error": out4["error"],
                **m4,
            }
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)

        # 6. Ablation Variants (if requested)
        if run_ablations:
            # Ablation 1: without Router
            out_abl1 = self._safe_invoke_ablation_no_router(question)
            m_abl1 = _evaluate_system_output(
                answer=out_abl1["answer"],
                context=out_abl1["context"],
                chunks=out_abl1["chunks"],
                question=question,
                expected_ans=expected_ans,
                expected_path=expected_path,
                expected_actor=expected_actor,
                category=category,
            )
            question_result["ablations"]["ablation_no_router"] = {
                "answer": out_abl1["answer"],
                "duration": out_abl1["duration"],
                "error": out_abl1["error"],
                **m_abl1,
            }
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)

            # Ablation 2: without BPMN Retriever
            out_abl2 = self._safe_invoke_ablation_no_bpmn(question)
            m_abl2 = _evaluate_system_output(
                answer=out_abl2["answer"],
                context=out_abl2["context"],
                chunks=out_abl2["chunks"],
                question=question,
                expected_ans=expected_ans,
                expected_path=expected_path,
                expected_actor=expected_actor,
                category=category,
            )
            question_result["ablations"]["ablation_no_bpmn"] = {
                "answer": out_abl2["answer"],
                "duration": out_abl2["duration"],
                "error": out_abl2["error"],
                **m_abl2,
            }
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)

            # Ablation 3: without Critic
            out_abl3 = self._safe_invoke_ablation_no_critic(question)
            m_abl3 = _evaluate_system_output(
                answer=out_abl3["answer"],
                context=out_abl3["context"],
                chunks=out_abl3["chunks"],
                question=question,
                expected_ans=expected_ans,
                expected_path=expected_path,
                expected_actor=expected_actor,
                category=category,
            )
            question_result["ablations"]["ablation_no_critic"] = {
                "answer": out_abl3["answer"],
                "duration": out_abl3["duration"],
                "error": out_abl3["error"],
                **m_abl3,
            }
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)

        return question_result
