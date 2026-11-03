"""
LCEL Generation Chain with Phase 4 Context Fusion and Phase 5 Critic Verification.
Combines query rewriting, intelligent routing, multi-source retrieval (SOP & BPMN),
context fusion with conflict resolution, Gemini answer generation, and
Critic-guided bounded self-correction verification loop ($N_{max\_repair} = 2$).
"""

import logging
import time
from typing import Any, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from agentic_rag_sop.config.settings import Settings
from agentic_rag_sop.core.llm import get_llm
from agentic_rag_sop.generation.prompts import (
    format_fused_context_block,
    format_docs_for_context,
    get_rag_prompt_template,
    get_revision_prompt_template,
)
from agentic_rag_sop.retrieval import BaseKnowledgeRetriever, SOPRetriever, BPMNRetriever
from agentic_rag_sop.agents.query_understanding import QueryUnderstandingAgent
from agentic_rag_sop.agents.router import RouterAgent, RouterDecision
from agentic_rag_sop.agents.critic import (
    CriticVerificationAgent,
    CriticDecision,
    VerificationStatus,
    CheckStatus,
)
from agentic_rag_sop.fusion.context_fusion import ContextFusionLayer, FusedContext

logger = logging.getLogger("SOPRAGChain")


class SOPRAGChain:
    """
    End-to-end RAG question-answering chain with:
    - Query Understanding Agent (Phase 2.1)
    - Router Agent (Phase 2.2)
    - Separated Retriever Layer: SOPRetriever & BPMNRetriever (Phase 2.3)
    - BPMN Process Graph Reasoning (Phase 3)
    - Context Fusion Layer with ICICoS Domain Guardrails (Phase 4)
    - Bounded Adversarial Critic Verification & Self-Repair Loop (Phase 5)
    """

    def __init__(
        self,
        sop_retriever: Optional[SOPRetriever] = None,
        bpmn_retriever: Optional[BPMNRetriever] = None,
        llm: Optional[BaseChatModel] = None,
        query_agent: Optional[QueryUnderstandingAgent] = None,
        router_agent: Optional[RouterAgent] = None,
        critic_agent: Optional[CriticVerificationAgent] = None,
        fusion_layer: Optional[ContextFusionLayer] = None,
        retriever: Optional[SOPRetriever] = None,
        max_repair_iterations: Optional[int] = None,
    ):
        self.settings = Settings()
        self.sop_retriever = sop_retriever or retriever or SOPRetriever()
        self.bpmn_retriever = bpmn_retriever or BPMNRetriever()
        self.retriever = self.sop_retriever  # Alias for backward compatibility
        self.llm = llm or get_llm()
        self.query_agent = query_agent
        self.router_agent = router_agent
        self.critic_agent = critic_agent
        self.fusion_layer = fusion_layer or ContextFusionLayer()
        self.prompt_template = get_rag_prompt_template()
        self.revision_template = get_revision_prompt_template()
        self.output_parser = StrOutputParser()
        self.max_repair_iterations = (
            max_repair_iterations
            if max_repair_iterations is not None
            else getattr(self.settings, "MAX_REPAIR_ITERATIONS", 2)
        )

        # Core generation and revision chains
        self._core_chain = self.prompt_template | self.llm | self.output_parser
        self._revision_chain = self.revision_template | self.llm | self.output_parser

    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Deduplicate and extract structured source citations from retrieved docs."""
        seen = set()
        sources = []

        for doc in documents:
            meta = doc.metadata
            doc_name = meta.get("document_name", "Dokumen Tidak Dikenal")
            doc_type = meta.get("document_type", "SOP").upper()
            page = meta.get("page_number", "N/A")
            proc = meta.get("process_name", "Umum")
            source_id = f"{doc_name}::{page}::{proc}"

            if source_id not in seen:
                seen.add(source_id)
                sources.append({
                    "document_name": doc_name,
                    "document_type": doc_type,
                    "page_number": page,
                    "process_name": proc,
                    "actor": meta.get("actor", "Terkait"),
                    "source_id": meta.get("source_id", "N/A"),
                })

        return sources

    def invoke(
        self,
        question: str,
        k: Optional[int] = None,
        document_type: Optional[str] = None,
        actor: Optional[str] = None,
        process_name: Optional[str] = None,
        use_query_agent: bool = True,
        use_router: bool = True,
        use_critic: bool = True,
        use_bpmn: bool = True,
        max_repair_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute full RAG generation pipeline with intelligent routing, context fusion,
        and downstream Critic verification with bounded iterative self-repair.

        Ablation Support:
        - use_query_agent=False: Disables query understanding rewrite
        - use_router=False: Disables adaptive intent router (falls back to fixed uniform BOTH quota)
        - use_bpmn=False: Disables BPMN retrieval (k_bpmn = 0)
        - use_critic=False: Disables Critic verification and closed-loop self-repair
        """
        t_start_total = time.time()

        # 1. Query Understanding Agent (Phase 2.1)
        rewritten_query = question
        if use_query_agent:
            if self.query_agent is None:
                self.query_agent = QueryUnderstandingAgent(llm=self.llm)
            rewritten_query = self.query_agent.rewrite_query(question)

        # 2. Router Agent (Phase 2.2)
        route_decision: Optional[RouterDecision] = None
        if use_router:
            if self.router_agent is None:
                self.router_agent = RouterAgent(llm=self.llm)
            route_decision = self.router_agent.route(rewritten_query)

        selected_retriever = route_decision.knowledge_source if route_decision else "BOTH"
        sop_docs: List[Document] = []
        bpmn_docs: List[Document] = []

        # Dual Retrieval Execution based on dynamic quota and ablation flags
        if not use_bpmn:
            # Ablation variant: Without BPMN Retrieval (k_bpmn = 0)
            sop_k = k if k is not None else 6
            sop_docs = self.sop_retriever.retrieve(
                query=rewritten_query,
                k=sop_k,
                document_type=document_type or "sop",
                actor=actor,
                process_name=process_name,
            )
            bpmn_docs = []
        elif selected_retriever == "SOP":
            sop_k = k if k is not None else 6
            sop_docs = self.sop_retriever.retrieve(
                query=rewritten_query,
                k=sop_k,
                document_type=document_type or "sop",
                actor=actor,
                process_name=process_name,
            )
            bpmn_docs = self.bpmn_retriever.retrieve(query=rewritten_query, k=2)

        elif selected_retriever == "BPMN":
            bpmn_k = k if k is not None else 4
            bpmn_docs = self.bpmn_retriever.retrieve(query=rewritten_query, k=bpmn_k)
            sop_docs = self.sop_retriever.retrieve(
                query=rewritten_query,
                k=3,
                document_type=document_type or "sop",
                actor=actor,
                process_name=process_name,
            )
        else:  # BOTH / HYBRID (or without_router ablation)
            sop_k = k if k is not None else 5
            bpmn_k = k if k is not None else 3
            sop_docs = self.sop_retriever.retrieve(
                query=rewritten_query,
                k=sop_k,
                document_type=document_type or "sop",
                actor=actor,
                process_name=process_name,
            )
            bpmn_docs = self.bpmn_retriever.retrieve(query=rewritten_query, k=bpmn_k)

        if not sop_docs and not bpmn_docs:
            sop_docs = self.sop_retriever.retrieve(
                query=rewritten_query,
                k=k or 4,
                document_type=None,
                actor=actor,
                process_name=process_name,
            )

        # 3. Context Fusion Layer (Phase 4)
        fused_context: FusedContext = self.fusion_layer.fuse(
            sop_documents=sop_docs,
            bpmn_documents=bpmn_docs,
            query=rewritten_query,
            route_decision=route_decision,
        )

        formatted_context = format_fused_context_block(fused_context)

        # 4. Initial Draft Generation
        t_start_draft = time.time()
        draft_answer = self._core_chain.invoke({
            "context": formatted_context,
            "question": rewritten_query,
        })
        draft_duration = round(time.time() - t_start_draft, 3)

        # 5. Bounded Adversarial Critic Verification & Self-Repair Loop (Phase 5)
        critic_decision: Optional[CriticDecision] = None
        current_answer = draft_answer
        repair_attempts = 0
        revision_trace: List[Dict[str, Any]] = []
        critic_total_latency = 0.0
        revision_total_latency = 0.0

        max_repairs = (
            max_repair_iterations
            if max_repair_iterations is not None
            else self.max_repair_iterations
        )

        if use_critic:
            if self.critic_agent is None:
                self.critic_agent = CriticVerificationAgent(llm=self.llm)

            # --- Pass 0: Initial Draft Verification ---
            t0 = time.time()
            critic_decision = self.critic_agent.verify(
                answer=current_answer,
                context=formatted_context,
                question=rewritten_query,
            )
            c_lat = round(time.time() - t0, 3)
            critic_total_latency += c_lat

            revision_trace.append({
                "attempt": 0,
                "action": "initial_verification",
                "answer": current_answer,
                "verification_status": critic_decision.verification_status.value,
                "is_valid": critic_decision.is_valid,
                "confidence_score": critic_decision.confidence_score,
                "failed_checks": list(critic_decision.failed_checks),
                "critique_summary": critic_decision.critique_summary,
                "revision_instructions": critic_decision.suggested_revision_instructions,
                "critic_latency": c_lat,
            })

            # --- Bounded Self-Repair Loop (up to N_max_repair iterations) ---
            while not critic_decision.is_valid and repair_attempts < max_repairs:
                # If Critic encountered an unrecoverable infrastructure error, halt loop
                if critic_decision.verification_status == VerificationStatus.VERIFICATION_ERROR:
                    logger.warning(
                        f"Critic encountered verification error: {critic_decision.critique_summary}. Halting repair loop."
                    )
                    break

                if not critic_decision.suggested_revision_instructions:
                    logger.warning("Critic rejected answer but provided no revision instructions. Halting repair loop.")
                    break

                repair_attempts += 1
                logger.info(
                    f"Self-repair iteration {repair_attempts}/{max_repairs} for query: {rewritten_query[:60]}..."
                )

                # Step A: Targeted Revision Pass
                t_rev = time.time()
                revised_answer = self._revision_chain.invoke({
                    "context": formatted_context,
                    "critique_instructions": critic_decision.suggested_revision_instructions,
                    "draft_answer": current_answer,
                    "question": rewritten_query,
                })
                rev_lat = round(time.time() - t_rev, 3)
                revision_total_latency += rev_lat
                current_answer = revised_answer

                # Step B: RE-VERIFICATION (MANDATORY AUDIT RULE)
                # The revised answer MUST be audited by the Critic before certification
                t_reverify = time.time()
                critic_decision = self.critic_agent.verify(
                    answer=current_answer,
                    context=formatted_context,
                    question=rewritten_query,
                )
                reverify_lat = round(time.time() - t_reverify, 3)
                critic_total_latency += reverify_lat

                revision_trace.append({
                    "attempt": repair_attempts,
                    "action": "revision_and_reverification",
                    "answer": current_answer,
                    "verification_status": critic_decision.verification_status.value,
                    "is_valid": critic_decision.is_valid,
                    "confidence_score": critic_decision.confidence_score,
                    "failed_checks": list(critic_decision.failed_checks),
                    "critique_summary": critic_decision.critique_summary,
                    "revision_instructions": critic_decision.suggested_revision_instructions,
                    "revision_latency": rev_lat,
                    "reverify_latency": reverify_lat,
                })

            # Terminal State Handling: If loop exhausted repair budget without passing
            if not critic_decision.is_valid and repair_attempts >= max_repairs:
                critic_decision.verification_status = VerificationStatus.UNVERIFIED
                logger.warning(
                    f"Self-repair budget exhausted ({repair_attempts} iterations). Final answer marked UNVERIFIED."
                )

        total_duration = round(time.time() - t_start_total, 3)

        # 6. Extract deduplicated citations across all sources
        all_docs = sop_docs + bpmn_docs
        sources = self._extract_sources(all_docs)

        final_status = (
            critic_decision.verification_status.value
            if critic_decision
            else ("DISABLED" if not use_critic else "SKIPPED")
        )

        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "route_decision": route_decision,
            "selected_retriever": selected_retriever,
            "sop_context": fused_context.sop_context,
            "bpmn_context": fused_context.bpmn_context,
            "fused_context": fused_context,
            "fused_text": fused_context.synthesized_context,
            "formatted_context": formatted_context,
            "draft_answer": draft_answer,
            "initial_answer": draft_answer,
            "answer": current_answer,
            "final_answer": current_answer,
            "critic_decision": critic_decision,
            "was_revised": repair_attempts > 0,
            "repair_iterations": repair_attempts,
            "final_verification_status": final_status,
            "revision_trace": revision_trace,
            "failed_checks": critic_decision.failed_checks if critic_decision else [],
            "generation_trace": {
                "total_duration": total_duration,
                "draft_duration": draft_duration,
                "critic_total_duration": round(critic_total_latency, 3),
                "revision_total_duration": round(revision_total_latency, 3),
                "repair_attempts": repair_attempts,
                "max_repairs": max_repairs,
            },
            "sources": sources,
            "documents": all_docs,
        }

    def stream(
        self,
        question: str,
        k: Optional[int] = None,
        document_type: Optional[str] = None,
        use_query_agent: bool = True,
        use_router: bool = True,
    ) -> Iterator[str]:
        """Stream generated tokens in real time."""
        rewritten_query = question
        if use_query_agent:
            if self.query_agent is None:
                self.query_agent = QueryUnderstandingAgent(llm=self.llm)
            rewritten_query = self.query_agent.rewrite_query(question)

        route_decision = None
        if use_router:
            if self.router_agent is None:
                self.router_agent = RouterAgent(llm=self.llm)
            route_decision = self.router_agent.route(rewritten_query)

        source = route_decision.knowledge_source if route_decision else "SOP"
        sop_docs: List[Document] = []
        bpmn_docs: List[Document] = []

        if source == "SOP":
            sop_k = k if k is not None else 6
            bpmn_k = 2
        elif source == "BPMN":
            bpmn_k = k if k is not None else 4
            sop_k = 3
        else:  # BOTH
            sop_k = k if k is not None else 5
            bpmn_k = k if k is not None else 3

        sop_docs = self.sop_retriever.retrieve(
            query=rewritten_query,
            k=sop_k,
            document_type=document_type or "sop",
        )
        bpmn_docs = self.bpmn_retriever.retrieve(
            query=rewritten_query,
            k=bpmn_k,
        )
        if not sop_docs and not bpmn_docs:
            sop_docs = self.sop_retriever.retrieve(query=rewritten_query, k=k or 4)

        fused_context = self.fusion_layer.fuse(
            sop_documents=sop_docs,
            bpmn_documents=bpmn_docs,
            query=rewritten_query,
            route_decision=route_decision,
        )
        formatted_context = format_fused_context_block(fused_context)

        for chunk in self._core_chain.stream({
            "context": formatted_context,
            "question": rewritten_query,
        }):
            yield chunk
