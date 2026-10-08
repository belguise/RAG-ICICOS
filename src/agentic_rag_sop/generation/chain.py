"""
LCEL Generation Chain with Phase 4 Context Fusion and Phase 5 Critic Verification.
Combines query rewriting, intelligent routing, multi-source retrieval (SOP & BPMN),
context fusion with conflict resolution, Gemini answer generation, and
Critic-guided self-correction verification.
"""

from typing import Any, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

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
from agentic_rag_sop.agents.critic import CriticVerificationAgent, CriticDecision
from agentic_rag_sop.fusion.context_fusion import ContextFusionLayer, FusedContext


class SOPRAGChain:
    """
    End-to-end RAG question-answering chain with:
    - Query Understanding Agent (Phase 2.1)
    - Router Agent (Phase 2.2)
    - Separated Retriever Layer: SOPRetriever & BPMNRetriever (Phase 2.3)
    - BPMN Process Graph Reasoning (Phase 3)
    - Context Fusion Layer with ICICoS Domain Guardrails (Phase 4)
    - Critic Verification Agent & Self-Correction Revision Loop (Phase 5)
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
    ):
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

        # Core generation and revision chains
        self._core_chain = self.prompt_template | self.llm | self.output_parser
        self._revision_chain = self.revision_template | self.llm | self.output_parser

    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Deduplicate and extract structured source citations from retrieved docs."""
        seen = set()
        sources = []

        for doc in documents:
            meta = doc.metadata
            doc_name = meta.get("document_name", "Unknown")
            page = meta.get("page_number", "N/A")
            proc = meta.get("process_name", "General")
            actor = meta.get("actor", "General")
            chunk_idx = meta.get("chunk_index", 0)

            key = (doc_name, str(page), proc)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "document_name": doc_name,
                    "document_type": meta.get("document_type", "sop"),
                    "category": meta.get("category", "general"),
                    "page_number": page,
                    "process_name": proc,
                    "actor": actor,
                    "chunk_index": chunk_idx,
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
    ) -> Dict[str, Any]:
        """
        Execute full RAG generation with upstream Query Understanding, Router Agents,
        separated Retriever dispatching, Context Fusion synthesis, draft generation,
        and downstream Critic verification with iterative repair.
        """
        # 0. Query Understanding Agent (Normalization, typo fixes, acronym expansion)
        rewritten_query = question
        if use_query_agent:
            if self.query_agent is None:
                self.query_agent = QueryUnderstandingAgent(llm=self.llm)
            rewritten_query = self.query_agent.rewrite_query(question)

        # 1. Router Agent (Intent & Knowledge Source Classification)
        route_decision = None
        selected_retriever = self.sop_retriever.retriever_name
        if use_router:
            if self.router_agent is None:
                self.router_agent = RouterAgent(llm=self.llm)
            route_decision = self.router_agent.route(rewritten_query)

        # 2. Dispatch to designated retriever(s) based on routing decision (Weighted Dual Retrieval)
        source = route_decision.knowledge_source if route_decision else "SOP"
        sop_docs: List[Document] = []
        bpmn_docs: List[Document] = []

        if source == "SOP":
            sop_k = k if k is not None else 6
            bpmn_k = 2
            selected_retriever = f"{self.sop_retriever.retriever_name} (Primary) + {self.bpmn_retriever.retriever_name} (Complementary)"
        elif source == "BPMN":
            bpmn_k = k if k is not None else 4
            sop_k = 3
            selected_retriever = f"{self.bpmn_retriever.retriever_name} (Primary) + {self.sop_retriever.retriever_name} (Complementary)"
        else:  # BOTH
            sop_k = k if k is not None else 5
            bpmn_k = k if k is not None else 3
            selected_retriever = f"{self.sop_retriever.retriever_name} + {self.bpmn_retriever.retriever_name}"

        # Dual retrieval with adaptive priority weighting
        sop_docs = self.sop_retriever.retrieve(
            query=rewritten_query,
            k=sop_k,
            document_type=document_type or "sop",
            actor=actor,
            process_name=process_name,
        )
        bpmn_docs = self.bpmn_retriever.retrieve(
            query=rewritten_query,
            k=bpmn_k,
            actor=actor,
            process_name=process_name,
        )

        # Fallback if both returned nothing
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

        # 4. Call Gemini LLM to produce draft answer
        draft_answer = self._core_chain.invoke({
            "context": formatted_context,
            "question": rewritten_query,
        })

        # 5. Critic Verification Agent (Phase 5)
        critic_decision: Optional[CriticDecision] = None
        final_answer = draft_answer
        was_revised = False

        if use_critic:
            if self.critic_agent is None:
                self.critic_agent = CriticVerificationAgent(llm=self.llm)

            critic_decision = self.critic_agent.verify(
                answer=draft_answer,
                context=formatted_context,
                question=rewritten_query,
            )

            # Targeted Revision Loop: If draft failed audit, trigger guided repair
            if not critic_decision.is_valid and critic_decision.suggested_revision_instructions:
                revised_answer = self._revision_chain.invoke({
                    "context": formatted_context,
                    "critique_instructions": critic_decision.suggested_revision_instructions,
                    "draft_answer": draft_answer,
                    "question": rewritten_query,
                })
                final_answer = revised_answer
                was_revised = True

        # 6. Extract deduplicated citations across all sources
        all_docs = sop_docs + bpmn_docs
        sources = self._extract_sources(all_docs)

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
            "answer": final_answer,
            "critic_decision": critic_decision,
            "was_revised": was_revised,
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
