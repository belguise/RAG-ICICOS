"""
BPMN Process Graph Retriever Implementation.
Phase 3 BPMN Knowledge Integration: Performs topological graph reasoning over
ICICoS business process models (tasks, gateways, sequence flows, and swimlane actors).
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document

from agentic_rag_sop.bpmn.models import BPMNNode, BPMNProcessGraph
from agentic_rag_sop.bpmn.storage import BPMNStorageManager
from agentic_rag_sop.retrieval.base import BaseKnowledgeRetriever


class BPMNRetriever(BaseKnowledgeRetriever):
    """
    Retriever for BPMN Process Models and Workflow Topology.
    Performs deterministic graph reasoning to answer workflow sequence,
    approval hierarchy, decision gateway, and actor responsibility inquiries.
    """

    def __init__(self, storage_manager: Optional[BPMNStorageManager] = None):
        self.storage_manager = storage_manager or BPMNStorageManager()
        self.graphs: List[BPMNProcessGraph] = self.storage_manager.load_all_graphs(auto_ingest=True)

    @property
    def retriever_name(self) -> str:
        return "BPMN Retriever"

    def _score_graph_relevance(self, graph: BPMNProcessGraph, query_words: set) -> float:
        """Calculate overlap score between user query words and graph elements."""
        score = 0.0

        # Match against process name
        proc_words = set(re.findall(r"\w+", graph.metadata.process_name.lower()))
        p_overlap = len(query_words.intersection(proc_words))
        score += p_overlap * 3.0

        # Match against actors
        for actor in graph.actors:
            act_words = set(re.findall(r"\w+", actor.name.lower()))
            if query_words.intersection(act_words):
                score += 2.0

        # Match against node names
        for node in graph.nodes.values():
            node_words = set(re.findall(r"\w+", node.name.lower()))
            overlap = len(query_words.intersection(node_words))
            if overlap:
                score += overlap * 1.5

        return score

    def _perform_graph_reasoning(self, graph: BPMNProcessGraph, query: str) -> str:
        """
        Analyze the query against the process graph and generate
        targeted topological reasoning text.
        """
        q = query.lower()
        reasoning_lines = [
            f"=== TOPOLOGI PROSES BISNIS BPMN: {graph.metadata.process_name.upper()} ===",
            f"Domain: {graph.metadata.document_domain} | Kategori: {graph.metadata.process_category}",
        ]

        # 1. Check for Next Step / Next Actor reasoning ("setelah", "berikutnya", "next", "approval")
        if any(w in q for w in ["setelah", "berikutnya", "lanjut ke", "approval"]):
            # Find which node is currently mentioned
            matched_nodes: List[Tuple[BPMNNode, int]] = []
            q_words = set(re.findall(r"\w+", q))
            for node in graph.nodes.values():
                n_words = set(re.findall(r"\w+", node.name.lower()))
                overlap = len(q_words.intersection(n_words))
                if overlap > 0:
                    matched_nodes.append((node, overlap))

            if matched_nodes:
                matched_nodes.sort(key=lambda x: x[1], reverse=True)
                current_node = matched_nodes[0][0]
                next_nodes = graph.get_next_nodes(current_node.node_id)

                reasoning_lines.append(f"\n[Analisis Alur Transisi Langkah]")
                reasoning_lines.append(f"• Titik/Tahapan Saat Ini: {current_node.name} (Penanggung Jawab: {current_node.actor})")

                if next_nodes:
                    reasoning_lines.append(f"• Langkah / Pihak Selanjutnya (Successor):")
                    for nxt in next_nodes:
                        reasoning_lines.append(f"  - {nxt.name} [Aktor Penanggung Jawab: {nxt.actor}] (Tipe: {nxt.node_type})")
                else:
                    reasoning_lines.append(f"• Tidak ada tahapan lanjutan (Titik Selesai/End Event).")

        # 2. Check for Decision Gateway / Rejection Branching ("ditolak", "tidak", "jika", "apabila", "cabang")
        gateways = [n for n in graph.nodes.values() if n.node_type == "gateway"]
        if gateways and any(w in q for w in ["tolak", "revisi", "tidak", "gagal", "cabang", "kondisi"]):
            reasoning_lines.append(f"\n[Analisis Percabangan Keputusan (Decision Gateways)]")
            for gw in gateways:
                branches = graph.get_decision_branches(gw.node_id)
                reasoning_lines.append(f"• Titik Keputusan: {gw.name} (Aktor: {gw.actor})")
                for b in branches:
                    reasoning_lines.append(f"  - Jalur Kondisi '{b['condition']}': Mengarah ke {b['target_name']} (Aktor: {b['target_actor']})")

        # 3. Check for Step Counting / Preceding Steps ("berapa tahapan", "berapa langkah", "sebelum")
        if any(w in q for w in ["berapa tahapan", "berapa langkah", "jumlah langkah", "sebelum"]):
            # Find target node
            matched = None
            q_words = set(re.findall(r"\w+", q))
            for node in graph.nodes.values():
                n_words = set(re.findall(r"\w+", node.name.lower()))
                if len(q_words.intersection(n_words)) >= 2:
                    matched = node
                    break

            if matched:
                preceding_count = graph.count_preceding_steps(matched.node_id)
                reasoning_lines.append(f"\n[Analisis Hitungan Tahapan Proses]")
                reasoning_lines.append(f"• Target Tahapan: {matched.name} (Aktor: {matched.actor})")
                reasoning_lines.append(f"• Jumlah Tahapan Sebelum Target: {preceding_count} langkah proses.")

        # 4. Append Complete Topological Sequence
        reasoning_lines.append(f"\n[Struktur Lengkap Alur Proses]")
        reasoning_lines.append(graph.to_narrative_context())

        return "\n".join(reasoning_lines)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        process_name: Optional[str] = None,
        actor: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Retrieve relevant BPMN process graphs and perform topological graph reasoning.
        Returns LangChain Document objects suitable for RAG generation.
        """
        if not self.graphs:
            self.graphs = self.storage_manager.load_all_graphs(auto_ingest=True)

        q_clean = query.strip()
        q_words = set(re.findall(r"\w+", q_clean.lower()))

        scored_graphs: List[Tuple[BPMNProcessGraph, float]] = []

        for g in self.graphs:
            # If explicit process_name filter passed
            if process_name and process_name.lower() not in g.metadata.process_name.lower():
                continue
            # If explicit actor filter passed
            if actor:
                act_names = [a.name.lower() for a in g.actors]
                if not any(actor.lower() in an for an in act_names):
                    continue

            score = self._score_graph_relevance(g, q_words)
            scored_graphs.append((g, score))

        # Sort by score descending
        scored_graphs.sort(key=lambda x: x[1], reverse=True)
        top_k = k or 2
        selected = [g for g, s in scored_graphs[:top_k] if s > 0]

        # If no strict keyword match, return top general graphs
        if not selected and scored_graphs:
            selected = [scored_graphs[0][0]]

        documents: List[Document] = []
        for g in selected:
            content = self._perform_graph_reasoning(g, q_clean)
            actor_names = ", ".join(set(a.name for a in g.actors if a.name)) or "Panitia ICICoS"

            doc = Document(
                page_content=content,
                metadata={
                    "document_name": f"{g.metadata.process_name} (BPMN Workflow)",
                    "document_type": "bpmn",
                    "category": g.metadata.process_category,
                    "process_name": g.metadata.process_name,
                    "department": "ICICoS Committee",
                    "actor": actor_names,
                    "page_number": "BPMN Process Graph",
                    "chunk_index": 0,
                },
            )
            documents.append(doc)

        return documents
