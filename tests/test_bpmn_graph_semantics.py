"""
BPMN Graph Semantics and Topological Retrieval Verification Tests.
Validates the authentic production implementation of BPMN knowledge integration:
1. Directionality: Forward traversal (get_next_nodes) vs backward traversal (get_previous_nodes).
2. Branch Handling: Exclusive (XOR) gateway branch extraction with condition guards.
3. Actor Attribution: Pool/lane role preservation on atomic activities and gateway decision points.
4. Cycle Resilience: Closed-loop rework transitions terminate safely without infinite recursion.
5. Depth Boundaries: Bounded path exploration respects max_depth traversal limits.
6. Context Serialization: Deterministic serialization to structured narrative text.
7. Source Provenance: BPMNRetriever document emission preserves document_type, actor, and category metadata.
"""

import unittest
from pathlib import Path

from agentic_rag_sop.bpmn.models import (
    BPMNActor,
    BPMNNode,
    BPMNProcessGraph,
    BPMNProcessMetadata,
    BPMNSequenceFlow,
)
from agentic_rag_sop.retrieval.bpmn_retriever import BPMNRetriever


def build_sample_process_with_cycle() -> BPMNProcessGraph:
    """
    Construct an authentic ICICoS paper review workflow containing:
    - Start Event
    - Task 1: Submit Paper (Author)
    - Gateway 1: Review Decision (Editor / Reviewer)
      - Branch A (Accepted) -> Task 2: Prepare Camera Ready (Author) -> End
      - Branch B (Revision Required) -> loops back to Task 1 (Author)
    """
    meta = BPMNProcessMetadata(
        process_id="icicos_review_01",
        process_name="Proses Review dan Revisi Makalah ICICoS",
        document_domain="ICICoS",
        process_category="ICICoS Author Process",
        description="Alur penelaahan naskah, keputusan editor, dan siklus perbaikan.",
    )
    actors = [
        BPMNActor(actor_id="act_author", name="Author", lane="Pemakalah"),
        BPMNActor(actor_id="act_editor", name="Editor / Reviewer", lane="Panitia Editorial"),
    ]
    nodes = {
        "start": BPMNNode(
            node_id="start",
            name="Mulai Pengajuan",
            node_type="start_event",
            actor="Author",
            outgoing_flows=["flow_start"],
        ),
        "task_submit": BPMNNode(
            node_id="task_submit",
            name="Unggah Naskah Lengkap",
            node_type="task",
            actor="Author",
            incoming_flows=["flow_start", "flow_revisi"],
            outgoing_flows=["flow_review"],
        ),
        "gw_decision": BPMNNode(
            node_id="gw_decision",
            name="Keputusan Penelaahan Reviewer",
            node_type="gateway",
            gateway_type="exclusive",
            actor="Editor / Reviewer",
            incoming_flows=["flow_review"],
            outgoing_flows=["flow_accept", "flow_revisi"],
        ),
        "task_cr": BPMNNode(
            node_id="task_cr",
            name="Unggah Versi Camera-Ready",
            node_type="task",
            actor="Author",
            incoming_flows=["flow_accept"],
            outgoing_flows=["flow_end"],
        ),
        "end": BPMNNode(
            node_id="end",
            name="Naskah Siap Prosiding",
            node_type="end_event",
            actor="Editor / Reviewer",
            incoming_flows=["flow_end"],
        ),
    }
    flows = {
        "flow_start": BPMNSequenceFlow(
            flow_id="flow_start",
            name="Mulai",
            source_node_id="start",
            target_node_id="task_submit",
        ),
        "flow_review": BPMNSequenceFlow(
            flow_id="flow_review",
            name="Ajukan untuk Review",
            source_node_id="task_submit",
            target_node_id="gw_decision",
        ),
        "flow_accept": BPMNSequenceFlow(
            flow_id="flow_accept",
            name="Diterima",
            condition="Nilai Review >= 75 (Accepted)",
            source_node_id="gw_decision",
            target_node_id="task_cr",
        ),
        "flow_revisi": BPMNSequenceFlow(
            flow_id="flow_revisi",
            name="Revisi Mayor / Minor",
            condition="Perlu Perbaikan (Revision Required)",
            source_node_id="gw_decision",
            target_node_id="task_submit",
        ),
        "flow_end": BPMNSequenceFlow(
            flow_id="flow_end",
            name="Selesai",
            source_node_id="task_cr",
            target_node_id="end",
        ),
    }
    return BPMNProcessGraph(metadata=meta, actors=actors, nodes=nodes, flows=flows)


class TestBPMNGraphSemantics(unittest.TestCase):
    """Forensic verification test suite for directed BPMN process graph semantics."""

    def setUp(self):
        self.graph = build_sample_process_with_cycle()

    def test_01_directionality(self):
        """1. Verify directed flow transitions preserve strict directionality."""
        # Successors of task_submit should only be gw_decision
        successors = self.graph.get_next_nodes("task_submit")
        self.assertEqual(len(successors), 1)
        self.assertEqual(successors[0].node_id, "gw_decision")

        # Predecessors of task_submit should be start AND gw_decision (from revision cycle)
        predecessors = self.graph.get_previous_nodes("task_submit")
        pred_ids = {p.node_id for p in predecessors}
        self.assertEqual(pred_ids, {"start", "gw_decision"})

    def test_02_branch_handling(self):
        """2. Verify XOR decision gateway extracts all outgoing conditional branches."""
        branches = self.graph.get_decision_branches("gw_decision")
        self.assertEqual(len(branches), 2)

        conditions = {b["condition"] for b in branches}
        self.assertIn("Nilai Review >= 75 (Accepted)", conditions)
        self.assertIn("Perlu Perbaikan (Revision Required)", conditions)

        # Check target nodes
        targets = {b["target_node_id"]: b["target_actor"] for b in branches}
        self.assertEqual(targets["task_cr"], "Author")
        self.assertEqual(targets["task_submit"], "Author")

    def test_03_actor_attribution(self):
        """3. Verify actor roles are accurately attributed to process activities."""
        self.assertEqual(self.graph.nodes["task_submit"].actor, "Author")
        self.assertEqual(self.graph.nodes["gw_decision"].actor, "Editor / Reviewer")
        self.assertEqual(self.graph.nodes["task_cr"].actor, "Author")

    def test_04_handling_of_cycles(self):
        """4. Verify cyclic workflow terminates safely in serialization and counting."""
        # Narrative context serialization must terminate without infinite loop
        narrative = self.graph.to_narrative_context()
        self.assertIn("Urutan Alur Langkah Proses", narrative)
        self.assertIn("Unggah Naskah Lengkap", narrative)
        self.assertIn("Keputusan Penelaahan Reviewer", narrative)

        # Preceding step count must terminate on cyclic graphs
        count = self.graph.count_preceding_steps("task_cr")
        # Preceding unique task/gateway steps: task_submit, gw_decision
        self.assertEqual(count, 2)

    def test_05_depth_boundaries(self):
        """5. Verify trace_path respects max_depth limit."""
        # With sufficient depth, path from start to end exists
        path_full = self.graph.trace_path("start", "end", max_depth=10)
        self.assertTrue(len(path_full) >= 4)
        self.assertEqual(path_full[0].node_id, "start")
        self.assertEqual(path_full[-1].node_id, "end")

        # With max_depth=2, end node cannot be reached
        path_shallow = self.graph.trace_path("start", "end", max_depth=2)
        self.assertEqual(len(path_shallow), 0)

    def test_06_graph_to_context_serialization(self):
        """6. Verify serialization includes gateway branching logic and metadata."""
        text = self.graph.to_narrative_context()
        self.assertIn("DIAGRAM PROSES BISNIS ICICOS", text)
        self.assertIn("Author", text)
        self.assertIn("Editor / Reviewer", text)
        self.assertIn("Titik Keputusan (Gateway)", text)
        self.assertIn("Jika 'Nilai Review >= 75 (Accepted)' -> Unggah Versi Camera-Ready (Author)", text)

    def test_07_source_provenance_and_retrieval(self):
        """7. Verify BPMNRetriever document emission preserves required metadata."""
        class MockStorageManager:
            def load_all_graphs(self, auto_ingest=True):
                return [build_sample_process_with_cycle()]

        retriever = BPMNRetriever(storage_manager=MockStorageManager())
        docs = retriever.retrieve(query="Bagaimana alur revisi dan siapa reviewer?", k=1)
        self.assertEqual(len(docs), 1)

        doc = docs[0]
        self.assertEqual(doc.metadata["document_type"], "bpmn")
        self.assertEqual(doc.metadata["process_name"], "Proses Review dan Revisi Makalah ICICoS")
        self.assertEqual(doc.metadata["category"], "ICICoS Author Process")
        self.assertIn("Author", doc.metadata["actor"])
        self.assertIn("ANALISIS PERCABANGAN KEPUTUSAN", doc.page_content.upper())


if __name__ == "__main__":
    unittest.main()
