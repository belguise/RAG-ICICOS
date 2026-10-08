"""
Unit tests for core software components of RAG-ICICOS.
Tests verify deterministic local modules without requiring external API keys.
"""

import os
import sys
import unittest
from pathlib import Path

# Add src to path for direct invocation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agentic_rag_sop.config.settings import Settings
from agentic_rag_sop.bpmn.models import (
    BPMNProcessMetadata,
    BPMNActor,
    BPMNNode,
    BPMNSequenceFlow,
    BPMNProcessGraph,
)
from langchain_core.documents import Document
from agentic_rag_sop.ingestion.chunker import ProcessAwareChunker


class TestSettings(unittest.TestCase):
    """Test configuration settings handling."""

    def test_settings_defaults(self):
        s = Settings()
        self.assertEqual(s.GEMINI_MODEL, "gemini-3.5-flash-lite")
        self.assertEqual(s.LLM_TEMPERATURE, 0.0)
        self.assertEqual(s.CHUNK_SIZE, 800)
        self.assertEqual(s.CHUNK_OVERLAP, 150)
        self.assertIsInstance(s.persist_path, Path)
        self.assertIsInstance(s.raw_path, Path)


class TestBPMNModels(unittest.TestCase):
    """Test BPMN graph data structures and topology."""

    def test_process_graph_construction(self):
        meta = BPMNProcessMetadata(
            process_id="proc_01",
            process_name="Test Submission Process",
            document_domain="ICICoS",
        )
        actors = [
            BPMNActor(actor_id="a1", name="Author", lane="Author"),
            BPMNActor(actor_id="a2", name="Reviewer", lane="Reviewer"),
        ]
        nodes = {
            "start": BPMNNode(node_id="start", name="Start Event", node_type="start_event", actor="Author", outgoing_flows=["f1"]),
            "task_1": BPMNNode(node_id="task_1", name="Submit Manuscript", node_type="task", actor="Author", incoming_flows=["f1"], outgoing_flows=["f2"]),
            "gw_1": BPMNNode(node_id="gw_1", name="Review Decision", node_type="gateway", gateway_type="exclusive", actor="Reviewer", incoming_flows=["f2"], outgoing_flows=["f3"]),
            "end": BPMNNode(node_id="end", name="Accepted", node_type="end_event", actor="Reviewer", incoming_flows=["f3"]),
        }
        flows = {
            "f1": BPMNSequenceFlow(flow_id="f1", source_node_id="start", target_node_id="task_1"),
            "f2": BPMNSequenceFlow(flow_id="f2", source_node_id="task_1", target_node_id="gw_1"),
            "f3": BPMNSequenceFlow(flow_id="f3", source_node_id="gw_1", target_node_id="end", condition="Score >= 3.0"),
        }

        graph = BPMNProcessGraph(metadata=meta, actors=actors, nodes=nodes, flows=flows)

        self.assertEqual(len(graph.nodes), 4)
        self.assertEqual(len(graph.flows), 3)

        # Test successors
        succs = graph.get_next_nodes("start")
        self.assertEqual(len(succs), 1)
        self.assertEqual(succs[0].node_id, "task_1")

        # Test gateway branches
        branches = graph.get_decision_branches("gw_1")
        self.assertEqual(len(branches), 1)
        self.assertEqual(branches[0]["target_node_id"], "end")
        self.assertEqual(branches[0]["condition"], "Score >= 3.0")

        # Test path tracing
        path = graph.trace_path("start", "end")
        self.assertEqual(len(path), 4)
        self.assertEqual([n.node_id for n in path], ["start", "task_1", "gw_1", "end"])


class TestProcessAwareChunker(unittest.TestCase):
    """Test document chunking logic."""

    def test_chunker_basic(self):
        chunker = ProcessAwareChunker(chunk_size=200, chunk_overlap=50)
        doc = Document(
            page_content=(
                "# Section 1: Overview\n"
                "This is an introductory paragraph explaining the operational policy. "
                "It outlines key responsibilities for staff members.\n\n"
                "## Section 1.1: Guidelines\n"
                "All financial requests must follow internal standard procedures. "
                "Supporting receipts are strictly required for reimbursement."
            ),
            metadata={"source_document": "test_sop.md", "document_type": "sop"}
        )
        chunks = chunker.split_documents([doc])
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertIsInstance(c.page_content, str)
            self.assertIn("document_name", c.metadata)


if __name__ == "__main__":
    unittest.main(verbosity=2)
