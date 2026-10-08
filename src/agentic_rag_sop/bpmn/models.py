"""
Canonical BPMN Intermediate Representation (IR) Models.
Phase 3 BPMN Knowledge Integration: Unified domain models representing
ICICoS business process workflows, actors, tasks, gateways, and sequence flows.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class BPMNProcessMetadata(BaseModel):
    """Metadata describing an ICICoS business process."""
    process_id: str
    process_name: str
    document_domain: str = "ICICoS"
    process_category: str = Field(
        default="ICICoS Operational Workflow",
        description="Category: ICICoS Author Process, ICICoS Payment Process, ICICoS Financial Process, ICICoS Committee Process, ICICoS Operational Workflow"
    )
    description: str = ""
    source_file: str = ""


class BPMNActor(BaseModel):
    """Organizational role or actor extracted from a BPMN pool or lane."""
    actor_id: str
    name: str = "Panitia ICICoS"
    pool: str = "ICICoS"
    lane: str = "Panitia"
    department: str = "ICICoS Committee"


class BPMNSequenceFlow(BaseModel):
    """Directed transition connecting two process nodes."""
    flow_id: str
    name: Optional[str] = None
    source_node_id: str
    target_node_id: str
    condition: Optional[str] = None


class BPMNNode(BaseModel):
    """Atomic step in a business process (task, gateway, start/end event)."""
    node_id: str
    name: str
    node_type: Literal["start_event", "end_event", "task", "gateway"] = "task"
    actor: str = "Semua Pihak / Terkait"
    gateway_type: Optional[Literal["exclusive", "parallel", "inclusive"]] = None
    incoming_flows: List[str] = Field(default_factory=list)
    outgoing_flows: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)


class BPMNProcessGraph(BaseModel):
    """
    Canonical Process Graph representation.
    Provides topological query methods for agentic workflow reasoning.
    """
    metadata: BPMNProcessMetadata
    actors: List[BPMNActor] = Field(default_factory=list)
    nodes: Dict[str, BPMNNode] = Field(default_factory=dict)
    flows: Dict[str, BPMNSequenceFlow] = Field(default_factory=dict)

    def get_next_nodes(self, node_id: str) -> List[BPMNNode]:
        """Find all immediate succeeding nodes reachable via outgoing sequence flows."""
        node = self.nodes.get(node_id)
        if not node:
            return []

        next_nodes = []
        for flow_id in node.outgoing_flows:
            flow = self.flows.get(flow_id)
            if flow and flow.target_node_id in self.nodes:
                next_nodes.append(self.nodes[flow.target_node_id])
        return next_nodes

    def get_previous_nodes(self, node_id: str) -> List[BPMNNode]:
        """Find all immediate preceding nodes leading into this node."""
        node = self.nodes.get(node_id)
        if not node:
            return []

        prev_nodes = []
        for flow_id in node.incoming_flows:
            flow = self.flows.get(flow_id)
            if flow and flow.source_node_id in self.nodes:
                prev_nodes.append(self.nodes[flow.source_node_id])
        return prev_nodes

    def get_decision_branches(self, gateway_id: str) -> List[Dict[str, Any]]:
        """Extract decision branches, condition labels, and target nodes for a gateway."""
        gw = self.nodes.get(gateway_id)
        if not gw or gw.node_type != "gateway":
            return []

        branches = []
        for flow_id in gw.outgoing_flows:
            flow = self.flows.get(flow_id)
            if flow and flow.target_node_id in self.nodes:
                target = self.nodes[flow.target_node_id]
                branches.append({
                    "condition": flow.condition or flow.name or "Default / Jalur Standar",
                    "target_node_id": target.node_id,
                    "target_name": target.name,
                    "target_actor": target.actor,
                })
        return branches

    def trace_path(self, start_node_id: str, end_node_id: str, max_depth: int = 15) -> List[BPMNNode]:
        """Perform breadth-first traversal to find the execution path between two nodes."""
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            return []

        queue = [[start_node_id]]
        visited = set()

        while queue:
            path = queue.pop(0)
            curr = path[-1]
            if curr == end_node_id:
                return [self.nodes[nid] for nid in path]

            if curr not in visited and len(path) < max_depth:
                visited.add(curr)
                for flow_id in self.nodes[curr].outgoing_flows:
                    flow = self.flows.get(flow_id)
                    if flow and flow.target_node_id not in visited and flow.target_node_id in self.nodes:
                        queue.append(path + [flow.target_node_id])

        return []

    def count_preceding_steps(self, target_node_id: str) -> int:
        """Count unique preceding task/gateway steps leading to target node."""
        if target_node_id not in self.nodes:
            return 0

        visited = set()
        stack = [target_node_id]

        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                node = self.nodes.get(curr)
                if node:
                    for flow_id in node.incoming_flows:
                        flow = self.flows.get(flow_id)
                        if flow and flow.source_node_id not in visited and flow.source_node_id in self.nodes:
                            stack.append(flow.source_node_id)

        # Exclude target node itself
        visited.discard(target_node_id)
        # Only count tasks and gateways
        return len([nid for nid in visited if self.nodes[nid].node_type in ("task", "gateway")])

    def to_narrative_context(self) -> str:
        """
        Render the process graph into a chronological, highly readable text format
        optimized for RAG context injection.
        """
        lines = [
            f"=== DIAGRAM PROSES BISNIS ICICOS: {self.metadata.process_name.upper()} ===",
            f"Kategori Domain: {self.metadata.process_category} (Domain: {self.metadata.document_domain})",
        ]
        if self.metadata.description:
            lines.append(f"Deskripsi: {self.metadata.description}")

        # List Actors
        actor_names = [a.name for a in self.actors if a.name]
        if actor_names:
            lines.append(f"Aktor & Pihak Terlibat: {', '.join(set(actor_names))}")

        lines.append("\nUrutan Alur Langkah Proses (Workflow Sequence):")

        # Find Start Events
        start_nodes = [n for n in self.nodes.values() if n.node_type == "start_event"]
        ordered_nodes = []
        visited = set()

        def traverse(node_id: str, depth: int = 0):
            if node_id in visited or depth > 30:
                return
            visited.add(node_id)
            node = self.nodes.get(node_id)
            if node:
                ordered_nodes.append(node)
                for flow_id in node.outgoing_flows:
                    flow = self.flows.get(flow_id)
                    if flow and flow.target_node_id:
                        traverse(flow.target_node_id, depth + 1)

        for sn in start_nodes:
            traverse(sn.node_id)

        # Add remaining unvisited nodes
        for nid, n in self.nodes.items():
            if nid not in visited:
                ordered_nodes.append(n)

        for idx, node in enumerate(ordered_nodes, start=1):
            if node.node_type == "start_event":
                lines.append(f"  [{idx}] Titik Awal (Start Event): {node.name}")
            elif node.node_type == "end_event":
                lines.append(f"  [{idx}] Titik Akhir (End Event): {node.name}")
            elif node.node_type == "gateway":
                branches = self.get_decision_branches(node.node_id)
                branch_desc = "; ".join(f"Jika '{b['condition']}' -> {b['target_name']} ({b['target_actor']})" for b in branches)
                lines.append(f"  [{idx}] Titik Keputusan (Gateway): {node.name} [Aktor: {node.actor}]")
                if branch_desc:
                    lines.append(f"       Cabang Keputusan: {branch_desc}")
            else:
                next_nodes = self.get_next_nodes(node.node_id)
                next_desc = f" -> Lanjut ke: {', '.join(f'{nxt.name} ({nxt.actor})' for nxt in next_nodes)}" if next_nodes else ""
                lines.append(f"  [{idx}] Aktivitas: {node.name} [Penanggung Jawab: {node.actor}]{next_desc}")

        return "\n".join(lines)


# Backward compatibility aliases
ProcessStep = BPMNNode
BPMNProcess = BPMNProcessGraph
BPMNActivity = BPMNNode
BPMNGateway = BPMNNode
BPMNLane = BPMNActor
