"""
Unified BPMN & Bizagi Process Parser.
Phase 3 BPMN Knowledge Integration: Extracts process workflows, actors,
tasks, decision gateways, and sequence flows from both:
1. Bizagi Modeler (.bpm) nested ZIP archives (WfMC XPDL 2.2 schema)
2. Standard BPMN 2.0 XML (.bpmn) files (OMG BPMN 2.0 schema)
"""

import io
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from agentic_rag_sop.bpmn.models import (
    BPMNActor,
    BPMNNode,
    BPMNProcessGraph,
    BPMNProcessMetadata,
    BPMNSequenceFlow,
)


def categorize_icicos_process(process_name: str) -> str:
    """Categorize process name into an ICICoS operational domain category."""
    pn = process_name.lower()
    if any(k in pn for k in ["va", "pembayaran", "bayar", "rekening", "mbanking", "pencairan"]):
        return "ICICoS Payment Process"
    elif any(k in pn for k in ["rab", "cmanis", "wcu", "anggaran", "sponsor"]):
        return "ICICoS Financial Process"
    elif any(k in pn for k in ["acceptence", "submit", "paper", "author", "edas", "present", "notifikasi"]):
        return "ICICoS Author Process"
    elif any(k in pn for k in ["sk ", "sk draft", "konsumsi rapat", "purchase", "surat keputusan", "rapat to pay"]):
        return "ICICoS Committee Process"
    elif any(k in pn for k in ["spj", "honor", "boc", "boa", "hotel", "akomodasi", "seminar kit"]):
        return "ICICoS Operational Workflow"
    return "ICICoS Operational Workflow"


class UnifiedBPMNParser:
    """
    Parser capable of parsing both standard OMG BPMN 2.0 XML (.bpmn)
    and Bizagi Modeler project packages (.bpm).
    """

    XPDL_NS = {"xpdl": "http://www.wfmc.org/2009/XPDL2.2"}
    OMG_NS = {"bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL"}

    def parse_file(self, file_path: Union[str, Path]) -> List[BPMNProcessGraph]:
        """
        Parse a .bpm or .bpmn file into one or more canonical BPMNProcessGraph instances.
        A single Bizagi .bpm archive typically contains multiple processes/diagrams.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".bpm":
            return self._parse_bizagi_bpm(path)
        elif ext in (".bpmn", ".xml"):
            graph = self._parse_omg_bpmn(path)
            return [graph] if graph else []
        else:
            raise ValueError(f"Unsupported BPMN file format: {ext} for {file_path}")

    def _parse_bizagi_bpm(self, path: Path) -> List[BPMNProcessGraph]:
        """Parse Bizagi Modeler (.bpm) nested ZIP archive containing XPDL 2.2 diagrams."""
        graphs: List[BPMNProcessGraph] = []

        try:
            with zipfile.ZipFile(path, "r") as z:
                # Find all .diag files in root
                diag_files = [n for n in z.namelist() if n.endswith(".diag")]

                for diag_name in diag_files:
                    diag_bytes = z.read(diag_name)
                    with zipfile.ZipFile(io.BytesIO(diag_bytes), "r") as diag_z:
                        if "Diagram.xml" not in diag_z.namelist():
                            continue

                        xml_data = diag_z.read("Diagram.xml")
                        diagram_graphs = self._parse_bizagi_diagram_xml(xml_data, source_name=path.name)
                        graphs.extend(diagram_graphs)

        except Exception as e:
            # If ZIP read fails, log and return empty
            return []

        # Deduplicate and filter out empty or meta processes (e.g. 'Main Process' with 0 nodes)
        filtered_graphs = []
        seen_names = set()
        for g in graphs:
            name = g.metadata.process_name.strip()
            if len(g.nodes) >= 2 and name not in ("Main Process", "Dokumen Lampiran", "Dokumen Pendukung"):
                if name not in seen_names:
                    seen_names.add(name)
                    filtered_graphs.append(g)

        return filtered_graphs

    def _parse_bizagi_diagram_xml(self, xml_bytes: bytes, source_name: str) -> List[BPMNProcessGraph]:
        """Parse individual Diagram.xml file from Bizagi XPDL archive."""
        root = ET.fromstring(xml_bytes)
        graphs = []

        # 1. Parse Swimlanes (Lanes & YCoordinate ranges)
        lane_bounds: List[Tuple[str, float, float]] = []  # (LaneName, y_min, y_max)
        actors: List[BPMNActor] = []

        for lane_elem in root.findall(".//xpdl:Lane", self.XPDL_NS):
            lane_id = lane_elem.get("Id", "")
            lane_name = lane_elem.get("Name", "Panitia").strip()
            if not lane_name:
                lane_name = "Panitia ICICoS"

            actors.append(BPMNActor(
                actor_id=lane_id,
                name=lane_name,
                pool="Panitia ICICoS",
                lane=lane_name,
                department="ICICoS Committee",
            ))

            coords = lane_elem.find(".//xpdl:Coordinates", self.XPDL_NS)
            ngi = lane_elem.find(".//xpdl:NodeGraphicsInfo", self.XPDL_NS)
            if coords is not None and ngi is not None:
                y_start = float(coords.get("YCoordinate", 0.0))
                height = float(ngi.get("Height", 300.0))
                lane_bounds.append((lane_name, y_start, y_start + height))

        # Default lane bounds fallback if none present
        if not lane_bounds and actors:
            lane_bounds.append((actors[0].name, 0.0, 99999.0))

        def resolve_actor_by_y(y_coord: float) -> str:
            for lname, y_min, y_max in lane_bounds:
                if y_min <= y_coord <= y_max:
                    return lname
            return actors[0].name if actors else "Panitia ICICoS"

        # 2. Parse WorkflowProcesses
        for proc in root.findall(".//xpdl:WorkflowProcess", self.XPDL_NS):
            proc_id = proc.get("Id", "")
            raw_name = proc.get("Name", "ICICoS Process").strip()
            # Clean newlines from process name
            clean_name = " ".join(raw_name.split())
            category = categorize_icicos_process(clean_name)

            metadata = BPMNProcessMetadata(
                process_id=proc_id,
                process_name=clean_name,
                document_domain="ICICoS",
                process_category=category,
                description=f"Alur proses bisnis operasional ICICoS: {clean_name}",
                source_file=source_name,
            )

            nodes: Dict[str, BPMNNode] = {}
            flows: Dict[str, BPMNSequenceFlow] = {}

            # Parse Activities
            for act in proc.findall(".//xpdl:Activity", self.XPDL_NS):
                act_id = act.get("Id", "")
                raw_act_name = act.get("Name", "").strip()
                if not raw_act_name:
                    continue
                act_name = " ".join(raw_act_name.split())

                # Resolve Actor via YCoordinate
                y_val = 0.0
                coords = act.find(".//xpdl:Coordinates", self.XPDL_NS)
                if coords is not None:
                    try:
                        y_val = float(coords.get("YCoordinate", 0.0))
                    except ValueError:
                        y_val = 0.0
                actor = resolve_actor_by_y(y_val)

                # Determine Node Type
                route = act.find(".//xpdl:Route", self.XPDL_NS)
                event = act.find(".//xpdl:Event", self.XPDL_NS)

                node_type = "task"
                gateway_type = None

                if route is not None:
                    node_type = "gateway"
                    raw_gw = (route.get("GatewayType") or "Exclusive").lower()
                    gateway_type = "exclusive" if "exclusive" in raw_gw else "parallel"
                elif event is not None:
                    if act.find(".//xpdl:StartEvent", self.XPDL_NS) is not None or "mulai" in act_name.lower() or "start" in act_name.lower():
                        node_type = "start_event"
                    else:
                        node_type = "end_event"
                elif "?" in act_name or "apakah" in act_name.lower() or "cocok" in act_name.lower() or "sesuai" in act_name.lower():
                    # Bizagi often defines decision points as activities with question marks
                    node_type = "gateway"
                    gateway_type = "exclusive"

                nodes[act_id] = BPMNNode(
                    node_id=act_id,
                    name=act_name,
                    node_type=node_type,
                    actor=actor,
                    gateway_type=gateway_type,
                )

            # Parse Transitions (Sequence Flows)
            for trans in proc.findall(".//xpdl:Transition", self.XPDL_NS):
                trans_id = trans.get("Id", "")
                from_id = trans.get("From", "")
                to_id = trans.get("To", "")
                raw_trans_name = trans.get("Name")
                condition = raw_trans_name.strip() if raw_trans_name else None

                # Check inside <Condition> tag if present
                cond_elem = trans.find(".//xpdl:Condition", self.XPDL_NS)
                if cond_elem is not None and cond_elem.text:
                    condition = cond_elem.text.strip()

                if from_id in nodes and to_id in nodes:
                    flow = BPMNSequenceFlow(
                        flow_id=trans_id,
                        name=condition,
                        source_node_id=from_id,
                        target_node_id=to_id,
                        condition=condition,
                    )
                    flows[trans_id] = flow
                    nodes[from_id].outgoing_flows.append(trans_id)
                    nodes[to_id].incoming_flows.append(trans_id)

            if nodes:
                graphs.append(BPMNProcessGraph(
                    metadata=metadata,
                    actors=actors,
                    nodes=nodes,
                    flows=flows,
                ))

        return graphs

    def _parse_omg_bpmn(self, path: Path) -> Optional[BPMNProcessGraph]:
        """Parse standard OMG BPMN 2.0 XML (.bpmn) file."""
        tree = ET.parse(str(path))
        root = tree.getroot()

        # Find process element
        proc_elem = None
        for elem in root.iter():
            tag_clean = elem.tag.split("}")[-1]
            if tag_clean == "process":
                proc_elem = elem
                break

        if proc_elem is None:
            return None

        raw_name = proc_elem.get("name") or proc_elem.get("id") or path.stem
        clean_name = " ".join(raw_name.split())
        category = categorize_icicos_process(clean_name)

        metadata = BPMNProcessMetadata(
            process_id=proc_elem.get("id", path.stem),
            process_name=clean_name,
            document_domain="ICICoS",
            process_category=category,
            description=f"Alur proses bisnis operasional ICICoS: {clean_name}",
            source_file=path.name,
        )

        actors_map: Dict[str, str] = {}  # node_id -> lane_name
        actors_list: List[BPMNActor] = []

        for lane in proc_elem.findall(".//{http://www.omg.org/spec/BPMN/20100524/MODEL}lane"):
            lane_name = lane.get("name", "Panitia").strip()
            lane_id = lane.get("id", "")
            actors_list.append(BPMNActor(
                actor_id=lane_id,
                name=lane_name,
                pool="ICICoS",
                lane=lane_name,
                department="ICICoS Committee",
            ))
            for fnr in lane.findall(".//{http://www.omg.org/spec/BPMN/20100524/MODEL}flowNodeRef"):
                if fnr.text:
                    actors_map[fnr.text.strip()] = lane_name

        nodes: Dict[str, BPMNNode] = {}
        flows: Dict[str, BPMNSequenceFlow] = {}

        for elem in proc_elem:
            tag = elem.tag.split("}")[-1]
            elem_id = elem.get("id", "")
            if not elem_id:
                continue

            elem_name = elem.get("name") or elem_id
            actor = actors_map.get(elem_id, "Semua Pihak / Terkait")

            if "task" in tag.lower():
                nodes[elem_id] = BPMNNode(
                    node_id=elem_id,
                    name=elem_name,
                    node_type="task",
                    actor=actor,
                )
            elif "gateway" in tag.lower():
                gw_type = "exclusive" if "exclusive" in tag.lower() else "parallel"
                nodes[elem_id] = BPMNNode(
                    node_id=elem_id,
                    name=elem_name,
                    node_type="gateway",
                    actor=actor,
                    gateway_type=gw_type,
                )
            elif "startevent" in tag.lower():
                nodes[elem_id] = BPMNNode(
                    node_id=elem_id,
                    name=elem_name,
                    node_type="start_event",
                    actor=actor,
                )
            elif "endevent" in tag.lower():
                nodes[elem_id] = BPMNNode(
                    node_id=elem_id,
                    name=elem_name,
                    node_type="end_event",
                    actor=actor,
                )
            elif tag == "sequenceFlow":
                src = elem.get("sourceRef", "")
                tgt = elem.get("targetRef", "")
                flow_name = elem.get("name")
                flows[elem_id] = BPMNSequenceFlow(
                    flow_id=elem_id,
                    name=flow_name,
                    source_node_id=src,
                    target_node_id=tgt,
                    condition=flow_name,
                )

        # Wire incoming and outgoing flows
        for fid, f in flows.items():
            if f.source_node_id in nodes:
                nodes[f.source_node_id].outgoing_flows.append(fid)
            if f.target_node_id in nodes:
                nodes[f.target_node_id].incoming_flows.append(fid)

        return BPMNProcessGraph(
            metadata=metadata,
            actors=actors_list,
            nodes=nodes,
            flows=flows,
        )


# Backward compatibility alias
BPMNXMLParser = UnifiedBPMNParser
