"""
BPMN process models and workflow topology route handlers.
Provides access to all 15 parsed ICICoS business process graphs.
"""

from fastapi import APIRouter, HTTPException
from agentic_rag_sop.bpmn.storage import BPMNStorageManager, slugify
from agentic_rag_sop.api.schemas import (
    BPMNProcessListResponse,
    BPMNProcessItemSchema,
)

router = APIRouter(prefix="/api/v1", tags=["BPMN Process Workflows"])

_storage_manager = None


def get_storage_manager() -> BPMNStorageManager:
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = BPMNStorageManager()
    return _storage_manager


@router.get("/bpmn/processes", response_model=BPMNProcessListResponse)
def list_bpmn_processes_endpoint() -> BPMNProcessListResponse:
    """
    List all parsed ICICoS BPMN business processes with actor, activity,
    and gateway counts.
    """
    sm = get_storage_manager()
    graphs = sm.load_all_graphs(auto_ingest=True)

    items = []
    for g in graphs:
        act_names = sorted(list(set(a.name for a in g.actors if a.name)))
        gateways_count = sum(1 for n in g.nodes.values() if getattr(n, "node_type", "") == "gateway")

        items.append(BPMNProcessItemSchema(
            process_id=g.metadata.process_id,
            process_name=g.metadata.process_name,
            document_domain=g.metadata.document_domain,
            category=g.metadata.process_category,
            actors=act_names,
            total_nodes=len(g.nodes),
            total_gateways=gateways_count,
            total_flows=len(g.flows),
        ))

    return BPMNProcessListResponse(
        total_processes=len(items),
        processes=items,
    )


@router.get("/bpmn/processes/{process_id}")
def get_bpmn_process_detail_endpoint(process_id: str):
    """
    Retrieve full graph topology (nodes, sequence flows, actors)
    for a specific BPMN process model to power React Flow graph visualization.
    """
    sm = get_storage_manager()
    graphs = sm.load_all_graphs(auto_ingest=True)

    target_id = process_id.lower().strip()
    for g in graphs:
        p_id = g.metadata.process_id.lower()
        slug = slugify(g.metadata.process_name).lower()
        p_name = g.metadata.process_name.lower()

        if target_id in (p_id, slug, p_name):
            return g.model_dump()

    raise HTTPException(status_code=404, detail=f"BPMN process '{process_id}' not found.")
