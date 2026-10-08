"""
Abstract interface for BPMN process and graph retrieval.
Future Architecture: Designed for Phase 2 Agentic layers (Router, Planner, Critic)
to interact with business processes as topological graphs rather than just text chunks.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field

from agentic_rag_sop.bpmn.models import BPMNProcess, ProcessStep


class BPMNStepDetail(BaseModel):
    """Detailed graph node representation for agent reasoning."""
    step_id: str
    step_name: str
    step_type: str            # 'task', 'gateway', 'start_event', 'end_event'
    actor: str                # Swimlane / Penanggung Jawab
    process_name: str         # Nama Diagram / Proses
    description: str = ""     # Penjelasan langkah
    conditions: List[str] = Field(default_factory=list)      # Aturan percabangan jika gateway
    incoming_steps: List[str] = Field(default_factory=list)  # Langkah prasyarat (preceding)
    outgoing_steps: List[str] = Field(default_factory=list)  # Langkah berikutnya (succeeding)
    artifacts: List[str] = Field(default_factory=list)       # Dokumen lampiran (invoice, nota, dsb.)


class BaseBPMNGraphRetriever(ABC):
    """
    Contract for structured BPMN process graph retrievers.
    Placeholder interface to be implemented in Phase 2 for agentic reasoning.
    """

    @abstractmethod
    def get_process_names(self) -> List[str]:
        """Return list of all available business process names."""
        pass

    @abstractmethod
    def get_process(self, process_name_or_id: str) -> Optional[BPMNProcess]:
        """Retrieve the full graph structure of a given process."""
        pass

    @abstractmethod
    def get_step_by_id(self, process_name: str, step_id: str) -> Optional[BPMNStepDetail]:
        """Retrieve full details of a specific process step."""
        pass

    @abstractmethod
    def get_next_steps(self, process_name: str, current_step_id: str) -> List[BPMNStepDetail]:
        """
        [Used by Planner Agent]:
        Answer: 'After step X, what are the next steps?' including branch conditions.
        """
        pass

    @abstractmethod
    def get_prerequisite_steps(self, process_name: str, current_step_id: str) -> List[BPMNStepDetail]:
        """
        [Used by Critic/Planner Agent]:
        Answer: 'What prerequisites must be satisfied before step Y?'
        """
        pass

    @abstractmethod
    def get_tasks_by_actor(self, process_name: str, actor_name: str) -> List[BPMNStepDetail]:
        """
        [Used by Router Agent]:
        Answer: 'What are the tasks and responsibilities of actor Z in process X?'
        """
        pass

    @abstractmethod
    def trace_full_path(
        self,
        process_name: str,
        start_step_id: str,
        end_step_id: str
    ) -> List[BPMNStepDetail]:
        """
        [Used by Planner Agent]:
        Trace the valid execution path (graph traversal) between two process nodes.
        """
        pass


# Alias for backward compatibility
BaseBPMNProcessRetriever = BaseBPMNGraphRetriever

