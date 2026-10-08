"""
BPMN Process Graph Storage Manager.
Phase 3 BPMN Knowledge Integration: Persists and loads canonical BPMN JSON process graphs
in data/processed/bpmn/ for fast, deterministic graph reasoning.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from agentic_rag_sop.config.settings import get_settings
from agentic_rag_sop.bpmn.models import BPMNProcessGraph
from agentic_rag_sop.bpmn.parser import UnifiedBPMNParser


def slugify(text: str) -> str:
    """Generate safe filename slug from process name."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[-\s]+", "_", text)


class BPMNStorageManager:
    """Manages serialization, disk persistence, and indexing of BPMN process graphs."""

    def __init__(self, processed_dir: Optional[Union[str, Path]] = None):
        settings = get_settings()
        self.processed_dir = Path(processed_dir or "./data/processed/bpmn")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.parser = UnifiedBPMNParser()
        self._cache: Dict[str, BPMNProcessGraph] = {}

    def save_graph(self, graph: BPMNProcessGraph) -> Path:
        """Save a single BPMNProcessGraph to JSON."""
        slug = slugify(graph.metadata.process_name)
        out_path = self.processed_dir / f"{slug}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(graph.model_dump(), f, indent=2, ensure_ascii=False)
        self._cache[graph.metadata.process_name.lower()] = graph
        return out_path

    def load_all_graphs(self, auto_ingest: bool = True) -> List[BPMNProcessGraph]:
        """
        Load all parsed process graphs from processed JSON directory.
        If empty and auto_ingest=True, parse raw BPMN files in data/raw/bpmn/ first.
        """
        json_files = list(self.processed_dir.glob("*.json"))

        if not json_files and auto_ingest:
            self.ingest_raw_bpmn()
            json_files = list(self.processed_dir.glob("*.json"))

        graphs: List[BPMNProcessGraph] = []
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    graph = BPMNProcessGraph.model_validate(data)
                    graphs.append(graph)
                    self._cache[graph.metadata.process_name.lower()] = graph
            except Exception:
                continue

        return graphs

    def ingest_raw_bpmn(self, raw_dir: Optional[Union[str, Path]] = None) -> List[BPMNProcessGraph]:
        """Parse all .bpm and .bpmn files in raw directory and save as canonical JSON."""
        dir_path = Path(raw_dir or "./data/raw/bpmn")
        if not dir_path.exists():
            return []

        all_graphs: List[BPMNProcessGraph] = []
        for file_path in dir_path.glob("*"):
            if file_path.suffix.lower() in (".bpm", ".bpmn", ".xml"):
                try:
                    graphs = self.parser.parse_file(file_path)
                    for g in graphs:
                        self.save_graph(g)
                        all_graphs.append(g)
                except Exception:
                    continue

        return all_graphs

    def find_process(self, query: str) -> Optional[BPMNProcessGraph]:
        """Find the best-matching BPMNProcessGraph for a given process name query."""
        if not self._cache:
            self.load_all_graphs()

        q_lower = query.lower()

        # 1. Exact or substring match
        for name, g in self._cache.items():
            if name in q_lower or q_lower in name:
                return g

        # 2. Keyword match against process name words
        q_words = set(re.findall(r"\w+", q_lower))
        best_match = None
        max_overlap = 0

        for name, g in self._cache.items():
            name_words = set(re.findall(r"\w+", name))
            overlap = len(q_words.intersection(name_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = g

        return best_match if max_overlap >= 2 else None
