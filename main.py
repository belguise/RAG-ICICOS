"""
Entry point for Agentic RAG SOP & BPMN Knowledge system.
"""

import sys
from pathlib import Path

# Ensure src directory is in Python path
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from agentic_rag_sop.cli import main

if __name__ == "__main__":
    main()
