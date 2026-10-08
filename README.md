# RAG-ICICOS

Adaptive Agentic Hybrid RAG with BPMN-Grounded Graph Reasoning for Enterprise Process Question Answering. This repository provides the core implementation of an agentic retrieval-augmented generation framework combining dense semantic vector search over Standard Operating Procedure (SOP) documents with deterministic topological graph traversal over Business Process Model and Notation (BPMN 2.0) models.

## Components

- `src/agentic_rag_sop/`: Core Python package containing:
  - `agents/`: Query Understanding Agent (QUA), Adaptive Intent Router, and Failure-Seeking Adversarial Critic with bounded self-repair.
  - `bpmn/`: ISO/IEC 19510 BPMN 2.0 XML parser, attributed process graph builder, and storage layer.
  - `retrieval/`: Dual-channel retrieval engine for SOP text embeddings (ChromaDB) and BPMN topological subgraphs.
  - `fusion/`: Cross-modal context fusion engine projecting reachability paths, gateway decision guards, and swimlane role attributions.
  - `generation/`: Grounded generation chains and structured prompt templates.
  - `config/`: Central settings management via `pydantic-settings`.
  - `api/`: FastAPI REST endpoints for document ingestion, process inspection, and interactive chat.
- `evaluation/`: Benchmark harness, metric calculation modules (RAGAS, process path accuracy, gateway decision accuracy, actor attribution recall), and paired statistical hypothesis testing.
- `tests/`: Automated unit tests and reproducible verification suites.

## Requirements

- Python >= 3.11
- Virtual environment manager (`venv` or `uv`)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/belguise/RAG-ICICOS.git
cd RAG-ICICOS

python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, install in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

## Configuration

1. Copy the example environment template:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and add your LLM provider API credentials:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```
   *(Optional: set `GROQ_API_KEY=your_groq_api_key_here` if evaluating open-weights backbones).*

> **SECURITY WARNING**: NEVER commit `.env` or expose API keys. The `.gitignore` file is pre-configured to exclude all environment and credential files.

## Running

Ingest documents and run query workflows via CLI:

```bash
# Ingest local SOP and BPMN knowledge sources
python main.py ingest

# Query the agentic RAG pipeline
python main.py query "What is the procedure for conference budget reimbursement?"

# Launch the interactive terminal chat session
python main.py chat

# Launch the REST API service (FastAPI)
python -m uvicorn agentic_rag_sop.api.main:app --host 0.0.0.0 --port 8000
```

## Evaluation

Run the evaluation and statistical verification suites:

```bash
# Run unit test suite
python -m unittest discover -s tests -p "test_*.py"

# Run statistical significance analysis
python evaluation/statistical_analysis.py
```

## Data

The benchmark dataset is distributed separately through Zenodo.
Raw SOP text manuals, BPMN 2.0 XML specifications, and annotated query pairs are archived for open access. The persistent Zenodo link will be added after archival deposit.

## Paper

This codebase implements the methodology described in:
*Adaptive Agentic Hybrid RAG with BPMN-Grounded Graph Reasoning for Explainable Enterprise Process Question Answering* (Target: Knowledge-Based Systems, Elsevier).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
