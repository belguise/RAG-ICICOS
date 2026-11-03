# ICICOS: Intent-Conditioned BPMN-SOP Hybrid RAG Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility: Verified](https://img.shields.io/badge/Reproducibility-Verified%20(58%2F58%20Tests)-brightgreen.svg)]()
[![Target: KBS Elsevier](https://img.shields.io/badge/Journal-Knowledge--Based%20Systems%20(Elsevier%20Q1)-orange.svg)]()

> **Official Implementation and Experimental Verification Suite for:**  
> *"ICICOS: An Intent-Conditioned BPMN-SOP Hybrid RAG Framework for Explainable Institutional Process Question Answering"*  
> Submitted to **Knowledge-Based Systems (Elsevier)**.

---

## Overview

**ICICOS** is a research-grade Agentic Retrieval-Augmented Generation (RAG) framework engineered to resolve the operational dichotomy between textual policy manuals and topological process specifications. By synthesizing dense semantic vector retrieval over Standard Operating Procedure (SOP) texts with deterministic topological graph traversal over ISO/IEC 19510 Business Process Model and Notation (BPMN 2.0) XML graphs, ICICOS enables explainable, factually grounded, and structurally compliant question answering for enterprise and institutional governance.

### Architectural Core

```
                         [ User Query ]
                               │
                               ▼
                 [ Query Understanding Agent ]
                               │
                               ▼
                   [ Adaptive Intent Router ]
                     /                     \
                    /                       \
                   ▼                         ▼
        [ SOP Vector Retrieval ]   [ BPMN Graph Traversal ]
         (ChromaDB Dense Top-k)     (Topological Reachability)
                    \                       /
                     \                     /
                      ▼                   ▼
                     [ Cross-Modal Context Fusion ]
                               │
                               ▼
                      [ Grounded Generator ]
                               │
                               ▼
                   [ Critic Verification Agent ]
                     (5 Multi-Dimensional Checks)
                     /                         \
           Passed / Conf >= 0.85      Failed / Conf < 0.85
                   │                           │
                   ▼                           ▼
            [ Final Answer ]        [ Bounded Self-Repair ]
          (Source Attribution)      (Max 2 Revision Cycles)
```

1. **Adaptive Intent Router**: Categorizes procedural intent (`SOP_ONLY`, `BPMN_ONLY`, `HYBRID`) using domain priors and dynamic thresholds to allocate optimal retrieval budgets.
2. **Dual-Channel Retrieval Engine**:
   - **Textual Channel**: Dense semantic similarity search over chunked SOP text collections persisted in ChromaDB.
   - **Structural Channel**: Topological graph retrieval extracting directed execution traces, decision gateway guards, and swimlane role attributions from BPMN 2.0 XML graphs.
3. **Cross-Modal Context Fusion**: Unifies unstructured textual clauses with structured sequence flows into a topologically aligned prompt representation.
4. **Failure-Seeking Adversarial Critic & Bounded Self-Repair**: A multi-dimensional evaluator inspecting factual grounding, domain compliance, workflow correctness, SOP consistency, and answer completeness, triggering bounded revision iterations (maximum 2 cycles) upon detected divergence.

---

## Repository Structure

```
.
├── src/agentic_rag_sop/            # Core ICICOS python package
│   ├── agents/                     # Query Understanding, Intent Router, Adversarial Critic
│   ├── bpmn/                       # BPMN 2.0 XML parser, attributed graph models, storage
│   ├── retrieval/                  # SOP dense vector retriever and BPMN graph retriever
│   ├── fusion/                     # Cross-modal context fusion engine
│   ├── generation/                 # Grounded LLM generation chains & prompt templates
│   ├── config/                     # Central configuration management (pydantic-settings)
│   └── api/                        # FastAPI REST service and route schemas
├── evaluation/                     # Reproducible evaluation & statistical verification
│   ├── metrics/                    # Reconstructed multi-engine evaluation harness
│   ├── statistical_analysis.py     # Non-parametric paired bootstrap & Holm-Bonferroni tests
│   ├── METRIC_SPECIFICATION.md     # Mathematical & procedural definitions of all metrics
│   ├── experiment_config.json      # Machine-readable experiment parameters
│   └── DATASET_MANIFEST_CHECKSUMS.sha256 # Cryptographic integrity verification
├── ICICOS_ProcessQA_Dataset/       # Canonical benchmark dataset & stratified evaluation cohorts
│   ├── benchmark/                  # Master dataset (338 procedural query-answer pairs)
│   └── cohorts/                    # Stratified subsets: primary_n70, procedural_n60, ablation_n20, cross_backbone_n34
├── tests/                          # 58 comprehensive unit, integration & ablation regression tests
│   ├── test_full_pipeline_integration.py # 10 full execution path & architecture category tests
│   ├── test_critic_self_repair.py        # 12 adversarial critic & repair convergence tests
│   ├── test_evaluation_pipeline.py       # 13 reconstructed evaluation metric tests
│   ├── test_scientific_consistency.py    # 10 statistical & numerical integrity tests
│   └── ...                         # Domain unit tests (BPMN parsing, router, vector store)
├── FINAL KBS FIX/                  # Authoritative journal manuscript & publication assets
├── CODE_VALIDATION_REPORT.md       # Full integration testing and code audit sign-off
├── FINAL_EXPERIMENT_EXECUTION_GUIDE.md # Step-by-step reproduction guide
├── REPRODUCIBILITY_CHECKLIST.md    # Journal reviewer & artifact auditor verification checklist
├── REPRODUCIBILITY_REPORT.md       # Forensic audit and environment freeze report
├── STATISTICAL_VALIDATION_REPORT.md# Detailed inferential statistical findings
├── requirements.txt                # Pinned primary package dependencies
├── environment.lock                # Pinned 154-package reproducible environment lockfile
└── pyproject.toml                  # Standard Python packaging configuration (v0.2.0)
```

---

## Requirements & Environment Setup

- **Python**: Version 3.11 or higher (tested on Python 3.11.9)
- **Operating Systems**: Linux (Ubuntu 22.04+), macOS (13+), Windows 10/11
- **API Credentials**: Google AI Studio API key (Gemini) or Groq Cloud API key (Qwen)

### Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/belguise/RAG-ICICOS.git
cd RAG-ICICOS

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

To install editable development mode:
```bash
pip install -e ".[dev]"
```

For strict bit-for-bit lockfile recreation using `pip`:
```bash
pip install -r environment.lock
```

### Environment Configuration

Copy the template configuration file:
```bash
cp .env.example .env
```

Populate `.env` with your API keys:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
LLM_TEMPERATURE=0.0

# Optional for cross-backbone evaluation:
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
```

> **SECURITY GUARANTEE**: Tracked files contain zero secrets. `.gitignore` strictly excludes `.env` and credential caches.

---

## Quickstart: Pipeline Execution

### 1. Ingest Knowledge Sources
Parse BPMN 2.0 XML graphs and embed SOP documents into ChromaDB:
```bash
python main.py ingest
```

### 2. Execute Single Query
Run an end-to-end procedural question answering workflow:
```bash
python main.py query "What is the procedure for conference budget reimbursement?"
```

### 3. Launch Interactive Terminal
Engage in multi-turn interactive consultation:
```bash
python main.py chat
```

### 4. Launch REST API Server
Start the FastAPI service:
```bash
python -m uvicorn agentic_rag_sop.api.main:app --host 0.0.0.0 --port 8000
```
Interactive OpenAPI documentation will be accessible at `http://localhost:8000/docs`.

---

## Experimental Reproducibility & Verification

### Running Automated Test Suite (58 Tests)
Execute all unit, integration, and ablation tests in isolated mock mode (no API charges):
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
Expected output: **58 tests ran — OK (0 failures, 0 errors)**.

### Running Statistical Analysis
Recompute all descriptive statistics, paired Wilcoxon signed-rank tests, 10,000 paired bootstrap 95% confidence intervals, and Holm-Bonferroni multiplicity corrections directly from raw per-query experimental trials:
```bash
python evaluation/statistical_analysis.py
```
Outputs generated in `results/`:
- `significance_tests.csv`: Exact paired p-values, Holm-Bonferroni adjusted p-values, Cohen's $d_z$ effect sizes.
- `bootstrap_ci_differences.csv`: Paired bootstrap empirical 95% confidence intervals ($\Delta = \text{ICICOS} - \text{Baseline}$).
- `architecture_comparison_table.csv`: Mean and standard deviation across all 5 evaluation cohorts.
- `ablation_table.csv`: Component contribution analysis (`without_router`, `without_bpmn`, `without_critic`).

---

## Empirical Findings & Statistical Rigor

All reported statistical conclusions are computed directly from raw per-query trial data ($N=70$ paired queries, 350 experimental runs):

| Metric | LLM-Only | Standard RAG | Hybrid RAG (No Critic) | **ICICOS (Full)** | Wilcoxon $p_{\text{adj}}$ | Cohen's $d_z$ | Paired Bootstrap 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Faithfulness** | 0.6015 | 0.7491 | 0.8174 | **0.8585** | **$5.95 \times 10^{-10}$** | **1.0809** (Large) | $[+0.0898, +0.1306]$ |
| **Answer Relevancy** | 0.8711 | **0.9317** | 0.9022 | 0.8963 | $0.1030$ (n.s.) | $-0.2977$ | $[-0.0610, -0.0101]$ |
| **Context Precision** | 0.0000 | 0.6974 | 0.7512 | **0.7937** | **$3.30 \times 10^{-6}$** | **0.6277** (Medium) | $[+0.0577, +0.1368]$ |
| **Context Recall** | 0.0000 | 0.7810 | 0.8129 | **0.8357** | **$0.0016$** | **0.4284** (Small) | $[+0.0229, +0.0871]$ |
| **Semantic F1 (SA-F1)** | 0.5891 | 0.7684 | 0.8120 | **0.8412** | **$3.46 \times 10^{-7}$** | **0.7410** (Medium) | $[+0.0475, +0.0988]$ |
| **Hallucination Rate** $\downarrow$ | 0.3985 | 0.2509 | 0.1826 | **0.1415** | **$5.95 \times 10^{-10}$** | **-1.0809** (Large) | $[-0.1306, -0.0898]$ |
| **Query Latency (s)** $\downarrow$ | **1.84** | 5.82 | 8.24 | 10.76 | $3.29 \times 10^{-13}$ | 1.8341 | $[+4.43, +5.47]$ |

*Note: Multiplicity controlled via Holm-Bonferroni correction ($\alpha = 0.05$). Answer Relevancy difference with Standard RAG is statistically non-significant under family-wise error control ($p_{\text{adj}} = 0.1030 > 0.05$).*

---

## Documentation Index

Detailed scientific documentation, execution guides, and audit reports are maintained in the repository:

1. [CODE_VALIDATION_REPORT.md](CODE_VALIDATION_REPORT.md): Comprehensive integration testing, architecture category verification, and static audit sign-off.
2. [FINAL_EXPERIMENT_EXECUTION_GUIDE.md](FINAL_EXPERIMENT_EXECUTION_GUIDE.md): Complete operational instructions for benchmark execution, cross-backbone testing, and evaluation replication.
3. [REPRODUCIBILITY_CHECKLIST.md](REPRODUCIBILITY_CHECKLIST.md): Research reproducibility checklist adhering to Nature/ACM/Elsevier artifact evaluation standards.
4. [REPRODUCIBILITY_REPORT.md](REPRODUCIBILITY_REPORT.md): Experimental environment audit, parameter provenance, and dataset cohort specifications.
5. [STATISTICAL_VALIDATION_REPORT.md](STATISTICAL_VALIDATION_REPORT.md): Exhaustive paired inferential statistics, bootstrap distributions, and latency breakdowns.
6. [evaluation/METRIC_SPECIFICATION.md](evaluation/METRIC_SPECIFICATION.md): Mathematical formulations, edge cases, and algorithmic implementations of all evaluation metrics.

---

## Citation & Academic Attribution

If you use ICICOS in your research, please cite our manuscript:

```bibtex
@article{icicos2026kbs,
  author    = {Belguise, et al.},
  title     = {ICICOS: An Intent-Conditioned BPMN-SOP Hybrid RAG Framework for Explainable Institutional Process Question Answering},
  journal   = {Knowledge-Based Systems},
  publisher = {Elsevier},
  year      = {2026},
  note      = {Under Review}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

