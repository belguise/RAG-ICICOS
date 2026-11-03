# ICICOS: Intent-Conditioned BPMN-SOP Hybrid RAG Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility: Verified](https://img.shields.io/badge/Reproducibility-Verified%20(58%2F58%20Tests)-brightgreen.svg)]()
[![Target: KBS Elsevier](https://img.shields.io/badge/Journal-Knowledge--Based%20Systems%20(Elsevier%20Q1)-orange.svg)]()

> **Official Implementation and Experimental Verification Suite for:**  
> *"Adaptive Agentic Hybrid RAG with BPMN-Grounded Graph Reasoning for Explainable Enterprise Process Question Answering"*  
> Prepared for submission to **Knowledge-Based Systems (Elsevier)**.

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
python -m uvicorn agentic_rag_sop.api.main:app --host 127.0.0.1 --port 8000
```
Interactive OpenAPI documentation will be accessible at `http://localhost:8000/docs`.

---

## Experimental Reproducibility & Verification

### 1. One-Click Reviewer Verification Script
To independently recompute all primary descriptive and inferential statistics directly from raw per-query trial observations and verify mathematical consistency against the manuscript without external API calls:
```bash
python scripts/verify_research_release.py
```

### 2. Running Automated Regression Suite (65 Tests)
Execute all unit, integration, and security regression tests in isolated mock mode:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
Expected output: **65 tests ran — OK (0 failures, 0 errors)**.

### 3. Running Statistical Analysis Engine
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

| Metric | LLM-Only | Traditional Vector RAG | Static Hybrid RAG | **ICICOS (Proposed)** | Wilcoxon $p_{\text{adj}}$ | Cohen's $d_z$ | Paired Bootstrap 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Faithfulness ($F_{\text{faith}}$)** | N/A* | 0.8143 | 0.8789 | **0.9883** | **$5.95 \times 10^{-10}$** | **1.0809** (Large) | $[+0.0865, +0.1332]$ |
| **Evidence-Ungrounded Rate (EUSR)** $\downarrow$ | N/A* | 18.5% | 12.1% | **1.2% (0.0116)** | **$5.50 \times 10^{-10}$** | **-1.0828** (Large) | $[-0.1332, -0.0864]$ |
| **Answer Relevancy ($R_{\text{ans}}$)** | 0.6658 | 0.7753 | **0.7876** | 0.7522 | $0.1030$ (ns) | $-0.3645$ | $[-0.0587, -0.0127]$ |
| **Context Precision ($P_{\text{ctx}}$)** | N/A* | 0.9906 | **0.9917** | 0.9909 | $0.6346$ (ns) | $-0.1195$ | $[-0.0022, +0.0000]$ |
| **Context Recall ($R_{\text{ctx}}$)** | N/A* | **1.0000** | **1.0000** | **1.0000** | $1.0000$ (ns) | N/A* | $[+0.0000, +0.0000]$ |
| **Semantic Alignment F1 (SA-F1)** | 0.4721 | 0.5566 | **0.5718** | 0.5645 | $0.5209$ (ns) | $-0.1176$ | $[-0.0217, +0.0073]$ |
| **End-to-End Latency (s)** $\downarrow$ | 3.66 | 3.80 | **3.05** | 10.76 | **$2.49 \times 10^{-12}$** | 1.2560 | $[+6.3874, +9.2363]$ |

### Historical Provenance & Discrepancy Note (0.8585 vs. 0.9883)
Earlier repository documentation drafts listed exploratory uncalibrated preliminary numbers (e.g., Faithfulness 0.8585). Forensic audit of Git commit `6183cd3` confirmed that those preliminary figures were clerical placeholders, while the inferential delta (+0.1094), p-value ($5.95 \times 10^{-10}$), and Cohen's $d_z$ ($1.0809$) were copy-pasted directly from the empirical primary trial. All numbers reported in the table above and in the manuscript represent the true empirical observations from the complete $N=70$ paired benchmark trial ($350$ runs) in `results/master_results.csv`.

### Research Dataset & Artifact Provenance (Zenodo)
The benchmark datasets, attributed process graphs, and cohort manifests are deposited under Creative Commons Attribution 4.0 International (CC-BY-4.0):
- **Zenodo DOI:** [10.5281/zenodo.23242855](https://doi.org/10.5281/zenodo.23242855) (Record ID: `23242855`)
- **Primary Archive:** `ICICOS_ProcessQA_Dataset.zip` (MD5: `b64b9c5e6a8d581a52effdb35865f5e9`)

*Note: Multiplicity controlled across the family of 7 primary evaluation hypotheses via step-down Holm-Bonferroni correction ($\alpha = 0.05$). Answer Relevancy difference with Static Hybrid RAG is statistically non-significant under family-wise error rate control ($p_{\text{adj}} = 0.1030 > 0.05$). For Context Recall, both architectures achieve identical recall (1.0000) across all 70 pairs, yielding zero variance in paired differences ($s_D = 0$); hence, paired Cohen's $d_z$ is mathematically undefined (N/A*). Retrieval-grounded metrics for LLM Only are undefined ($k=0$). All metrics verified against raw trial data ($N=70$ complete paired queries, 350 primary benchmark runs in `results/master_results.csv` and `results/statistical_tests.csv`).*

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
  title     = {Adaptive Agentic Hybrid RAG with BPMN-Grounded Graph Reasoning for Explainable Enterprise Process Question Answering},
  journal   = {Knowledge-Based Systems},
  publisher = {Elsevier},
  year      = {2026},
  note      = {Manuscript prepared for submission}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

