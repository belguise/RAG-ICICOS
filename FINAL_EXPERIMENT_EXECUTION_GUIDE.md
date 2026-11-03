# FINAL EXPERIMENT EXECUTION GUIDE: ICICOS EVALUATION & REPRODUCIBILITY

**Target Publication:** *Knowledge-Based Systems* (Elsevier, Q1)  
**Project:** ICICOS (Intent-Conditioned BPMN-SOP Hybrid RAG Framework)  
**Authoritative SHA:** `4defd04678f66ea64f674ea0bf107fc7f907edc3`  
**Guide Version:** 2.0.0 (Post-Audit Code Freeze)  

---

## 1. Overview & Objective

This guide provides end-to-end operational instructions for reproducing all benchmark evaluations, ablation studies, and statistical significance tests reported in the *Knowledge-Based Systems* manuscript. 

The reproduction workflow is divided into two distinct tiers:
1. **Tier 1: Zero-Cost Verification (Local & Deterministic)**: Run the full test suite (58 tests) and recompute all inferential statistical analyses directly from raw per-query trial records without consuming external API credits.
2. **Tier 2: Full-Scale API Benchmark Replication**: Re-run the active generation and evaluation pipelines across LLM backbones (Gemini 2.5 Flash / Gemini 3.5 Flash Lite and Groq Qwen 2.5 32B / Qwen 3.8 27B).

---

## 2. Environment Provisioning

### System Requirements
- **Operating System:** Linux (Ubuntu 22.04+ recommended), macOS (13+), or Windows 10/11.
- **Python:** 3.11.x (tested on 3.11.9).
- **RAM:** Minimum 8 GB (16 GB recommended for dense embedding models).
- **Disk:** ~2 GB for dependencies, vector store indices, and dataset caches.

### Step-by-Step Setup

```bash
# 1. Clone repository
git clone https://github.com/belguise/RAG-ICICOS.git
cd RAG-ICICOS

# 2. Initialize virtual environment
python -m venv .venv

# 3. Activate environment
# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# (Optional) For bit-for-bit lockfile reproducibility:
# pip install -r environment.lock
```

### Configuration (`.env`)

Copy the configuration template:
```bash
cp .env.example .env
```

Edit `.env` to include required API credentials:
```env
# Primary LLM Provider (Google AI Studio)
GEMINI_API_KEY=your_actual_gemini_api_key
GOOGLE_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
LLM_TEMPERATURE=0.0

# Cross-Backbone Provider (Groq Cloud - Optional)
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=qwen/qwen3.8-27b

# Critic Verification & Bounded Self-Repair
MAX_REPAIR_ITERATIONS=2
CRITIC_CONFIDENCE_THRESHOLD=0.85

# Local Sentence Transformer Embeddings
EMBEDDING_PROVIDER=sentence-transformer
SENTENCE_TRANSFORMER_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# Storage Paths
CHROMA_PERSIST_DIR=./data/vector_store
CHROMA_COLLECTION_NAME=sop_bpmn_knowledge
RAW_DATA_DIR=./data/raw
```

---

## 3. Dataset Integrity & Checksum Verification

Before executing any evaluation, verify the cryptographic integrity of the benchmark datasets:

```bash
# Verify dataset manifest checksums (SHA-256)
sha256sum -c evaluation/DATASET_MANIFEST_CHECKSUMS.sha256
```
*(On Windows PowerShell, use `Get-FileHash` or the provided Python helper):*
```bash
python -c "
import hashlib
from pathlib import Path

manifest = Path('evaluation/DATASET_MANIFEST_CHECKSUMS.sha256')
for line in manifest.read_text().splitlines():
    if not line or line.startswith('#'): continue
    expected_hash, filepath = line.split()
    actual_hash = hashlib.sha256(Path(filepath).read_bytes()).hexdigest()
    assert actual_hash == expected_hash, f'Mismatch in {filepath}'
    print(f'[OK] {filepath}')
print('All dataset checksums verified successfully!')
"
```

Expected output:
- `[OK] ICICOS_ProcessQA_Dataset/benchmark/icicos_process_qa.json`
- `[OK] ICICOS_ProcessQA_Dataset/benchmark/icicos_process_qa.csv`
- `[OK] ICICOS_ProcessQA_Dataset/cohorts/primary_n70.csv`
- `[OK] ICICOS_ProcessQA_Dataset/cohorts/procedural_n60.csv`
- `[OK] ICICOS_ProcessQA_Dataset/cohorts/ablation_n20.csv`
- `[OK] ICICOS_ProcessQA_Dataset/cohorts/cross_backbone_evaluations_n34.csv`

---

## 4. Tier 1: Zero-Cost Local Reproduction

### A. Run Automated Regression Test Suite (58 Tests)
Execute all unit, integration, and ablation tests in isolated mock mode:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
**Expected Outcome:**
- 58 tests executed across 7 test suites.
- Status: `OK` (0 failures, 0 errors).
- Execution time: ~15 seconds.

### B. Reproduce Inferential Statistical Analysis
Compute paired Wilcoxon signed-rank tests, 10,000 paired bootstrap 95% confidence intervals, and Holm-Bonferroni corrections directly from the 350 raw per-query trial rows in `results/master_results.csv`:
```bash
python evaluation/statistical_analysis.py
```

**Generated Outputs in `results/`:**
1. `significance_tests.csv`: Exact paired p-values, Holm-Bonferroni adjusted p-values, and Cohen's $d_z$ effect sizes.
2. `bootstrap_ci_differences.csv`: Empirical 95% bootstrap confidence intervals for all paired metric differences ($\Delta = \text{ICICOS} - \text{Baseline}$).
3. `architecture_comparison_table.csv`: Mean and standard deviation for all 5 architectures across primary metrics.
4. `ablation_table.csv`: Metric contributions of Router, BPMN, and Critic components.

---

## 5. Tier 2: Full-Scale Benchmark Evaluation (Live APIs)

> **PRE-FLIGHT BUDGET & RESOURCE AUDIT:**  
> Executing live API evaluation incurs external service costs and takes time. Below is the exact call-count and estimated expenditure profile before initiating full evaluation.

### Resource & Budget Estimation

| Benchmark Cohort | Queries ($N$) | Architectures Evaluated | Total Pipeline Runs | Expected LLM Calls (Draft + Critic + Repair) | Estimated API Cost (USD) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary Benchmark** | 70 | 5 | 350 | ~700 - 850 | ~$1.20 - $1.80 |
| **Procedural Diagnostic** | 60 | 1 (ICICOS) | 60 | ~120 - 150 | ~$0.25 - $0.35 |
| **Ablation Study** | 20 | 4 variants | 80 | ~120 - 160 | ~$0.20 - $0.30 |
| **Cross-Backbone Qwen**| 34 (7 unique) | 2 | 34 | ~70 - 85 | Free tier (Groq) |
| **Total Full Run** | **184** | — | **524** | **~1,010 - 1,245** | **~$1.65 - $2.45** |

### Execution Commands

#### 1. Ingest Raw SOP Text and BPMN XML Models
```bash
python main.py ingest
```
This indexes SOP documents into `data/vector_store` and registers BPMN process graphs in `data/bpmn_storage`.

#### 2. Execute Primary Cohort Benchmark ($N=70$)
```bash
python -m evaluation.evaluator \
    --config evaluation_config.yaml \
    --cohort primary_benchmark \
    --output results/new_primary_run.csv
```

#### 3. Execute Ablation Study ($N=20$)
```bash
python -m evaluation.evaluator \
    --config evaluation_config.yaml \
    --cohort ablation_diagnostic \
    --output results/new_ablation_run.csv
```

#### 4. Run Cross-Backbone Evaluation ($N=34$)
Ensure `GROQ_API_KEY` is active in `.env`:
```bash
python -m evaluation.evaluator \
    --config evaluation_config.yaml \
    --cohort cross_backbone_diagnostic \
    --output results/new_cross_backbone_run.csv
```

---

## 6. Reconstructed Metric Evaluation Pipeline

The evaluation pipeline (`evaluation/metrics/reconstructed_evaluator.py`) executes five automated metric engines:

1. **Faithfulness**:
   - Computes claim-level grounding ratio against retrieved SOP and BPMN context.
   - Evaluated using either RAGAS or local sentence-transformer NLI verification.
2. **Answer Relevancy**:
   - Assesses query-answer semantic alignment via reverse question generation or cosine embedding similarity.
3. **Context Precision & Recall**:
   - Evaluates signal-to-noise ratio and reference clause coverage.
4. **Semantic Answer F1 (SA-F1)**:
   - Token-level and semantic overlap between predicted answer and canonical ground truth.
5. **Procedural BPMN Metrics**:
   - **Process Path Accuracy**: Correctness of sequential process transitions.
   - **Gateway Decision Accuracy**: Accuracy in conditional branching and gateway logic.
   - **Actor Attribution Recall**: Correct identification of swimlane roles and responsibilities.

---

## 7. Troubleshooting & FAQ

### Q1: Received `ResourceExhausted` (Google AI Studio 429)
- **Cause**: Google AI Studio free-tier RPM (Requests Per Minute) limit reached.
- **Remedy**: The evaluation harness includes exponential backoff retry. You can increase request delay by setting `REQUEST_DELAY_SECONDS=2.0` or upgrading to a paid tier.

### Q2: SQLite / ChromaDB Concurrency Lock
- **Cause**: Running parallel test runners accessing `data/vector_store` simultaneously.
- **Remedy**: Run benchmarks sequentially or assign separate `CHROMA_PERSIST_DIR` paths per worker process.

### Q3: Why does Answer Relevancy show a slight drop (-0.0354) for ICICOS?
- **Scientific Rationale**: As detailed in the manuscript and `STATISTICAL_VALIDATION_REPORT.md`, the failure-seeking critic penalizes over-general or verbose conversational filler in favor of strict, concise compliance. When family-wise error rate is controlled via Holm-Bonferroni correction, this difference is **statistically non-significant** ($p_{\text{adj}} = 0.1030 > 0.05$).
