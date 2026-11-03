# REPRODUCIBILITY REPORT: EXPERIMENTAL CONFIGURATION & RESEARCH INTEGRITY HARDENING
**Project:** ICICOS — Intent-Conditioned BPMN-SOP Hybrid RAG Framework for Explainable Institutional Process Question Answering  
**Target Venue:** *Knowledge-Based Systems* (Elsevier, Q1)  
**Authoritative Git Commit SHA:** `4defd04678f66ea64f674ea0bf107fc7f907edc3`  
**Date:** November 2026  
**Auditor:** Senior ML Infrastructure & Experimental Reproducibility Engineer  

---

## 1. Executive Summary

This reproducibility report establishes a comprehensive, auditable, and mathematically grounded foundation for all experimental configurations, datasets, sampling manifests, dependencies, and provenance tracking across the ICICOS research project.

Prior to this engineering pass:
- Legacy configuration files (`evaluation_config.yaml`) claimed an IEEE conference venue, legacy model identifiers (`gemini-1.5-flash`), and an aspirational cohort size ($N=338$), which diverged from the actual evaluated primary cohort ($N=70$, 350 complete executions) in the manuscript.
- Random seeds (e.g., `seed = 42`) were conflated across dataset sampling, bootstrap statistical resampling, and LLM decoding, masking the fundamental non-determinism of cloud-hosted GPU/TPU matrix operations.
- Dependency tracking lacked an exact, pinned lockfile, with `uv.lock` representing an early 87-package snapshot, while the true historical execution environment (`.venv/`) contained 154 packages including `ragas==0.4.3`, `groq==0.37.1`, and `chromadb==1.5.9`.

### Key Hardening Actions Completed
1. **Model Audit & Truthful Parameterization:** Verified all model identifiers, provider endpoints, temperature settings ($T=0.0$), timeouts, retries, and documented the exact physical boundaries of LLM decoding determinism.
2. **Sampling Manifest & Cohort Verification:** Audited all 5 experimental cohorts against the canonical 338-query corpus; mathematically proved cohort disjointness between the primary ($N=70$) and procedural ($N=60$) suites ($|S_{\text{primary}} \cap S_{\text{procedural}}| = 0$).
3. **Dataset Integrity & Cryptographic Checksums:** Inspected all 338 queries for schema compliance, duplicate IDs (0 found), and missing fields (0 found); generated authoritative SHA-256 checksums in [`evaluation/DATASET_MANIFEST_CHECKSUMS.sha256`](evaluation/DATASET_MANIFEST_CHECKSUMS.sha256).
4. **Structured Experiment Provenance:** Formulated [`evaluation/run_metadata_schema.json`](evaluation/run_metadata_schema.json) and generated [`evaluation/reproducibility_run_metadata.json`](evaluation/reproducibility_run_metadata.json) linking the exact Git commit SHA, environment hashes, and verified metrics.
5. **Dependency Pinning:** Extracted and generated [`environment.lock`](environment.lock) from the authoritative execution environment; reconciled [`requirements.txt`](requirements.txt) and [`pyproject.toml`](pyproject.toml).
6. **Execution Cost & API Call Budget:** Calculated complete token requirements and API cost estimates for experimental verification runs, presented below for author approval.

---

## 2. Model Configuration Audit

### Table 1: Model Specifications Across Pipeline Roles

| Pipeline Role | Configured Model Identifier | Actual Provider & Endpoint | Temperature | Top-p | Max Tokens | Timeout / Retries | Deterministic Decoding Support |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary Generator** | `gemini-3.5-flash-lite` | Google AI Studio (`v1beta`) | $0.0$ | $1.0$ | 1024 / Default | 60s / 2 retries | Greedy argmax; non-deterministic across parallel GPU/TPU kernels |
| **Critic Verification Agent** | `gemini-3.5-flash-lite` | Google AI Studio (`v1beta`) | $0.0$ | $1.0$ | Default | 60s / 2 retries | Pydantic schema-constrained; fail-closed upon error |
| **Secondary Backbone (Diagnostic)** | `qwen/qwen3.8-27b` | Groq Cloud API (`openai/v1`) | $0.0$ | $1.0$ | Default | 60s / 2 retries | Greedy argmax; subject to Groq on-demand OTPM throttling |
| **Dense Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace Local (PyTorch/CPU) | N/A | N/A | 384 dim | N/A | **Fully Bitwise Deterministic** on same hardware architecture |
| **Cloud Embeddings (Fallback)** | `models/text-embedding-004` | Google AI Studio (`v1beta`) | N/A | N/A | 768 dim | 60s / 2 retries | Deterministic embedding vectors |
| **Automated Evaluator** | Reconstructed Evaluator Engine | Local Rule Heuristics + MiniLM Cosine | N/A | N/A | N/A | Local | **Fully Bitwise Deterministic**; replicates historical SA-F1 |

### 2.1 Critical Distinction Between Random Seeds
Scientific integrity requires explicitly distinguishing between three independent seed mechanisms:
1. **Dataset Sampling Seed (`seed = 42`):** Used in `sample_manifest.py` (`random.seed(42)`). Fully deterministic; controls stratified sampling of the 200 target queries from the 338-query master corpus.
2. **Bootstrap Resampling Seed (`seed = 42`):** Used in `calculate_query_level_statistics.py` (`np.random.seed(42)`). Fully deterministic; generates 10,000 paired difference resamples for empirical 95% confidence intervals and non-parametric validation.
3. **LLM Decoding Seed:** **Not supported** by `langchain_google_genai.ChatGoogleGenerativeAI` or `langchain_groq.ChatGroq`. While $T = 0.0$ enforces greedy argmax selection, cloud-hosted LLM endpoints exhibit subtle token-level variation due to floating-point non-associativity across multi-threaded CUDA warps and distributed mixture-of-experts routing. **Claiming bitwise deterministic LLM generation is scientifically invalid and must not be asserted in the manuscript.**

---

## 3. Sampling Manifests & Cohort Architecture

The ICICOS evaluation benchmark is organized into five verifiable, documented cohorts derived from the canonical 338-query master corpus.

```
                          ┌────────────────────────────────────────────────────────┐
                          │         Full Corpus: ICICOS ProcessQA (N = 338)        │
                          │   WF: 108  |  HYB: 81  |  DEC: 80  |  DOC: 69          │
                          └───────────────────────────┬────────────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       │                                                             │
        [Stratified Sampling (Seed 42)]                                  [Domain Filter]
                       │                                                             │
                       v                                                             v
        ┌─────────────────────────────┐                               ┌─────────────────────────────┐
        │ Primary Benchmark (N = 70)  │                               │ Procedural Diagnostic (N=60)│
        │ 350 Executions (5 Systems)  │ <────── Disjoint ──────>      │ Procedural metrics: PPA/GDA │
        │ DOC: 18 | WF: 18            │       (Overlap = 0)           │ WF: 20 | DEC: 20 | HYB: 20  │
        │ DEC: 17 | HYB: 17           │                               │ DOC: 0                      │
        └──────────────┬──────────────┘                               └──────────────┬──────────────┘
                       │                                                             │
                       ├─────────────────────────────┐                               │
                       │                             │                               │
                       v                             v                               v
        ┌─────────────────────────────┐  ┌───────────────────────┐   ┌───────────────────────────────┐
        │ Cross-Backbone Diag. (N=34) │  │  Component Ablation   │ <─┤ Shared overlap:               │
        │ Qwen 3.8 27B on Groq API    │  │  Cohort (N = 20)      │   │ 3 queries from Primary        │
        │ 7 Unique Queries (Subset)   │  │  5 per category       │   │ 7 queries from Procedural     │
        └─────────────────────────────┘  └───────────────────────┘   └───────────────────────────────┘
```

### Table 2: Authoritative Cohort Specification Matrix

| Cohort Identifier | Purpose & Scope | Total Queries ($N$) | Total Executions | Category Distribution | Sampling Source & Seed | Selection Algorithm | Overlap with Primary ($N=70$) |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- | :---: |
| **Full Corpus** | Intent-conditioned routing token savings | 338 | 338 | DOC: 69, WF: 108, DEC: 80, HYB: 81 | `ICICOS_ProcessQA_Dataset` | Full institutional manual coverage | 70 / 70 (100%) |
| **Primary Benchmark** | Main 5-architecture comparative evaluation | 70 | 350 | DOC: 18, WF: 18, DEC: 17, HYB: 17 | `results/primary_benchmark_manifest.json` (Seed 42) | Stratified random sample; first 70 queries completed | **Authoritative Cohort** |
| **Procedural Diagnostic** | Process metrics (PPA, GDA, AAA) | 60 | 60 | WF: 20, DEC: 20, HYB: 20, DOC: 0 | `ICICOS_ProcessQA_Dataset/cohorts/procedural_n60.csv` | Stratified procedural query filter | **0 (Strictly Disjoint)** |
| **Component Ablation** | Component sensitivity ($w/o\text{ BPMN}$, etc.) | 20 | 80 | DOC: 5, WF: 5, DEC: 5, HYB: 5 | `ICICOS_ProcessQA_Dataset/cohorts/ablation_n20.csv` | Balanced 5-per-category diagnostic subset | 3 queries |
| **Cross-Backbone Diagnostic** | Open-weights validation on Qwen 3.8 27B | 7 unique | 34 | DOC: 4, DEC: 1, WF: 1, HYB: 1 | `results/master_results.csv` | Subset halted after query 7 by Groq rate limits | 7 / 7 (100% subset) |

### 3.1 Historical Execution Truncation Forensics
- **Target vs. Actual Evaluated Primary Cohort:** The initial benchmark design targeted $N=200$ queries (50 per category) via `results/primary_benchmark_manifest.json`. On 2026-10-06 at 15:04 UTC, execution halted at query index 71 (`ICICOS_DEC_234`) when the Google Generative Language API free tier reached its hard daily ceiling of 500 requests (`RESOURCE_EXHAUSTED`, HTTP 429).
- **Integrity of Evaluated $N=70$ Sample:** The first 70 queries executed across all five architectures without error, generating exactly $70 \times 5 = 350$ complete, paired rows in `results/master_results.csv`. This 70-query cohort constitutes the true empirical basis of the manuscript's primary evaluation tables.
- **Cross-Backbone Qwen Truncation:** On Groq Cloud, `qwen/qwen3.8-27b` completed 6 queries across all 5 architectures (30 executions). On query 7 (`ICICOS_WF_146`), the 4 baseline architectures completed, but `proposed_agentic` triggered HTTP 429 (`OTPM limit 1000 exceeded, requested 1130 tokens`). Total completed executions = 34.

---

## 4. Dataset Integrity & Cryptographic Checksums

The master dataset resides in [`ICICOS_ProcessQA_Dataset/benchmark/icicos_process_qa.json`](ICICOS_ProcessQA_Dataset/benchmark/icicos_process_qa.json).

### Quality Audit Results
- **Total Records:** 338
- **Unique Query IDs:** 338 (0 duplicate IDs)
- **Null / Empty Fields:** 0 missing values across all required attributes (`id`, `question`, `category`, `expected_answer`, `expected_source`, `expected_actor`, `expected_process`, `expected_bpmn_path`).
- **Category Stratification:** 69 Document Requirement, 108 Workflow, 80 Decision Gateway, 81 Hybrid.
- **Data Leakage:** No train/test leakage exists; ICICOS is a zero-shot, in-context agentic reasoning framework with frozen foundation weights.

### Table 3: Authoritative Cryptographic SHA-256 Checksums
All checksums have been stored in [`evaluation/DATASET_MANIFEST_CHECKSUMS.sha256`](evaluation/DATASET_MANIFEST_CHECKSUMS.sha256):

```text
3ff56beac177b42706e34abac269d58ce4f0ef4af699cad47a6e71c1aecc356a  ICICOS_ProcessQA_Dataset/benchmark/icicos_process_qa.json
5cef9731f7a768c8cf1f78575d38f1485166213d8395f2a33979c74cbe2fba3b  ICICOS_ProcessQA_Dataset/benchmark/icicos_process_qa.csv
29c943a6e75d136625f93a81b82989c168a02ffee12a5778c366d1d70343049d  ICICOS_ProcessQA_Dataset/cohorts/primary_n70.csv
00af2ee8e2ba0724033fef0b4266428f5ab19e91e39b77b83c6df6ee73bb78c7  ICICOS_ProcessQA_Dataset/cohorts/procedural_n60.csv
1c91a12148cb7c184bce1521f4f7b852a62c9346bbe56e47fc67c5db136c6f2e  ICICOS_ProcessQA_Dataset/cohorts/ablation_n20.csv
3efc4d09e2d34d42d6193b7192361a79b08aa34268fda68cc4ec47a7a05ab8ba  ICICOS_ProcessQA_Dataset/cohorts/cross_backbone_evaluations_n34.csv
1a7511b310c2b15a4dc9ec42abecfb18b4bf952381e2b8b34dc396c910f76cb7  ICICOS_ProcessQA_Dataset/cohorts/full_corpus_router_n338.csv
d848f2548b3da31e67a6b0e6862c85a3631047ff2e8d22309f0c54ad2420aa9f  results/primary_benchmark_manifest.json
186f10766eeb2d6be175ab72bb0fd91ee712ff18b8bbb65edcabab518f6bf6ff  results/sampling_manifest.json
fbcc8b50e5514ee097d4f9dd9632dd2d4a57d73f0c8b76a670219aae13ebbbf6  results/master_results.csv
79bee445c3a46c4ef1080aa977c018ab04a3dfbbec960447998b92400a0e09f2  results/benchmark_results_primary_gemini.json
49e87222ea1d69c66ea6b7af6d554a06a23a8f15ad96a7927dd172221f1d67b1  results/benchmark_results_primary_qwen.json
68479d67ce93327bfa503a52fab81087163012940902d6913ccf0f3c52e22674  results/failed_queries.json
1c709b33f37ff5a732cf54194322c5e3587845a72744a579c0bbe85cb7d23506  results/execution_log.json
8f59dacdce5f9b3a6ad883f51971db62545a015942bd7b2014c31170496eebc0  evaluation_config.yaml
```

---

## 5. Experiment Provenance & Run Metadata Tracking

To eliminate future ambiguity in benchmark replication, every execution must be bound to:
1. **Git Commit SHA:** Fixed at `4defd04678f66ea64f674ea0bf107fc7f907edc3`.
2. **Standardized Schema:** Formal JSON Schema (draft 2020-12) created at [`evaluation/run_metadata_schema.json`](evaluation/run_metadata_schema.json).
3. **Structured Metadata Record:** Created at [`evaluation/reproducibility_run_metadata.json`](evaluation/reproducibility_run_metadata.json) linking dataset hashes, model parameters, hardware environment, and metric outcomes.

---

## 6. Dependency Management & Exact Environment Locking

### 6.1 Generated `environment.lock`
The dedicated virtual environment (`.venv/`) was audited and frozen directly into [`environment.lock`](environment.lock). Key packages pinned include:
- `langchain==1.3.16`
- `langchain-core==1.6.1`
- `langchain-community==0.4.2`
- `langchain-google-genai==4.4.0`
- `langchain-groq==1.1.3`
- `groq==0.37.1`
- `ragas==0.4.3`
- `chromadb==1.5.9`
- `sentence-transformers==6.0.0`
- `torch==2.13.0`
- `scipy==1.17.1`
- `numpy==2.4.6`
- `networkx==3.6.1`
- `pydantic==2.13.4`

### 6.2 Updated Project Configuration Files
- [`requirements.txt`](requirements.txt): Updated with justified dependencies for cross-backbone testing (`langchain-groq`, `groq`), statistical validation (`scipy`, `pandas`), and graph analysis (`networkx`).
- [`pyproject.toml`](pyproject.toml): Version bumped to `0.2.0`; core dependencies reconciled with optional groups for `cross-backbone` and `eval`.
- [`evaluation_config.yaml`](evaluation_config.yaml): Reconciled with Elsevier KBS standards, exact $N=70$ primary sample size, and $T=0.0$ greedy sampling parameters.

---

## 7. Experiments That Cannot Be Reconstructed From Historical Logs

In adherence to Non-Negotiable Principle 1 & 4 (Never fabricate logs or claim execution without evidence):

1. **Component Ablation Diagnostic ($N=20$):**
   - In `evaluation/generate_ablation_analysis.py`, lines 19-72 contain hardcoded summary numbers without raw query-level outputs (`ablation_results.csv` contains only aggregate mean rows).
   - *Verdict:* While the relative degradation trend ($w/o\text{ BPMN} = 0.984$, $w/o\text{ Router} = 0.925$, $\text{Dense} = 0.742$) is directionally consistent, raw query-level outputs cannot be audited without re-running the 20-query ablation suite.
2. **Cross-Backbone Cohort for Qwen ($N=63$ remaining queries):**
   - Only 7 queries (34 executions) exist in `master_results.csv`. The remaining 63 queries of the primary cohort were never executed on Qwen due to Groq OTPM rate limits.
   - *Verdict:* Any claim of cross-backbone generalization across all 70 queries is scientifically invalid; the manuscript's claim must remain strictly bounded to the 34-execution diagnostic subset.

---

## 8. Resource & Cost Estimation for Proposed Experimental Runs

In strict accordance with the Master Directive, **no external API calls will be initiated without explicit author approval**. Below is the resource, token, and cost budget for executing candidate benchmark runs:

### Option A: Complete Primary Benchmark Re-evaluation ($N=70$ Queries, 5 Architectures)
- **Target:** 70 queries across LLM-Only, Traditional RAG, Graph RAG, Static Hybrid RAG, and Proposed Agentic.
- **Estimated API Calls:**
  - 4 baseline architectures: $70 \times 4 = 280$ calls.
  - Proposed Agentic (Router + Draft + Verification + Revision): $70 \times \sim 2.4 = 168$ calls.
  - **Total API Calls:** $\approx 448$ calls.
- **Token Budget:**
  - Total Input Tokens: $\approx 896,000$ tokens ($\approx 0.90\text{M}$ tokens).
  - Total Output Tokens: $\approx 179,200$ tokens ($\approx 0.18\text{M}$ tokens).
- **Estimated Financial Cost:**
  - On Google Gemini 3.5 Flash-Lite (Paid Tier: $\$0.075 / 1\text{M}$ input, $\$0.30 / 1\text{M}$ output): **$\approx \$0.12$ USD**.
  - On Google AI Studio Free Tier (500 requests/day limit): **$\approx \$0.00$ USD** (requires 1 day of quota or 2 API keys).
- **Estimated Execution Time:** $\approx 25 - 35$ minutes (with exponential backoff delay).

### Option B: Component Ablation Suite Execution ($N=20$ Queries, 4 Conditions)
- **Target:** 20 queries evaluated across: Proposed Agentic, $w/o\text{ BPMN}$, $w/o\text{ Dynamic Quotas}$, $w/o\text{ Critic Repair}$.
- **Total API Calls:** $\approx 160$ calls.
- **Estimated Financial Cost:** **$\approx \$0.04$ USD** (or within daily free-tier quota).
- **Estimated Execution Time:** $\approx 10 - 15$ minutes.

### Option C: Complete Cross-Backbone Qwen-2.5-7B Suite ($N=70$ Queries)
- **Target:** Complete the remaining 63 queries on Qwen via Groq Cloud API.
- **Total API Calls:** $\approx 400$ calls.
- **Rate Limit Requirement:** Groq on-demand OTPM limit is 1,000 tokens/min. Requires an intentional 3-second sleep between requests or Groq Dev Tier upgrade.
- **Estimated Financial Cost:** **$\approx \$0.08$ USD** on Groq Dev Tier.

---

## 9. Deliverables Sign-Off

- [x] Full model configuration audit completed with determinism boundaries established.
- [x] Verified cohort manifests created and documented with exact category distributions.
- [x] Dataset integrity verified: 338 unique queries, 0 duplicate IDs, 0 null fields.
- [x] Cryptographic SHA-256 checksums recorded in `evaluation/DATASET_MANIFEST_CHECKSUMS.sha256`.
- [x] Structured run metadata schema (`evaluation/run_metadata_schema.json`) and run record (`evaluation/reproducibility_run_metadata.json`) created.
- [x] Exact environment locked in `environment.lock`; `pyproject.toml` and `requirements.txt` reconciled.
- [x] Incomplete/truncated experiments forensically isolated and documented.
- [x] API call budget and financial cost presented for author authorization.
