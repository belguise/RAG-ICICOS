# RESEARCH REPRODUCIBILITY CHECKLIST

**Target Publication:** *Knowledge-Based Systems* (Elsevier, Q1)  
**Manuscript Title:** ICICOS: An Intent-Conditioned BPMN-SOP Hybrid RAG Framework for Explainable Institutional Process Question Answering  
**Evaluation Standard:** ACM Artifact Review and Badging / Elsevier Reproducibility Guidelines  
**Audited Commit SHA:** `4defd04678f66ea64f674ea0bf107fc7f907edc3`  
**Audit Completion Date:** November 3, 2026  

---

## 1. Research Integrity & Non-Negotiable Rules

- [x] **No Fabricated Data**: All reported metrics and confidence intervals are computed directly from raw per-query experimental trials.
- [x] **No Hardcoded Statistical Summaries**: `evaluation/statistical_analysis.py` has been completely purged of synthetic summary arrays.
- [x] **No Cosmetic Matching**: Discrepancies between draft models and empirical outputs have been documented transparently rather than artificially patched.
- [x] **No In-Place Overwrites of Historical Records**: Original trial logs in `results/master_results.csv` are preserved as immutable experimental baselines.
- [x] **No Unsupported Determinism Claims**: Acknowledged that LLM decoding temperature $T=0.0$ does not guarantee bit-for-bit identical token generation across different cloud provider API revisions.

---

## 2. Artifact Identification & Versioning

- [x] **Git Repository Version**: Tagged at release candidate commit `4defd04678f66ea64f674ea0bf107fc7f907edc3`.
- [x] **Branch Name**: `public-release`.
- [x] **Machine-Readable Metadata**: `evaluation/experiment_config.json` and `evaluation/reproducibility_run_metadata.json` define execution schemas and provenance.
- [x] **Zenodo Archival Metadata**: Complete metadata prepared in `ZENODO_UPLOAD_METADATA.md` ready for DOI assignment upon manuscript acceptance.
- [x] **License**: Permissive open-source license ([MIT License](LICENSE)) included.

---

## 3. Dependency & Environment Completeness

- [x] **Python Runtime**: Strictly verified on Python 3.11.9.
- [x] **Primary Dependencies**: `requirements.txt` specifies minimum and tested package versions.
- [x] **Packaging Standard**: `pyproject.toml` configured with standard PEP 517/518 build metadata (version `0.2.0`).
- [x] **Fully Pinned Lockfile**: `environment.lock` records all 154 installed packages in the verified `.venv`.
- [x] **Cross-Platform Compatibility**: Code runs on Linux, macOS, and Windows with normalized path separators (`pathlib.Path`).

---

## 4. Dataset Provenance & Integrity

- [x] **Canonical Master Dataset**: 338 curated institutional process queries located in `ICICOS_ProcessQA_Dataset/benchmark/`.
- [x] **Cryptographic Hash Manifest**: `evaluation/DATASET_MANIFEST_CHECKSUMS.sha256` provides SHA-256 signatures for all data files.
- [x] **Stratified Evaluation Cohorts**:
  - `cohorts/primary_n70.csv`: 70 queries across 4 categories (18 Document, 18 Workflow, 17 Gateway, 17 Hybrid).
  - `cohorts/procedural_n60.csv`: 60 queries strictly disjoint with the primary benchmark.
  - `cohorts/ablation_n20.csv`: 20 balanced queries for component contribution validation.
  - `cohorts/cross_backbone_evaluations_n34.csv`: 34 evaluations across open-weights backbones.
- [x] **Data Quality Audit**: No duplicate query IDs, missing values, or corrupted labels exist in canonical manifests.

---

## 5. Architectural Pipeline & Code Modularity

- [x] **Query Understanding Agent**: Decomposes user intent and extracts procedural entities (`src/agentic_rag_sop/agents/query_understanding.py`).
- [x] **Adaptive Intent Router**: Categorizes query into `SOP_ONLY`, `BPMN_ONLY`, or `HYBRID` (`src/agentic_rag_sop/agents/router.py`).
- [x] **Dense Vector Retrieval**: ChromaDB integration with cosine semantic search (`src/agentic_rag_sop/retrieval/sop_retriever.py`).
- [x] **BPMN Topological Retrieval**: Directed reachability, gateway extraction, and swimlane role lookup (`src/agentic_rag_sop/bpmn/storage.py`, `src/agentic_rag_sop/retrieval/bpmn_retriever.py`).
- [x] **Cross-Modal Fusion**: Merges graph traces with textual clauses into unified prompts (`src/agentic_rag_sop/fusion/context_fusion.py`).
- [x] **Adversarial Critic**: 5-dimensional evaluation (factual grounding, domain compliance, workflow correctness, SOP consistency, completeness) (`src/agentic_rag_sop/agents/critic.py`).
- [x] **Bounded Self-Repair**: Maximum 2 revision iterations with fail-safe fallback (`src/agentic_rag_sop/generation/chain.py`).
- [x] **Clean Import Graph**: Zero circular imports across all core modules.
- [x] **Zero Credential Leaks**: Scanned tracked files; zero secrets or active API keys present.

---

## 6. Evaluation Metrics & Statistical Methods

- [x] **Metric Specification Document**: `evaluation/METRIC_SPECIFICATION.md` establishes mathematical formulations, input variables, and edge cases.
- [x] **Reconstructed Evaluator**: `evaluation/metrics/reconstructed_evaluator.py` implements multi-engine metric evaluations without synthetic shortcuts.
- [x] **Paired Experimental Design**: All comparative tests evaluate identical queries across competing architectures.
- [x] **Non-Parametric Statistical Testing**: Two-sided Wilcoxon signed-rank tests applied to all paired comparisons.
- [x] **Multiplicity Control**: Family-wise error rate controlled via step-down Holm-Bonferroni correction ($\alpha = 0.05$).
- [x] **Empirical Bootstrap Confidence Intervals**: 10,000 paired bootstrap resamples (`seed=42`) used to compute exact 95% CIs.
- [x] **Standardized Effect Sizes**: Cohen's $d_z$ computed for paired observations to quantify practical significance.

---

## 7. Execution & Verification Verification Checklist

- [x] **Automated Regression Suite**: 58/58 unit and integration tests passing (`python -m unittest discover tests`).
- [x] **Statistical Script Execution**: `python evaluation/statistical_analysis.py` executes without errors and generates 4 authoritative CSV tables.
- [x] **Table-to-Manuscript Parity**: Every numerical value in manuscript Tables 2, 3, 4, and 5 matches `results/` outputs.
- [x] **Ablation Study Consistency**: Component contributions (`without_router`, `without_bpmn`, `without_critic`) validated.
- [x] **Latency Breakdown**: Conditional repair latency verified (9.97s compliant, 13.22s repaired, +3.26s repair overhead).
- [x] **Relevancy Trade-off Defense**: Validated that the $-0.0354$ Answer Relevancy shift is statistically non-significant under Holm-Bonferroni correction ($p_{\text{adj}} = 0.1030$).

---

## 8. Artifact Reviewer Instructions

To verify the artifact independently:

1. **Verify dependencies and environment**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Verify dataset integrity**:
   ```bash
   python -c "
   import hashlib, pathlib
   for line in pathlib.Path('evaluation/DATASET_MANIFEST_CHECKSUMS.sha256').read_text().splitlines():
       if line and not line.startswith('#'):
           h, p = line.split()
           assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() == h
   print('ALL DATASETS VERIFIED OK')
   "
   ```
3. **Run regression test suite**:
   ```bash
   python -m unittest discover -s tests -p "test_*.py" -v
   ```
4. **Run paired statistical analysis**:
   ```bash
   python evaluation/statistical_analysis.py
   ```
5. **Inspect generated statistical tables**:
   - `results/significance_tests.csv`
   - `results/bootstrap_ci_differences.csv`
   - `results/architecture_comparison_table.csv`
   - `results/ablation_table.csv`
