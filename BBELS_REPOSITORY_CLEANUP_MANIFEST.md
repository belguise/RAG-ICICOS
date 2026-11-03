# BBELS REPOSITORY CLEANUP MANIFEST
## File Classification, Preservation Strategy, and Ignore Policy Audit

**Target Journal:** *Knowledge-Based Systems* (Elsevier, Q1)  
**Canonical Manuscript Title:** *"Adaptive Agentic Hybrid RAG with BPMN-Grounded Graph Reasoning for Explainable Enterprise Process Question Answering"*  
**Repository:** `https://github.com/belguise/RAG-ICICOS`  
**Lead Auditor:** Senior Git Release Engineer & Research Reproducibility Auditor (BBELS Ultra-Maximum Mode)  
**Audit Timestamp:** November 4, 2026, 02:02:00 UTC+7  
**Status:** REPOSITORY CLEANED & CLASSIFIED — ZERO DESTRUCTIVE BULK DELETIONS

---

## 1. COMPREHENSIVE FILE CLASSIFICATION TAXONOMY

In accordance with Phase D directives, every file group in the workspace is assigned a distinct release category:

| Directory / File Category | Release Classification | Rationale & Retention Policy |
| :--- | :---: | :--- |
| **`src/agentic_rag_sop/`** | `PUBLISH_REQUIRED` | Core ICICOS agentic architecture, agents, retrieval modules, and settings. Must be public. |
| **`evaluation/`** | `PUBLISH_REQUIRED` | Reconstructed evaluator, statistical analysis engine, and manifest checksums. Essential for reproducibility. |
| **`tests/`** | `PUBLISH_REQUIRED` | Unit, integration, and security regression tests (`test_api_cors_security.py`). Essential for verification. |
| **`scripts/verify_research_release.py`** | `PUBLISH_REQUIRED` | Automated reviewer verification suite to reproduce Table 2 statistics without manual setup. |
| **`results/master_results.csv`** | `PUBLISH_REQUIRED` | The 388 raw trial observations underlying all primary empirical claims. Lightweight (45 KB). |
| **`results/statistical_tests.csv`** | `PUBLISH_REQUIRED` | Recomputed inferential statistics and Holm-Bonferroni correction values. |
| **`results/final_comparison_table.csv`** | `PUBLISH_REQUIRED` | Descriptive summary table across all 5 architectures on $N=70$ paired queries. |
| **`README.md`, `LICENSE`, `pyproject.toml`** | `PUBLISH_REQUIRED` | Authoritative project introduction, licensing (MIT), and dependency specifications. |
| **`BBELS_*.md`, `BBELS_*.csv`** | `PUBLISH_REQUIRED` | Forensic audit reports, evidence matrices, and mathematical reproduction records. |
| **`results/benchmark_results_primary_gemini.json`** | `LOCAL_ONLY` / `ZENODO` | Heavyweight JSON execution traces (884 KB). Preserved locally; registered on Zenodo. |
| **`ICICOS_ProcessQA_Dataset.zip`** | `LOCAL_ONLY` / `ZENODO` | Primary dataset archive (83,176 B). Deposited on Zenodo under DOI `10.5281/zenodo.23242855`. |
| **`.env`** | `SENSITIVE_PRIVATE` | Local development API keys. Ignored by `.gitignore`. Never committed to Git. |
| **`FINAL KBS FIX/*.pdf` (Manuscript & Figures)**| `LOCAL_ONLY` / `AUDIT` | Authoritative 22-page manuscript and vector figures. Retained locally for journal submission package. |
| **Temporary LaTeX Intermediates (`.aux`, `.log`)**| `BUILD_ARTIFACT` | Disposable compiler artifacts. Excluded from Git tracking. |
| **Ephemeral Caches (`chroma_db/`, `__pycache__/`)**| `BUILD_ARTIFACT` | Runtime vector caches and Python bytecode. Ignored by `.gitignore`. |
| **External Recoverable Backup** | `HISTORICAL_PRESERVE` | Located outside repository at `C:\FILE LINATNG\BACKUP_RAG_BELAJAR_RELEASE_20261104`. |

---

## 2. `.GITIGNORE` POLICY REFINEMENT

The `.gitignore` configuration was audited and hardened to:
1. **Prevent Leaking Sensitive Credentials:** `.env`, `.env.*`, `*.pem`, `*.key`, `credentials*`, `secrets*`.
2. **Prevent Leaking Ephemeral Databases:** `chroma/`, `chroma_db/`, `.chroma/`, `data/vector_store/`, `*.sqlite3`, `*.db`.
3. **Allow Public Verification Tools:** Explicitly permits `scripts/verify_research_release.py`.
4. **Allow Essential Empirical Tables:** Explicitly permits `results/master_results.csv`, `results/statistical_tests.csv`, and `results/final_comparison_table.csv`.
5. **Exclude Disposable Compiler & Draft PDFs:** Redundant intermediate PDF renders and LaTeX scratch files are ignored to avoid repository bloat.

---

## 3. AUDIT OF REMOVED & PRESERVED ASSETS

- **Destructive Deletion Policy:** Zero historical CSVs, JSON logs, or python scripts were deleted.
- **External Archiving:** All pre-release git states and uncommitted diffs were bundled into `C:\FILE LINATNG\BACKUP_RAG_BELAJAR_RELEASE_20261104\repo_bundle_all_refs.bundle` (2.4 MB).
- **Workspace Hygiene:** Scratch test files (`test_frontmatter.pdf`, `test_frontmatter.tex`) remain untracked and excluded from staging.