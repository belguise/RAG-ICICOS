# Zenodo Upload Metadata: ICICOS ProcessQA Benchmark Dataset

Use the following metadata fields when manually creating the deposit on [Zenodo](https://zenodo.org/deposit):

---

## 1. Basic Metadata

- **Resource Type:** Dataset
- **Title:** ICICOS ProcessQA Dataset: Enterprise Process Question Answering over Standard Operating Procedures and BPMN 2.0 Graphs
- **Publication Date:** 2026-10-08 (or current upload date)
- **Digital Object Identifier (DOI):** Leave blank to let Zenodo assign a new DOI (or reserve one).

---

## 2. Authors and Affiliations

*(Enter the author names, institutional affiliations, and ORCID identifiers matching the submitted Knowledge-Based Systems manuscript.)*

---

## 3. Description / Abstract

```text
The ICICOS ProcessQA benchmark is a curated empirical dataset designed for evaluating retrieval-augmented generation (RAG) and multi-agent systems on enterprise Standard Operating Procedures (SOPs) and Business Process Model and Notation (BPMN 2.0) workflows.

The dataset comprises 338 operational queries categorized across four distinct process reasoning domains:
1. Workflow Ordering (108 queries): Multi-step execution sequences and role handoffs.
2. Hybrid Reasoning (81 queries): Cross-modal synthesis linking unstructured SOP textual policy and structured BPMN graph topology.
3. Decision Gateways (80 queries): Conditional XOR branching logic, exception handling, and fallback pathways.
4. Document Requirements (69 queries): Declarative checklists, registration prerequisites, and fee schedules.

Package Contents:
- Canonical Benchmark: 338 queries with ground-truth answers, expected actors, source regulations, and execution paths in both JSON and CSV formats.
- Process Graph Models: 15 machine-readable attributed directed graph representations in JSON format, alongside a standard OMG BPMN 2.0 XML process model.
- Experimental Cohort Manifests: Query-level index manifests for the primary comparative benchmark (N=70), procedural diagnostic cohort (N=60, query-disjoint), ablation diagnostic cohort (N=20), cross-backbone diagnostic evaluations (34 evaluations across 7 queries), and full corpus router evaluation (N=338).
- Formal JSON Schema (Draft 2020-12) and cryptographic SHA-256 checksums.

Associated Research Manuscript:
"Adaptive Agentic Hybrid RAG with BPMN-Grounded Graph Reasoning for Explainable Enterprise Process Question Answering" (Knowledge-Based Systems, Elsevier).
```

---

## 4. Keywords

- `Enterprise Process Question Answering`
- `Retrieval-Augmented Generation`
- `BPMN 2.0`
- `Business Process Management`
- `Process-Aware Retrieval`
- `Agentic AI`
- `Knowledge Graphs`
- `Standard Operating Procedures`

---

## 5. Related Identifiers

- **Software Code Repository:** `https://github.com/belguise/RAG-ICICOS`
  - *Relation:* Is supplemented by this software
- **Target Manuscript:** Update with publication DOI upon acceptance in *Knowledge-Based Systems* (Elsevier)
  - *Relation:* Is cited by this publication

---

## 6. Access and License

- **Access Right:** Open Access
- **License:** Creative Commons Attribution 4.0 International (CC-BY 4.0)
