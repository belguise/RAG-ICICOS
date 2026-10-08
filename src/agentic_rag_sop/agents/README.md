# Phase 2 Agentic Layer Integration Guide

This directory is reserved for Phase 2: Agentic RAG.

## Architecture Blueprint

```text
User Query
    │
    ▼
[Query Understanding Agent]
    │ - Query rewriting & acronym normalization
    │ - Intent classification (SOP fact lookup vs Process flow path)
    ▼
[Router Agent]
    ├─── If Fact/Step Lookup ────────► [Retriever (Phase 1)]
    └─── If Graph/Path Reasoning ────► [BPMN Process Graph Retriever (Phase 1)]
    │
    ▼
[Planner Agent]
    │ - Decomposes complex multi-document workflows into sub-tasks
    │ - Coordinates sub-retrievals across multiple SOPs
    ▼
[Gemini LLM Synthesis (Phase 1)]
    │
    ▼
[Critic Agent]
    │ - Evaluates answer adherence against retrieved chunks (Self-Correction)
    │ - Flags hallucinated steps or missing actor constraints
    ▼
Final Verified Answer
```

## Phase 1 Components Ready for Agents:
1. `agentic_rag_sop.retrieval.SOPRetriever`: Ready to be exposed as a LangChain `Tool` for the Planner & Router.
2. `agentic_rag_sop.bpmn.BaseBPMNProcessRetriever`: Interface ready for graph path traversal and decision-tree reasoning.
3. `agentic_rag_sop.core.llm.get_llm()`: Shared LLM backbone across all agents.
