# BBELS METRIC GROUND TRUTH SPECIFICATION
## Comprehensive Methodological & Algorithmic Audit of Evaluation Metrics

**Date of Verification:** November 4, 2026  
**Auditor:** Scientific Methodology & Software Verification Unit (BBELS Mode)  
**Target Codebase:** ICICOS Evaluation Suite  

---

### 1. Architectural Call Chain & Implementation Map

All evaluation metrics reported in the KBS manuscript and primary benchmark results (`results/master_results.csv`) are implemented in:
- Primary Engine: [`evaluation/metrics/reconstructed_evaluator.py`](file:///c:/FILE%20LINATNG/RAG%20BELAJAR/evaluation/metrics/reconstructed_evaluator.py)
- Standalone Verification Harness: [`evaluation/metrics/standard_rag_metrics.py`](file:///c:/FILE%20LINATNG/RAG%20BELAJAR/evaluation/metrics/standard_rag_metrics.py)
- Statistical Test Harness: [`evaluation/statistical_analysis.py`](file:///c:/FILE%20LINATNG/RAG%20BELAJAR/evaluation/statistical_analysis.py)

```
[ Model Generation Output (Answer A, Context C, Query Q) ]
                            │
                            ▼
        [ evaluation/metrics/reconstructed_evaluator.py ]
         ├── calculate_faithfulness(answer, context)
         ├── calculate_evidence_grounding(answer, context)  --> EUSR = 1.0 - EGS
         ├── calculate_answer_relevancy(query, answer)
         ├── calculate_context_precision(retrieved_ids, reference_ids)
         ├── calculate_context_recall(retrieved_ids, reference_ids)
         ├── calculate_semantic_f1(answer, reference_answer)
         └── calculate_procedural_diagnostics(answer, gold_bpmn_trace)
                            │
                            ▼
               [ results/master_results.csv ]
                            │
                            ▼
             [ evaluation/statistical_analysis.py ]
         ├── Non-parametric Paired Bootstrap (B=10,000)
         ├── Paired Wilcoxon Signed-Rank Test
         ├── Paired Cohen's d_z = mean(D) / std(D)
         └── Step-Down Holm-Bonferroni Multiplicity Control (7 Hypotheses)
```

---

### 2. Precise Mathematical & Operational Definitions

#### A. Faithfulness ($F_{\text{faith}}$)
- **Function:** `calculate_faithfulness(answer: str, context: str) -> float`
- **Inputs:** Generated response text $A$, concatenated retrieved passages context $C$. Gold references are strictly withheld.
- **Operational Algorithm:**
  1. Split answer $A$ into substantive sentences $S = \{s_1, \dots, s_m\}$ where $\text{len}(s_i) > 15$ characters.
  2. For each sentence $s_i$, compute substantive token overlap against context $C$:
     $$\text{TokenOverlap}(s_i, C) = \frac{|\text{Tokens}(s_i) \cap \text{Tokens}(C)|}{|\text{Tokens}(s_i)|}$$
     where tokens exclude punctuation and language-specific stop words.
  3. Sentence $s_i$ is scored as supported if $\text{TokenOverlap}(s_i, C) \ge 0.45$ OR if $s_i$ contains an exact 4-gram phrase appearing verbatim in $C$.
  4. Aggregate metric:
     $$F_{\text{faith}} = \frac{1}{|S|}\sum_{i=1}^{|S|} \mathbb{I}\left[ \text{TokenOverlap}(s_i, C) \ge 0.45 \lor \text{HasExact4Gram}(s_i, C) \right]$$
- **Construct Validity & Limitations:**
  - Guarantees deterministic evaluation with zero evaluator variance.
  - Measures surface lexical containment against retrieved premises.
  - Does **NOT** execute natural language inference (NLI), LLM-as-a-judge prompt evaluation, or formal proposition entailment.
  - A score of `0.9883` reflects 98.83% lexical sentence containment, which should not be conflated with 98.83% verified semantic factual accuracy.

#### B. Evidence-Ungrounded Sentence Rate (EUSR)
- **Function:** `calculate_evidence_grounding(answer: str, context: str) -> float`
- **Inputs:** Generated response text $A$, retrieved context $C$.
- **Operational Algorithm:**
  1. Evaluates substantive token containment using a dedicated sentence parser.
  2. Computes the Evidence Grounding Score (EGS) as the fraction of sentences satisfying substantive containment threshold $\ge 0.45$.
  3. Computes EUSR as the complement:
     $$\text{EUSR} = 1.0 - \text{EGS}$$
- **Mathematical Dependency:**
  - EUSR is evaluated through a dedicated sentence parser distinct from $F_{\text{faith}}$, but both operate on substantive token containment against $C$.
  - While complementary in direction ($\Delta F_{\text{faith}} = +0.1094, \Delta \text{EUSR} = -0.1094$), EUSR reflects ungrounded lexical phrasing rather than human-verified semantic hallucination.

#### C. Answer Relevancy ($R_{\text{ans}}$)
- **Function:** `calculate_answer_relevancy(query: str, answer: str) -> float`
- **Inputs:** Normalized user query $q'$, generated answer $A$. Gold reference answers are strictly withheld.
- **Operational Algorithm:**
  1. Dense Semantic Similarity: Compute sentence embedding cosine similarity between $q'$ and $A$ using `paraphrase-multilingual-MiniLM-L12-v2` ($d=384$):
     $$\text{Sim}_{\text{embed}}(q', A) = \cos(\mathbf{e}_{q'}, \mathbf{e}_A) = \frac{\mathbf{e}_{q'} \cdot \mathbf{e}_A}{\|\mathbf{e}_{q'}\| \|\mathbf{e}_A\|}$$
  2. Interrogative Keyword Coverage: Compute lexical coverage of key interrogative and topic tokens from $q'$ within $A$:
     $$\text{Cov}_{\text{key}}(q', A) = \frac{|\text{KeyTokens}(q') \cap \text{Tokens}(A)|}{|\text{KeyTokens}(q')|}$$
  3. Weighted Composite:
     $$R_{\text{ans}} = 0.70 \cdot \text{Sim}_{\text{embed}}(q', A) + 0.30 \cdot \text{Cov}_{\text{key}}(q', A)$$
- **Construct Validity & Limitations:**
  - Calculations do **NOT** invoke LLM reverse-question generation (e.g. generating synthetic questions from $A$ and computing cosine similarity with $q'$).
  - The drop in Answer Relevancy from $0.7876$ to $0.7522$ ($\Delta = -0.0354$) reflects strict conciseness enforcement by the Critic.
  - Step-down Holm-Bonferroni correction confirms this difference is statistically non-significant ($p_{\text{adj}} = 0.1030 > 0.05$). Non-significance indicates absence of evidence for a significant difference, not affirmative proof of equivalence.

#### D. Context Precision ($P_{\text{ctx}}$) and Context Recall ($R_{\text{ctx}}$)
- **Functions:** `calculate_context_precision(retrieved_ids, reference_ids)`, `calculate_context_recall(retrieved_ids, reference_ids)`
- **Inputs:** Retrieved chunk IDs, author-curated gold reference chunk IDs.
- **Reference Access Policy:** Accessed **strictly post-hoc** during offline evaluation. Completely withheld from runtime generation and Critic verification.
- **Degeneracy Finding for Context Recall:**
  - Across all 70 pairs in the primary comparative suite, both Proposed ICICOS and Static Hybrid RAG retrieved 100% of the curated reference chunk IDs ($R_{\text{ctx}} = 1.0000$ identically for all 70 pairs).
  - Paired difference variance $s_D^2 = 0$. Hence, paired $t$-statistic and Cohen's $d_z$ are $0/0$ (mathematically undefined) and are formally designated `N/A*`.

#### E. Semantic Alignment F1 (SA-F1)
- **Function:** `calculate_semantic_f1(answer: str, reference: str) -> float`
- **Inputs:** Generated answer $A$, gold reference answer $A^*$. Accessed strictly during offline evaluation.
- **Operational Algorithm:**
  1. Splits $A$ and $A^*$ into individual sentences.
  2. Encodes sentences using `paraphrase-multilingual-MiniLM-L12-v2`.
  3. Computes greedy bipartite matching across sentence embedding cosine similarities to compute Precision, Recall, and harmonic mean F1:
     $$\text{SA-F1} = \frac{2 \cdot P_{\text{align}} \cdot R_{\text{align}}}{P_{\text{align}} + R_{\text{align}}}$$
- **Construct Validity:** Does **NOT** invoke HuggingFace `bert_score` or BERT token embeddings. Operates at the sentence-embedding level.

#### F. Procedural Diagnostics (PPA, GDA, AAR)
- **Cohort:** Evaluated on the specialized 60-query procedural subset ($N=20$ workflow, $N=20$ decision gateway, $N=20$ hybrid).
- **Process Path Accuracy (PPA):**
  $$\text{PPA} = 0.6 \cdot \text{Cov}_{\text{task}} + 0.4 \cdot \text{Ord}_{\text{task}}$$
  evaluates topological task node coverage and pairwise execution order preservation against gold NetworkX execution traces.
- **Gateway Decision Accuracy (GDA):** Evaluates whether the generated text adheres to the serialized gateway condition expression (e.g. $[Score < 3.0] \to \text{Rejection}$).
- **Actor Attribution Recall (AAR):** Macro recall of assigned swimlane roles against gold organizational roles.

---

### 3. Metric Summary Table

| Metric | Code Function | Input Premises | Output Range | Historical Execution Script | Provenance Status |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **Faithfulness ($F_{\text{faith}}$)** | `calculate_faithfulness` | Generated $A$, Context $C$ | $[0, 1]$ | `reconstructed_evaluator.py` | **VERIFIED** |
| **EUSR (Ungrounded Rate)** | `calculate_evidence_grounding` | Generated $A$, Context $C$ | $[0, 1]$ | `reconstructed_evaluator.py` | **VERIFIED** |
| **Answer Relevancy ($R_{\text{ans}}$)** | `calculate_answer_relevancy` | Query $q$, Generated $A$ | $[0, 1]$ | `reconstructed_evaluator.py` | **VERIFIED** |
| **Context Precision ($P_{\text{ctx}}$)** | `calculate_context_precision` | Retrieved IDs, Gold IDs | $[0, 1]$ | `reconstructed_evaluator.py` | **VERIFIED** |
| **Context Recall ($R_{\text{ctx}}$)** | `calculate_context_recall` | Retrieved IDs, Gold IDs | $[0, 1]$ | `reconstructed_evaluator.py` | **VERIFIED (DEGENERATE)** |
| **Semantic Alignment F1** | `calculate_semantic_f1` | Generated $A$, Gold $A^*$ | $[0, 1]$ | `reconstructed_evaluator.py` | **VERIFIED** |
| **Path Accuracy (PPA)** | `calculate_path_accuracy` | Generated $A$, Gold Path | $[0, 1]$ | `reconstructed_evaluator.py` | **VERIFIED (DIAGNOSTIC)** |
| **Gateway Accuracy (GDA)** | `calculate_gateway_accuracy` | Generated $A$, Gold Guards | $[0, 1]$ | `reconstructed_evaluator.py` | **VERIFIED (DIAGNOSTIC)** |
| **Actor Attribution Recall** | `calculate_actor_accuracy` | Generated $A$, Gold Roles | $[0, 1]$ | `reconstructed_evaluator.py` | **VERIFIED (DIAGNOSTIC)** |
