# ICICOS METRIC SPECIFICATION: SCIENTIFIC DEFINITION, FORMULATIONS, AND IMPLEMENTATION AUDIT
**Project:** ICICOS (Intent-Conditioned BPMN-SOP Hybrid RAG Framework)  
**Target Venue:** *Knowledge-Based Systems* (Elsevier, Q1)  
**Document Purpose:** Authoritative metric specification defining mathematical formulations, input variables, reference requirements, actual Python implementations, thresholds, edge-case protocols, and methodological boundaries for all evaluation metrics.

---

## 1. TAXONOMY AND EXECUTIVE SUMMARY

The ICICOS evaluation framework spans three complementary analytical dimensions:
1. **Primary Factual Grounding & Retrieval Alignment**: Faithfulness, Answer Relevancy, Context Precision, Context Recall.
2. **Semantic Text Generation Quality**: Semantic Alignment F1 (SA-F1 / Sentence-BERT F1), Evidence-Ungrounded Sentence Rate (EUSR) / Hallucination Rate, Wall-Clock Latency.
3. **Procedural & Topological Graph Reasoning**: Process Path Accuracy (PPA), Gateway Decision Accuracy (GDA), Actor Attribution Accuracy/Recall (AAA/AAR).

### Summary Table of Metric Specifications

| Metric Name | Manuscript Label | Methodology Family | Ground Truth Required? | Model Dependency | Output Range |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Faithfulness** | $F_{\text{faith}}$ / RAGAS Faithfulness | Lexical-Semantic Claim Overlap (Heuristic) | No (Context only) | Regex + Tokenizer | $[0.0, 1.0]$ |
| **Answer Relevancy** | $R_{\text{ans}}$ / Answer Relevancy | Embedding Cosine + Keyword Coverage | No (Question only) | `paraphrase-multilingual-MiniLM-L12-v2` | $[0.0, 1.0]$ |
| **Context Precision** | $P_{\text{ctx}}$ / Context Precision | Rank-Weighted Average Precision | Yes ($A_{\text{gold}}$ or key facts) | Lexical Tokenizer | $[0.0, 1.0]$ |
| **Context Recall** | $R_{\text{ctx}}$ / Context Recall | Propositional Context Coverage | Yes ($A_{\text{gold}}$) | Lexical Tokenizer | $[0.0, 1.0]$ |
| **Semantic Alignment F1** | SA-F1 (misnamed BERTScore) | Sentence-Level Embedding Max-Pooling F1 | Yes ($A_{\text{gold}}$) | `paraphrase-multilingual-MiniLM-L12-v2` | $[0.0, 1.0]$ |
| **EUSR / Hallucination Rate** | EUSR / HR ($1.0 - \text{EGS}$) | Sentence-Level Ungrounded Ratio | No (Context only) | Regex + Tokenizer | $[0.0, 1.0]$ |
| **Process Path Accuracy** | PPA | Topological Coverage + Concordant Pairs | Yes ($G_{\text{gold}}$ task sequence) | Regex + Procedural Matcher | $[0.0, 1.0]$ |
| **Gateway Decision Accuracy**| GDA | Piecewise Condition Guard + Branch Overlap| Yes ($G_{\text{gold}}$ branch guard) | Regex + Procedural Matcher | $[0.0, 1.0]$ |
| **Actor Attribution Accuracy**| AAA / AAR | Multi-Actor Recall with Synonym Matching | Yes ($G_{\text{gold}}$ swimlane roles) | Domain Synonym Dictionary | $[0.0, 1.0]$ |
| **System Latency** | Latency (s) | Wall-clock execution time | No | Python `time.time()` | $[0.0, \infty)$ |

---

## 2. DETAILED METRIC SPECIFICATIONS

### 2.1 FAITHFULNESS ($F_{\text{faith}}$)

#### 1. Official Metric Name
- **Theoretical Target**: RAGAS Faithfulness (Es et al., 2024).
- **Actual Operational Implementation**: Calibrated Sentence-Level Token-Overlap and N-Gram Grounding (`calculate_faithfulness`).

#### 2. Scientific Definition
The proportion of factual statements or atomic propositions generated in candidate answer $A$ that can be verified as directly entailed by and grounded within the retrieved context $C$. It penalizes parametric confabulation and ungrounded extrapolation.

#### 3. Mathematical Formulation
Given generated answer $A$ segmented into $M$ non-trivial declarative statements $\mathcal{S}(A) = \{s_1, s_2, \dots, s_M\}$, and retrieved context $C$:
$$F_{\text{faith}} = \begin{cases} 
\frac{1}{M} \sum_{i=1}^M \mathbb{I}(\text{Grounded}(s_i, C)), & M > 0 \\
1.0, & M = 0 \text{ and } A \neq \emptyset \\
0.0, & A = \emptyset \text{ or } C = \emptyset
\end{cases}$$
Where the grounding indicator $\mathbb{I}(\text{Grounded}(s_i, C))$ is defined operationally as:
$$\text{Grounded}(s_i, C) = \begin{cases} 
1, & \text{if } \frac{|\mathcal{T}(s_i) \cap \mathcal{T}(C)|}{|\mathcal{T}(s_i)|} \ge \theta_{\text{token}} \quad \text{OR} \quad \exists w \in \text{4-grams}(s_i) : w \in C \\
0, & \text{otherwise}
\end{cases}$$
With parameter $\theta_{\text{token}} = 0.45$, where $\mathcal{T}(\cdot)$ denotes cleaned, lowercased content tokens (length $> 2$, stop words omitted).

#### 4. Required Input Variables
- `generated_answer` ($A$): String, the textual output produced by the system.
- `retrieved_context` ($C$): String, the concatenation of all retrieved SOP passages and BPMN graph serializations provided to the LLM prompt.

#### 5. Reference / Ground-Truth Requirements
- **None**. Gold reference answers ($A_{\text{gold}}$) are strictly withheld to evaluate retrieval faithfulness rather than semantic overlap.

#### 6. Actual Implementation & Location
- Location: [`evaluation/metrics/standard_rag_metrics.py:45-141`](evaluation/metrics/standard_rag_metrics.py#L45-L141).
- Statement segmentation: Regex `re.split(r"(?:[\n;]|\.\s+)", generated_answer)` filtered by length $> 15$ characters.
- Conversational filtering: Strips conversational headers matching `["berdasarkan", "referensi sumber", "catatan:", "ringkasan:"]`.
- Grounding verification: Evaluates substantive token overlap threshold ($\ge 0.45$) or exact 4-gram sliding window string containment.

#### 7. Model Used for Evaluation
- **None** (Deterministic algorithmic heuristic). No external LLM-as-a-judge is invoked in the historical codebase.

#### 8. Thresholds & Parameters
- `min_statement_length`: 15 characters.
- `token_overlap_threshold` ($\theta_{\text{token}}$): 0.45 (45% content token overlap).
- `ngram_window_size`: 4 consecutive words.

#### 9. Output Range
- Bounded real value in $[0.0, 1.0]$. Higher is better ($\uparrow$).

#### 10. Edge Cases & Handling
- **Empty Generated Answer**: Returns $0.0$ (`total_claims = 0, supported_claims = 0`).
- **Empty Context (Zero-Shot / LLM Only)**: Without evidence, external grounding is impossible. Returns $0.0$ (`total_claims = 1, supported_claims = 0`).
- **Punctuation-only / Whitespace Answer**: Handled by strip filtering, returns $0.0$.
- **Conversational Preamble Only**: If all sentences are conversational disclaimers, returns $1.0$ (no ungrounded factual assertions made).

#### 11. Known Limitations
- Does not perform true Natural Language Inference (NLI) directional entailment. Lexical overlap can be fooled by negation inversion (e.g., "SOP does not require receipt" vs "SOP requires receipt").

---

### 2.2 ANSWER RELEVANCY ($R_{\text{ans}}$)

#### 1. Official Metric Name
- **Theoretical Target**: RAGAS Answer Relevancy (Reverse-Question Generation).
- **Actual Operational Implementation**: Dual-Component Semantic Embedding Cosine Similarity and Interrogative Keyword Coverage (`calculate_answer_relevancy`).

#### 2. Scientific Definition
Evaluates whether candidate answer $A$ directly and concisely addresses the inquiry posed in question $q$, penalizing evasive responses, incomplete answers, or tangential digressions.

#### 3. Mathematical Formulation
$$R_{\text{ans}} = \alpha \cdot \text{Sim}_{\text{emb}}(q, A) + (1 - \alpha) \cdot \text{Cov}_{\text{intent}}(q, A)$$
Where:
- $\alpha = 0.70$ (weight assigned to dense embedding alignment).
- Dense semantic similarity:
  $$\text{Sim}_{\text{emb}}(q, A) = \max\left(0.0, \frac{\mathbf{e}_q \cdot \mathbf{e}_A}{\|\mathbf{e}_q\| \|\mathbf{e}_A\|}\right)$$
  Where $\mathbf{e}_q, \mathbf{e}_A \in \mathbb{R}^{384}$ are contextual representations from `paraphrase-multilingual-MiniLM-L12-v2`.
- Interrogative intent keyword coverage:
  $$\text{Cov}_{\text{intent}}(q, A) = \frac{|\mathcal{K}(q) \cap \mathcal{T}(A)|}{|\mathcal{K}(q)|}$$
  Where $\mathcal{K}(q) = \{w \in \mathcal{T}(q) \mid |w| > 3\}$.

#### 4. Required Input Variables
- `question` ($q$): String, user input prompt.
- `generated_answer` ($A$): String, system output answer.

#### 5. Reference / Ground-Truth Requirements
- **Strictly Withheld**. Reference answers are not provided to prevent data leakage.

#### 6. Actual Implementation & Location
- Location: [`evaluation/metrics/standard_rag_metrics.py:148-189`](evaluation/metrics/standard_rag_metrics.py#L148-L189).
- SentenceTransformer: Cached singleton `_EMBEDDING_MODEL` loading `paraphrase-multilingual-MiniLM-L12-v2` with offline fallback.
- Intent extractor: Content words of length $> 3$ from question checked against answer text.

#### 7. Model Used for Evaluation
- `SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")` ($d=384$). Local execution on CPU/GPU.

#### 8. Thresholds & Parameters
- $\alpha = 0.70$, $1 - \alpha = 0.30$.
- Keyword length threshold: $> 3$ characters.

#### 9. Output Range
- Bounded real value in $[0.0, 1.0]$. Higher is better ($\uparrow$).

#### 10. Edge Cases & Handling
- **Empty Answer**: Returns `relevancy_score = 0.0`.
- **Empty Question**: If $q = \emptyset$, returns `0.0`.
- **Embedding Failure / Offline Unavailable**: Falls back gracefully to lexical Jaccard token overlap between $q$ and $A$.

#### 11. Known Limitations
- Official RAGAS generates $M$ synthetic questions $\{q_j\}$ from $A$ using an LLM and measures $\frac{1}{M}\sum \cos(q, q_j)$. The ICICOS implementation directly computes $\cos(q, A)$, which measures question-answer semantic proximity rather than reverse question entailment.

---

### 2.3 CONTEXT PRECISION ($P_{\text{ctx}}$)

#### 1. Official Metric Name
- **Theoretical Target**: Rank-Weighted Context Precision (RAGAS Context Precision).
- **Actual Operational Implementation**: Mean Average Precision (MAP) over Top-$K$ Retrieved Chunks with Lexical Relevance Assignment (`calculate_context_precision`).

#### 2. Scientific Definition
Evaluates whether the retriever prioritizes relevant information at earlier ranks in the retrieved chunk list $C = \langle c_1, c_2, \dots, c_K \rangle$, rewarding systems that place signal above noise.

#### 3. Mathematical Formulation
Given $K$ retrieved passages ordered by rank $k \in \{1, \dots, K\}$, and binary relevance indicator $v_k \in \{0, 1\}$:
$$P_{\text{ctx}} = \begin{cases}
\frac{1}{\sum_{k'=1}^K v_{k'}} \sum_{k=1}^K v_k \cdot \text{Precision@}k, & \sum_{k'=1}^K v_{k'} > 0 \\
0.0, & \sum_{k'=1}^K v_{k'} = 0
\end{cases}$$
Where:
$$\text{Precision@}k = \frac{\sum_{j=1}^k v_j}{k}$$
And the operational chunk relevance $v_k$ is assigned as:
$$v_k = \begin{cases} 
1, & \text{if } \frac{|\mathcal{T}(c_k) \cap \mathcal{T}(A_{\text{gold}})|}{|\mathcal{T}(A_{\text{gold}})|} \ge 0.30 \quad \text{OR} \quad \left( \frac{|\mathcal{T}(c_k) \cap \mathcal{T}(A_{\text{gold}})|}{|\mathcal{T}(A_{\text{gold}})|} \ge 0.20 \ \land \ \frac{|\mathcal{T}(c_k) \cap \mathcal{T}(q)|}{|\mathcal{T}(q)|} \ge 0.35 \right) \\
0, & \text{otherwise}
\end{cases}$$

#### 4. Required Input Variables
- `retrieved_chunks`: List of strings $[c_1, \dots, c_K]$ in order of retriever ranking.
- `expected_answer` ($A_{\text{gold}}$): String, gold ground-truth reference answer.
- `question` ($q$): Optional string, user inquiry.

#### 5. Reference / Ground-Truth Requirements
- **Required**. Requires gold reference answer $A_{\text{gold}}$ to establish ground-truth relevance of chunks.

#### 6. Actual Implementation & Location
- Location: [`evaluation/metrics/standard_rag_metrics.py:195-256`](evaluation/metrics/standard_rag_metrics.py#L195-L256).

#### 7. Model Used for Evaluation
- **None** (Deterministic algorithmic token overlap).

#### 8. Thresholds & Parameters
- Direct Reference Threshold: $\ge 30\%$ token overlap with $A_{\text{gold}}$.
- Hybrid Relevance Threshold: $\ge 20\%$ reference overlap AND $\ge 35\%$ question overlap.

#### 9. Output Range
- Bounded real value in $[0.0, 1.0]$. Higher is better ($\uparrow$).

#### 10. Edge Cases & Handling
- **Empty Retrieved Chunks ($K=0$)**: Returns $0.0$.
- **No Relevant Chunks Retrieved ($\sum v_k = 0$)**: Returns $0.0$.
- **Missing / Empty Reference Answer**: *Bug in legacy code*: previously returned 1.0. *Corrected standard*: must return structured error / invalid (`status: MISSING_REFERENCE`, score: `None` or `0.0`).

#### 11. Known Limitations
- Lexical overlap with $A_{\text{gold}}$ serves as a proxy for chunk relevance; chunks containing gold facts phrased in completely different vocabulary may be marked as non-relevant ($v_k = 0$).

---

### 2.4 CONTEXT RECALL ($R_{\text{ctx}}$)

#### 1. Official Metric Name
- **Theoretical Target**: RAGAS Context Recall.
- **Actual Operational Implementation**: Sentence-Level Proposition Coverage of Reference Answer in Context (`calculate_context_recall`).

#### 2. Scientific Definition
Evaluates whether the retrieved context passages $C$ contain all necessary factual claims required to formulate the author-curated reference answer $A_{\text{gold}}$.

#### 3. Mathematical Formulation
Given gold reference answer $A_{\text{gold}}$ decomposed into $N_{\text{ref}}$ declarative sentences $\{r_1, \dots, r_{N_{\text{ref}}}\}$:
$$R_{\text{ctx}} = \frac{|\{r_j \in \mathcal{S}(A_{\text{gold}}) \mid \text{Covered}(r_j, C)\}|}{N_{\text{ref}}}$$
Where:
$$\text{Covered}(r_j, C) = \begin{cases}
1, & \text{if } \frac{|\mathcal{T}(r_j) \cap \mathcal{T}(C)|}{|\mathcal{T}(r_j)|} \ge 0.40 \\
0, & \text{otherwise}
\end{cases}$$

#### 4. Required Input Variables
- `retrieved_chunks`: List of strings $[c_1, \dots, c_K]$.
- `expected_answer` ($A_{\text{gold}}$): String, gold reference answer.

#### 5. Reference / Ground-Truth Requirements
- **Required**. Ground-truth reference answer is the sole source of evaluation propositions.

#### 6. Actual Implementation & Location
- Location: [`evaluation/metrics/standard_rag_metrics.py:262-307`](evaluation/metrics/standard_rag_metrics.py#L262-L307).

#### 7. Model Used for Evaluation
- **None** (Deterministic algorithmic token overlap).

#### 8. Thresholds & Parameters
- Proposition coverage threshold: $\ge 40\%$ content token containment in concatenated context.
- Minimum reference proposition length: $> 10$ characters.

#### 9. Output Range
- Bounded real value in $[0.0, 1.0]$. Higher is better ($\uparrow$).

#### 10. Edge Cases & Handling
- **Empty Retrieved Chunks ($K=0$)**: Returns $0.0$.
- **Empty Reference Answer**: *Bug in legacy code*: returned 1.0. *Corrected standard*: must return structured missing error (`status: MISSING_REFERENCE`).
- **Context Identical to Reference**: Returns $1.0$.

#### 11. Known Limitations
- In the benchmark runs, $R_{\text{ctx}} = 1.000$ across all 70 queries for retrieval architectures because the SOP corpus chunks are comprehensive and institutional questions are focused.

---

### 2.5 SEMANTIC ALIGNMENT F1 (SA-F1 / Misattributed BERTScore)

#### 1. Official Metric Name
- **Theoretical Target**: BERTScore (Zhang et al., 2020).
- **Actual Operational Implementation**: Sentence-Level Contextual Embedding Alignment F1 (SA-F1) (`calculate_bert_score`).

#### 2. Scientific Definition
Measures the semantic equivalence between generated answer $A$ and gold reference answer $A_{\text{gold}}$ by computing pairwise cosine similarities of sentence representations and taking the harmonic mean of max-pooled precision and recall.

#### 3. Mathematical Formulation
Given answer $A$ split into sentences $\{u_1, \dots, u_m\}$ and reference $A_{\text{gold}}$ split into sentences $\{v_1, \dots, v_n\}$:
1. Contextual sentence embeddings: $\mathbf{x}_i = E(u_i) \in \mathbb{R}^d, \quad \mathbf{y}_j = E(v_j) \in \mathbb{R}^d$ (with $\|\mathbf{x}_i\|_2 = \|\mathbf{y}_j\|_2 = 1$).
2. Cosine similarity matrix: $\mathbf{S}_{ij} = \mathbf{x}_i^\top \mathbf{y}_j \in [-1.0, 1.0]$.
3. Max-pooled precision and recall:
   $$P_{\text{SA}} = \frac{1}{m} \sum_{i=1}^m \max_{1 \le j \le n} \max(0.0, \mathbf{S}_{ij})$$
   $$R_{\text{SA}} = \frac{1}{n} \sum_{j=1}^n \max_{1 \le i \le m} \max(0.0, \mathbf{S}_{ij})$$
4. Harmonic Mean:
   $$\text{SA-F1} = \begin{cases}
   \frac{2 \cdot P_{\text{SA}} \cdot R_{\text{SA}}}{P_{\text{SA}} + R_{\text{SA}}}, & P_{\text{SA}} + R_{\text{SA}} > 0 \\
   0.0, & \text{otherwise}
   \end{cases}$$

#### 4. Required Input Variables
- `generated_answer` ($A$): String, candidate response.
- `reference_answer` ($A_{\text{gold}}$): String, gold ground truth.

#### 5. Reference / Ground-Truth Requirements
- **Required**.

#### 6. Actual Implementation & Location
- Location: [`evaluation/metrics/standard_rag_metrics.py:313-372`](evaluation/metrics/standard_rag_metrics.py#L313-L372).
- Normalized embeddings via `model.encode(sents, normalize_embeddings=True)`.
- Sentence segmentation: `re.split(r"[\n.]+", text)` filtered by length $> 8$ characters.

#### 7. Model Used for Evaluation
- `SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")` ($d=384$).

#### 8. Thresholds & Parameters
- Minimum sentence length: $> 8$ characters.
- Non-negative truncation: $\max(0.0, \mathbf{S}_{ij})$ to prevent negative cosine distortions.

#### 9. Output Range
- Bounded real value in $[0.0, 1.0]$. Higher is better ($\uparrow$).

#### 10. Edge Cases & Handling
- **Empty Answer or Reference**: Returns $0.0$.
- **Exact Identical Text**: Returns $1.0000$.
- **Model Offline / Unavailable**: Falls back to token-level set F1 score.

#### 11. Known Limitations
- Official BERTScore (Zhang et al., 2020) aligns embeddings at the *token level* using BERT/RoBERTa contextual layers with greedy matching and IDF weighting. The ICICOS implementation aligns at the *sentence level* using Sentence-BERT. It is scientifically valid, but must be called **SA-F1**, not BERTScore.

---

### 2.6 EVIDENCE-UNGROUNDED SENTENCE RATE (EUSR) / HALLUCINATION RATE (HR)

#### 1. Official Metric Name
- **Theoretical Target**: Evidence-Ungrounded Sentence Rate (EUSR) / Hallucination Rate.
- **Operational Implementation**: Direct mathematical dual of Evidence-Grounded Sentences ($1.0 - \text{EGS}$).

#### 2. Scientific Definition
The proportion of generated sentences in candidate answer $A$ that lack direct evidentiary grounding in the retrieved context $C$.

#### 3. Mathematical Formulation
$$\text{EUSR} = \text{Hallucination Rate} = 1.0 - \text{EGS}$$
Where $\text{EGS}$ (Evidence-Grounded Sentences) is:
$$\text{EGS} = \frac{|\{s \in \mathcal{S}(A) \mid \text{Grounded}(s, C)\}|}{|\mathcal{S}(A)|}$$
Hence:
$$\text{EUSR} + \text{EGS} \equiv 1.0$$

#### 4. Required Input Variables
- `generated_answer` ($A$): String.
- `retrieved_context` ($C$): String.

#### 5. Reference / Ground-Truth Requirements
- **None**. Evaluated strictly against retrieved context $C$.

#### 6. Actual Implementation & Location
- Location: [`evaluation/metrics/rag_metrics.py:calculate_evidence_grounding`](evaluation/metrics/rag_metrics.py).
- In [`results/master_results.csv`](results/master_results.csv), the column is labeled `hallucination_rate`. In the manuscript (Tables 1, 2, 3), it is presented as **EUSR**.

#### 7. Output Range
- Bounded percentage $[0.0\%, 100.0\%]$ or $[0.0, 1.0]$. Lower is better ($\downarrow$).

#### 8. Edge Cases & Handling
- **Zero-Shot LLM ($k=0$, no context)**: $\text{EUSR} = 1.0$ (100.0%). As noted in Table 1 footnote, this follows structurally from the absence of external evidence and does not mean every claim is factually false.
- **Fully Grounded Answer**: $\text{EUSR} = 0.0$ (0.0%).

---

### 2.7 PROCESS PATH ACCURACY (PPA)

#### 1. Official Metric Name
- **Operational Implementation**: Process Path Accuracy (`calculate_path_accuracy` in [`evaluation/metrics/bpmn_process_metrics.py`](evaluation/metrics/bpmn_process_metrics.py#L65-L140)).

#### 2. Scientific Definition
Testbed-specific procedural diagnostic evaluating whether the execution sequence of business process activities in generated answer $A$ matches the ground-truth BPMN workflow path $G_{\text{gold}}$ in both step coverage and chronological sequence flow.

#### 3. Mathematical Formulation
$$\text{PPA} = 0.6 \cdot \text{Coverage} + 0.4 \cdot \text{OrderPreservation}$$
Where:
- Step Coverage:
  $$\text{Coverage} = \frac{|\mathcal{M}(A, G_{\text{gold}})|}{|G_{\text{gold}}|}$$
  Where $\mathcal{M}(A, G_{\text{gold}})$ is the set of expected BPMN activity nodes detected in $A$ via exact substring match or $\ge 50\%$ substantive content-token overlap.
- Relative Order Preservation (Concordance):
  Given ordered positions of matched steps in answer $A$:
  $$\text{OrderPreservation} = \begin{cases}
  \frac{\sum_{i < j} \mathbb{I}(\text{pos}(s_i) < \text{pos}(s_j))}{\binom{|\mathcal{M}|}{2}}, & |\mathcal{M}| \ge 2 \\
  1.0, & |\mathcal{M}| = 1 \\
  0.0, & |\mathcal{M}| = 0
  \end{cases}$$

#### 4. Required Input Variables
- `predicted_answer`: String.
- `expected_bpmn_path`: List of strings $[s_1, \dots, s_p]$ representing the topological sequence of task and event labels.

#### 5. Output Range
- Bounded real value in $[0.0, 1.0]$. Higher is better ($\uparrow$).

---

### 2.8 GATEWAY DECISION ACCURACY (GDA)

#### 1. Official Metric Name
- **Operational Implementation**: Gateway Decision Accuracy (`calculate_gateway_accuracy` in [`evaluation/metrics/bpmn_process_metrics.py`](evaluation/metrics/bpmn_process_metrics.py#L143-L237)).

#### 2. Scientific Definition
Testbed-specific diagnostic measuring whether the model correctly identifies XOR/OR conditional branching guards, decision predicates, and resulting terminal consequences.

#### 3. Mathematical Formulation
$$\text{GDA} = \begin{cases}
1.0, & \text{if } \text{ConditionIdentified} \ \land \ \text{Overlap} \ge 0.40 \\
0.5 + 0.5 \cdot \text{Overlap}, & \text{if } \text{ConditionIdentified} \ \land \ \text{Overlap} < 0.40 \\
0.4 + 0.4 \cdot \text{Overlap}, & \text{if } \neg\text{ConditionIdentified} \ \land \ \text{Overlap} \ge 0.40 \\
0.2 \cdot \text{Overlap}, & \text{otherwise}
\end{cases}$$
Where $\text{ConditionIdentified} \in \{0, 1\}$ detects presence of institutional branching guards (e.g., "sesuai", "tidak sesuai", "fix cost", "open va", "ditolak", "diterima"), and $\text{Overlap} = \frac{|\mathcal{T}(A) \cap \mathcal{T}(A_{\text{gold}})|}{|\mathcal{T}(A_{\text{gold}})|}$.

#### 4. Output Range
- Bounded real value in $[0.0, 1.0]$. Higher is better ($\uparrow$).

---

### 2.9 ACTOR ATTRIBUTION ACCURACY / RECALL (AAA / AAR)

#### 1. Official Metric Name
- **Operational Implementation**: Actor Attribution Recall / Accuracy (`calculate_actor_accuracy` in [`evaluation/metrics/bpmn_process_metrics.py`](evaluation/metrics/bpmn_process_metrics.py#L240-L330)).

#### 2. Scientific Definition
Evaluates whether the organizational roles (BPMN swimlanes) responsible for executing specific procedural activities are correctly attributed in the generated answer.

#### 3. Mathematical Formulation
$$\text{AAA} = \frac{|\{a \in \text{Actors}(G_{\text{gold}}) \mid \text{Attributed}(a, A)\}|}{|\text{Actors}(G_{\text{gold}})|}$$
Where $\text{Attributed}(a, A)$ checks direct substring containment, domain synonym match (e.g., "bendahara" $\leftrightarrow$ "keuangan panitia", "author" $\leftrightarrow$ "pemakalah"), or $\ge 50\%$ role token overlap.

#### 4. Output Range
- Bounded real value in $[0.0, 1.0]$. Higher is better ($\uparrow$).

---

## 3. PROVENANCE, DATA INTEGRITY & CODE AUDIT PROTOCOL

1. **RAGAS Claim vs. Reality**: The manuscript claims `ragas==0.4.3` was used. Code audit confirms `ragas` is not installed; historical runs used the deterministic lexical overlap and Sentence-BERT formulations specified above.
2. **BERTScore Claim vs. Reality**: The manuscript claims official BERTScore (Zhang et al., 2020). Code audit confirms sentence-level max-pooling using Sentence-BERT MiniLM. This specification formalizes this metric under its correct descriptive name: **SA-F1 (Semantic Alignment F1)**.
3. **Missing Value Rule**: Missing reference or context must produce an explicit `None` or structured error, never a default `1.0`.
