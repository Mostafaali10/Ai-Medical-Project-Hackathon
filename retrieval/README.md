# 🔍 Retrieval Evaluation: "Correct Evidence in Top-K"

## 1. Purpose

This module evaluates the performance of the clinical retrieval pipeline in retrieving **correct, guideline-verified evidence** for clinical screening questions across varying cutoff thresholds: **Top-1, Top-3, and Top-5**.

The objective is to establish whether the top-ranked retrieved context contains sufficient evidence for the LLM to generate an accurate, safe clinical answer with proper citations.

---

## 2. Architecture & Pipeline Flow

```
evaluation/evaluation_set.py (Canonical 16-question Benchmark)
        ↓
retrieval/retrieval_experiment.py (Runner)
        ↓
Production Pipeline (src/loader.py → src/chunking.py → src/indexing.py)
        ↓
Top-K Chunks with Cosine Similarity Scores (K = 1, 3, 5)
        ↓
evaluation/metrics.py (Relevance & IR Metric Engine)
        ↓
retrieval/ Artifacts (JSON, CSV, Failure Analysis)
```

---

## 3. Experimental Setup

- **Benchmark Dataset:** [`evaluation/evaluation_set.py`](../evaluation/evaluation_set.py) (16 total: 13 in-scope clinical queries + 3 out-of-scope negative controls).
- **Chunk Configuration:** **Configuration A (Optimal)** — `chunk_size = 850`, `chunk_overlap = 150` (identified as winner by the chunking experiment).
- **Total Indexed Chunks:** 205 chunks across 22 pages of USPSTF guidelines.
- **Embedding Model:** `BAAI/bge-small-en-v1.5` (Cosine similarity distance).
- **Evaluated Cutoffs (Top-K):** $K \in \{1, 3, 5\}$.

---

## 4. Definition of "Correct Evidence in Top-K"

For each query, retrieved chunks are verified against ground-truth guideline annotations:

A chunk is marked as **Correct Evidence** ($\text{is\_correct} = \text{True}$) if and only if:
1. **In-Scope Query:** The question targets a covered USPSTF guideline (`expected_doc` is not `None`).
2. **Document Match:** The chunk's `document_id` matches the expected guideline document ID.
3. **Keyword Verification:** The chunk contains at least one essential ground-truth phrase or medical concept.

### Distinction: Correct Evidence Rate vs. Precision
- **Correct Evidence Rate @ K (Hit Rate @ K):** A binary indicator of whether **at least one** correct evidence chunk exists within the Top-K results. Measures pipeline recall and whether the LLM will have the needed evidence.
- **Precision @ K:** The fraction of chunks in the Top-K that contain valid evidence ($\frac{\text{Relevant Chunks in Top-K}}{K}$). Measures signal-to-noise ratio in the context window.

---

## 5. Benchmark Results

Measured across the **13 in-scope clinical guideline questions**:

| Metric | Top-1 ($K=1$) | Top-3 ($K=3$) | Top-5 ($K=5$) |
|:---|:---:|:---:|:---:|
| **Correct Evidence Rate (Hit Rate)** | **76.92%** (10/13) | **84.62%** (11/13) | **92.31%** (12/13) |
| **Precision** | **76.92%** | **58.97%** | **50.77%** |
| **Mean Reciprocal Rank (MRR)** | — | — | **0.8269** |

### Negative Controls (Safety Refusal Evaluation)
- **Queries Evaluated:** `Q13` (Metastatic Melanoma), `Q14` (Pediatric Asthma), `Q15` (Unguided Nodule Rx).
- **False Positive Evidence Rate:** **0.00%** (0/3 marked as evidence; correctly flagged as out-of-scope for refusal).

---

## 6. Key Findings & Root Cause Analysis

1. **High Top-1 Accuracy (76.92% Hit@1, MRR 0.8269):**
   - For 10 out of 13 in-scope queries, the exact clinical recommendation chunk was ranked at **Rank 1**.
2. **Expansion to Top-5 Captures 92.31% of Guidelines:**
   - At $K=5$, 12 out of 13 queries successfully surfaced the necessary evidence chunk.
3. **Edge Cases:**
   - `Q16` (Lynch syndrome exclusion criteria) is placed at **Rank 2**.
   - `Q05` (NLST & NELSON trials) is placed at **Rank 4**.
   - `Q11` (Non-recommended CRC screening modalities) is in dense table footnotes across pages 2 and 12, ranking outside Top-5 under dense semantic embedding alone.

---

## 7. Artifacts & Outputs

- [`retrieval_results.json`](retrieval_results.json) — Full machine-readable evaluation data, chunk metadata, similarity scores, and aggregate metrics.
- [`retrieval_results.csv`](retrieval_results.csv) — Per-question table with Rank 1 scores, Top-K hit indicators, and precisions.
- [`failure_cases.md`](failure_cases.md) — Qualitative root-cause analysis for queries where evidence was not retrieved at Rank 1, 3, or 5.

---

## 8. Reproducibility

To re-run the retrieval evaluation experiment:

```bash
python -m retrieval.retrieval_experiment
```
