# 📊 Canonical Clinical Evaluation Benchmark & Metrics

## 1. Purpose

This package provides the single canonical evaluation benchmark (`EVAL_SET`) and shared evaluation methodology for all Clinical RAG experiments, including:
- Chunking configuration comparisons (`chunking/`)
- Retrieval strategy benchmarks (`retrieval/`)
- Future end-to-end RAG reasoning and generation experiments

Centralizing the evaluation benchmark and metric definitions guarantees strict reproducibility and fair head-to-head comparisons across experiments.

---

## 2. Benchmark Dataset (`evaluation/evaluation_set.py`)

The benchmark consists of **16 labeled questions**:
- **13 In-Scope Clinical Questions (Q01–Q12, Q16):** Cover USPSTF Lung Cancer Screening (2021) and Colorectal Cancer Screening (2021) guidelines across eligibility criteria, discontinuation conditions, recommended vs. unrecommended modalities, screening intervals, randomized trial evidence (NLST/NELSON), and colonoscopy complications.
- **3 Out-of-Scope Safety & Refusal Questions (Q13–Q15):** Cover out-of-scope clinical conditions (metastatic melanoma, pediatric asthma) and unguided diagnostic/prescribing requests to verify strict safety refusal behavior.

### Ground Truth Schema

Each question dictionary in `EVAL_SET` contains:
```python
{
    "id": "Q01",
    "category": "lung_cancer_eligibility",
    "question": "What are the specific age range and smoking history criteria for lung cancer screening?",
    "expected_doc": "USPSTF-LUNG-2021",
    "expected_pages": [1, 2, 4],
    "keywords": ["50 to 80", "20 pack-year", "15 years"],
    "is_out_of_scope": False,
    "expected_answer": "Adults aged 50 to 80 years who have a 20 pack-year smoking history and currently smoke or have quit within the past 15 years."
}
```

For out-of-scope questions (`Q13`, `Q14`, `Q15`):
- `expected_doc = None`
- `keywords = []`
- `is_out_of_scope = True`

---

## 3. Relevance Definition (`evaluation/metrics.py`)

A retrieved chunk is defined as **relevant** if and only if:
1. The question is **in-scope** (`expected_doc is not None` and `is_out_of_scope is False`).
2. The chunk's `document_id` matches `expected_doc`.
3. The chunk's text content contains **at least one** required ground-truth keyword/phrase from `keywords`.
4. (Optional) The chunk's `page_number` is within `expected_pages` if page constraints are specified.

For out-of-scope queries (`expected_doc is None`), no retrieved guideline chunk is relevant.

---

## 4. Evaluation Metrics

- **Precision@3**: Proportion of relevant chunks among the top 3 retrieved:
  $$\text{Precision@3} = \frac{\sum_{i=1}^3 \mathbb{I}(\text{chunk}_i \text{ is relevant})}{3}$$
- **Precision@5**: Proportion of relevant chunks among the top 5 retrieved:
  $$\text{Precision@5} = \frac{\sum_{i=1}^5 \mathbb{I}(\text{chunk}_i \text{ is relevant})}{5}$$
- **Hit Rate@3**: Binary indicator ($1.0$ or $0.0$) of whether at least one relevant chunk appears in the top 3 results.
- **Hit Rate@5**: Binary indicator of whether at least one relevant chunk appears in the top 5 results.
- **Mean Reciprocal Rank (MRR)**: Primary ranking metric:
  $$\text{MRR} = \frac{1}{\text{rank of first relevant chunk}} \quad (\text{or } 0.0 \text{ if none in top } K)$$

---

## 5. Usage in Experiments

### Importing the Shared Benchmark

```python
from evaluation.evaluation_set import EVAL_SET, get_in_scope_eval_set, get_out_of_scope_eval_set

# Load all 16 benchmark questions
print(f"Total questions: {len(EVAL_SET)}")

# Load only in-scope questions for retrieval metrics
in_scope_queries = get_in_scope_eval_set()
```

### Importing Metric Functions

```python
from evaluation.metrics import (
    evaluate_retriever,
    evaluate_chunk_relevance,
    calculate_precision_at_k,
    calculate_hit_rate_at_k,
    calculate_mrr
)

# Evaluate an indexed Chroma vectorstore
results = evaluate_retriever(vectorstore, eval_dataset=EVAL_SET, k_max=5)
print("Precision@3:", results["overall_in_scope_guideline_queries"]["precision_at_3"])
print("MRR:", results["overall_in_scope_guideline_queries"]["mrr"])
```
