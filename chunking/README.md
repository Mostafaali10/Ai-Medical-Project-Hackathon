# 🔬 Chunking Configuration Experiment

## 1. Task Objective

Empirically evaluate and compare multiple text chunking strategies (varying `chunk_size` and `chunk_overlap`) against an annotated clinical decision support benchmark dataset, and determine the optimal configuration for clinical guideline retrieval.

---

## 2. Configurations Tested

The experiment evaluates five distinct configurations under strictly identical ingestion, embedding, and retrieval conditions:

| Configuration | Description | Chunk Size (chars) | Overlap (chars) | Total Chunks Generated | Average Chunk Length |
|:---|:---|---:|---:|---:|---:|
| **Config A (Winner)** | Baseline | 850 | 150 | 205 | 784.3 chars |
| **Config B** | Small Granular | 450 | 50 | 350 | 408.2 chars |
| **Config C** | Medium-Low | 600 | 100 | 282 | 536.8 chars |
| **Config D** | Medium-High | 700 | 100 | 238 | 632.7 chars |
| **Config E** | Large Context | 1000 | 150 | 170 | 908.6 chars |

---

## 3. Evaluation Metrics

- **Precision@3**: Proportion of relevant chunks among the Top 3 retrieved ($\frac{\text{Relevant in Top 3}}{3}$).
- **Precision@5**: Proportion of relevant chunks among the Top 5 retrieved ($\frac{\text{Relevant in Top 5}}{5}$).
- **Hit Rate@3**: Binary indicator ($1$ or $0$) of whether at least one relevant chunk appears in the Top 3 results.
- **Hit Rate@5**: Binary indicator of whether at least one relevant chunk appears in the Top 5 results.
- **Mean Reciprocal Rank (MRR)**: Primary ranking metric defined as $\frac{1}{\text{rank of first relevant chunk}}$, measuring how high the best evidence is ranked.

---

## 4. Evaluation Dataset

The benchmark dataset ([`data/evaluation_set.json`](../data/evaluation_set.json)) contains **16 labeled questions**:
- **13 In-Scope Clinical Questions**: Covering USPSTF Lung Cancer Screening (2021) and USPSTF Colorectal Cancer Screening (2021) guidelines (eligibility criteria, age thresholds, screening intervals, RCT evidence, colonoscopy risks, and unrecommended modalities).
- **3 Out-of-Scope Safety & Refusal Questions**: Testing model safety behavior (metastatic melanoma chemotherapy, pediatric asthma, and unguided nodule prescription).

> **Evaluation Rule:** Retrieval quality metrics (Precision@K, Hit Rate@K, and MRR) are computed strictly across the **13 in-scope clinical guideline questions** with ground-truth evidence. Out-of-scope questions are evaluated separately to verify safety refusal behavior and avoid biasing retrieval metrics.

---

## 5. Benchmark Results

Measured across the 13 in-scope clinical guideline questions:

| Configuration | Size / Overlap | Precision@3 | Precision@5 | Hit Rate@3 | Hit Rate@5 | MRR (Primary Metric) |
|:---|---:|---:|---:|---:|---:|:---:|
| **Config A (Winner)** | **850 / 150** | **58.97%** | **50.77%** | **84.62%** | **92.31%** | **0.8269** |
| **Config D** | 700 / 100 | 56.41% | 46.15% | 92.31% | 92.31% | **0.6795** |
| **Config C** | 600 / 100 | 48.72% | 40.00% | 84.62% | 84.62% | **0.6667** |
| **Config B** | 450 / 50 | 38.46% | 30.77% | 61.54% | 76.92% | **0.6115** |
| **Config E** | 1000 / 150 | 38.46% | 33.85% | 69.23% | 76.92% | **0.5833** |

---

## 6. Benchmark Winner

### **Winner: Configuration A (`chunk_size = 850`, `chunk_overlap = 150`)**

### Primary Justification:
* **Highest MRR (`0.8269`):** Configuration A consistently places the primary evidence at **Rank 1** for the majority of clinical queries.
* **Highest Precision@3 (`58.97%`) & Precision@5 (`50.77%`):** Maximizes high-density evidence while minimizing irrelevant noise sent to the LLM context prompt.
* **Hit Rate Analysis:** While **Config D** achieved a slightly higher Hit Rate@3 (92.31% vs 84.62%), **Config A remains the decisive overall winner** because its primary ranking power (MRR 0.8269 vs 0.6795) and precision (58.97% vs 56.41%) are significantly superior.

### Clinical & Semantic Rationale:
Clinical screening guidelines feature complex qualifying conditions (e.g. *"annual LDCT in adults aged 50-80 years with $\ge$20 pack-year history who currently smoke or quit within 15 years"*). Shorter chunks (`450/50` and `600/100`) fragment these multi-clause conditions, while oversized chunks (`1000/150`) dilute the semantic vector with adjacent recommendations. Configuration A hits the optimal balance of context preservation and semantic specificity.

---

## 7. Experiment Files

* [`chunk_experiment.py`](chunk_experiment.py) — Automated benchmark runner.
* [`chunk_comparison.json`](chunk_comparison.json) — Full machine-readable results including per-query rankings across all 5 configurations.
* [`chunk_comparison.csv`](chunk_comparison.csv) — Tabular per-query metric comparison.
* [`failure_cases.md`](failure_cases.md) — Qualitative failure analysis and edge-case breakdown.

---

## 8. Reproducibility

To re-run the benchmark experiment and update the comparison artifacts:

```bash
python -m chunking.chunk_experiment
```
