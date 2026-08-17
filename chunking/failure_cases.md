# 🔬 Clinical RAG 5-Way Chunking Benchmark & Failure Analysis

## 1. Executive Summary & Aggregate Benchmark Results

All metrics evaluated across 13 in-scope clinical guideline queries with stable ground-truth criteria:

| Configuration | Chunk Size | Overlap | Total Chunks | Precision@3 | Precision@5 | Hit Rate@3 | Hit Rate@5 | MRR (Primary) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Config A** | 850 | 150 | 205 | **58.97%** | **50.77%** | **84.62%** | **92.31%** | **0.8269** |
| **Config B** | 450 | 50 | 350 | 38.46% | 30.77% | 61.54% | 76.92% | 0.6115 |
| **Config C** | 600 | 100 | 282 | 48.72% | 40.00% | 84.62% | 84.62% | 0.6667 |
| **Config D** | 700 | 100 | 238 | 56.41% | 46.15% | 92.31% | 92.31% | 0.6795 |
| **Config E** | 1000 | 150 | 170 | 38.46% | 33.85% | 69.23% | 76.92% | 0.5833 |

### Overall Winner: **Config A (850/150)**

- **Primary Metric (MRR):** `0.8269`
- **Precision@3:** `58.97%`
- **Hit Rate@3:** `84.62%`

## 2. In-Scope Win Breakdown

- **Config A:** 3 questions
- **Config B:** 1 questions
- **Config C:** 0 questions
- **Config D:** 2 questions
- **Config E:** 1 questions
- **Ties:** 6 questions

## 3. Key Findings & Chunking Sensitivity Analysis

1. **Context Fragmentation in Small Chunks (Config B - 450/50):**
   - Produces 350 chunks. Shorter chunks sever multi-clause eligibility criteria (e.g. splitting age brackets from smoking pack-years).
   - Lowest MRR (0.6115) and lowest Precision@3 (38.46%).

2. **Sweet Spot in Moderate Chunk Ranges (Config A - 850/150 and Config D - 700/100):**
   - Config A (850/150) preserves complete clinical thoughts and table footnotes within single chunks, maximizing retrieval accuracy.
   - Config D (700/100) provides a strong alternative with high precision and slightly lower chunk count.

3. **Context Dilution in Extra-Large Chunks (Config E - 1000/150):**
   - At 1000 characters (181 chunks), each chunk contains multiple unrelated clinical recommendations, leading to semantic vector dilution.

## 4. Per-Question Results Across All 5 Configurations

### Question Q01: What are the specific age range and smoking history criteria for lung cancer screening?

- **Category:** `lung_cancer_eligibility`
- **Out of Scope:** `False`
- **Winner:** **TIE (A, B)**

| Config | Chunk Size/Overlap | Precision@3 | Precision@5 | Hit Rate@3 | MRR |
|:---|---:|---:|---:|---:|---:|
| Config A | 850/150 | 1.00 | 0.80 | 1.0 | 1.00 |
| Config B | 450/50 | 1.00 | 0.60 | 1.0 | 1.00 |
| Config C | 600/100 | 0.33 | 0.60 | 1.0 | 0.50 |
| Config D | 700/100 | 0.67 | 0.80 | 1.0 | 0.50 |
| Config E | 1000/150 | 0.67 | 0.60 | 1.0 | 1.00 |

---

### Question Q02: When should lung cancer screening with LDCT be discontinued?

- **Category:** `lung_cancer_discontinuation`
- **Out of Scope:** `False`
- **Winner:** **TIE (A, D, E)**

| Config | Chunk Size/Overlap | Precision@3 | Precision@5 | Hit Rate@3 | MRR |
|:---|---:|---:|---:|---:|---:|
| Config A | 850/150 | 0.67 | 0.40 | 1.0 | 1.00 |
| Config B | 450/50 | 0.33 | 0.20 | 1.0 | 1.00 |
| Config C | 600/100 | 0.33 | 0.20 | 1.0 | 1.00 |
| Config D | 700/100 | 0.67 | 0.40 | 1.0 | 1.00 |
| Config E | 1000/150 | 0.67 | 0.40 | 1.0 | 1.00 |

---

### Question Q03: What screening modality is recommended for lung cancer screening, and which tests are explicitly not recommended?

- **Category:** `lung_cancer_modality`
- **Out of Scope:** `False`
- **Winner:** **Config A**

| Config | Chunk Size/Overlap | Precision@3 | Precision@5 | Hit Rate@3 | MRR |
|:---|---:|---:|---:|---:|---:|
| Config A | 850/150 | 0.33 | 0.40 | 1.0 | 1.00 |
| Config B | 450/50 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config C | 600/100 | 0.33 | 0.20 | 1.0 | 0.50 |
| Config D | 700/100 | 0.33 | 0.40 | 1.0 | 0.33 |
| Config E | 1000/150 | 0.33 | 0.20 | 1.0 | 0.50 |

---

### Question Q04: How frequently should lung cancer screening be performed?

- **Category:** `lung_cancer_frequency`
- **Out of Scope:** `False`
- **Winner:** **Config A**

| Config | Chunk Size/Overlap | Precision@3 | Precision@5 | Hit Rate@3 | MRR |
|:---|---:|---:|---:|---:|---:|
| Config A | 850/150 | 1.00 | 0.80 | 1.0 | 1.00 |
| Config B | 450/50 | 0.33 | 0.20 | 1.0 | 0.50 |
| Config C | 600/100 | 0.33 | 0.20 | 1.0 | 1.00 |
| Config D | 700/100 | 0.33 | 0.40 | 1.0 | 0.50 |
| Config E | 1000/150 | 0.00 | 0.00 | 0.0 | 0.00 |

---

### Question Q05: What randomized clinical trials (RCTs) provided the primary evidence for the mortality benefit of LDCT screening?

- **Category:** `lung_cancer_trials`
- **Out of Scope:** `False`
- **Winner:** **Config E**

| Config | Chunk Size/Overlap | Precision@3 | Precision@5 | Hit Rate@3 | MRR |
|:---|---:|---:|---:|---:|---:|
| Config A | 850/150 | 0.00 | 0.40 | 0.0 | 0.25 |
| Config B | 450/50 | 0.00 | 0.20 | 0.0 | 0.25 |
| Config C | 600/100 | 0.33 | 0.40 | 1.0 | 0.33 |
| Config D | 700/100 | 0.33 | 0.20 | 1.0 | 0.50 |
| Config E | 1000/150 | 0.67 | 0.60 | 1.0 | 0.50 |

---

### Question Q06: What are the potential harms associated with LDCT screening for lung cancer?

- **Category:** `lung_cancer_harms`
- **Out of Scope:** `False`
- **Winner:** **TIE (C, D)**

| Config | Chunk Size/Overlap | Precision@3 | Precision@5 | Hit Rate@3 | MRR |
|:---|---:|---:|---:|---:|---:|
| Config A | 850/150 | 0.67 | 0.40 | 1.0 | 1.00 |
| Config B | 450/50 | 0.67 | 0.40 | 1.0 | 1.00 |
| Config C | 600/100 | 1.00 | 0.60 | 1.0 | 1.00 |
| Config D | 700/100 | 1.00 | 0.60 | 1.0 | 1.00 |
| Config E | 1000/150 | 0.33 | 0.40 | 1.0 | 1.00 |

---

### Question Q07: What is the recommended starting age for colorectal cancer screening in average-risk adults?

- **Category:** `colorectal_starting_age`
- **Out of Scope:** `False`
- **Winner:** **Config D**

| Config | Chunk Size/Overlap | Precision@3 | Precision@5 | Hit Rate@3 | MRR |
|:---|---:|---:|---:|---:|---:|
| Config A | 850/150 | 0.67 | 0.60 | 1.0 | 1.00 |
| Config B | 450/50 | 0.33 | 0.20 | 1.0 | 1.00 |
| Config C | 600/100 | 0.67 | 0.60 | 1.0 | 1.00 |
| Config D | 700/100 | 1.00 | 0.80 | 1.0 | 1.00 |
| Config E | 1000/150 | 0.67 | 0.40 | 1.0 | 1.00 |

---

### Question Q08: What are the recommendation grades for colorectal cancer screening across different age groups (45-49, 50-75, 76-85)?

- **Category:** `colorectal_grades`
- **Out of Scope:** `False`
- **Winner:** **Config D**

| Config | Chunk Size/Overlap | Precision@3 | Precision@5 | Hit Rate@3 | MRR |
|:---|---:|---:|---:|---:|---:|
| Config A | 850/150 | 0.67 | 0.60 | 1.0 | 1.00 |
| Config B | 450/50 | 0.33 | 0.60 | 1.0 | 1.00 |
| Config C | 600/100 | 0.67 | 0.80 | 1.0 | 1.00 |
| Config D | 700/100 | 1.00 | 0.80 | 1.0 | 1.00 |
| Config E | 1000/150 | 0.67 | 0.60 | 1.0 | 1.00 |

---

### Question Q09: What are the recommended direct visualization screening tests and their respective intervals for colorectal cancer?

- **Category:** `colorectal_modalities`
- **Out of Scope:** `False`
- **Winner:** **Config A**

| Config | Chunk Size/Overlap | Precision@3 | Precision@5 | Hit Rate@3 | MRR |
|:---|---:|---:|---:|---:|---:|
| Config A | 850/150 | 0.33 | 0.20 | 1.0 | 1.00 |
| Config B | 450/50 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config C | 600/100 | 0.33 | 0.20 | 1.0 | 0.33 |
| Config D | 700/100 | 0.33 | 0.20 | 1.0 | 0.50 |
| Config E | 1000/150 | 0.00 | 0.00 | 0.0 | 0.00 |

---

### Question Q10: What are the recommended stool-based screening tests and their testing intervals?

- **Category:** `colorectal_stool_tests`
- **Out of Scope:** `False`
- **Winner:** **TIE (A, B, C)**

| Config | Chunk Size/Overlap | Precision@3 | Precision@5 | Hit Rate@3 | MRR |
|:---|---:|---:|---:|---:|---:|
| Config A | 850/150 | 1.00 | 0.60 | 1.0 | 1.00 |
| Config B | 450/50 | 1.00 | 0.60 | 1.0 | 1.00 |
| Config C | 600/100 | 1.00 | 0.60 | 1.0 | 1.00 |
| Config D | 700/100 | 0.67 | 0.40 | 1.0 | 1.00 |
| Config E | 1000/150 | 0.67 | 0.60 | 1.0 | 1.00 |

---

### Question Q11: Why does the USPSTF not recommend serum tests, urine tests, or capsule endoscopy for colorectal cancer screening?

- **Category:** `colorectal_unrecommended`
- **Out of Scope:** `False`
- **Winner:** **Config B**

| Config | Chunk Size/Overlap | Precision@3 | Precision@5 | Hit Rate@3 | MRR |
|:---|---:|---:|---:|---:|---:|
| Config A | 850/150 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config B | 450/50 | 0.00 | 0.20 | 0.0 | 0.20 |
| Config C | 600/100 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config D | 700/100 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config E | 1000/150 | 0.00 | 0.00 | 0.0 | 0.00 |

---

### Question Q12: What are the serious harms and complications associated with screening colonoscopy?

- **Category:** `colorectal_harms`
- **Out of Scope:** `False`
- **Winner:** **TIE (A, B, C)**

| Config | Chunk Size/Overlap | Precision@3 | Precision@5 | Hit Rate@3 | MRR |
|:---|---:|---:|---:|---:|---:|
| Config A | 850/150 | 1.00 | 1.00 | 1.0 | 1.00 |
| Config B | 450/50 | 1.00 | 0.80 | 1.0 | 1.00 |
| Config C | 600/100 | 1.00 | 0.80 | 1.0 | 1.00 |
| Config D | 700/100 | 0.67 | 0.80 | 1.0 | 1.00 |
| Config E | 1000/150 | 0.33 | 0.40 | 1.0 | 0.33 |

---

### Question Q13: What is the first-line chemotherapy regimen for stage IV metastatic melanoma?

- **Category:** `out_of_scope`
- **Out of Scope:** `True`
- **Winner:** **N/A (Out of Scope Refusal)**

| Config | Chunk Size/Overlap | Precision@3 | Precision@5 | Hit Rate@3 | MRR |
|:---|---:|---:|---:|---:|---:|
| Config A | 850/150 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config B | 450/50 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config C | 600/100 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config D | 700/100 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config E | 1000/150 | 0.00 | 0.00 | 0.0 | 0.00 |

---

### Question Q14: What are the diagnostic criteria and treatment guidelines for pediatric asthma exacerbation?

- **Category:** `out_of_scope`
- **Out of Scope:** `True`
- **Winner:** **N/A (Out of Scope Refusal)**

| Config | Chunk Size/Overlap | Precision@3 | Precision@5 | Hit Rate@3 | MRR |
|:---|---:|---:|---:|---:|---:|
| Config A | 850/150 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config B | 450/50 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config C | 600/100 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config D | 700/100 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config E | 1000/150 | 0.00 | 0.00 | 0.0 | 0.00 |

---

### Question Q15: Can you diagnose my patient with a 7 mm ground-glass lung nodule and prescribe the appropriate antibiotic?

- **Category:** `out_of_scope_refusal`
- **Out of Scope:** `True`
- **Winner:** **N/A (Out of Scope Refusal)**

| Config | Chunk Size/Overlap | Precision@3 | Precision@5 | Hit Rate@3 | MRR |
|:---|---:|---:|---:|---:|---:|
| Config A | 850/150 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config B | 450/50 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config C | 600/100 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config D | 700/100 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config E | 1000/150 | 0.00 | 0.00 | 0.0 | 0.00 |

---

### Question Q16: What is the recommended screening strategy for individuals with prior personal history of colorectal cancer or Lynch syndrome?

- **Category:** `colorectal_high_risk`
- **Out of Scope:** `False`
- **Winner:** **TIE (A, D)**

| Config | Chunk Size/Overlap | Precision@3 | Precision@5 | Hit Rate@3 | MRR |
|:---|---:|---:|---:|---:|---:|
| Config A | 850/150 | 0.33 | 0.40 | 1.0 | 0.50 |
| Config B | 450/50 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config C | 600/100 | 0.00 | 0.00 | 0.0 | 0.00 |
| Config D | 700/100 | 0.33 | 0.20 | 1.0 | 0.50 |
| Config E | 1000/150 | 0.00 | 0.20 | 0.0 | 0.25 |

---

