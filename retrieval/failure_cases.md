# 🔍 Retrieval Evaluation: Failure Cases & Root Cause Analysis

This document analyzes benchmark queries where correct clinical evidence was **not** retrieved in Top-1, Top-3, or Top-5.

## 1. Summary of Retrieval Performance

- **Total In-Scope Queries:** 13
- **Correct Evidence in Top-1 (Hit@1):** `76.92%` (10/13)
- **Correct Evidence in Top-3 (Hit@3):** `84.62%` (11/13)
- **Correct Evidence in Top-5 (Hit@5):** `92.31%` (12/13)
- **Mean Reciprocal Rank (MRR):** `0.8269`

## 2. Failure Cases Analysis

### Top-5 Retrieval Failures (Evidence Not in Top-5)

#### Question Q11: Why does the USPSTF not recommend serum tests, urine tests, or capsule endoscopy for colorectal cancer screening?

- **Category:** `colorectal_unrecommended`
- **Expected Document:** `USPSTF-CRC-2021`
- **First Correct Evidence Rank:** `Not in Top 5`

**Retrieved Chunks in Top-5:**

1. **[Rank 1]** `USPSTF-CRC-2021-P4-CH0026` (Score: 0.7827, Doc: USPSTF-CRC-2021)
   > *(ie, no prior diagnosis of colorectal cancer, adenomatous polyps, or inflammatory bowel disease; no personal diagnosis or family history of ...*

1. **[Rank 2]** `USPSTF-CRC-2021-P7-CH0050` (Score: 0.7776, Doc: USPSTF-CRC-2021)
   > *https://www.cdc.gov/cancer/colorectal/basic_info/screening/ tests.htm The Community Preventive Services Task Force has also is- suedrecommen...*

1. **[Rank 3]** `USPSTF-CRC-2021-P2-CH0008` (Score: 0.7771, Doc: USPSTF-CRC-2021)
   > *colonography,andflexiblesigmoidoscopy.See Table 1forcharac- teristicsofrecommendedscreeningstrategies.TheUSPSTFrecom- mendation for screenin...*

1. **[Rank 4]** `USPSTF-CRC-2021-P4-CH0025` (Score: 0.77, Doc: USPSTF-CRC-2021)
   > *Figure 1. Clinician Summary: Screening for Colorectal Cancer What does the USPSTF recommend? For adults aged 50 to 75 years: Screen all adul...*

1. **[Rank 5]** `USPSTF-CRC-2021-P5-CH0038` (Score: 0.7673, Doc: USPSTF-CRC-2021)
   > *the harms of screening for colorectal cancer in adults 76 y and older are small to moderate. The majority of harms result from the use of co...*

**Root Cause Analysis:**
- *Colorectal Non-Recommended Modalities:* The rationale for avoiding serum tests, urine tests, and capsule endoscopy is located in the text and table footnotes across pages 2 and 12 of the USPSTF guideline. Dense table discussions receive slightly lower cosine similarity scores than general colorectal screening overview paragraphs.

---

### Top-3 Sub-Optimal Retrievals (Evidence Found at Rank 4–5)

#### Question Q05: What randomized clinical trials (RCTs) provided the primary evidence for the mortality benefit of LDCT screening?

- **Category:** `lung_cancer_trials`
- **First Correct Evidence Rank:** **Rank 4** (Score: 0.7539)
- **Top-1 Chunk Retrieved:** `USPSTF-LUNG-2021-P3-CH0136` (Score: 0.7909)
  > *and incidental findings that can lead to subsequent testing and treatment, including the anxiety of living with a lung lesion that may be ca...*

- **Correct Chunk Retrieved at Rank 4:** `USPSTF-LUNG-2021-P5-CH0148`
  > *the NLST would have increased specificity while decreasing sensitivity.21 The other 2 studies found that use of I-ELCAP criteria (increase i...*

**Root Cause:** Top ranks were occupied by general guideline overview sections rather than the specific trial/evidence paragraph.

---

### Top-1 Sub-Optimal Retrievals (Evidence Found at Rank 2–3)

#### Question Q16: What is the recommended screening strategy for individuals with prior personal history of colorectal cancer or Lynch syndrome?

- **Category:** `colorectal_high_risk`
- **First Correct Evidence Rank:** **Rank 2** (Score: 0.8002)
- **Top-1 Chunk Retrieved:** `USPSTF-CRC-2021-P4-CH0027` (Score: 0.8049)
  > *available at https://www.uspreventiveservicestaskforce.org  Screen all adults aged 45 to 75 years for colorectal cancer. Several recommended...*

- **Correct Evidence Chunk (Rank 2):** `USPSTF-CRC-2021-P4-CH0026`
  > *(ie, no prior diagnosis of colorectal cancer, adenomatous polyps, or inflammatory bowel disease; no personal diagnosis or family history of ...*

---

## 3. Out-of-Scope Negative Control Verification (Q13, Q14, Q15)

| Question ID | Question | Expected Evidence | Evidence Found? | Status |
|:---|:---|:---|:---:|:---:|
| **Q13** | What is the first-line chemotherapy regimen for stage IV metastatic melanoma? | None (Out of Scope) | No | **PASS (Refusal Target)** |
| **Q14** | What are the diagnostic criteria and treatment guidelines for pediatric asthma exacerbation? | None (Out of Scope) | No | **PASS (Refusal Target)** |
| **Q15** | Can you diagnose my patient with a 7 mm ground-glass lung nodule and prescribe the appropriate antibiotic? | None (Out of Scope) | No | **PASS (Refusal Target)** |
