# Clinical Decision Support RAG System

A modular, evidence-based Clinical Retrieval-Augmented Generation (RAG) system built with **LangChain**, **FastEmbed**, **ChromaDB**, and **Groq LLM** (`llama-3.3-70b-versatile`).

This system parses clinical screening guidelines (such as USPSTF Lung Cancer and Colorectal Cancer screening recommendations), indexes text chunks into a vector database, and generates strictly cited answers adhering to clinical safety rules.

---

## 📁 Architecture Overview

```
clinical_rag_project/
├── README.md
├── requirements.txt
├── .env
├── .env.example
├── data/
│   ├── lung-cancer-screening-final-recommendation.pdf
│   ├── colorectal-cancer-screening-final-recommendation-updated.pdf
│   └── evaluation_set.json     # 16 labeled clinical benchmark questions
├── chunking/
│   ├── README.md               # Detailed chunking benchmark documentation
│   ├── chunk_experiment.py     # Automated 5-way chunking benchmark runner
│   ├── chunk_comparison.json   # Machine-readable benchmark results & metrics
│   ├── chunk_comparison.csv    # Per-query precision & MRR comparison table
│   └── failure_cases.md        # Qualitative failure analysis & error breakdown
├── notebooks/
│   └── AIHACK1.ipynb           # Exploratory development notebook
└── src/
    ├── __init__.py
    ├── config.py               # Canonical settings, metadata mappings & chunk configurations
    ├── loader.py               # PDF loading & validation with metadata preservation
    ├── chunking.py             # Recursive text splitter with deterministic chunk IDs
    ├── indexing.py             # Chroma vectorstore creation & similarity search
    ├── llm.py                  # Groq client initialization
    ├── rag.py                  # RAG prompt template, citation formatting & chain
    ├── evaluation.py           # Precision@K, Hit Rate, MRR calculation engine
    └── main.py                 # CLI entry point (interactive & one-shot mode)
```

---

## 🔑 System Features

1. **Strict Medical Safety Prompting**:
   - Responds strictly using provided guideline context.
   - States *"The provided guideline evidence is insufficient to answer this question."* if information is absent.
   - Appends mandatory clinical decision support disclaimers to all model outputs.

2. **Deterministic Citation & Auditability**:
   - Every chunk is assigned a deterministic ID (e.g., `USPSTF-LUNG-2021-P2-CH0124`).
   - Standardized source citations included in responses: `[Document Name | Page X | Chunk ID]`.

3. **Robust Loading & Fail-Fast Validation**:
   - `src/loader.py` validates page extraction across all target PDFs in `data/` and raises explicit runtime errors if pages fail to parse or return empty results.

4. **Fast Local Embeddings**:
   - Embeds text using `FastEmbedEmbeddings` with `BAAI/bge-small-en-v1.5` cosine distance index.

5. **Automated Retrieval Benchmarking**:
   - Integrated evaluation suite (`src/evaluation.py` & `chunking/chunk_experiment.py`) measuring Precision@K, Hit Rate@K, and Mean Reciprocal Rank (MRR).

---

## 🚀 Quick Start

### 1. Environment Setup

Clone or open the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and add your Groq API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b
```

---

## 💻 Usage

### Running via CLI

#### Interactive Mode
Run the system interactively:

```bash
python -m src.main
```

#### One-Shot Query Mode
Evaluate a single question directly:

```bash
python -m src.main --query "What are the specific age range and smoking history criteria for lung cancer screening?"
```

### Running Chunking Benchmark Experiment

Execute the automated comparative benchmark across all 5 chunk configurations (A, B, C, D, E):

```bash
python -m chunking.chunk_experiment
```

---

## 🛠️ Module Reference

- [`src/config.py`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/src/config.py): Canonical document metadata mappings, 5-way chunk configurations (A through E), and LLM hyperparameters.
- [`src/loader.py`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/src/loader.py): Scans `data/` for PDFs, assigns canonical document IDs, title, and page numbers, and validates page extraction.
- [`src/chunking.py`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/src/chunking.py): Splits documents using `RecursiveCharacterTextSplitter` and tags each chunk with a unique `chunk_id`.
- [`src/indexing.py`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/src/indexing.py): Builds isolated in-memory Chroma vector databases with cosine distance search.
- [`src/llm.py`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/src/llm.py): Initializes the Groq LLM client.
- [`src/rag.py`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/src/rag.py): Defines the clinical system prompt, document citation formatter, and RAG execution chain.
- [`src/evaluation.py`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/src/evaluation.py): Evaluates retriever precision, hit rate, and MRR against labeled ground-truth questions.
- [`chunking/chunk_experiment.py`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/chunking/chunk_experiment.py): Automated 5-way benchmark comparing Configurations A, B, C, D, and E.
- [`src/main.py`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/src/main.py): Command-line entry point.

---

## 🔬 Chunking Configuration Experiment

To empirically determine the optimal chunking strategy for clinical decision support retrieval, we evaluated five distinct chunk configurations against an annotated clinical benchmark (`data/evaluation_set.json`) containing 13 in-scope USPSTF guideline queries and 3 out-of-scope negative controls.

### Evaluated Configurations

| Config | Name | Chunk Size | Overlap | Total Chunks | Avg Length |
|:---|:---|---:|---:|---:|---:|
| **Config A** | Baseline | 850 | 150 | 205 | 784.3 chars |
| **Config B** | Small Granular | 450 | 50 | 350 | 408.2 chars |
| **Config C** | Medium-Low | 600 | 100 | 282 | 536.8 chars |
| **Config D** | Medium-High | 700 | 100 | 238 | 632.7 chars |
| **Config E** | Large Context | 1000 | 150 | 170 | 908.6 chars |

### Experimental Benchmark Results (13 In-Scope Queries)

| Configuration | Chunk Size | Overlap | Precision@3 | Precision@5 | Hit Rate@3 | Hit Rate@5 | MRR (Primary) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **Config A (Winner)** | **850** | **150** | **58.97%** | **50.77%** | **84.62%** | **92.31%** | **0.8269** |
| **Config B** | 450 | 50 | 38.46% | 30.77% | 61.54% | 76.92% | 0.6115 |
| **Config C** | 600 | 100 | 48.72% | 40.00% | 84.62% | 84.62% | 0.6667 |
| **Config D** | 700 | 100 | 56.41% | 46.15% | 92.31% | 92.31% | 0.6795 |
| **Config E** | 1000 | 150 | 38.46% | 33.85% | 69.23% | 76.92% | 0.5833 |

### Benchmark Winner & Analysis

**Overall Winner: Configuration A (`chunk_size = 850`, `chunk_overlap = 150`)**

- **Highest MRR (0.8269):** Places the most relevant evidence directly at Rank 1 in the majority of clinical queries.
- **Highest Precision@3 (58.97%) & Precision@5 (50.77%):** Minimizes irrelevant context passed to the LLM prompt.
- **Why it won:** Clinical screening recommendations contain multi-clause criteria (age ranges, cumulative pack-years, cessation intervals, screening intervals). Small chunks (`450/50` and `600/100`) fragment these qualifying conditions across chunk boundaries, whereas oversized chunks (`1000/150`) dilute the semantic embedding with unrelated adjacent recommendations. Configuration A provides the optimal semantic density for USPSTF guideline retrieval.

Detailed per-query comparisons and failure analysis are saved in [`evaluation/failure_cases.md`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/evaluation/failure_cases.md), [`evaluation/chunk_comparison.json`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/evaluation/chunk_comparison.json), and [`evaluation/chunk_comparison.csv`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/evaluation/chunk_comparison.csv).

---

## ⚖️ Disclaimer

For educational and clinical decision-support use only. This system does not replace the judgment of a qualified healthcare professional.
