# Clinical Decision Support RAG System

A modular, evidence-based Clinical Retrieval-Augmented Generation (RAG) system built with **LangChain**, **FastEmbed**, **ChromaDB**, and **Groq LLM** (`llama-3.3-70b-versatile`).

This system parses clinical screening guidelines (such as USPSTF Lung Cancer and Colorectal Cancer screening recommendations), indexes text chunks into a vector database, and generates strictly cited answers adhering to clinical safety rules.

---

## 📁 Architecture Overview

```
clinical_rag_project/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── lung-cancer-screening-final-recommendation.pdf
│   └── colorectal-cancer-screening-final-recommendation-updated.pdf
└── src/
    ├── __init__.py
    ├── config.py      # Document metadata, chunking/retrieval/LLM settings
    ├── loader.py      # PDF loading + metadata, fails loudly on partial loads
    ├── chunking.py     # RecursiveCharacterTextSplitter + stable chunk_id
    ├── indexing.py     # Chroma vectorstore (fresh each run) + retrieval
    ├── llm.py          # Groq-backed ChatOpenAI/ChatGroq client
    ├── rag.py          # format_docs, prompt template, ask_clinical_rag
    └── main.py         # CLI entry point
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

---

## 🛠️ Module Reference

- [`src/config.py`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/src/config.py): Contains canonical document mappings (`USPSTF-LUNG-2021`, `USPSTF-CRC-2021`), chunk size/overlap settings, and LLM hyperparameters.
- [`src/loader.py`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/src/loader.py): Scans `data/` for PDFs, assigns canonical document IDs, title, and page numbers, and validates page counts.
- [`src/chunking.py`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/src/chunking.py): Splits documents using `RecursiveCharacterTextSplitter` and tags each chunk with a unique `chunk_id`.
- [`src/indexing.py`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/src/indexing.py): Builds an in-memory Chroma vector database with cosine distance search.
- [`src/llm.py`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/src/llm.py): Initializes the Groq LLM client (`llama-3.3-70b-versatile`).
- [`src/rag.py`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/src/rag.py): Defines the clinical system prompt, document citation formatter, and RAG execution chain.
- [`src/main.py`](file:///c:/Users/aa683/Desktop/Ai%20Hackathon/Day%201/clinical_rag_project/src/main.py): Command-line entry point.

---

## ⚖️ Disclaimer

For educational and clinical decision-support use only. This system does not replace the judgment of a qualified healthcare professional.
