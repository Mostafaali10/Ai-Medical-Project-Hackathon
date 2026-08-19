# Clinical RAG FastAPI Backend

Production-style REST API wrapper around the evidence-grounded Clinical Retrieval-Augmented Generation (RAG) system for Non-Small Cell Lung Cancer guidelines.

---

## 🏛️ Architecture Overview

The backend uses a singleton lifespan initialization model:

```text
FastAPI Startup (Lifespan)
    ↓
Initialize ClinicalRAGPipeline once (app.state.pipeline)
    ↓
Persistent ChromaDB Vectorstore Loaded (721 Chunks)
    ↓
REST Endpoints (POST /api/ask, GET /health, GET /api/documents)
    ↓
Pydantic Request & Response Validation
```

All RAG logic (safety guardrails, lexical filtering, similarity retrieval, Groq generation, and citation validation) is executed directly via `src.pipeline.ClinicalRAGPipeline`.

---

## 🚀 Installation & Setup

### 1. Install Dependencies

From the project root:

```bash
pip install -r backend/requirements.txt
```

### 2. Environment Configuration

Ensure `.env` exists in the project root with your Groq API credentials:

```env
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b
```

---

## 💻 Running the Server

Run Uvicorn from the **project root**:

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 📚 API Documentation & Endpoints

Interactive Swagger UI documentation is available at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Summary of Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Real-time system health, LLM status, and indexed chunk count |
| `POST` | `/api/ask` | Submit clinical queries to the grounded RAG pipeline |
| `GET` | `/api/documents` | List indexed clinical guidelines and document metadata |

---

## 📋 Endpoint Examples

### 1. Submit Clinical Question (`POST /api/ask`)

#### Request Body
```json
{
  "question": "What are the recommendations for lung cancer screening?",
  "k": 5
}
```

#### Response (HTTP 200)
```json
{
  "question": "What are the recommendations for lung cancer screening?",
  "answer": {
    "status": "answered",
    "recommendation": "Screen high-risk individuals with low-dose helical CT scanning; chest radiography and sputum cytology are not recommended because they have not demonstrated mortality benefit.",
    "supporting_evidence": [
      {
        "claim": "Low-dose helical CT scanning is the only screening modality for early detection that has been shown to alter mortality in high-risk patients.",
        "citations": [
          "[NCI Non-Small Cell Lung Cancer Treatment (PDQ) | Page 4 | NCI-NSCLC-PDQ-P4-CH0010]"
        ]
      }
    ],
    "citations": [
      {
        "document": "NCI Non-Small Cell Lung Cancer Treatment (PDQ)",
        "section": "Clinical Presentation",
        "page": 4,
        "chunk_id": "NCI-NSCLC-PDQ-P4-CH0010"
      }
    ],
    "confidence": "High",
    "missing_information": [],
    "safety_note": "Educational information only; not a diagnosis or medical advice."
  },
  "schema_valid": true,
  "citation_report": {
    "citations_used": [
      "[NCI Non-Small Cell Lung Cancer Treatment (PDQ) | Page 4 | NCI-NSCLC-PDQ-P4-CH0010]"
    ],
    "invented_citations": [],
    "claims_missing_citation": [],
    "valid": true
  },
  "chunks_used": [
    {
      "rank": 1,
      "document_id": "NCI-NSCLC-PDQ",
      "document_name": "NCI Non-Small Cell Lung Cancer Treatment (PDQ)",
      "section": "Clinical Presentation",
      "page_number": 4,
      "chunk_id": "NCI-NSCLC-PDQ-P4-CH0010",
      "similarity_score": 0.7699,
      "text": "..."
    }
  ]
}
```

### 2. Health Check (`GET /health`)

#### Response (HTTP 200)
```json
{
  "status": "healthy",
  "pipeline": "ready",
  "vectorstore": "ready",
  "llm": "live (openai/gpt-oss-120b)",
  "collection_name": "nci_lung_cancer_rag",
  "indexed_chunks": 721
}
```

### 3. List Guidelines (`GET /api/documents`)

#### Response (HTTP 200)
```json
{
  "documents": [
    {
      "document_id": "NCI-NSCLC-PDQ",
      "document_name": "NCI Non-Small Cell Lung Cancer Treatment (PDQ)",
      "source": "National Cancer Institute",
      "document_type": "Health Professional PDQ",
      "pages": 203,
      "chunks": 721,
      "source_file": "Non-Small Cell Lung Cancer Treatment (PDQ®) - NCI.pdf"
    }
  ],
  "collection_name": "nci_lung_cancer_rag",
  "total_chunks": 721
}
```

---

## 🔒 CORS Configuration

Configured for development with React and Vite frontend clients:
- `http://localhost:5173`
- `http://127.0.0.1:5173`
- `http://localhost:3000`
- `http://127.0.0.1:3000`
