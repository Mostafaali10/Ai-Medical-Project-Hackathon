import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Known Document Canonical Mapping
DOCUMENT_METADATA_MAP = {
    "lung": {
        "document_id": "USPSTF-LUNG-2021",
        "document_name": "USPSTF Lung Cancer Screening Recommendation (2021)",
    },
    "colorectal": {
        "document_id": "USPSTF-CRC-2021",
        "document_name": "USPSTF Colorectal Cancer Screening Recommendation (2021)",
    },
}

# Default Active Chunking Settings
CHUNK_SIZE = 850
CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Benchmark Experiment Configurations (A, B, C, D, E)
CHUNK_CONFIGURATIONS = [
    {"name": "A", "chunk_size": 850, "chunk_overlap": 150, "description": "Baseline (850/150)"},
    {"name": "B", "chunk_size": 450, "chunk_overlap": 50, "description": "Small Granular (450/50)"},
    {"name": "C", "chunk_size": 600, "chunk_overlap": 100, "description": "Medium-Low (600/100)"},
    {"name": "D", "chunk_size": 700, "chunk_overlap": 100, "description": "Medium-High (700/100)"},
    {"name": "E", "chunk_size": 1000, "chunk_overlap": 150, "description": "Large Context (1000/150)"},
]

# Embedding & VectorStore Settings
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "clinical_cancer_screening"
VECTORSEARCH_K = 4

# LLM Settings
LLM_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
TEMPERATURE = 0.0
