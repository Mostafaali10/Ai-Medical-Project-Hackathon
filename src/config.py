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

# Chunking Settings
CHUNK_SIZE = 850
CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Embedding & VectorStore Settings
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "clinical_cancer_screening"
VECTORSEARCH_K = 4

# LLM Settings
LLM_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.0
