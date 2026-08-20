import os
from pathlib import Path
from dotenv import load_dotenv

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Automatically load environment variables from project root .env
load_dotenv(BASE_DIR / ".env")


# ============================================================
# Document Metadata
# ============================================================
#
# Your current data folder contains:
#
# Non-Small Cell Lung Cancer Treatment (PDQ#U00ae) - NCI.pdf
#
# This replaces the old USPSTF document configuration.
#

DOCUMENT_METADATA_MAP = {
    "non-small cell lung cancer treatment": {
        "document_id": "NCI-NSCLC-PDQ",
        "document_name": "NCI Non-Small Cell Lung Cancer Treatment (PDQ)",
        "source": "National Cancer Institute",
        "document_type": "Health Professional PDQ",
    },
}


# ============================================================
# Chunking
# ============================================================

CHUNK_SIZE = 850
CHUNK_OVERLAP = 150

CHUNK_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]


# ============================================================
# Chunking Experiments
# ============================================================

CHUNK_CONFIGURATIONS = [
    {
        "name": "A",
        "chunk_size": 850,
        "chunk_overlap": 150,
        "description": "Baseline (850/150)",
    },
    {
        "name": "B",
        "chunk_size": 450,
        "chunk_overlap": 50,
        "description": "Small Granular (450/50)",
    },
    {
        "name": "C",
        "chunk_size": 600,
        "chunk_overlap": 100,
        "description": "Medium-Low (600/100)",
    },
    {
        "name": "D",
        "chunk_size": 700,
        "chunk_overlap": 100,
        "description": "Medium-High (700/100)",
    },
    {
        "name": "E",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "description": "Large Context (1000/150)",
    },
]


# ============================================================
# Embeddings / Vector Store
# ============================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# New collection name so you don't accidentally reuse
# an old USPSTF collection.
COLLECTION_NAME = "nci_lung_cancer_rag"

# Retrieve more candidates first.
# We will perform a second relevance-ranking step afterward.
VECTORSEARCH_K = 5

# Number of final chunks given to the LLM.
FINAL_CONTEXT_K = 5


# ============================================================
# Retrieval / Relevance
# ============================================================

# Minimum vector similarity required before considering
# the retrieved evidence useful.
DEFAULT_SIMILARITY_THRESHOLD = 0.30

# Minimum relevance score after our combined ranking.
DEFAULT_RELEVANCE_THRESHOLD = 0.25


# ============================================================
# LLM & Environment Resolvers
# ============================================================

def get_groq_api_key() -> str | None:
    """
    Robust resolver for GROQ_API_KEY from environment variables.
    Handles case-insensitivity, whitespace in key names/values, and surrounding quotes.
    """
    # 1. Exact match
    raw = os.environ.get("GROQ_API_KEY")
    if raw:
        v = raw.strip().strip("'\"")
        if v:
            return v

    # 2. Case-insensitive / whitespace-tolerant search
    for k, val in os.environ.items():
        k_norm = k.strip().upper()
        if k_norm in ("GROQ_API_KEY", "GROQ_APIKEY", "GROQ_KEY", "GROQAPI_KEY", "GROQ_SECRET", "GROQ_TOKEN"):
            v = val.strip().strip("'\"")
            if v:
                return v
        if "GROQ" in k_norm and ("KEY" in k_norm or "TOKEN" in k_norm or "SECRET" in k_norm):
            v = val.strip().strip("'\"")
            if v:
                return v

    return None


def get_groq_model() -> str:
    """
    Robust resolver for GROQ_MODEL from environment variables.
    """
    raw = os.environ.get("GROQ_MODEL")
    if raw:
        v = raw.strip().strip("'\"")
        if v:
            return v

    for k, val in os.environ.items():
        k_norm = k.strip().upper()
        if k_norm in ("GROQ_MODEL", "GROQMODEL", "GROQ_MODEL_NAME"):
            v = val.strip().strip("'\"")
            if v:
                return v

    return "openai/gpt-oss-120b"


LLM_MODEL = get_groq_model()

TEMPERATURE = 0.0