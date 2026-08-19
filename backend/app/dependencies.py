import sys
from pathlib import Path
from fastapi import Request, HTTPException, status

# Ensure root repository is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline import ClinicalRAGPipeline


def get_pipeline(request: Request) -> ClinicalRAGPipeline:
    """
    Dependency that returns the singleton ClinicalRAGPipeline instance
    stored in app.state.pipeline during application startup.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Clinical RAG Pipeline is not initialized or unavailable."
        )
    return pipeline
