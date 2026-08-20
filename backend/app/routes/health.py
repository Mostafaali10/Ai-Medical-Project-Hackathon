import os
from fastapi import APIRouter, Request, Depends
from backend.app.schemas import HealthResponse
from backend.app.dependencies import get_pipeline
from src.config import COLLECTION_NAME, LLM_MODEL
from src.pipeline import ClinicalRAGPipeline

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="System Health Check")
@router.get("/api/health", response_model=HealthResponse, include_in_schema=False)
def check_health(pipeline: ClinicalRAGPipeline = Depends(get_pipeline)) -> HealthResponse:
    """
    Returns the real-time operational status of the Clinical RAG system,
    including the persistent vectorstore chunk count and LLM status.
    """
    vectorstore_status = "ready"
    chunk_count = 0
    try:
        if pipeline.vectorstore and hasattr(pipeline.vectorstore, "_collection"):
            chunk_count = pipeline.vectorstore._collection.count()
        else:
            vectorstore_status = "unavailable"
    except Exception:
        vectorstore_status = "error"

    if pipeline.llm is not None:
        model_name = getattr(pipeline.llm, "model_name", None) or getattr(pipeline.llm, "model", None) or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        llm_status = f"live ({model_name})"
    else:
        llm_status = "simulation_mode (no GROQ_API_KEY)"

    return HealthResponse(
        status="healthy",
        pipeline="ready",
        vectorstore=vectorstore_status,
        llm=llm_status,
        collection_name=COLLECTION_NAME,
        indexed_chunks=chunk_count,
    )
