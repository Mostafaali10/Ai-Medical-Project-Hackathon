import logging
from fastapi import APIRouter, Depends, HTTPException, status
from backend.app.schemas import AskRequest, AskResponse
from backend.app.dependencies import get_pipeline
from src.pipeline import ClinicalRAGPipeline

logger = logging.getLogger("clinical_rag_api")
router = APIRouter(prefix="/api", tags=["RAG"])


@router.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit Clinical Question to RAG Pipeline"
)
def ask_question(
    request: AskRequest,
    pipeline: ClinicalRAGPipeline = Depends(get_pipeline)
) -> AskResponse:
    """
    Submits a clinical question to the grounded Clinical RAG pipeline:
    1. Pre-retrieval safety gate (intercepts patient-specific diagnosis/prescription/dosage).
    2. Vector similarity retrieval with cosine relevance scoring.
    3. Post-retrieval lexical overlap & confidence threshold validation.
    4. Grounded answer generation using Groq LLM.
    5. Post-generation citation audit and JSON schema validation.
    """
    try:
        # Override k if specified in request and different from pipeline default
        if request.k is not None and request.k != pipeline.k:
            old_k = pipeline.k
            pipeline.k = request.k
            try:
                result = pipeline.ask(request.question)
            finally:
                pipeline.k = old_k
        else:
            result = pipeline.ask(request.question)

        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            schema_valid=result["schema_valid"],
            citation_report=result["citation_report"],
            chunks_used=result["chunks_used"],
        )
    except Exception as e:
        logger.error(f"Error executing Clinical RAG pipeline for query '{request.question}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the clinical question. Please try again."
        )
