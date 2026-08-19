from typing import List, Union, Optional
from pydantic import BaseModel, Field, field_validator


# ============================================================
# Request Schemas
# ============================================================

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        description="The clinical question to evaluate against indexed medical guidelines.",
        examples=["What are the recommendations for lung cancer screening?"]
    )
    k: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of evidence chunks to retrieve (between 1 and 10)."
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Clinical question must not be empty or whitespace only.")
        v_stripped = v.strip()
        if len(v_stripped) > 1000:
            raise ValueError("Clinical question cannot exceed 1000 characters.")
        return v_stripped


# ============================================================
# Response Schemas (Matching schema/response_schema.json)
# ============================================================

class SupportingEvidenceItem(BaseModel):
    claim: str = Field(..., description="One supported evidence claim in plain language.")
    citations: List[str] = Field(default_factory=list, description="Exact bracketed citation tags backing this claim.")


class CitationItem(BaseModel):
    document: str = Field(..., description="Canonical document title.")
    section: str = Field(..., description="Section heading from the source text.")
    page: Union[int, str] = Field(..., description="Page number of the citation.")
    chunk_id: str = Field(..., description="Deterministic chunk identifier.")


class AnswerObject(BaseModel):
    status: str = Field(..., description="Outcome status: answered, insufficient_evidence, or safety_refusal.")
    recommendation: str = Field(..., description="Direct clinical answer or refusal explanation.")
    supporting_evidence: List[SupportingEvidenceItem] = Field(default_factory=list)
    citations: List[CitationItem] = Field(default_factory=list)
    confidence: str = Field(..., description="Evidence confidence: High, Medium, Low, or Insufficient Evidence.")
    missing_information: List[str] = Field(default_factory=list)
    safety_note: str = Field(..., description="Standard medical decision support disclaimer.")


class CitationReport(BaseModel):
    citations_used: List[str] = Field(default_factory=list)
    invented_citations: List[str] = Field(default_factory=list)
    claims_missing_citation: List[str] = Field(default_factory=list)
    valid: bool = Field(..., description="True if all citations are grounded and no claims lack citations.")


class ChunkUsed(BaseModel):
    rank: int = Field(..., description="Retrieval ranking (1-indexed).")
    document_id: str = Field(..., description="Canonical document identifier.")
    document_name: str = Field(..., description="Name of the source document.")
    section: str = Field(..., description="Identified section heading.")
    page_number: Union[int, str] = Field(..., description="Page number in the PDF.")
    chunk_id: str = Field(..., description="Unique deterministic chunk ID.")
    similarity_score: float = Field(..., description="Cosine similarity score.")
    text: str = Field(..., description="Full text content of the retrieved chunk.")


class AskResponse(BaseModel):
    question: str = Field(..., description="The original clinical query submitted.")
    answer: AnswerObject = Field(..., description="Structured grounded clinical answer.")
    schema_valid: bool = Field(..., description="Whether the answer validated against response_schema.json.")
    citation_report: CitationReport = Field(..., description="Post-generation citation validation audit report.")
    chunks_used: List[ChunkUsed] = Field(default_factory=list, description="All chunks retrieved during vector search.")


# ============================================================
# Health & Document Metadata Schemas
# ============================================================

class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall system health status.")
    pipeline: str = Field(..., description="Status of the ClinicalRAGPipeline instance.")
    vectorstore: str = Field(..., description="Status of the persistent ChromaDB index.")
    llm: str = Field(..., description="Status of the LLM client (live vs simulation mode).")
    collection_name: str = Field(..., description="Active Chroma collection name.")
    indexed_chunks: int = Field(..., description="Number of indexed chunks in the vector database.")


class DocumentMetadata(BaseModel):
    document_id: str = Field(..., description="Canonical document ID.")
    document_name: str = Field(..., description="Document title.")
    source: str = Field(..., description="Source institution or publisher.")
    document_type: str = Field(..., description="Classification type of the document.")
    pages: int = Field(..., description="Total pages in the source document.")
    chunks: int = Field(..., description="Total indexed chunks from this document.")
    source_file: str = Field(..., description="Filename of the source PDF.")


class DocumentsResponse(BaseModel):
    documents: List[DocumentMetadata] = Field(default_factory=list)
    collection_name: str = Field(..., description="Active ChromaDB collection name.")
    total_chunks: int = Field(..., description="Total number of chunks across all documents.")
